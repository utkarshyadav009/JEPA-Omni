"""scripts/ego4d_av_relevance_filter.py — item 3: score Ego4D candidate
10s windows for AV congruency using signals INDEPENDENT of M2 (per
instruction: scoring with the current M2 model would select for what it
already believes and bias the corpus toward what it least needs).

Three independent signals:
  1. Silero VAD (torch.hub, snakers4/silero-vad) -- wearer-speech-dominance
     fraction. LOW is good (we want non-speech-dominant segments).
  2. MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer,
     527-class AudioSet tagger) -- confidence of the best NON-speech,
     NON-noise-like event class. HIGH is good (a confident, describable
     event, not silence/hum/static).
  3. Energy dynamics -- coefficient of variation of short-time RMS energy.
     HIGH is good (a discrete event has an onset/transient; constant hum
     or handling noise is flat).

Congruency polarity is the OPPOSITE of the EasyCom turn-taking filter
(that one wants speech; this one wants non-speech events with a plausible
visible cause) -- do not reuse that filter or its thresholds.

score = top_nonspeech_event_prob * (1 - vad_speech_fraction) * energy_cov_norm

Candidate generation: per source file, up to --per-file-cap windows,
EVENLY spread (not consecutive) across the file's available 10s slots --
guarantees full 1,808-file representation in the pool that gets scored,
independent of how the ranking cut lands.

Usage:
    python scripts/ego4d_av_relevance_filter.py --per-file-cap 50 --top-n 45000
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EGO4D_DIRS = {
    "video_540ss": "/mnt/Raid-Storage-2/utkarsh-data/ego4d_probe/v2/video_540ss",
    "clips": "/mnt/Raid-Storage-2/utkarsh-data/ego4d_probe/v2/clips",
}
WINDOW_SEC = 10.0
AUDIO_SR = 16000

SPEECH_IDX = [0, 1, 2, 3, 7]                      # Speech, Male/Female/Child speech, Speech synthesizer
NOISE_LIKE_IDX = [70, 285, 496, 500, 513, 514, 515, 516, 520, 521]
# Hubbub/speech-babble, mic wind noise, Hum, Silence, Noise, Environmental
# noise, Static, Mains hum, White noise, Pink noise -- excluded from
# "describable event" credit (not speech, but not a nameable causal event
# either).


def get_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    return float(out)


def has_audio_stream(path: str) -> bool:
    """34% of video_540ss (420/1236) have NO audio track at all -- video-only
    egocentric recordings. Confirmed via ffprobe -select_streams a; these
    must be excluded before candidate generation, not discovered as a
    per-candidate extraction failure."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    return len(out) > 0


def uniform_window_starts(n_windows: int, cap: int) -> list:
    """Evenly-spaced window indices (no consecutive runs) -- same
    linspace-with-rounding pattern used elsewhere in this project
    (data/video_text_dataset.py's _uniform_frame_indices)."""
    if n_windows <= cap:
        return list(range(n_windows))
    idx = np.linspace(0, n_windows - 1, cap)
    return sorted(set(int(round(i)) for i in idx))


def extract_audio(path: str, start_sec: float, device="cpu") -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_sec:.3f}", "-t", f"{WINDOW_SEC}",
           "-i", path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    import soundfile as sf_io
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    assert sr == AUDIO_SR
    if audio.shape[0] < AUDIO_SR:  # too short (near EOF) -- pad
        audio = np.pad(audio, (0, AUDIO_SR - audio.shape[0]))
    return audio


def energy_cov(audio: np.ndarray, frame_len=1600, hop=800) -> float:
    """Coefficient of variation of short-time RMS energy -- high = eventful
    (onsets/transients), low = flat/constant (hum, handling noise)."""
    n = audio.shape[0]
    frames = [audio[i:i + frame_len] for i in range(0, n - frame_len, hop)]
    if len(frames) < 2:
        return 0.0
    rms = np.array([np.sqrt(np.mean(f.astype(np.float64) ** 2) + 1e-12) for f in frames])
    mean = rms.mean()
    if mean < 1e-6:
        return 0.0
    return float(rms.std() / mean)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-file-cap", type=int, default=50)
    p.add_argument("--top-n", type=int, default=45000)
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/ego4d_av_filter_scores.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[filter] device={device}", flush=True)

    print("[filter] loading Silero VAD + AST audio tagger (both independent of M2)...", flush=True)
    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = vad_utils

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_fe = AutoFeatureExtractor.from_pretrained(ast_name)
    ast_model = AutoModelForAudioClassification.from_pretrained(ast_name).to(device).eval()

    print("[filter] building candidate pool (per-file cap, evenly-spread, audio-having files only)...", flush=True)
    candidates = []
    n_no_audio = 0
    for source, d in EGO4D_DIRS.items():
        files = sorted(glob.glob(os.path.join(d, "*.mp4")))
        for f in files:
            try:
                dur = get_duration(f)
            except Exception:
                continue
            if not has_audio_stream(f):
                n_no_audio += 1
                continue
            n_windows = int(dur // WINDOW_SEC)
            if n_windows < 1:
                continue
            idxs = uniform_window_starts(n_windows, args.per_file_cap)
            # source_id = the video file itself -- used downstream to keep
            # windows from the same source video out of the same training
            # batch's negative set (item 4: same-source false negatives).
            source_id = os.path.splitext(os.path.basename(f))[0]
            for i in idxs:
                candidates.append({"source": source, "source_id": source_id,
                                    "path": f, "start_sec": i * WINDOW_SEC})
    print(f"[filter] candidate pool: {len(candidates)} windows across "
          f"{len(set(c['path'] for c in candidates))} files "
          f"({n_no_audio} files skipped -- no audio stream)", flush=True)

    cache_path = args.out.replace(".json", "_scores_incremental.pt")
    scored = []
    if os.path.isfile(cache_path):
        scored = torch.load(cache_path, weights_only=False)
        print(f"[filter] RESUMING: {len(scored)} candidates already scored", flush=True)

    t_start = time.time()
    for i, c in enumerate(candidates):
        if i < len(scored):
            continue
        try:
            audio = extract_audio(c["path"], c["start_sec"])
            # VAD
            wav_t = torch.from_numpy(audio)
            speech_ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=AUDIO_SR)
            speech_samples = sum(t["end"] - t["start"] for t in speech_ts)
            vad_speech_frac = speech_samples / audio.shape[0]
            # AST tagger
            inputs = ast_fe(audio, sampling_rate=AUDIO_SR, return_tensors="pt")
            with torch.no_grad():
                logits = ast_model(**{k: v.to(device) for k, v in inputs.items()}).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            mask = np.ones(probs.shape[0], dtype=bool)
            mask[SPEECH_IDX] = False
            mask[NOISE_LIKE_IDX] = False
            top_event_idx = int(np.argmax(probs * mask))
            top_event_prob = float(probs[top_event_idx])
            speech_prob = float(probs[SPEECH_IDX].sum())
            # energy dynamics
            cov = energy_cov(audio)
            cov_norm = min(1.0, cov / 2.0)  # empirical normalization, cov~0-2 typical range

            score = top_event_prob * (1.0 - vad_speech_frac) * cov_norm
            scored.append({**c, "score": score, "top_event_idx": top_event_idx,
                            "top_event_prob": top_event_prob, "vad_speech_frac": vad_speech_frac,
                            "speech_prob": speech_prob, "energy_cov": cov})
        except Exception as e:
            print(f"[filter] candidate {i} FAILED ({c['path']} @ {c['start_sec']:.1f}s): {e}", flush=True)
            continue

        if (i + 1) % 200 == 0 or i == len(candidates) - 1:
            torch.save(scored, cache_path)
            elapsed = time.time() - t_start
            print(f"[filter] {len(scored)}/{len(candidates)} scored, elapsed={elapsed/60:.1f}min", flush=True)

    torch.save(scored, cache_path)

    scored.sort(key=lambda x: -x["score"])
    kept = scored[:args.top_n]
    dropped = scored[args.top_n:]
    retention = len(kept) / max(1, len(scored))

    from transformers import AutoModelForAudioClassification as _M
    id2label = ast_model.config.id2label

    result = {
        "n_candidates": len(scored),
        "top_n": args.top_n,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "retention_rate": retention,
        "kept_files_covered": len(set(c["path"] for c in kept)),
        "candidate_files_covered": len(set(c["path"] for c in scored)),
        "score_stats": {
            "kept_score_min": kept[-1]["score"] if kept else None,
            "kept_score_max": kept[0]["score"] if kept else None,
            "dropped_score_max": dropped[0]["score"] if dropped else None,
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out.replace(".json", "_kept.json"), "w") as f:
        json.dump(kept, f, indent=2)
    with open(args.out.replace(".json", "_dropped.json"), "w") as f:
        json.dump(dropped, f, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    print(f"[filter] wrote {args.out} + _kept.json + _dropped.json", flush=True)


if __name__ == "__main__":
    main()
