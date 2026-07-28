"""scripts/audioset_av_relevance_filter.py — P4: AV-relevance filter for
AudioSet-Strong, SAME methodology as scripts/ego4d_av_relevance_filter.py
(imports its SPEECH_IDX/NOISE_LIKE_IDX constants directly, not
reimplemented) so retention rates are directly comparable across corpora:
Silero VAD (speech fraction), MIT AST tagger (top non-speech/non-noise
event confidence), energy-dynamics CoV -- combined score = top_event_prob
* (1 - vad_speech_frac) * energy_cov_norm.

AudioSet-Strong is flat (35,247 already-downloaded whole short clips, no
per-file windowing needed unlike Ego4D's long-video candidate generation)
-- each file is scored once, whole-clip.

Usage:
    python scripts/audioset_av_relevance_filter.py --top-n 15000
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

from scripts.ego4d_av_relevance_filter import SPEECH_IDX, NOISE_LIKE_IDX, energy_cov

CLIPS_DIR = "/home/utkarsh/raid2-data/audioset_mp4/strong"
AUDIO_SR = 16000
FLOOR = 0.10
VAD_SPEECH_EXCLUDE = 0.84


def extract_audio(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    import soundfile as sf_io
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    assert sr == AUDIO_SR
    if audio.shape[0] < AUDIO_SR // 2:  # too short to be meaningful (<0.5s)
        raise ValueError(f"audio too short: {audio.shape[0]} samples")
    return audio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=15000)
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/audioset_av_filter_scores.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audioset-filter] device={device}", flush=True)

    print("[audioset-filter] loading Silero VAD + AST audio tagger (both independent of M2)...", flush=True)
    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = vad_utils

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_fe = AutoFeatureExtractor.from_pretrained(ast_name)
    ast_model = AutoModelForAudioClassification.from_pretrained(ast_name).to(device).eval()
    id2label = ast_model.config.id2label

    files = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))
    print(f"[audioset-filter] {len(files)} candidate files", flush=True)

    cache_path = args.out.replace(".json", "_scores_incremental.pt")
    scored = []
    if os.path.isfile(cache_path):
        scored = torch.load(cache_path, weights_only=False)
        print(f"[audioset-filter] RESUMING: {len(scored)} candidates already scored", flush=True)

    t_start = time.time()
    for i, path in enumerate(files):
        if i < len(scored):
            continue
        try:
            audio = extract_audio(path)
            wav_t = torch.from_numpy(audio)
            speech_ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=AUDIO_SR)
            speech_samples = sum(t["end"] - t["start"] for t in speech_ts)
            vad_speech_frac = speech_samples / audio.shape[0]

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

            cov = energy_cov(audio)
            cov_norm = min(1.0, cov / 2.0)

            score = top_event_prob * (1.0 - vad_speech_frac) * cov_norm
            scored.append({
                "path": path, "score": score, "top_event_idx": top_event_idx,
                "top_event_label": id2label[top_event_idx], "top_event_prob": top_event_prob,
                "vad_speech_frac": vad_speech_frac, "speech_prob": speech_prob, "energy_cov": cov,
            })
        except Exception as e:
            print(f"[audioset-filter] candidate {i} FAILED ({path}): {e!r}", flush=True)
            continue

        if (i + 1) % 500 == 0 or i == len(files) - 1:
            torch.save(scored, cache_path)
            elapsed = time.time() - t_start
            print(f"[audioset-filter] {len(scored)}/{len(files)} scored, elapsed={elapsed/60:.1f}min", flush=True)

    torch.save(scored, cache_path)

    # base exclusion mask: floor + vad_speech + (top_event is itself a noise-like/speech idx already excluded from top_event selection)
    kept_pool = [c for c in scored if c["top_event_prob"] >= FLOOR and c["vad_speech_frac"] < VAD_SPEECH_EXCLUDE]
    dropped_pool = [c for c in scored if c not in kept_pool]

    kept_pool.sort(key=lambda x: -x["score"])
    kept = kept_pool[:args.top_n]
    dropped = kept_pool[args.top_n:] + dropped_pool

    from collections import Counter
    top20 = Counter(c["top_event_label"] for c in kept).most_common(20)

    result = {
        "n_candidates": len(scored),
        "n_after_floor_and_vad_excl": len(kept_pool),
        "top_n": args.top_n,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "retention_rate_of_scored": len(kept) / max(1, len(scored)),
        "retention_rate_of_floor_passed": len(kept) / max(1, len(kept_pool)),
        "top20_tag_histogram": top20,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out.replace(".json", "_kept.json"), "w") as f:
        json.dump(kept, f, indent=2)
    with open(args.out.replace(".json", "_dropped.json"), "w") as f:
        json.dump(dropped, f, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    print(f"[audioset-filter] wrote {args.out} + _kept.json + _dropped.json", flush=True)


if __name__ == "__main__":
    main()
