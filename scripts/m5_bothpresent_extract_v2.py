"""scripts/m5_bothpresent_extract_v2.py — Phase 2.2: re-extract A1's
World-States using the CORRECTED, verified world_state_builder (Phase 1.2
gate PASSED: mean cos=0.9985, min cos=0.9932, tbins exact match, vs the
cached-feature reference). Replaces scripts/m5_bothpresent_extract.py's
inline construction, which had the same three bugs as the streaming loop
(unpooled 8192 vision tokens, linspace tbins, no ambient/WavJEPA at all).

New path (checkpoints/m4_decision_head_3class_bothpresent_v2/) -- does
NOT overwrite the original v1 caches, so both remain on disk for direct
before/after comparison.

Same tick sampling as v1 (same seed, same session-disjoint split, same
per-class caps) -- feature CONSTRUCTION is the only variable.

Usage:
    python scripts/m5_bothpresent_extract_v2.py --split test --n-per-class 100
    python scripts/m5_bothpresent_extract_v2.py --split train --cap 800
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m4_speech import WhisperSpeechEncoder
from models.world_state_builder import build_world_state_features
from data.m4_easycom_turntaking import build_ticks
from data.m4_speech_dataset import EASYCOM_ROOT, VIDEO_FPS, WHISPER_SR

from scripts.m5_ood_falsifier import decode_video_window, chunk_of

WINDOW_SEC = 10.0
AUDIO_SR = 16000


def decode_audio_window(audio_path: str, center_sec: float, chunk_dur_sec: float = 60.0):
    """Decode the SAME 10s window as vision (same center, same clip), at
    16kHz mono, for WavJEPA -- separate from the Whisper-facing
    speech-activity slice, which stays as-is."""
    import soundfile as sf_io
    import librosa
    import numpy as np

    audio, sr = sf_io.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
        sr = AUDIO_SR
    t0 = max(0.0, center_sec - WINDOW_SEC / 2)
    t1 = min(chunk_dur_sec, t0 + WINDOW_SEC)
    t0 = max(0.0, t1 - WINDOW_SEC)
    i0, i1 = int(t0 * sr), int(t1 * sr)
    clip = audio[i0:i1]
    if clip.size == 0:
        clip = np.zeros(int(WINDOW_SEC * sr), dtype=np.float32)
    return torch.from_numpy(clip.astype("float32")), clip.shape[0] / sr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--cap", type=int, default=800)
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--out-dir", default="checkpoints/m4_decision_head_3class_bothpresent_v2")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract-v2-{args.split}] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    train_ticks, test_ticks = build_ticks()
    ticks = train_ticks if args.split == "train" else test_ticks

    by_label = {"speak": [], "silence": [], "backchannel": []}
    for t in ticks:
        chunk = chunk_of(t.audio_path)
        video_path = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{t.session}", chunk + ".mp4")
        if os.path.isfile(video_path):
            by_label[t.label3].append((t, video_path))

    import random
    if args.split == "train":
        rng = random.Random(0)
        sample = []
        for lbl in ["speak", "silence", "backchannel"]:
            pool = by_label[lbl][:]
            rng.shuffle(pool)
            sample.extend(pool[:args.cap])
    else:
        rng = random.Random(11)
        sample = []
        for lbl in ["speak", "silence", "backchannel"]:
            pool = by_label[lbl][:]
            rng.shuffle(pool)
            sample.extend(pool[:args.n_per_class])
    rng.shuffle(sample)
    print(f"[extract-v2-{args.split}] sampling {len(sample)} ticks (same seed/caps as v1)", flush=True)

    print(f"[extract-v2-{args.split}] loading real encoders (ViT-L, WavJEPA-base, WavJEPA-nat, M2, Whisper)...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    cache_path = os.path.join(args.out_dir, f"{args.split}_bothpresent_v2_cache.pt")
    cache_batch = []
    if os.path.isfile(cache_path):
        cache_batch = torch.load(cache_path, weights_only=False)
        print(f"[extract-v2-{args.split}] RESUMING: {len(cache_batch)} ticks already cached", flush=True)

    t_start = time.time()
    for i, (t, video_path) in enumerate(sample):
        if i < len(cache_batch):
            continue
        center = (t.start_sec + t.end_sec) / 2.0
        try:
            frames = decode_video_window(video_path, center, device)  # (64,3,256,256) uint8
            ambient_audio, true_dur = decode_audio_window(t.audio_path, center)

            result = build_world_state_features(frames, ambient_audio, true_dur, vision_enc,
                                                 base_enc, nat_enc, predictor_cfg.max_tdm_bins, device)
            with torch.no_grad():
                ws = predictor.encode_world_state(result.feats, result.tbins)[0].cpu()

                import soundfile as sf_io
                import numpy as np
                audio, sr = sf_io.read(t.audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                i0, i1 = max(0, int(t.start_sec * sr)), min(len(audio), int(t.end_sec * sr))
                clip = audio[i0:i1]
                if clip.size == 0:
                    clip = np.zeros(int(0.02 * sr), dtype=np.float32)
                if sr != WHISPER_SR:
                    import librosa
                    clip = librosa.resample(clip, orig_sr=sr, target_sr=WHISPER_SR)
                    sr = WHISPER_SR
                dur = clip.shape[0] / sr
                hidden, valid_frames = whisper([clip.astype(np.float32)], [dur], device)
                vf = int(valid_frames[0].item())
                sf_feat = hidden[0, :vf].float().mean(dim=0).cpu()
        except Exception as e:
            print(f"[extract-v2-{args.split}] tick {i} FAILED ({video_path} @ {center:.1f}s): {e!r}", flush=True)
            continue

        cache_batch.append({"ws": ws, "sf": sf_feat, "label3": t.label3, "session": t.session,
                             "video_path": video_path, "center_sec": center})
        if (i + 1) % 20 == 0 or i == len(sample) - 1:
            torch.save(cache_batch, cache_path)
            elapsed = time.time() - t_start
            print(f"[extract-v2-{args.split}] {len(cache_batch)}/{len(sample)} cached, "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    torch.save(cache_batch, cache_path)
    label_counts = {}
    for c in cache_batch:
        label_counts[c["label3"]] = label_counts.get(c["label3"], 0) + 1
    print(f"[extract-v2-{args.split}] DONE. n={len(cache_batch)} label_counts={label_counts} "
          f"wrote {cache_path}", flush=True)


if __name__ == "__main__":
    main()
