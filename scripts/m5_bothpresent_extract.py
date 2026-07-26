"""scripts/m5_bothpresent_extract.py — A1 step 1: extract REAL
both-modalities-present features (World-State from real decoded
Video_Compressed footage + real Whisper speech-feat) for EasyCom
train/test ticks, to retrain the 3-class decision head on the actual
streaming-loop input regime (both real, not one-zeroed).

Caches incrementally (torch.save after every tick, resumable) --
lesson learned from the first m5_ood_falsifier.py run that was killed by
an external `timeout` with zero saved progress.

Train cache: capped at --train-cap per class (backchannel is the natural
ceiling at 682 matched ticks; speak/silence capped down to match order of
magnitude rather than left at their full ~2600-2700, which would just add
wall-clock without changing what the retrain is testing).
Test cache: EXACTLY matches m5_falsifier_bothpresent.py's eval sampling
(same seed, same n-per-class) so extraction and evaluation are
guaranteed consistent -- no separate sampling logic to drift apart.

Usage:
    python scripts/m5_bothpresent_extract.py --split train --cap 800
    python scripts/m5_bothpresent_extract.py --split test --n-per-class 100
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m4_speech import WhisperSpeechEncoder
from data.m4_easycom_turntaking import build_ticks
from data.m4_speech_dataset import EASYCOM_ROOT, WHISPER_SR

from scripts.m5_ood_falsifier import decode_video_window, chunk_of

SEED_TEST = 11  # must match m5_ood_falsifier.py's default seed


def gather_matched(ticks):
    by_label = {"speak": [], "silence": [], "backchannel": []}
    for t in ticks:
        chunk = chunk_of(t.audio_path)
        video_path = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{t.session}", chunk + ".mp4")
        if os.path.isfile(video_path):
            by_label[t.label3].append((t, video_path))
    return by_label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--cap", type=int, default=800, help="train: max ticks per class")
    p.add_argument("--n-per-class", type=int, default=100, help="test: exact ticks per class (matches eval)")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--out-dir", default="checkpoints/m4_decision_head_3class_bothpresent")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract-{args.split}] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    train_ticks, test_ticks = build_ticks()
    ticks = train_ticks if args.split == "train" else test_ticks
    by_label = gather_matched(ticks)
    counts = {k: len(v) for k, v in by_label.items()}
    print(f"[extract-{args.split}] matched-video pool: {counts}", flush=True)

    import random
    if args.split == "train":
        rng = random.Random(0)
        sample = []
        for lbl in ["speak", "silence", "backchannel"]:
            pool = by_label[lbl][:]
            rng.shuffle(pool)
            sample.extend(pool[:args.cap])
    else:
        rng = random.Random(SEED_TEST)
        sample = []
        for lbl in ["speak", "silence", "backchannel"]:
            pool = by_label[lbl][:]
            rng.shuffle(pool)
            sample.extend(pool[:args.n_per_class])
    rng.shuffle(sample)
    print(f"[extract-{args.split}] sampling {len(sample)} ticks "
          f"({ {lbl: min(len(by_label[lbl]), (args.cap if args.split=='train' else args.n_per_class)) for lbl in by_label} })", flush=True)

    print(f"[extract-{args.split}] loading real V-JEPA2 ViT-L, M2 predictor, Whisper...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    cache_path = os.path.join(args.out_dir, f"{args.split}_bothpresent_cache.pt")
    cache_batch = []
    if os.path.isfile(cache_path):
        cache_batch = torch.load(cache_path, weights_only=False)
        print(f"[extract-{args.split}] RESUMING: {len(cache_batch)} ticks already cached", flush=True)

    t_start = time.time()
    for i, (t, video_path) in enumerate(sample):
        if i < len(cache_batch):
            continue
        center = (t.start_sec + t.end_sec) / 2.0
        try:
            frames = decode_video_window(video_path, center, device)
            with torch.no_grad():
                v = vision_enc.encode(frames.unsqueeze(0))
                n_tok = v.shape[1]
                bin_idx = torch.linspace(0, predictor_cfg.max_tdm_bins - 1, n_tok, device=device).round().long()
                feats = {"vision": v.float()}
                tbins = {"vision": bin_idx.unsqueeze(0)}
                ws = predictor.encode_world_state(feats, tbins)[0].cpu()

                import soundfile as sf_io
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
            print(f"[extract-{args.split}] tick {i} FAILED ({video_path} @ {center:.1f}s): {e}", flush=True)
            continue

        cache_batch.append({"ws": ws, "sf": sf_feat, "label3": t.label3, "session": t.session,
                             "video_path": video_path, "center_sec": center})
        if (i + 1) % 20 == 0 or i == len(sample) - 1:
            torch.save(cache_batch, cache_path)
            elapsed = time.time() - t_start
            rate = (i + 1 - len(cache_batch) + len(cache_batch)) / max(1e-9, elapsed)
            print(f"[extract-{args.split}] {len(cache_batch)}/{len(sample)} cached, "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    torch.save(cache_batch, cache_path)
    label_counts = {}
    for c in cache_batch:
        label_counts[c["label3"]] = label_counts.get(c["label3"], 0) + 1
    print(f"[extract-{args.split}] DONE. n={len(cache_batch)} label_counts={label_counts} "
          f"wrote {cache_path}", flush=True)


if __name__ == "__main__":
    main()
