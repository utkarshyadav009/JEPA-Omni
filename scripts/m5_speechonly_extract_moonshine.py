"""scripts/m5_speechonly_extract_moonshine.py -- task 150: re-extract the
speech-only decision head's cached features using Moonshine instead of
Whisper-medium. Only SpeechOnlyThreeClassHead is retrained here (models/
m4_decision_head.py's ThreeClassHead / the M3/M4b deep-grounding path is
NOT touched -- confirmed by reading scripts/bmo_jetson_startup.py that the
currently-deployed production stack only loads SpeechOnlyThreeClassHead,
never wires up DuplexLoop/AsyncThinker/the M4b projector at all, so this
is the actual load-bearing piece worth retraining first).

Simplified from scripts/m5_bothpresent_extract_v2.py: SpeechOnlyThreeClassHead
takes ONLY speech_feat, no World-State -- skips loading VisionEncoder/
WavJEPA/M2 predictor entirely (not needed, faster extraction).

SAME tick sampling (same seeds, same per-class caps) as the original
Whisper-based extraction, so this is a clean apples-to-apples comparison --
only the encoder changed.

Usage:
    python scripts/m5_speechonly_extract_moonshine.py --split train --cap 800
    python scripts/m5_speechonly_extract_moonshine.py --split test --n-per-class 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoProcessor, MoonshineModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.m4_easycom_turntaking import build_ticks
from data.m4_speech_dataset import WHISPER_SR  # 16000, reused for Moonshine too (same native rate)

MOONSHINE_SR = 16000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--cap", type=int, default=800)
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--moonshine-model", default="UsefulSensors/moonshine-base")
    p.add_argument("--out-dir", default="checkpoints/m4_decision_head_3class_speechonly_moonshine")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract-moonshine-{args.split}] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    train_ticks, test_ticks = build_ticks()
    ticks = train_ticks if args.split == "train" else test_ticks

    by_label = {"speak": [], "silence": [], "backchannel": []}
    for t in ticks:
        by_label[t.label3].append(t)

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
    print(f"[extract-moonshine-{args.split}] sampling {len(sample)} ticks (same seed/caps as the Whisper extraction)", flush=True)

    print(f"[extract-moonshine-{args.split}] loading {args.moonshine_model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.moonshine_model)
    model = MoonshineModel.from_pretrained(args.moonshine_model, dtype=torch.bfloat16).to(device).eval()
    hidden_size = model.config.hidden_size

    cache_path = os.path.join(args.out_dir, f"{args.split}_speechonly_moonshine_cache.pt")
    cache_batch = []
    if os.path.isfile(cache_path):
        cache_batch = torch.load(cache_path, weights_only=False)
        print(f"[extract-moonshine-{args.split}] RESUMING: {len(cache_batch)} ticks already cached", flush=True)

    import soundfile as sf_io
    import librosa

    t_start = time.time()
    for i, t in enumerate(sample):
        if i < len(cache_batch):
            continue
        try:
            audio, sr = sf_io.read(t.audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            i0, i1 = max(0, int(t.start_sec * sr)), min(len(audio), int(t.end_sec * sr))
            clip = audio[i0:i1]
            if clip.size == 0:
                clip = np.zeros(int(0.02 * sr), dtype=np.float32)
            if sr != MOONSHINE_SR:
                clip = librosa.resample(clip, orig_sr=sr, target_sr=MOONSHINE_SR)
                sr = MOONSHINE_SR

            with torch.no_grad():
                inputs = processor(clip, sampling_rate=sr, return_tensors="pt")
                input_values = inputs.input_values.to(device, dtype=torch.bfloat16)
                enc_out = model.encoder(input_values)
                sf_feat = enc_out.last_hidden_state[0].float().mean(dim=0).cpu()  # mean-pool over time, same convention as the Whisper extraction
        except Exception as e:
            print(f"[extract-moonshine-{args.split}] tick {i} FAILED ({t.audio_path} @ {t.start_sec:.1f}-{t.end_sec:.1f}s): {e!r}", flush=True)
            continue

        cache_batch.append({"sf": sf_feat, "label3": t.label3, "session": t.session,
                             "audio_path": t.audio_path, "start_sec": t.start_sec, "end_sec": t.end_sec})
        if (i + 1) % 50 == 0 or i == len(sample) - 1:
            torch.save(cache_batch, cache_path)
            elapsed = time.time() - t_start
            print(f"[extract-moonshine-{args.split}] {len(cache_batch)}/{len(sample)} cached, "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    torch.save(cache_batch, cache_path)
    label_counts = {}
    for c in cache_batch:
        label_counts[c["label3"]] = label_counts.get(c["label3"], 0) + 1
    print(f"[extract-moonshine-{args.split}] DONE. n={len(cache_batch)} hidden_size={hidden_size} "
          f"label_counts={label_counts} wrote {cache_path}", flush=True)


if __name__ == "__main__":
    main()
