"""scripts/audioset_decode_parallel_bench.py — P3: measure real decode+encode
throughput on AudioSet-Strong (35,254 short, variable-duration mp4 clips
already downloaded to /home/utkarsh/raid2-data/audioset_mp4/strong/) before
and after parallelizing across Mercury's 4 GPUs.

AudioSet-Strong clips are WHOLE short clips (1-10s+, not fixed-duration
windows into a longer video like VGGSound/Ego4D), so this reuses
scripts/extract_features_av.py's existing _decode_video_raw/_decode_audio_raw
(whole-clip decode, uniform 64-frame sampling across whatever the real
duration is) directly -- no new decode logic needed, unlike Ego4D's
windowed case.

Two conditions on the SAME 1000 real clips:
  BEFORE: single process, sequential decode + full encode (ViT-L +
    WavJEPA-base/nat), matching what a naive single-threaded extraction
    would do.
  AFTER: sharded across 4 GPUs (4 independent processes, matching
    scripts/extract_features_ego4d_train.py's already-proven pattern),
    each with its own decode (CPU, genuinely parallel across processes)
    and its own GPU for encode.

Usage:
    python scripts/audioset_decode_parallel_bench.py --mode before --n 1000
    python scripts/audioset_decode_parallel_bench.py --mode after --n 1000 --shard-idx 0 --num-shards 4
    (run the 4 "after" shards concurrently, one per GPU)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import (
    _decode_video_raw, _decode_audio_raw, _spatial_pool, _vision_ts, _audio_ts,
    VISION_DIM, VISION_SPAT, NUM_FRAMES, RESOLUTION, AUDIO_SR,
)
from models.audio_encoder import WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO

CLIPS_DIR = "/home/utkarsh/raid2-data/audioset_mp4/strong"
SAMPLE_SEED = 0
RESULTS_DIR = "checkpoints/vjepa21_shelved"


def sample_clips(n: int):
    all_files = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(all_files)
    return all_files[:n]


def run_shard(files, device, out_path):
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    n_ok, n_fail = 0, 0
    t_start = time.time()
    for i, path in enumerate(files):
        try:
            frames = _decode_video_raw(path, NUM_FRAMES, RESOLUTION)
            audio = _decode_audio_raw(path, AUDIO_SR)
            with torch.no_grad():
                raw = vision_enc.encode([frames])
            raw = raw[0]
            vis_full = raw.view(-1, 256, VISION_DIM)
            _ = _spatial_pool(vis_full).to(torch.bfloat16)
            wav1 = audio.unsqueeze(0).unsqueeze(0)
            wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0)
            with torch.no_grad():
                _ = base_enc.encode(wav1.to(device))
                _ = nat_enc.encode(wav2.to(device))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[bench] {path} FAILED: {e!r}", flush=True)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"[bench] {i+1}/{len(files)}  elapsed={elapsed:.1f}s  "
                  f"rate={((i+1)/elapsed):.3f} clips/s", flush=True)

    elapsed = time.time() - t_start
    result = {"n_attempted": len(files), "n_ok": n_ok, "n_fail": n_fail,
              "elapsed_s": elapsed, "clips_per_sec": len(files) / elapsed}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["before", "after"], required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sampled = sample_clips(args.n)

    if args.mode == "before":
        out_path = os.path.join(RESULTS_DIR, "AUDIOSET_DECODE_BENCH_BEFORE.json")
        run_shard(all_sampled, device, out_path)
    else:
        my_files = [f for i, f in enumerate(all_sampled) if i % args.num_shards == args.shard_idx]
        out_path = os.path.join(RESULTS_DIR, f"AUDIOSET_DECODE_BENCH_AFTER_shard{args.shard_idx}.json")
        run_shard(my_files, device, out_path)


if __name__ == "__main__":
    main()
