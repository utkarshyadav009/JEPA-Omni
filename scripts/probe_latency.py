"""Probe encode latency / throughput of the M1 spine vision path.

Measures the wall-clock cost of ``SpineM1.embed_video`` on synthetic clips so
the vision encoder can be benchmarked without a dataset. Uses the same public
interface as training/eval (``embed_video`` takes a *list* of ``[T, C, H, W]``
uint8 clips).

Usage
-----
    python scripts/probe_latency.py --config configs/m1.yaml --batch-size 8 --iters 20
"""

from __future__ import annotations

import argparse
import time
from typing import List

import torch

from models import SpineConfig, SpineM1
from utils import load_config


def _synthetic_clips(batch_size: int, num_frames: int, resolution: int) -> List[torch.Tensor]:
    return [
        torch.randint(0, 256, (num_frames, 3, resolution, resolution), dtype=torch.uint8)
        for _ in range(batch_size)
    ]


@torch.no_grad()
def probe(config_path: str, batch_size: int, iters: int, warmup: int) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spine = SpineM1(SpineConfig(**cfg["model"])).to(device).eval()
    num_frames = int(cfg["num_frames"])
    resolution = int(cfg["resolution"])

    clips = _synthetic_clips(batch_size, num_frames, resolution)

    for _ in range(max(0, warmup)):
        spine.embed_video(clips)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        spine.embed_video(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times_t = torch.tensor(times)
    per_batch_ms = times_t.mean().item() * 1000.0
    per_clip_ms = per_batch_ms / batch_size
    clips_per_s = batch_size / times_t.mean().item()
    print(
        f"[probe] device={device} batch_size={batch_size} "
        f"num_frames={num_frames} resolution={resolution}"
    )
    print(
        f"[probe] per_batch={per_batch_ms:.2f} ms  per_clip={per_clip_ms:.2f} ms  "
        f"throughput={clips_per_s:.2f} clips/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe M1 vision encode latency.")
    parser.add_argument("--config", default="configs/m1.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    probe(args.config, args.batch_size, args.iters, args.warmup)


if __name__ == "__main__":
    main()
