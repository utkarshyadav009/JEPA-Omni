"""
scripts/probe_latency.py

Measures a single V-JEPA 2 ViT-L forward pass: latency (ms), peak VRAM (GB), and the
token count N. THIS NUMBER GATES M5 (real-time streaming feasibility): if one encoder
forward at the target chunk rate is >> the chunk period, re-encoding a sliding window
every chunk is not real-time and a causal/cached variant or distillation is required.

Run on the H100 cluster first:
    python scripts/probe_latency.py --frames 64 --res 256 --iters 30
"""

from __future__ import annotations

import argparse
import time

import torch

from models.vision_encoder import VisionEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Run this on the GPU node."
    enc = VisionEncoder(args.repo)

    vid = (torch.rand(args.batch, args.frames, 3, args.res, args.res) * 255).to(torch.uint8)

    for _ in range(args.warmup):
        _ = enc.encode(vid)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(args.iters):
        start.record()
        out = enc.encode(vid)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("=" * 56)
    print(f" repo          : {args.repo}")
    print(f" input         : B={args.batch} T={args.frames} {args.res}x{args.res}")
    print(f" tokens N      : {out.shape[1]}  (x{out.shape[-1]} dim)")
    print(f" latency p50   : {p50:.1f} ms   p90: {p90:.1f} ms")
    print(f" peak VRAM     : {peak_gb:.2f} GB (encoder fwd only)")
    print("=" * 56)
    print(" M5 read: compare p50 to your chunk period (e.g. 100 ms).")
    print(" If p50 >> chunk period -> re-encode-per-chunk is NOT real-time.")


if __name__ == "__main__":
    main()
