"""scripts/ego4d_decode_parallel_bench.py — item 5b: decode is 88% of
per-clip extraction cost and is CPU-bound while the GPU idles (measured
in the VGGSound ViT-B throughput test: 0.630s decode vs 0.069s forward).
scripts/extract_features_av.py's own Phase 1 decodes ONE clip at a time
(a subprocess per clip, spawned and joined serially -- confirmed by
reading the source, this is for timeout-safety, not throughput). This
script measures serial vs multiprocessing-Pool-parallel decode throughput
on real Ego4D windows (video_540ss, the harder/slower long-form case) to
report windows/sec before and after, as requested.

Does NOT modify scripts/extract_features_av.py (the script that produced
the frozen, validated 764GiB VGGSound cache) -- this is a standalone
benchmark; the real Ego4D extraction script (built separately) will adopt
whichever pattern this benchmark shows is faster.

Usage:
    python scripts/ego4d_decode_parallel_bench.py --n 40 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

NUM_FRAMES = 64
RESOLUTION = 256
WINDOW_SEC = 10.0
AUDIO_SR = 16000


def decode_one(args) -> bool:
    path, start_sec = args
    try:
        from torchcodec.decoders import VideoDecoder, AudioDecoder
        from data.video_text_dataset import _uniform_frame_indices

        vdec = VideoDecoder(path, device="cpu")
        fps = 30.0  # Ego4D video_540ss native fps (confirmed via ffprobe: 30 tbr)
        f0 = int(round(start_sec * fps))
        f1 = int(round((start_sec + WINDOW_SEC) * fps))
        num_total = getattr(vdec.metadata, "num_frames", None) or len(vdec)
        f1 = min(f1, num_total)
        idx = _uniform_frame_indices(f1 - f0, NUM_FRAMES)
        abs_idx = sorted(set(i + f0 for i in idx))
        batch = vdec.get_frames_at(indices=abs_idx)
        decoded = batch.data
        remap = {o: i for i, o in enumerate(abs_idx)}
        gi = torch.tensor([remap[i + f0] for i in idx], dtype=torch.long)
        frames = decoded.index_select(0, gi)
        if frames.shape[-2] != RESOLUTION or frames.shape[-1] != RESOLUTION:
            x = frames.float()
            x = F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear",
                               align_corners=False, antialias=True)
            frames = x.round_().clamp_(0, 255).to(torch.uint8)

        # NOTE: torchcodec's AudioDecoder.get_all_samples() decodes the
        # WHOLE audio track before slicing -- fine for VGGSound/EasyCom's
        # short clips, but catastrophic for Ego4D's 10-60+ minute files (a
        # real, separate finding from the "parallelize decode" ask: the
        # naive per-window audio decode doesn't scale to long-form video
        # at all). Use ffmpeg seek-before-decode instead (matches
        # scripts/ego4d_av_relevance_filter.py's extract_audio, already
        # measured at ~0.06s regardless of source file length).
        import subprocess, io, soundfile as sf_io
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_sec:.3f}", "-t", f"{WINDOW_SEC}",
               "-i", path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True, timeout=30)
        _audio, _sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    audio_list_path = "/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad/audio_files_540ss.txt"
    if os.path.isfile(audio_list_path):
        files = [l.strip() for l in open(audio_list_path) if l.strip()]
    else:
        files = sorted(glob.glob("/mnt/Raid-Storage-2/utkarsh-data/ego4d_probe/v2/video_540ss/*.mp4"))
    candidates = [(f, 60.0) for f in files[: args.n]]

    print(f"[decode-bench] {len(candidates)} candidate windows, "
          f"video_540ss (long-form, the harder case)", flush=True)

    # ---- serial (current extract_features_av.py pattern: one at a time) ----
    t0 = time.time()
    n_ok_serial = sum(1 for c in candidates if decode_one(c))
    t1 = time.time()
    serial_wps = n_ok_serial / (t1 - t0)
    print(f"[decode-bench] SERIAL: {n_ok_serial}/{len(candidates)} ok, "
          f"{t1-t0:.2f}s, {serial_wps:.3f} windows/sec", flush=True)

    # ---- parallel (multiprocessing.Pool) ----
    # NOTE: plain mp.Pool() (fork start method) HUNG here -- torchcodec's
    # internal ffmpeg/thread-pool state is not fork-safe (the child
    # inherits a copy of locks held by threads that don't exist in it,
    # deadlocking on first decode call). Matches extract_features_av.py's
    # own documented "pre-CUDA, safe fork" caution, just for a different
    # underlying reason here. Fix: spawn context (fresh interpreter per
    # worker, no inherited thread/lock state) -- slower to start workers,
    # but the only combination that actually completed.
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(decode_one, candidates)
    t1 = time.time()
    n_ok_parallel = sum(results)
    parallel_wps = n_ok_parallel / (t1 - t0)
    print(f"[decode-bench] PARALLEL ({args.workers} workers): "
          f"{n_ok_parallel}/{len(candidates)} ok, {t1-t0:.2f}s, "
          f"{parallel_wps:.3f} windows/sec", flush=True)

    speedup = parallel_wps / serial_wps if serial_wps > 0 else float("nan")
    print(f"[decode-bench] SPEEDUP: {speedup:.2f}x", flush=True)


if __name__ == "__main__":
    main()
