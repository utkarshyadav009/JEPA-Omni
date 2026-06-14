"""Quick stress-test: can we use num_workers>0 with file_descriptor sharing?

Run in a SEPARATE terminal while training continues on GPU 0.
This test uses CPU-only decoding, so it won't interfere with your running job.

Usage:
    python test_workers.py --config configs/m1_scale.yaml --workers 4 --batches 3
"""
import argparse, time, os, sys

# --- The critical fix: bypass /dev/shm entirely ---
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

import torch
from torch.utils.data import DataLoader
from data.video_text_dataset import build_dataset, collate_fn
from utils import load_config, cfg_get

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_scale.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=3, help="Number of batches to fetch")
    parser.add_argument("--limit", type=int, default=64, help="Subset of dataset to use")
    args = parser.parse_args()

    cfg = load_config(args.config)
    batch_size = int(cfg_get(cfg, "train.batch_size", default=256))

    print(f"[test] /dev/shm size: {os.popen('df -h /dev/shm').read().strip()}")
    print(f"[test] sharing_strategy: {torch.multiprocessing.get_sharing_strategy()}")
    print(f"[test] batch_size={batch_size}, num_workers={args.workers}, limit={args.limit}")
    print(f"[test] Building dataset (small subset)...", flush=True)

    ds = build_dataset(cfg, "train", limit=args.limit, decode_device="cpu")
    print(f"[test] Dataset ready: {len(ds)} samples")

    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=False,           # avoid any CUDA interaction
        persistent_workers=args.workers > 0,
        # NOT using spawn — use default fork, lighter and avoids re-importing
    )

    print(f"[test] DataLoader created. Fetching {args.batches} batches...", flush=True)
    for i, (clips, captions) in enumerate(loader):
        if i >= args.batches:
            break
        shapes = [c.shape for c in clips[:3]]
        t = time.time()
        print(f"[test] Batch {i+1}/{args.batches}: {len(clips)} clips, "
              f"first shapes={shapes}, captions[0]={captions[0][:60]}...")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"[test] SUCCESS — num_workers={args.workers} with file_descriptor sharing works!")
    print(f"[test] Safe to apply this config to the real training run.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
