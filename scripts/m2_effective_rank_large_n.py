"""scripts/m2_effective_rank_large_n.py — correct the rank record.

The "19-21/47" effective-rank figures on file were measured with
effective_rank()'s covariance estimated from a small batch (ws_sub =
ws[:rank_ceil+1] in train_m2.py's logging path), so the reported ceiling was
min(batch_size-1, d_model) = 47, NOT the true dimensionality (d_model=1024).
That made the numbers look like a near-collapse signal when they were
actually just batch-size-limited. This script re-measures with N >> 1024 so
the ceiling is the real d_model=1024, and the number is directly comparable
to it.

Usage:
    python scripts/m2_effective_rank_large_n.py --ckpt checkpoints/m2_fusion_20k_best/step19000_peak.pt --n 5000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor, effective_rank
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m3 import load_vgg_split_ids
from torch.utils.data import DataLoader

CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_vgg51k"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)
    print(f"[eff-rank] loaded {args.ckpt} step={ckpt.get('step')}", flush=True)

    train_ids = sorted(load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "train.csv")))
    test_ids = sorted(load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "test.csv")))
    all_ids = train_ids + test_ids
    rng.shuffle(all_ids)
    pool_ids = all_ids[:args.n]
    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=pool_ids, max_tdm_bins=512, audio_mode="mean")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, collate_fn=av_collate_fn, drop_last=False)

    all_ws = []
    with torch.no_grad():
        for batch in loader:
            feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
            tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                ws = predictor.encode_world_state(feats, tbins)
            all_ws.append(ws.float().cpu())
    ws_all = torch.cat(all_ws, 0)
    N, D = ws_all.shape
    ceiling = min(N - 1, D)
    er = effective_rank(ws_all)
    print(f"[eff-rank] N={N}  d_model={D}  ceiling=min(N-1,D)={ceiling}  effective_rank={er:.2f}", flush=True)
    print(f"[eff-rank] effective_rank / d_model = {er/D:.4f}  ({er:.1f} of {D})", flush=True)

    # subsample sweep, to show the batch-size artifact explicitly for the record
    for n_sub in (48, 128, 512, 1500, N):
        if n_sub > N:
            continue
        sub = ws_all[:n_sub]
        sub_ceiling = min(n_sub - 1, D)
        sub_er = effective_rank(sub)
        print(f"[eff-rank]   at N={n_sub:5d}: ceiling={sub_ceiling:5d}  effective_rank={sub_er:6.2f}  "
              f"({'CEILING-LIMITED' if sub_ceiling < D else 'true d_model ceiling'})", flush=True)

    result = {"ckpt": args.ckpt, "step": ckpt.get("step"), "n": N, "d_model": D,
              "effective_rank": er, "effective_rank_over_d_model": er / D}
    out = args.out or f"checkpoints/m2_fusion_20k_best/effective_rank_large_n.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eff-rank] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
