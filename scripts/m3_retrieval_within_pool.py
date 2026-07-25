"""scripts/m3_retrieval_within_pool.py — clean, apples-to-apples version of
the M2 retrieval trace.

scripts/m3_failure_trace.py's retrieval check inserted the 200 M3-eval clips
as extra queries into the FIXED 1545-clip standard eval gallery. That number
turned out confounded: nearly every inserted clip ranked #1, which does NOT
match the fixed gallery's own internal R@1 (~52%, reproduced independently
via scripts/eval_checkpoint_gallery.py). The likely reason is gallery-
composition, not perception quality: the 1545 set is small and internally
clustered (many near-duplicate categories competing with each other), so an
externally-inserted query is disproportionately likely to be the closest
match to ITSELF against that specific fixed distractor pool, regardless of
how well or poorly M2 actually represents it.

This script instead builds the gallery from the clips' OWN distribution: all
(or a large sample of) the M3 common-test pool (13,679 clips with all 5
caption granularities) is BOTH gallery and query set -- the same
self-contained setup the standard R@1 baseline uses, just on a different
(larger) clip pool. Every clip, not just the "failure" ones, gets a rank;
this lets us correlate M3 caption quality (average F1 percentile across the
5 granularities) against M2 retrieval rank properly, continuously, over all
200 M3-eval clips, instead of anecdotally on a handful.

Usage:
    python scripts/m3_retrieval_within_pool.py --gallery-size 4000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, ALL_GRANULARITIES
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import pool_and_project
from torch.utils.data import DataLoader

CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_vgg51k"


@torch.no_grad()
def extract_contrastive(predictor, vision_proj, ambient_proj, clip_ids, device, batch_size=128):
    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                         collate_fn=av_collate_fn, drop_last=False)
    predictor.eval(); vision_proj.eval(); ambient_proj.eval()
    zv_all, za_all, seen = [], [], []
    for batch in loader:
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
        zv_all.append(z_v.cpu()); za_all.append(z_a.cpu()); seen.extend(batch["clip_ids"])
    return torch.cat(zv_all, 0), torch.cat(za_all, 0), seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--multigran-results", default="checkpoints/m3_multigran/multigran_falsifier_results.json")
    p.add_argument("--gallery-size", type=int, default=4000,
                    help="Total gallery size (must include the 200 M3-eval clips + random fill "
                         "from the rest of the M3 common-test pool).")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="checkpoints/m3_multigran/retrieval_within_pool_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    with open(args.multigran_results) as f:
        mg = json.load(f)
    eval_clips = [row["clip_id"] for row in mg["all_clips"]]
    print(f"[pool-retrieval] {len(eval_clips)} M3-eval clips to trace", flush=True)

    print("[pool-retrieval] rebuilding M3 common-test pool (all 5 granularities present)...", flush=True)
    _, test_pairs = build_splits(ALL_GRANULARITIES)
    test_by_clip = defaultdict(dict)
    for cid, field, text in test_pairs:
        test_by_clip[cid][field] = text
    common_test_clips = [cid for cid, d in test_by_clip.items() if len(d) == len(ALL_GRANULARITIES)]
    print(f"[pool-retrieval] common-test pool = {len(common_test_clips)} clips", flush=True)

    rest = [c for c in common_test_clips if c not in set(eval_clips)]
    rng.shuffle(rest)
    fill = rest[:max(0, args.gallery_size - len(eval_clips))]
    gallery = eval_clips + fill
    rng.shuffle(gallery)   # order doesn't matter for retrieval, just avoids block-structure
    print(f"[pool-retrieval] gallery N={len(gallery)} ({len(eval_clips)} target + {len(fill)} fill)", flush=True)

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)
    contrast_dim = ckpt["vision_proj"]["weight"].shape[0]
    vision_proj = nn.Linear(1024, contrast_dim).to(device)
    ambient_proj = nn.Linear(1024, contrast_dim).to(device)
    vision_proj.load_state_dict(ckpt["vision_proj"])
    ambient_proj.load_state_dict(ckpt["ambient_proj"])
    print(f"[pool-retrieval] loaded M2 ckpt step={ckpt.get('step')}  contrast_dim={contrast_dim}", flush=True)

    z_v, z_a, seen = extract_contrastive(predictor, vision_proj, ambient_proj, gallery, device)
    N = z_v.shape[0]
    print(f"[pool-retrieval] extracted {N} clips", flush=True)
    id_to_idx = {c: i for i, c in enumerate(seen)}
    sim = z_v @ z_a.T
    gt = torch.arange(N)
    ranked_v2a = (-sim).argsort(1)
    ranked_a2v = (-sim.T).argsort(1)
    r1 = (ranked_v2a[:, :1] == gt.unsqueeze(1)).any(1).float().mean().item()
    r5 = (ranked_v2a[:, :5] == gt.unsqueeze(1)).any(1).float().mean().item()
    print(f"[pool-retrieval] WHOLE-POOL sanity: vision->ambient R@1={r1*100:.2f}% R@5={r5*100:.2f}% "
          f"(context for the per-clip ranks below)", flush=True)

    def rank_of(i, ranked_matrix):
        return int((ranked_matrix[i] == i).nonzero(as_tuple=True)[0].item()) + 1

    per_clip = {}
    for cid in eval_clips:
        if cid not in id_to_idx:
            continue
        i = id_to_idx[cid]
        r_v2a = rank_of(i, ranked_v2a)
        r_a2v = rank_of(i, ranked_a2v)
        per_clip[cid] = {
            "rank_vision_to_ambient": r_v2a, "rank_ambient_to_vision": r_a2v,
            "percentile_v2a": round(100.0 * (1 - (r_v2a - 1) / (N - 1)), 2),
            "percentile_a2v": round(100.0 * (1 - (r_a2v - 1) / (N - 1)), 2),
            "gallery_size": N,
        }

    results = {"gallery_size": N, "whole_pool_R1_v2a": r1, "whole_pool_R5_v2a": r5, "per_clip": per_clip}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[pool-retrieval] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
