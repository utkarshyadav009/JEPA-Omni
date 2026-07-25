"""scripts/m4_shift_metric_eval.py — M4 proactive-trigger shift-metric selection.

Ego4D-independent. Compares three candidate "has the World-State shifted
enough to warrant a speak/silence check" metrics -- Euclidean, cosine
distance, and PCA-whitened Mahalanobis (using a covariance measured from a
large independent sample of real World-States) -- against a labeled proxy
for turn-taking boundaries: VGGSound clip transitions in a synthetic
concatenated pseudo-timeline.

Why this proxy, and why NOT the SIGReg-isotropy argument: the earlier
proposal argued Euclidean distance is justified because SIGReg shapes the
World-State toward N(0,I). Our own measured effective_rank(world_state)
plateaus around ~19-21 (against a small-batch ceiling of 47), i.e. the
representation is clearly anisotropic in practice, not isotropic -- the
premise was falsified by our own training logs. This script drops that
argument entirely and picks the shift metric empirically instead.

Methodology:
  1. Build K-clip "sessions" by concatenating distinct VGGSound test clips.
  2. Within each clip, compute a STREAMING sequence of World-States at
     increasing token-time cutoffs (25/50/75/100% of the TDM axis) --
     filtering feats/tbins to tokens with tbin < cutoff simulates "what a
     live system would have seen so far," a causal/streaming-consistent way
     to get multiple ticks out of a single (otherwise static) clip.
  3. Stitch each session's per-clip tick sequences into one timeline. Every
     adjacent tick pair is labeled BOUNDARY (crosses from one clip to the
     next -- our only available proxy for "something changed enough that a
     turn-taking decision should be reconsidered") or WITHIN (same clip).
  4. Separately, extract a large independent pool of whole-clip World-States
     to measure a real covariance matrix (and re-confirm effective_rank at a
     MUCH larger N than the "19-21/47" figure, which was rank-ceiling-limited
     by a small batch).
  5. For each metric, score all boundary vs. within-clip tick-pairs and
     report ROC-AUC (does the shift score separate the two classes) + Cohen's
     d. Recommend whichever metric wins.

Usage:
    python scripts/m4_shift_metric_eval.py --n-sessions 300 --clips-per-session 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor, effective_rank
from data.av_cached_dataset import AVCachedDataset
from train_m3 import load_vgg_split_ids

CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_vgg51k"
MAX_TDM_BINS = 512
TICK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
MIN_TOKENS = 8   # skip a tick if too few tokens survive the cutoff filter


@torch.no_grad()
def clip_tick_sequence(predictor, item, device) -> List[torch.Tensor]:
    """Returns a list of World-State vectors (d_model,) for this clip at each
    TICK_FRACTIONS cutoff, filtering feats/tbins causally (tbin < cutoff)."""
    ticks = []
    for frac in TICK_FRACTIONS:
        cutoff = int(frac * MAX_TDM_BINS)
        feats_f, tbins_f = {}, {}
        ok = True
        for m in ("vision", "ambient"):
            tb = item["tbins"][m]
            keep = tb < cutoff
            if keep.sum().item() < MIN_TOKENS:
                ok = False
                break
            feats_f[m] = item["feats"][m][keep].unsqueeze(0).to(device).float()
            tbins_f[m] = tb[keep].unsqueeze(0).to(device)
        if not ok:
            continue
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ws = predictor.encode_world_state(feats_f, tbins_f)
        ticks.append(ws.float().squeeze(0).cpu())
    return ticks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--n-sessions", type=int, default=300)
    p.add_argument("--clips-per-session", type=int, default=4)
    p.add_argument("--n-cov-clips", type=int, default=3000, help="pool size for covariance/effective-rank estimate")
    p.add_argument("--pca-explained-var", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="checkpoints/m4/shift_metric_eval_results.json")
    p.add_argument("--save-basis", default="checkpoints/m4/shift_metric_pca_basis.pt",
                    help="Save the mu/components/whiten_scale basis so models/m4_shift_trigger.py "
                         "can reuse it without recomputing the covariance pool.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)
    print(f"[shift-metric] loaded M2 ckpt step={ckpt.get('step')}", flush=True)

    test_ids = sorted(load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "test.csv")))
    rng.shuffle(test_ids)
    n_needed = args.n_cov_clips + args.n_sessions * args.clips_per_session
    pool_ids = test_ids[:min(n_needed, len(test_ids))]
    print(f"[shift-metric] loading {len(pool_ids)} clips from cache...", flush=True)
    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=pool_ids, max_tdm_bins=MAX_TDM_BINS, audio_mode="mean")
    id_to_item = {}
    for i in range(len(ds)):
        item = ds[i]
        id_to_item[item["clip_id"]] = item
    available_ids = list(id_to_item.keys())
    print(f"[shift-metric] {len(available_ids)} clips actually present in cache", flush=True)

    # ── covariance / effective-rank pool (whole-clip World-States) ─────────
    cov_ids = available_ids[:args.n_cov_clips]
    print(f"[shift-metric] extracting {len(cov_ids)} whole-clip World-States for covariance...", flush=True)
    cov_ws = []
    for cid in cov_ids:
        item = id_to_item[cid]
        ticks = clip_tick_sequence(predictor, item, device)
        if ticks:
            cov_ws.append(ticks[-1])   # full (100%) tick
    cov_ws = torch.stack(cov_ws, 0)   # (N, d)
    eff_rank_large_n = effective_rank(cov_ws)
    print(f"[shift-metric] cov pool N={cov_ws.shape[0]}  effective_rank={eff_rank_large_n:.2f} "
          f"(of ceiling {min(cov_ws.shape[0]-1, cov_ws.shape[1])}, d_model={cov_ws.shape[1]})", flush=True)

    # PCA whitening basis (top-K by explained variance) for Mahalanobis
    mu = cov_ws.mean(0, keepdim=True)
    X = (cov_ws - mu).numpy().astype(np.float64)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    explained = np.cumsum(var) / var.sum()
    K = int(np.searchsorted(explained, args.pca_explained_var) + 1)
    K = max(2, min(K, len(S)))
    print(f"[shift-metric] PCA: K={K} components explain {explained[K-1]*100:.1f}% variance "
          f"(target {args.pca_explained_var*100:.0f}%)", flush=True)
    components = Vt[:K]                    # (K, d)
    singular = S[:K]
    whiten_scale = 1.0 / (singular / np.sqrt(max(1, X.shape[0] - 1)) + 1e-8)   # (K,)
    mu_np = mu.numpy().astype(np.float64).squeeze(0)

    def mahalanobis_whiten(w: torch.Tensor) -> np.ndarray:
        x = w.numpy().astype(np.float64) - mu_np
        proj = components @ x            # (K,)
        return proj * whiten_scale        # whitened coords

    if args.save_basis:
        os.makedirs(os.path.dirname(args.save_basis), exist_ok=True)
        torch.save({
            "mu": torch.tensor(mu_np, dtype=torch.float32),
            "components": torch.tensor(components, dtype=torch.float32),
            "whiten_scale": torch.tensor(whiten_scale, dtype=torch.float32),
            "K": K, "explained_var": float(explained[K - 1]),
            "d_model": cov_ws.shape[1], "cov_pool_n": cov_ws.shape[0],
            "source_ckpt": args.m2_ckpt,
        }, args.save_basis)
        print(f"[shift-metric] saved PCA whitening basis to {args.save_basis}", flush=True)

    # ── sessions: K clips each, streaming ticks, boundary-labeled pairs ────
    session_pool = [c for c in available_ids if c not in set(cov_ids)]
    rng.shuffle(session_pool)
    n_sessions = min(args.n_sessions, len(session_pool) // args.clips_per_session)
    print(f"[shift-metric] building {n_sessions} sessions of {args.clips_per_session} clips each...", flush=True)

    boundary_scores = {"euclidean": [], "cosine": [], "mahalanobis": []}
    within_scores = {"euclidean": [], "cosine": [], "mahalanobis": []}

    idx = 0
    for s in range(n_sessions):
        session_clip_ids = session_pool[idx: idx + args.clips_per_session]
        idx += args.clips_per_session
        timeline = []   # list of (tick_vector, clip_index)
        for ci, cid in enumerate(session_clip_ids):
            item = id_to_item[cid]
            ticks = clip_tick_sequence(predictor, item, device)
            for t in ticks:
                timeline.append((t, ci))
        for i in range(1, len(timeline)):
            w0, c0 = timeline[i - 1]
            w1, c1 = timeline[i]
            is_boundary = (c0 != c1)
            d_euc = float(torch.norm(w1 - w0).item())
            cos_sim = float(torch.nn.functional.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item())
            d_cos = 1.0 - cos_sim
            m0 = mahalanobis_whiten(w0)
            m1 = mahalanobis_whiten(w1)
            d_mah = float(np.linalg.norm(m1 - m0))
            bucket = boundary_scores if is_boundary else within_scores
            bucket["euclidean"].append(d_euc)
            bucket["cosine"].append(d_cos)
            bucket["mahalanobis"].append(d_mah)
        if (s + 1) % 50 == 0:
            print(f"[shift-metric] {s+1}/{n_sessions} sessions processed", flush=True)

    def cohens_d(a, b):
        a, b = np.array(a), np.array(b)
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return (a.mean() - b.mean()) / (pooled_std + 1e-12)

    report = {"effective_rank_large_n": eff_rank_large_n, "cov_pool_n": cov_ws.shape[0],
              "pca_K": K, "pca_explained_var": float(explained[K-1]),
              "n_boundary_pairs": len(boundary_scores["euclidean"]),
              "n_within_pairs": len(within_scores["euclidean"]), "metrics": {}}
    print(f"\n[shift-metric] n_boundary_pairs={len(boundary_scores['euclidean'])}  "
          f"n_within_pairs={len(within_scores['euclidean'])}\n", flush=True)
    for name in ("euclidean", "cosine", "mahalanobis"):
        y = np.array([1] * len(boundary_scores[name]) + [0] * len(within_scores[name]))
        scores = np.array(boundary_scores[name] + within_scores[name])
        auc = roc_auc_score(y, scores)
        d = cohens_d(boundary_scores[name], within_scores[name])
        report["metrics"][name] = {
            "roc_auc": float(auc), "cohens_d": float(d),
            "boundary_mean": float(np.mean(boundary_scores[name])), "boundary_std": float(np.std(boundary_scores[name])),
            "within_mean": float(np.mean(within_scores[name])), "within_std": float(np.std(within_scores[name])),
        }
        print(f"[shift-metric] {name:12s}  AUC={auc:.4f}  Cohen's_d={d:.4f}  "
              f"boundary={np.mean(boundary_scores[name]):.4f}±{np.std(boundary_scores[name]):.4f}  "
              f"within={np.mean(within_scores[name]):.4f}±{np.std(within_scores[name]):.4f}", flush=True)

    best = max(report["metrics"], key=lambda k: report["metrics"][k]["roc_auc"])
    report["recommended_metric"] = best
    print(f"\n[shift-metric] RECOMMENDED: {best} (highest AUC)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[shift-metric] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
