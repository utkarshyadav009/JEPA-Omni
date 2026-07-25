"""scripts/m4_shift_metric_easycom.py — shift-metric selection on REAL
graded turn transitions (EasyCom), replacing the earlier VGGSound clip-
transition proxy (scripts/m4_shift_metric_eval.py), which the module
docstring for models/m4_shift_trigger.py already flagged as too coarse
(all-different-clip boundaries are trivially easy, AUC>=0.9999 for all
three metrics, no real discrimination).

World-State doesn't exist for EasyCom (no M2/vision features), so this
tests the metrics on the Whisper speech-activity feature space instead
(mean-pooled frozen whisper-medium hidden state per segment) -- the
feature space we DO have for EasyCom, and the one the M4 decision head
actually reads for the EasyCom side.

Boundary definition (real, graded, not synthetic): walk the chronological
sequence of ALL usable speech segments within a (session, chunk) 60s
window (any participant), across CONSECUTIVE segment pairs:
  - SAME participant speaking again (no real turn change) = "within"
  - DIFFERENT participant (a genuine speaker turn) = "boundary"
This is graded in the sense that some speaker changes are quick back-and-
forth exchanges (subtle) vs long monologue swaps (dramatic) -- unlike the
all-or-nothing "totally different VGGSound clip" proxy.

Usage:
    python scripts/m4_shift_metric_easycom.py
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.m4_speech_dataset import build_segments, EasyComSpeechDataset
from models.m4_speech import WhisperSpeechEncoder


@torch.no_grad()
def extract_all_segment_features(whisper, all_segs, device):
    ds = EasyComSpeechDataset(all_segs)
    feats = []
    for i in range(len(ds)):
        item = ds[i]
        hidden, valid_frames = whisper([item["waveform"]], [item["duration_sec"]], device)
        vf = int(valid_frames[0].item())
        pooled = hidden[0, :vf].float().mean(dim=0)
        feats.append(pooled.cpu())
        if (i + 1) % 500 == 0:
            print(f"[shift-easycom] extracted {i+1}/{len(ds)}", flush=True)
    return torch.stack(feats, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--pca-explained-var", type=float, default=0.95)
    p.add_argument("--out", default="checkpoints/m4_decision_head/shift_metric_easycom_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[shift-easycom] loading frozen whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[shift-easycom] building all usable EasyCom segments (train+test sessions combined "
          "-- this is a metric-selection analysis, not a model eval, no train/test split needed)...", flush=True)
    train_segs, test_segs = build_segments()
    all_segs = train_segs + test_segs
    print(f"[shift-easycom] {len(all_segs)} total usable segments", flush=True)

    print("[shift-easycom] extracting per-segment Whisper features (this is the expensive step)...", flush=True)
    feats = extract_all_segment_features(whisper, all_segs, device)   # (N, 1024)
    d = feats.shape[1]

    # group by (session, chunk), sort by start_sec, walk consecutive pairs
    by_chunk = defaultdict(list)
    for seg, feat in zip(all_segs, feats):
        by_chunk[(seg.session, seg.chunk)].append((seg.start_sec, seg.participant_id, feat))

    boundary_scores = {"euclidean": [], "cosine": [], "mahalanobis": []}
    within_scores = {"euclidean": [], "cosine": [], "mahalanobis": []}

    # PCA/Mahalanobis basis fit on the FULL feature pool (Whisper-space, distinct from World-State space)
    X = feats.numpy().astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    explained = np.cumsum(var) / var.sum()
    K = int(np.searchsorted(explained, args.pca_explained_var) + 1)
    K = max(2, min(K, len(S)))
    print(f"[shift-easycom] Whisper-feature-space PCA: K={K} components explain "
          f"{explained[K-1]*100:.1f}% variance (target {args.pca_explained_var*100:.0f}%)", flush=True)
    components = Vt[:K]
    whiten_scale = 1.0 / (S[:K] / np.sqrt(max(1, X.shape[0] - 1)) + 1e-8)

    def mahalanobis_whiten(v: np.ndarray) -> np.ndarray:
        return (components @ (v - mu)) * whiten_scale

    n_boundary_pairs = n_within_pairs = 0
    for (sess, chunk), items in by_chunk.items():
        items.sort(key=lambda x: x[0])
        for i in range(1, len(items)):
            _, pid0, f0 = items[i - 1]
            _, pid1, f1 = items[i]
            is_boundary = pid0 != pid1
            f0n, f1n = f0.numpy().astype(np.float64), f1.numpy().astype(np.float64)
            d_euc = float(np.linalg.norm(f1n - f0n))
            cos_sim = float(np.dot(f0n, f1n) / (np.linalg.norm(f0n) * np.linalg.norm(f1n) + 1e-8))
            d_cos = 1.0 - cos_sim
            m0, m1 = mahalanobis_whiten(f0n), mahalanobis_whiten(f1n)
            d_mah = float(np.linalg.norm(m1 - m0))
            bucket = boundary_scores if is_boundary else within_scores
            bucket["euclidean"].append(d_euc)
            bucket["cosine"].append(d_cos)
            bucket["mahalanobis"].append(d_mah)
            if is_boundary:
                n_boundary_pairs += 1
            else:
                n_within_pairs += 1

    print(f"\n[shift-easycom] n_boundary_pairs (speaker changed) = {n_boundary_pairs}  "
          f"n_within_pairs (same speaker again) = {n_within_pairs}\n", flush=True)

    def cohens_d(a, b):
        a, b = np.array(a), np.array(b)
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return (a.mean() - b.mean()) / (pooled_std + 1e-12)

    report = {"n_boundary_pairs": n_boundary_pairs, "n_within_pairs": n_within_pairs,
              "pca_K": K, "pca_explained_var": float(explained[K-1]), "metrics": {}}
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
        print(f"[shift-easycom] {name:12s}  AUC={auc:.4f}  Cohen's_d={d:.4f}  "
              f"boundary={np.mean(boundary_scores[name]):.4f}±{np.std(boundary_scores[name]):.4f}  "
              f"within={np.mean(within_scores[name]):.4f}±{np.std(within_scores[name]):.4f}", flush=True)

    best = max(report["metrics"], key=lambda k: report["metrics"][k]["roc_auc"])
    report["selected_metric"] = best
    print(f"\n[shift-easycom] SELECTED: {best} (highest AUC on real graded turn transitions)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[shift-easycom] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
