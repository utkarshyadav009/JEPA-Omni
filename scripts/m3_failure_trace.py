"""scripts/m3_failure_trace.py — trace M3 clips that failed ALL 5 caption
granularities back to M2 (retrieval rank + linear-probe classification).

Hypothesis under test: M3's clustered failures (same clip wrong across every
granularity) originate in the frozen M2 World-State for those clips, not in
the connector or in caption noise. If true, these clips should ALSO rank
poorly on M2's own contrastive retrieval and misclassify under M2's linear
probe -- i.e. M2 already "doesn't understand" these clips, independent of
M3 entirely.

Two measurements, both against the SAME frozen M2 checkpoint used to train
M3 (checkpoints/m2_fusion_20k_best/step19000_peak.pt):

1. Retrieval rank: insert the failure clips into the FIXED 1545-clip eval
   gallery (data/vggsound_eval_1545.txt) as extra gallery members, then for
   each failure clip report the rank of its own true cross-modal match
   (vision->ambient and ambient->vision) among all N=1545+k gallery clips --
   same contrastive vision_proj/ambient_proj heads used in train_m2.py's
   contrastive_retrieval_eval.
2. Linear probe: train the SAME cheap 1-layer probe as scripts/m3_linear_probe.py
   (full VGGSound train/test), then report each failure clip's predicted vs
   true class and where the true class ranks in the probe's own sorted logits.

Usage:
    python scripts/m3_failure_trace.py --clip-ids clip1,clip2,... \
        --labels-csv-note "from M3 all-granularity-failure list"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import pool_and_project

CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_vgg51k"


def load_labels(csv_path: str) -> Dict[str, str]:
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                out[os.path.splitext(row[0].strip())[0]] = row[1].strip()
    return out


@torch.no_grad()
def extract_world_states(predictor, clip_ids, device, batch_size=128):
    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                         collate_fn=av_collate_fn, drop_last=False)
    predictor.eval()
    all_ws, seen = [], []
    for batch in loader:
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ws = predictor.encode_world_state(feats, tbins)
        all_ws.append(ws.float().cpu())
        seen.extend(batch["clip_ids"])
    return torch.cat(all_ws, 0), seen


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
            feats_f = {k: v.float() for k, v in feats.items()}
            z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats_f, tbins)
        zv_all.append(z_v.cpu()); za_all.append(z_a.cpu()); seen.extend(batch["clip_ids"])
    return torch.cat(zv_all, 0), torch.cat(za_all, 0), seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip-ids", required=True, help="comma-separated failure clip_ids")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--gallery", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--out", default="checkpoints/m3_multigran/failure_trace_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failure_clips = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    print(f"[trace] {len(failure_clips)} failure clips: {failure_clips}", flush=True)

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
    print(f"[trace] loaded M2 ckpt step={ckpt.get('step')}  contrast_dim={contrast_dim}", flush=True)

    train_labels = load_labels(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_labels = load_labels(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    all_labels = {**train_labels, **test_labels}

    # ── 1. Retrieval rank: fixed 1545 gallery + failure clips as extras ────
    with open(args.gallery) as f:
        gallery_ids = [l.strip() for l in f if l.strip()]
    extra = [c for c in failure_clips if c not in set(gallery_ids)]
    full_gallery = gallery_ids + extra
    print(f"[trace] gallery={len(gallery_ids)}  +{len(extra)} failure clips not already in it  "
          f"-> N={len(full_gallery)}", flush=True)

    t0 = time.time()
    z_v, z_a, seen_ids = extract_contrastive(predictor, vision_proj, ambient_proj, full_gallery, device)
    print(f"[trace] extracted contrastive embeddings for {len(seen_ids)} clips in {time.time()-t0:.1f}s", flush=True)
    id_to_idx = {cid: i for i, cid in enumerate(seen_ids)}
    N = z_v.shape[0]
    sim = z_v @ z_a.T   # (N, N), row=vision query, col=ambient gallery

    def rank_of(cid: str, direction: str) -> int:
        i = id_to_idx[cid]
        if direction == "v2a":
            row = sim[i]
        else:
            row = sim[:, i]
        order = (-row).argsort()
        # 1-indexed rank of the true match (itself) in the sorted list
        return int((order == i).nonzero(as_tuple=True)[0].item()) + 1

    retrieval_report = []
    for cid in failure_clips:
        if cid not in id_to_idx:
            retrieval_report.append({"clip_id": cid, "error": "not found in cache"})
            continue
        r_v2a = rank_of(cid, "v2a")
        r_a2v = rank_of(cid, "a2v")
        retrieval_report.append({
            "clip_id": cid, "label": all_labels.get(cid),
            "rank_vision_to_ambient": r_v2a, "rank_ambient_to_vision": r_a2v,
            "percentile_v2a": round(100.0 * r_v2a / N, 1),
            "percentile_a2v": round(100.0 * r_a2v / N, 1),
            "gallery_size": N,
        })
        print(f"[trace] {cid}  v2a_rank={r_v2a}/{N} ({100.0*r_v2a/N:.1f}%ile)  "
              f"a2v_rank={r_a2v}/{N} ({100.0*r_a2v/N:.1f}%ile)", flush=True)

    # ── 2. Linear probe: full train, evaluate specifically on failure clips ─
    vocab = sorted(set(train_labels.values()) | set(test_labels.values()))
    label2idx = {l: i for i, l in enumerate(vocab)}
    idx2label = {i: l for l, i in label2idx.items()}
    print(f"[trace] {len(vocab)}-class linear probe, training on full VGGSound train split...", flush=True)

    cache_ids = set()
    for shard in os.listdir(CACHE_DIR):
        shard_dir = os.path.join(CACHE_DIR, shard)
        if not os.path.isdir(shard_dir):
            continue
        for fn in os.listdir(shard_dir):
            if fn.endswith(".pt"):
                cache_ids.add(fn[:-3])
    train_ids = [v for v in train_labels if v in cache_ids]
    test_ids = [v for v in test_labels if v in cache_ids]
    # make sure every failure clip is scored even if it's oddly not in test.csv
    extra_probe = [c for c in failure_clips if c not in test_ids and c in cache_ids]

    train_ws, train_seen = extract_world_states(predictor, train_ids, device)
    test_ws, test_seen = extract_world_states(predictor, test_ids + extra_probe, device)
    y_train = torch.tensor([label2idx[train_labels[v]] for v in train_seen], dtype=torch.long)
    y_test = torch.tensor([label2idx[all_labels[v]] for v in test_seen], dtype=torch.long)

    mu = train_ws.mean(dim=0, keepdim=True)
    sigma = train_ws.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_ws = ((train_ws - mu) / sigma).to(device)
    test_ws = ((test_ws - mu) / sigma).to(device)
    y_train = y_train.to(device); y_test = y_test.to(device)

    probe = nn.Linear(train_ws.shape[1], len(vocab)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.probe_epochs)
    n_train = train_ws.shape[0]
    bs = 512
    best_top1 = 0.0
    for epoch in range(args.probe_epochs):
        probe.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(probe(train_ws[idx]), y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        probe.eval()
        with torch.no_grad():
            top1 = (probe(test_ws).argmax(dim=-1) == y_test).float().mean().item()
        best_top1 = max(best_top1, top1)
        if (epoch + 1) % 10 == 0 or epoch == args.probe_epochs - 1:
            print(f"[trace]   probe epoch {epoch+1}/{args.probe_epochs}  test_top1={top1:.4f}", flush=True)
    print(f"[trace] probe best test_top1={best_top1:.4f} (sanity check vs known ~0.54)", flush=True)

    probe.eval()
    with torch.no_grad():
        logits = probe(test_ws)
        probs = F.softmax(logits, dim=-1)
        ranks = (-logits).argsort(dim=-1)

    seen_idx = {cid: i for i, cid in enumerate(test_seen)}
    probe_report = []
    for cid in failure_clips:
        if cid not in seen_idx:
            probe_report.append({"clip_id": cid, "error": "not found in cache/labels"})
            continue
        i = seen_idx[cid]
        true_idx = y_test[i].item()
        pred_idx = logits[i].argmax().item()
        true_rank = int((ranks[i] == true_idx).nonzero(as_tuple=True)[0].item()) + 1
        probe_report.append({
            "clip_id": cid, "true_label": idx2label[true_idx], "predicted_label": idx2label[pred_idx],
            "correct": bool(pred_idx == true_idx), "true_class_rank": true_rank,
            "true_class_prob": round(probs[i, true_idx].item(), 4),
            "pred_class_prob": round(probs[i, pred_idx].item(), 4),
        })
        print(f"[trace] {cid}  true={idx2label[true_idx]!r}  pred={idx2label[pred_idx]!r}  "
              f"correct={pred_idx==true_idx}  true_rank={true_rank}/{len(vocab)}", flush=True)

    results = {
        "failure_clips": failure_clips,
        "retrieval": retrieval_report,
        "linear_probe": {"probe_test_top1": best_top1, "n_classes": len(vocab), "per_clip": probe_report},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[trace] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
