"""scripts/m3_linear_probe.py — M3 cheap linear probe on VGGSound classification.

Freezes a trained AVJepaPredictor checkpoint's World-State encoder
(encode_world_state) and trains ONE linear layer on top of it against
VGGSound's 309-class ambient labels (from data/train.csv / data/test.csv).
Reports top-1 accuracy. Probe only -- no connector, no fine-tuning of the
trunk. Tells us if the representation is classification-grade independent
of retrieval R@1.

Usage:
    python scripts/m3_linear_probe.py --ckpt checkpoints/m2_fusion_fullscale/best.pt \\
        --cache-dir /home/utkarsh/raid2-data/feature_cache_vgg51k
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn


def load_labels(csv_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            fname = row[0].strip()
            vid = os.path.splitext(fname)[0]
            out[vid] = row[1].strip()
    return out


@torch.no_grad()
def extract_world_states(
    predictor: AVJepaPredictor,
    cache_dir: str,
    clip_ids: List[str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    ds = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         collate_fn=av_collate_fn, drop_last=False)
    predictor.eval()
    all_ws: List[torch.Tensor] = []
    seen_ids: List[str] = []
    t0 = time.time()
    n_done = 0
    for batch in loader:
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ws = predictor.encode_world_state(feats, tbins)
        all_ws.append(ws.float().cpu())
        seen_ids.extend(batch["clip_ids"])
        n_done += ws.shape[0]
        if n_done % (batch_size * 20) == 0:
            print(f"[m3-probe] extracted {n_done}/{len(clip_ids)}  "
                  f"{n_done / (time.time() - t0):.1f} clips/s", flush=True)
    print(f"[m3-probe] extracted {n_done}/{len(clip_ids)} total, {time.time()-t0:.1f}s", flush=True)
    return torch.cat(all_ws, dim=0), seen_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--limit-train", type=int, default=None,
                   help="cap train clips for a fast smoke test")
    p.add_argument("--limit-test", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── label vocab (union of train+test labels, deterministic order) ──────
    train_labels = load_labels(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_labels = load_labels(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    vocab = sorted(set(train_labels.values()) | set(test_labels.values()))
    label2idx = {l: i for i, l in enumerate(vocab)}
    print(f"[m3-probe] {len(vocab)} classes", flush=True)

    # ── restrict to clips actually present in the feature cache ────────────
    cache_ids = set()
    for shard in os.listdir(args.cache_dir):
        shard_dir = os.path.join(args.cache_dir, shard)
        if not os.path.isdir(shard_dir):
            continue
        for fn in os.listdir(shard_dir):
            if fn.endswith(".pt"):
                cache_ids.add(fn[:-3])

    train_ids = [v for v in train_labels if v in cache_ids]
    test_ids = [v for v in test_labels if v in cache_ids]
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_test:
        test_ids = test_ids[: args.limit_test]
    print(f"[m3-probe] train clips={len(train_ids)}  test clips={len(test_ids)}", flush=True)

    # ── build + load frozen predictor ───────────────────────────────────────
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                  max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    print(f"[m3-probe] loaded {args.ckpt}  (step={ckpt.get('step')})", flush=True)

    # ── extract World-State embeddings (frozen, no grad) ────────────────────
    train_ws, train_seen = extract_world_states(predictor, args.cache_dir, train_ids, device,
                                                 batch_size=args.batch_size)
    test_ws, test_seen = extract_world_states(predictor, args.cache_dir, test_ids, device,
                                               batch_size=args.batch_size)
    y_train = torch.tensor([label2idx[train_labels[v]] for v in train_seen], dtype=torch.long)
    y_test = torch.tensor([label2idx[test_labels[v]] for v in test_seen], dtype=torch.long)

    # ── standardize using train stats (helps a plain linear probe converge) ─
    mu = train_ws.mean(dim=0, keepdim=True)
    sigma = train_ws.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_ws = (train_ws - mu) / sigma
    test_ws = (test_ws - mu) / sigma

    train_ws, y_train = train_ws.to(device), y_train.to(device)
    test_ws, y_test = test_ws.to(device), y_test.to(device)

    # ── train ONE linear layer (the probe) ──────────────────────────────────
    d_model = train_ws.shape[1]
    probe = nn.Linear(d_model, len(vocab)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_train = train_ws.shape[0]
    bs = 512
    best_top1 = 0.0
    for epoch in range(args.epochs):
        probe.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = probe(train_ws[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        sched.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_ws)
            top1 = (test_logits.argmax(dim=-1) == y_test).float().mean().item()
            train_logits = probe(train_ws)
            train_top1 = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
        best_top1 = max(best_top1, top1)
        print(f"[m3-probe] epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={total_loss/n_train:.4f}  train_top1={train_top1:.4f}  "
              f"test_top1={top1:.4f}  best_top1={best_top1:.4f}", flush=True)

    print(f"[m3-probe] DONE. best test top-1 = {best_top1*100:.2f}%  "
          f"({len(vocab)}-way VGGSound classification, linear probe on frozen World-State)",
          flush=True)


if __name__ == "__main__":
    main()
