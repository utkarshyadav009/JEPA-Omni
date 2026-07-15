"""Standalone eval: compare a trained M2 checkpoint's retrieval R@1/5/10 on
the normal 1545-clip gallery vs a FRESH, never-trained-on, never-in-gallery
held-out slice, to test whether the model is data-limited.

Reuses contrastive_retrieval_eval / pool_and_project / AVJepaConfig etc.
directly from train_m2.py rather than duplicating logic (same pattern as
scripts/verify_grad_sync.py).

Usage:
    python scripts/eval_fresh_holdout.py \
        --ckpt checkpoints/m2_diag192/step6000.pt \
        --fresh-cache-dir /dev/shm/jepa_m2_freshcache
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from utils import load_config, cfg_get
from train_m2 import contrastive_retrieval_eval


def build_loader(cache_dir: str, max_tdm_bins: int, audio_mode: str,
                  clip_ids=None, batch_size: int = 64) -> DataLoader:
    dataset = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                               max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       collate_fn=av_collate_fn, drop_last=False,
                       pin_memory=torch.cuda.is_available())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m2.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fresh-cache-dir", required=True)
    p.add_argument("--gallery-eval-subset", default="data/vggsound_eval_1545.txt")
    p.add_argument("--contrast-dim", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    predictor_cfg = AVJepaConfig(
        d_model      = int(cfg_get(cfg, "model.d_model",       default=1024)),
        depth        = int(cfg_get(cfg, "model.depth",         default=8)),
        heads        = int(cfg_get(cfg, "model.heads",         default=8)),
        mlp_ratio    = float(cfg_get(cfg, "model.mlp_ratio",   default=4.0)),
        max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins",  default=512)),
        dropout      = float(cfg_get(cfg, "model.dropout",     default=0.0)),
    )
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    d_model = predictor_cfg.d_model
    vision_proj = nn.Linear(d_model, args.contrast_dim).to(device)
    ambient_proj = nn.Linear(d_model, args.contrast_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"])
    vision_proj.load_state_dict(ckpt["vision_proj"])
    ambient_proj.load_state_dict(ckpt["ambient_proj"])
    print(f"[eval_fresh_holdout] loaded {args.ckpt} (step={ckpt.get('step')})", flush=True)

    max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512))
    audio_mode   = str(cfg_get(cfg, "model.audio_mode",   default="mean"))
    train_cache_dir = str(cfg_get(cfg, "data.av_cache_dir", default="/dev/shm/jepa_m2_cache"))

    with open(args.gallery_eval_subset) as f:
        gallery_ids = [l.strip() for l in f if l.strip()]
    gallery_loader = build_loader(train_cache_dir, max_tdm_bins, audio_mode,
                                   clip_ids=gallery_ids)
    fresh_loader = build_loader(args.fresh_cache_dir, max_tdm_bins, audio_mode,
                                 clip_ids=None)
    n_fresh = len(fresh_loader.dataset)
    print(f"[eval_fresh_holdout] gallery clips={len(gallery_ids)}  fresh clips={n_fresh}",
          flush=True)

    gallery_results = contrastive_retrieval_eval(
        predictor, vision_proj, ambient_proj, gallery_loader, device,
        max_clips=len(gallery_ids),
    )
    print("=== GALLERY (1545, trained distribution) ===", flush=True)
    for k, v in sorted(gallery_results.items()):
        print(f"  {k}={v}", flush=True)

    fresh_results = contrastive_retrieval_eval(
        predictor, vision_proj, ambient_proj, fresh_loader, device,
        max_clips=n_fresh,
    )
    print(f"=== FRESH HOLDOUT ({n_fresh}, never trained/gallery) ===", flush=True)
    for k, v in sorted(fresh_results.items()):
        print(f"  {k}={v}", flush=True)


if __name__ == "__main__":
    main()
