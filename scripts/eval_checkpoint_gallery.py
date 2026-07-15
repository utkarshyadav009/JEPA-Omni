"""Re-baseline a saved M2 checkpoint's retrieval R@1/5/10 through the FIXED
multi-GPU eval path (build_dataloader(..., distributed_sampler=False)),
exercising the actual is_distributed()=True codepath that had the
DistributedSampler-sharding bug -- not just a single-process script that
was never sharded to begin with.

Run with torchrun so the fix is verified under real distributed conditions;
only rank 0 does the eval (matches the production pattern in train_m2.py).

Usage:
    torchrun --nproc_per_node=4 scripts/eval_checkpoint_gallery.py \
        --ckpt checkpoints/m2_diag192/step6000.pt

    # sanity-check the fresh holdout cache through the SAME fixed path:
    torchrun --nproc_per_node=4 scripts/eval_checkpoint_gallery.py \
        --ckpt checkpoints/m2_diag192/step6000.pt \
        --eval-subset data/vggsound_fresh_holdout_1700.txt \
        --cache-dir-override /dev/shm/jepa_m2_freshcache
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_m2 import (
    build_dataloader,
    contrastive_retrieval_eval,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset as _DS
from utils import load_config, cfg_get


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m2.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-subset", default="data/vggsound_eval_1545.txt")
    p.add_argument("--cache-dir-override", default=None,
                    help="Point the eval gallery at a different cache dir "
                         "(e.g. the fresh-holdout cache) for the two-path "
                         "agreement sanity check.")
    p.add_argument("--contrast-dim", type=int, default=256)
    args = p.parse_args()

    device = setup_distributed()
    cfg = load_config(args.config)
    if args.cache_dir_override:
        cfg["data"]["av_cache_dir"] = args.cache_dir_override

    predictor_cfg = AVJepaConfig(
        d_model      = int(cfg_get(cfg, "model.d_model",       default=1024)),
        depth        = int(cfg_get(cfg, "model.depth",         default=8)),
        heads        = int(cfg_get(cfg, "model.heads",         default=8)),
        mlp_ratio    = float(cfg_get(cfg, "model.mlp_ratio",   default=4.0)),
        max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins",  default=512)),
        dropout      = float(cfg_get(cfg, "model.dropout",     default=0.0)),
    )
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    vision_proj = nn.Linear(predictor_cfg.d_model, args.contrast_dim).to(device)
    ambient_proj = nn.Linear(predictor_cfg.d_model, args.contrast_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"])
    vision_proj.load_state_dict(ckpt["vision_proj"])
    ambient_proj.load_state_dict(ckpt["ambient_proj"])

    with open(args.eval_subset) as f:
        eval_clip_ids = [l.strip() for l in f if l.strip()]

    cache_dir = str(cfg_get(cfg, "data.av_cache_dir", default="/dev/shm/jepa_m2_cache"))
    _probe = _DS(cache_dir=cache_dir, clip_ids=eval_clip_ids,
                 max_tdm_bins=predictor_cfg.max_tdm_bins, audio_mode="mean")
    n_eval_avail = len(_probe.clip_ids)

    eval_loader, _ = build_dataloader(
        cfg, clip_ids=_probe.clip_ids, batch_size_override=64,
        num_workers_override=0, distributed_sampler=False, drop_last_override=False,
    )

    if is_main_process():
        print(f"[eval_checkpoint_gallery] ckpt={args.ckpt} (step={ckpt.get('step')})  "
              f"cache_dir={cache_dir}  dataset_len={n_eval_avail}", flush=True)
        cret = contrastive_retrieval_eval(predictor, vision_proj, ambient_proj, eval_loader, device,
                                           max_clips=n_eval_avail)
        n_clips_seen = int(cret.pop("n_clips"))
        assert n_clips_seen == n_eval_avail, (
            f"eval_loader yielded {n_clips_seen} clips, expected the full "
            f"gallery ({n_eval_avail}) -- distributed_sampler leak?"
        )
        print(f"[eval_checkpoint_gallery]   dataset_len={n_eval_avail}  clips_seen={n_clips_seen}  (full-gallery OK)",
              flush=True)
        for k, v in sorted(cret.items()):
            if "R@" in k:
                print(f"[eval_checkpoint_gallery]   {k}={v:.2f}%", flush=True)
            else:
                print(f"[eval_checkpoint_gallery]   {k}={v:.4f}", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
