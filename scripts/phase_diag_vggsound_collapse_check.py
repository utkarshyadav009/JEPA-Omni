"""scripts/phase_diag_vggsound_collapse_check.py — Diagnostic 1a: is the
high within-modality pairwise cosine similarity seen on the EasyCom frozen
eval (vision=0.805, ambient=0.870) actually diagnostic of representation
collapse, or is it just what m2_fusion_20k_best's embeddings look like on
ANY gallery, including the in-domain 1545-clip VGGSound gallery it was
validated against? Same checkpoint, same pool_and_project pattern
(imported, not reimplemented), same metric (mean off-diagonal pairwise
cosine within z_v and within z_a).

If VGGSound's within-modality clustering is comparable to EasyCom's, high
clustering alone is not diagnostic of collapse and the "representation
collapse" claim in falsifier_tracking.md must be retracted.

Usage:
    python scripts/phase_diag_vggsound_collapse_check.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m2 import build_dataloader, pool_and_project
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from utils import load_config, cfg_get

M2_CKPT = "checkpoints/m2_fusion_20k_best/step19000_peak.pt"
PERSISTENT_CACHE = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k"
EVAL_SUBSET = "data/vggsound_eval_1545.txt"
EXPECTED_N = 1545


def mean_offdiag_cosine(z: torch.Tensor) -> float:
    sim = z @ z.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vgg-collapse-check] device={device}", flush=True)

    cfg = load_config("configs/m2.yaml")
    cfg["data"]["av_cache_dir"] = PERSISTENT_CACHE

    with open(EVAL_SUBSET) as f:
        eval_clip_ids = [l.strip() for l in f if l.strip()]

    eval_loader, _ = build_dataloader(
        cfg, clip_ids=eval_clip_ids, batch_size_override=64,
        num_workers_override=4, distributed_sampler=False, drop_last_override=False,
    )

    predictor_cfg = AVJepaConfig(
        d_model=int(cfg_get(cfg, "model.d_model", default=1024)),
        depth=int(cfg_get(cfg, "model.depth", default=8)),
        heads=int(cfg_get(cfg, "model.heads", default=8)),
        mlp_ratio=float(cfg_get(cfg, "model.mlp_ratio", default=4.0)),
        max_tdm_bins=int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
        dropout=float(cfg_get(cfg, "model.dropout", default=0.0)),
    )
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    vision_proj = nn.Linear(predictor_cfg.d_model, 256).to(device)
    ambient_proj = nn.Linear(predictor_cfg.d_model, 256).to(device)

    ckpt = torch.load(M2_CKPT, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    vision_proj.load_state_dict(ckpt["vision_proj"])
    ambient_proj.load_state_dict(ckpt["ambient_proj"])
    predictor.eval(); vision_proj.eval(); ambient_proj.eval()

    zv_all, za_all = [], []
    n_clips = 0
    clips_seen = 0
    with torch.no_grad():
        for batch in eval_loader:
            feats = {k: v.to(device) for k, v in batch["feats"].items()}
            tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
            zv_all.append(z_v.cpu()); za_all.append(z_a.cpu())
            clips_seen += z_v.shape[0]

    z_v = torch.cat(zv_all, 0)
    z_a = torch.cat(za_all, 0)
    n_clips = z_v.shape[0]
    print(f"[vgg-collapse-check] clips_seen={clips_seen}  n_clips={n_clips}  expected={EXPECTED_N}", flush=True)

    sim = z_v @ z_a.T
    gt = torch.arange(n_clips)
    matched_sim = sim.diagonal().mean().item()
    perm = torch.roll(torch.arange(n_clips), 1)
    shuffled_sim = sim[torch.arange(n_clips), perm].mean().item()

    results = {
        "clips_seen": clips_seen,
        "n_clips": n_clips,
        "clips_seen_matches_expected_1545": clips_seen == EXPECTED_N,
        "vision_within_modality_mean_offdiag_cosine": round(mean_offdiag_cosine(z_v), 4),
        "ambient_within_modality_mean_offdiag_cosine": round(mean_offdiag_cosine(z_a), 4),
        "matched_cos_sim": round(matched_sim, 4),
        "shuffled_cos_sim": round(shuffled_sim, 4),
        "shuffle_sanity_gap": round(matched_sim - shuffled_sim, 4),
        "easycom_comparison": {
            "easycom_vision_within_modality": 0.8048,
            "easycom_ambient_within_modality": 0.8703,
            "easycom_shuffle_sanity_gap": 0.0071,
        },
    }
    print(json.dumps(results, indent=2), flush=True)

    out_path = "checkpoints/vjepa21_shelved/VGGSOUND_COLLAPSE_CHECK_1545.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[vgg-collapse-check] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
