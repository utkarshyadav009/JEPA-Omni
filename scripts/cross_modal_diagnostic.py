"""scripts/cross_modal_diagnostic.py — STOP-2: prove cross-modal necessity.

Evaluates trained windowed vs whole models with two controls on a held-out batch:
  (a) TEMPORAL-SHUFFLE: predict clip i's masked tokens using context from clip j≠i
  (b) VISION-DROPOUT:   zero the vision tokens, predict masked ambient from ambient-context only

Usage (run AFTER both 1500-step training runs complete):
    conda run -n jepa-omni python scripts/cross_modal_diagnostic.py \
        --ckpt-windowed checkpoints/m2_windowed/last.pt \
        --ckpt-whole    checkpoints/m2_whole/last.pt \
        --cache-dir /dev/shm/jepa_m2_cache \
        --n-eval 64
"""

from __future__ import annotations
import argparse
import os
import sys
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor, effective_rank
from data.av_cached_dataset import AVCachedDataset, av_collate_fn


def load_predictor(ckpt_path: str, device: str) -> AVJepaPredictor:
    cfg   = AVJepaConfig()
    model = AVJepaPredictor(cfg).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = sd.get("model", sd)
        model.load_state_dict(state, strict=True)
        print(f"  loaded {ckpt_path}", flush=True)
    else:
        print(f"  WARNING: {ckpt_path} not found — using random init", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_whole_mask(tbins: Dict[str, Tensor], masked_mod: str = "ambient") -> Dict[str, Tensor]:
    """Mask 100% of one modality."""
    mask = {}
    for m, bins in tbins.items():
        if m == masked_mod:
            mask[m] = torch.ones_like(bins, dtype=torch.bool)
        else:
            mask[m] = torch.zeros_like(bins, dtype=torch.bool)
    return mask


def compute_loss(
    model: AVJepaPredictor,
    feats: Dict[str, Tensor],
    tbins: Dict[str, Tensor],
    mask:  Dict[str, Tensor],
    device: str,
) -> float:
    feats  = {k: v.to(device) for k, v in feats.items()}
    tbins  = {k: v.to(device) for k, v in tbins.items()}
    mask   = {k: v.to(device) for k, v in mask.items()}
    with torch.no_grad():
        loss, _ = model(feats, tbins, mask)
    return loss.item()


@torch.no_grad()
def run_diagnostics(
    model: AVJepaPredictor,
    batches: List[Dict],
    device: str,
    label: str,
) -> Dict[str, float]:
    """Run normal / shuffle / vision-dropout on held-out batches.

    Always masks ambient (100% whole mask) — removes ambiguity about which
    modality is being tested.
    """
    normal_losses    = []
    shuffle_losses   = []
    vdropout_losses  = []

    for batch in batches:
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        B     = feats["vision"].shape[0]

        # Whole mask on ambient (full cross-modal test)
        mask = make_whole_mask(tbins, masked_mod="ambient")

        # (1) Normal: predict clip-i ambient from clip-i vision
        with torch.no_grad():
            loss_normal, _ = model(feats, tbins, mask)
        normal_losses.append(loss_normal.item())

        # (2) Temporal-shuffle: roll vision by 1 → predict clip-i ambient from clip-j vision
        feats_shuffled = dict(feats)
        feats_shuffled["vision"] = torch.roll(feats["vision"], shifts=1, dims=0)
        with torch.no_grad():
            loss_shuffle, _ = model(feats_shuffled, tbins, mask)
        shuffle_losses.append(loss_shuffle.item())

        # (3) Vision-dropout: zero the vision tokens
        feats_nodrop = dict(feats)
        feats_nodrop["vision"] = torch.zeros_like(feats["vision"])
        with torch.no_grad():
            loss_vdrop, _ = model(feats_nodrop, tbins, mask)
        vdropout_losses.append(loss_vdrop.item())

    n = len(normal_losses)
    avg_normal   = sum(normal_losses)   / n
    avg_shuffle  = sum(shuffle_losses)  / n
    avg_vdropout = sum(vdropout_losses) / n

    results = {
        "normal":           round(avg_normal,  4),
        "shuffle_delta":    round(avg_shuffle  - avg_normal, 4),
        "vdropout_delta":   round(avg_vdropout - avg_normal, 4),
        "shuffle_abs":      round(avg_shuffle,  4),
        "vdropout_abs":     round(avg_vdropout, 4),
    }

    print(f"\n[{label}]")
    print(f"  normal loss:            {results['normal']:.4f}")
    print(f"  shuffle loss:           {results['shuffle_abs']:.4f}  (Δ={results['shuffle_delta']:+.4f})")
    print(f"  vision-dropout loss:    {results['vdropout_abs']:.4f}  (Δ={results['vdropout_delta']:+.4f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-windowed", default="checkpoints/m2_windowed/last.pt")
    parser.add_argument("--ckpt-whole",    default="checkpoints/m2_whole/last.pt")
    parser.add_argument("--cache-dir",     default="/dev/shm/jepa_m2_cache")
    parser.add_argument("--n-eval",        type=int, default=64,
                        help="Number of clips for diagnostic (batched).")
    parser.add_argument("--batch-size",    type=int, default=16)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load eval clips
    ds = AVCachedDataset(args.cache_dir, max_tdm_bins=512, audio_mode="mean")
    rng = random.Random(args.seed)
    eval_ids = list(ds.clip_ids)
    rng.shuffle(eval_ids)
    eval_ids = eval_ids[:args.n_eval]
    ds.clip_ids = eval_ids

    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=av_collate_fn, shuffle=False)
    batches = list(loader)
    print(f"Diagnostic: {args.n_eval} clips, {len(batches)} batches")

    # Load both models
    print("\nLoading windowed model...")
    m_win  = load_predictor(args.ckpt_windowed, args.device)
    print("Loading whole model...")
    m_whole = load_predictor(args.ckpt_whole,   args.device)

    # Run diagnostics
    r_win   = run_diagnostics(m_win,   batches, args.device, "windowed")
    r_whole = run_diagnostics(m_whole, batches, args.device, "whole")

    print("\n" + "=" * 70)
    print("STOP-2 DIAGNOSTIC RESULTS  (whole-mask on ambient, vision is context)")
    print("=" * 70)
    print(f"{'Model':<12} {'normal':>8} {'shuffle-Δ':>12} {'vdropout-Δ':>14}")
    print("-" * 70)
    for label, r in [("windowed", r_win), ("whole", r_whole)]:
        print(f"{label:<12} {r['normal']:>8.4f} {r['shuffle_delta']:>+12.4f} {r['vdropout_delta']:>+14.4f}")
    print("=" * 70)

    # Decision rule
    print("\nDECISION RULE:")
    print("  Genuine cross-modal → shuffle_delta LARGE, vdropout_delta LARGE")
    print("  Shortcutting (within-modal interp) → shuffle_delta small OR vdropout_delta ~0")
    print()
    for label, r in [("windowed", r_win), ("whole", r_whole)]:
        shuffle_ok   = r["shuffle_delta"]  > 0.05 * r["normal"]
        vdropout_ok  = r["vdropout_delta"] > 0.05 * r["normal"]
        verdict = "CROSS-MODAL" if (shuffle_ok and vdropout_ok) else "SHORTCUT-SUSPECTED"
        print(f"  {label}: shuffle_delta={r['shuffle_delta']:+.4f} vdropout_delta={r['vdropout_delta']:+.4f} → {verdict}")


if __name__ == "__main__":
    main()
