"""scripts/m4_decision_head_3class_swap_control.py — swap-control for the
3-class decision head (SPEAK/SILENCE/BACKCHANNEL), same discipline as the
2-class version: feed a stale/wrong World-State and confirm the decision
tracks the DONOR's label, not the target's.

Usage:
    python scripts/m4_decision_head_3class_swap_control.py
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, IDX_TO_LABEL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", default="checkpoints/m4_decision_head_3class/best.pt")
    p.add_argument("--cache-path", default="checkpoints/m4_decision_head/features_cache.pt")
    p.add_argument("--out", default="checkpoints/m4_decision_head_3class/swap_control_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=False)
    cfg = DecisionHeadConfig(**ckpt["cfg"])
    head = ThreeClassHead(cfg).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    cache = torch.load(args.cache_path, weights_only=False)
    ws = cache["vgg_test_ws"].to(device)
    y = cache["vgg_test_y"].to(device).long()   # 0=silence, 1=speak (VGG has no backchannel)
    zero_sf = torch.zeros(ws.shape[0], cfg.speech_feat_dim, device=device)

    N = ws.shape[0]
    swapped_idx = torch.roll(torch.arange(N), shifts=1)
    ws_swapped = ws[swapped_idx]
    y_donor = y[swapped_idx]

    with torch.no_grad():
        preds_normal = head(ws, zero_sf).argmax(dim=-1)
        preds_swapped = head(ws_swapped, zero_sf).argmax(dim=-1)

    acc_normal_vs_own = (preds_normal == y).float().mean().item()
    acc_swapped_vs_target = (preds_swapped == y).float().mean().item()
    acc_swapped_vs_donor = (preds_swapped == y_donor).float().mean().item()
    frac_changed = (preds_normal != preds_swapped).float().mean().item()

    results = {
        "n": N, "acc_normal_vs_own_label": acc_normal_vs_own,
        "acc_swapped_vs_target_label": acc_swapped_vs_target,
        "acc_swapped_vs_donor_label": acc_swapped_vs_donor,
        "fraction_decision_changed_under_swap": frac_changed,
    }
    print(json.dumps(results, indent=2))
    print(f"\nNormal accuracy:                {acc_normal_vs_own:.3f}")
    print(f"Swapped vs TARGET (want ~chance): {acc_swapped_vs_target:.3f}")
    print(f"Swapped vs DONOR (want ~normal):  {acc_swapped_vs_donor:.3f}")
    print(f"Fraction changed under swap:      {frac_changed:.3f}")
    print(f"{'PASS' if frac_changed > 0.05 and acc_swapped_vs_donor > acc_swapped_vs_target else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
