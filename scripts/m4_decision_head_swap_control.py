"""scripts/m4_decision_head_swap_control.py — swap-control for the M4
speak/silence decision head, same falsifier discipline used throughout
this project: feed a stale/wrong World-State and confirm the decision
CHANGES accordingly (tracks the donor's own label, not the target's) --
proof the head is actually reading its input, not memorizing a fixed prior.

Uses the cached features from train_decision_head.py's feature cache
(checkpoints/m4_decision_head/features_cache.pt) -- no LLM, no M2/Whisper
forward pass needed here, just the cached World-State vectors + the
trained head.

Usage:
    python scripts/m4_decision_head_swap_control.py
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from models.m4_decision_head import SpeakSilenceHead, DecisionHeadConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", default="checkpoints/m4_decision_head/best.pt")
    p.add_argument("--cache-path", default="checkpoints/m4_decision_head/features_cache.pt")
    p.add_argument("--out", default="checkpoints/m4_decision_head/swap_control_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=False)
    cfg = DecisionHeadConfig(**ckpt["cfg"])
    head = SpeakSilenceHead(cfg).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    threshold = ckpt["threshold"]

    cache = torch.load(args.cache_path, weights_only=False)
    ws = cache["vgg_test_ws"].to(device)          # (N, 1024) real VGGSound World-States
    y = cache["vgg_test_y"].to(device)             # (N,) 1=speak/0=silence
    zero_sf = torch.zeros(ws.shape[0], cfg.speech_feat_dim, device=device)

    N = ws.shape[0]
    swapped_idx = torch.roll(torch.arange(N), shifts=1)
    ws_swapped = ws[swapped_idx]
    y_donor = y[swapped_idx]   # the label that ACTUALLY belongs to the swapped-in World-State

    with torch.no_grad():
        logits_normal = head(ws, zero_sf)
        preds_normal = (torch.sigmoid(logits_normal) > threshold).float()
        logits_swapped = head(ws_swapped, zero_sf)
        preds_swapped = (torch.sigmoid(logits_swapped) > threshold).float()

    acc_normal_vs_own = (preds_normal == y).float().mean().item()
    acc_swapped_vs_target = (preds_swapped == y).float().mean().item()          # should be poor/chance-ish
    acc_swapped_vs_donor = (preds_swapped == y_donor).float().mean().item()      # should be close to acc_normal_vs_own
    frac_decision_changed = (preds_normal != preds_swapped).float().mean().item()

    results = {
        "n": N,
        "acc_normal_vs_own_label": acc_normal_vs_own,
        "acc_swapped_vs_target_label": acc_swapped_vs_target,
        "acc_swapped_vs_donor_label": acc_swapped_vs_donor,
        "fraction_decision_changed_under_swap": frac_decision_changed,
    }
    print(json.dumps(results, indent=2))
    print()
    print(f"Normal accuracy (own World-State vs own label):      {acc_normal_vs_own:.3f}")
    print(f"Swapped accuracy vs TARGET's label (want ~chance):   {acc_swapped_vs_target:.3f}")
    print(f"Swapped accuracy vs DONOR's label (want close to normal): {acc_swapped_vs_donor:.3f}")
    print(f"Fraction of decisions that changed when swapped:     {frac_decision_changed:.3f}")
    print(f"{'PASS' if frac_decision_changed > 0.05 and acc_swapped_vs_donor > acc_swapped_vs_target else 'FAIL'}: "
          f"decision head {'is' if frac_decision_changed > 0.05 else 'is NOT'} reading its World-State input "
          f"(a constant-output head would show frac_changed≈0)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
