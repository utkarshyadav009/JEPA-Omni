"""scripts/m4_backchannel_detection_test.py — DETECTION-side comparison for
backchannel handling: false-halt rate on EasyCom backchannel test segments
under OLD (2-class SpeakSilenceHead: any speech-activity -> halt) vs NEW
(3-class ThreeClassHead: only SPEAK triggers halt, BACKCHANNEL does not).

Reuses the already-extracted 3-class EasyCom test features (label3-tagged)
from checkpoints/m4_decision_head_3class/easycom_3class_features_cache.pt --
no re-extraction needed, both heads consume the same [World-State;
speech-feat] input contract (World-State is the zero-vector for EasyCom
ticks in both cases, same as training).

Usage:
    python scripts/m4_backchannel_detection_test.py
"""
from __future__ import annotations

import json
import os

import torch

from models.m4_decision_head import SpeakSilenceHead, ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = torch.load("checkpoints/m4_decision_head_3class/easycom_3class_features_cache.pt",
                        weights_only=False)
    sf_test, y3_test = cache["ec_test_sf"].to(device), cache["ec_test_y3"].to(device)

    bc_mask = y3_test == LABEL_TO_IDX["backchannel"]
    speak_mask = y3_test == LABEL_TO_IDX["speak"]
    n_bc = bc_mask.sum().item()
    n_speak = speak_mask.sum().item()
    print(f"[bc-detect] EasyCom test: {n_bc} backchannel ticks, {n_speak} real-speak ticks", flush=True)

    ws_dim = 1024
    zero_ws = torch.zeros(sf_test.shape[0], ws_dim, device=device)

    # OLD: 2-class SpeakSilenceHead. Any tick classified "speak" (sigmoid>0.5) -> halt.
    old_ckpt = torch.load("checkpoints/m4_decision_head/best.pt", map_location=device, weights_only=False)
    old_cfg = DecisionHeadConfig(**old_ckpt["cfg"])
    old_head = SpeakSilenceHead(old_cfg).to(device)
    old_head.load_state_dict(old_ckpt["state_dict"])
    old_head.eval()

    # NEW: 3-class ThreeClassHead. Only argmax==SPEAK(1) -> halt. BACKCHANNEL(2) does not.
    new_ckpt = torch.load("checkpoints/m4_decision_head_3class/best.pt", map_location=device, weights_only=False)
    new_cfg = DecisionHeadConfig(**new_ckpt["cfg"])
    new_head = ThreeClassHead(new_cfg).to(device)
    new_head.load_state_dict(new_ckpt["state_dict"])
    new_head.eval()

    with torch.no_grad():
        old_logits = old_head(zero_ws, sf_test)
        old_halts = (torch.sigmoid(old_logits) > 0.5)   # True = would halt

        new_logits = new_head(zero_ws, sf_test)
        new_pred = new_logits.argmax(dim=-1)
        new_halts = (new_pred == LABEL_TO_IDX["speak"])   # only SPEAK halts

    # false-halt rate: fraction of BACKCHANNEL ticks that trigger a halt
    old_false_halt = old_halts[bc_mask].float().mean().item()
    new_false_halt = new_halts[bc_mask].float().mean().item()

    # sanity/regression check: real SPEAK ticks should STILL halt under both policies
    old_true_halt_on_speak = old_halts[speak_mask].float().mean().item()
    new_true_halt_on_speak = new_halts[speak_mask].float().mean().item()

    results = {
        "n_backchannel": n_bc,
        "n_real_speak": n_speak,
        "false_halt_rate_on_backchannel": {"old_binary": old_false_halt, "new_3class": new_false_halt},
        "true_halt_rate_on_real_speak": {"old_binary": old_true_halt_on_speak, "new_3class": new_true_halt_on_speak},
    }
    print(json.dumps(results, indent=2))
    print(f"\nFalse-halt on backchannel: OLD binary={old_false_halt:.3f}  NEW 3-class={new_false_halt:.3f}")
    print(f"True-halt on real speak:   OLD binary={old_true_halt_on_speak:.3f}  NEW 3-class={new_true_halt_on_speak:.3f}")

    out_path = "checkpoints/m4_decision_head_3class/backchannel_detection_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
