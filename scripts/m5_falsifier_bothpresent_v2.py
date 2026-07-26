"""scripts/m5_falsifier_bothpresent.py — A1 step 3: evaluate the
both-present-retrained 3-class decision head (checkpoints/
m4_decision_head_3class_bothpresent/best.pt) at n>=300 (100/class),
using the SAME real EasyCom test cache the model was scored against
during training (test_bothpresent_v2_cache.pt -- guarantees the eval set
here is identical to what training reported, no separate sampling path
to drift out of sync).

Three conditions, exactly as specified:
  (a) real WS (fresh, correctly paired) + real SF
  (b) WS zeroed (control -- the OLD training regime's exact input)
  (c) WS swapped (rolled by 1 -- mismatched pairing, same real SF)

PASS = (a) beats both (b) and (c) by a margin that clears binomial/
bootstrap noise at this n. Reports full confusion matrices for all three
plus a paired bootstrap CI on the accuracy gaps (a-b) and (a-c).

Usage:
    python scripts/m5_falsifier_bothpresent.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX, IDX_TO_LABEL


def eval_condition(decision_head, ws, sf, y, device):
    with torch.no_grad():
        logits = decision_head(ws.to(device), sf.to(device))
        preds = logits.argmax(dim=-1).cpu()
    n_classes = 3
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y.tolist(), preds.tolist()):
        conf[t, p] += 1
    acc = (preds == y).float().mean().item()
    per_class_recall, per_class_f1 = {}, {}
    for c in range(n_classes):
        tp = conf[c, c].item()
        n_true = conf[c].sum().item()
        n_pred = conf[:, c].sum().item()
        recall = tp / max(1, n_true)
        precision = tp / max(1, n_pred)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class_recall[IDX_TO_LABEL[c]] = recall
        per_class_f1[IDX_TO_LABEL[c]] = f1
    macro_f1 = sum(per_class_f1.values()) / n_classes
    return {
        "accuracy": acc, "macro_f1": macro_f1,
        "per_class_recall": per_class_recall, "per_class_f1": per_class_f1,
        "confusion_matrix_rows_true_cols_pred": conf.tolist(),
        "label_order": [IDX_TO_LABEL[i] for i in range(n_classes)],
    }, preds


def make_within_session_perm(sessions: torch.Tensor, seed: int) -> torch.Tensor:
    """Derangement restricted to same-session partners: for each index i,
    pick a DIFFERENT index j with sessions[j] == sessions[i]. Raises if any
    session has only 1 member (no valid within-session partner exists)."""
    g = torch.Generator().manual_seed(seed)
    n = sessions.shape[0]
    perm = torch.arange(n)
    for sess in sessions.unique().tolist():
        idx = (sessions == sess).nonzero(as_tuple=True)[0]
        if idx.shape[0] < 2:
            raise ValueError(f"session {sess} has only {idx.shape[0]} test items -- no within-session swap partner possible")
        shuffled = idx[torch.randperm(idx.shape[0], generator=g)]
        # ensure no fixed points (derangement) by rotating if any survive
        while (shuffled == idx).any():
            shuffled = idx[torch.randperm(idx.shape[0], generator=g)]
        perm[idx] = shuffled
    return perm


def make_cross_session_perm(sessions: torch.Tensor, seed: int) -> torch.Tensor:
    """For each index i, pick a random index j with sessions[j] != sessions[i]
    (a genuinely different scene/session, not just a different tick)."""
    g = torch.Generator().manual_seed(seed)
    n = sessions.shape[0]
    perm = torch.empty(n, dtype=torch.long)
    for i in range(n):
        candidates = (sessions != sessions[i]).nonzero(as_tuple=True)[0]
        j = candidates[torch.randint(0, candidates.shape[0], (1,), generator=g)]
        perm[i] = j
    return perm


def bootstrap_acc_gap(preds_a, preds_b, y, n_boot=10000, seed=0):
    """Paired bootstrap over the n test items: resample indices with
    replacement, recompute accuracy(a)-accuracy(b) each time, report the
    95% CI of the gap. If the CI excludes 0, the gap is not noise."""
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    correct_a = (preds_a == y).float()
    correct_b = (preds_b == y).float()
    gaps = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        gaps.append((correct_a[idx].mean() - correct_b[idx].mean()).item())
    gaps.sort()
    lo = gaps[int(0.025 * n_boot)]
    hi = gaps[int(0.975 * n_boot)]
    return {"point_estimate": (correct_a.mean() - correct_b.mean()).item(),
            "ci95_lo": lo, "ci95_hi": hi, "excludes_zero": (lo > 0 or hi < 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/m4_decision_head_3class_bothpresent_v2")
    p.add_argument("--out", default="checkpoints/m4_decision_head_3class_bothpresent_v2/A1_FALSIFIER_RESULTS_V2.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-dimension mean/std for the matched-statistics random control and
    # the dataset-mean control are computed from the TRAIN cache (n=2282,
    # the larger, more representative pool) -- features already extracted,
    # no new encoding needed.
    train_cache_path = os.path.join(args.ckpt_dir, "train_bothpresent_v2_cache.pt")
    train_batch = torch.load(train_cache_path, weights_only=False)
    train_ws = torch.stack([c["ws"] for c in train_batch], 0)
    ws_mean_per_dim = train_ws.mean(dim=0)
    ws_std_per_dim = train_ws.std(dim=0)
    ws_dataset_mean = ws_mean_per_dim.clone()
    print(f"[a1-falsifier] train WS stats: n={train_ws.shape[0]} "
          f"mean_norm={ws_mean_per_dim.norm().item():.3f} "
          f"mean_per_dim_std={ws_std_per_dim.mean().item():.3f}", flush=True)

    test_cache_path = os.path.join(args.ckpt_dir, "test_bothpresent_v2_cache.pt")
    batch = torch.load(test_cache_path, weights_only=False)
    ws = torch.stack([c["ws"] for c in batch], 0)
    sf = torch.stack([c["sf"] for c in batch], 0)
    y = torch.tensor([LABEL_TO_IDX[c["label3"]] for c in batch], dtype=torch.long)
    sessions = torch.tensor([c["session"] for c in batch], dtype=torch.long)
    print(f"[a1-falsifier] session counts: "
          f"{ {int(s): int((sessions==s).sum()) for s in sessions.unique()} }", flush=True)
    print(f"[a1-falsifier] n={ws.shape[0]}  label counts: "
          f"{ {lbl: int((y == idx).sum()) for lbl, idx in LABEL_TO_IDX.items()} }", flush=True)

    ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pt"), map_location=device, weights_only=False)
    cfg = DecisionHeadConfig(**ckpt["cfg"])
    head = ThreeClassHead(cfg).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    ws_dim = cfg.world_state_dim
    results = {}

    print("\n[a1-falsifier] === condition (a) real fresh WS ===", flush=True)
    stats_a, preds_a = eval_condition(head, ws, sf, y, device)
    print(json.dumps(stats_a, indent=2), flush=True)
    results["a_real_fresh"] = stats_a

    print("\n[a1-falsifier] === condition (b) WS zeroed (control) ===", flush=True)
    ws_zero = torch.zeros_like(ws)
    stats_b, preds_b = eval_condition(head, ws_zero, sf, y, device)
    print(json.dumps(stats_b, indent=2), flush=True)
    results["b_ws_zeroed"] = stats_b

    print("\n[a1-falsifier] === condition (c-within) WS swapped, WITHIN-SESSION (the real number) ===", flush=True)
    perm_within = make_within_session_perm(sessions, seed=3)
    ws_swapped_within = ws[perm_within]
    stats_c_within, preds_c_within = eval_condition(head, ws_swapped_within, sf, y, device)
    print(json.dumps(stats_c_within, indent=2), flush=True)
    results["c_ws_swapped_within_session"] = stats_c_within

    print("\n[a1-falsifier] === condition (c-cross) WS swapped, CROSS-SESSION (scene-ID, reported for contrast) ===", flush=True)
    perm_cross = make_cross_session_perm(sessions, seed=4)
    ws_swapped_cross = ws[perm_cross]
    stats_c_cross, preds_c_cross = eval_condition(head, ws_swapped_cross, sf, y, device)
    print(json.dumps(stats_c_cross, indent=2), flush=True)
    results["c_ws_swapped_cross_session"] = stats_c_cross

    print("\n[a1-falsifier] === condition (e) random vector, matched per-dim mean/std to real WS ===", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    ws_random_matched = torch.randn(ws.shape, generator=g) * ws_std_per_dim.unsqueeze(0) + ws_mean_per_dim.unsqueeze(0)
    stats_e, preds_e = eval_condition(head, ws_random_matched, sf, y, device)
    print(json.dumps(stats_e, indent=2), flush=True)
    results["e_random_matched_stats"] = stats_e

    print("\n[a1-falsifier] === condition (f) dataset-mean World-State (same vector for every row) ===", flush=True)
    ws_dataset_mean_bcast = ws_dataset_mean.unsqueeze(0).expand_as(ws)
    stats_f, preds_f = eval_condition(head, ws_dataset_mean_bcast, sf, y, device)
    print(json.dumps(stats_f, indent=2), flush=True)
    results["f_dataset_mean"] = stats_f

    print("\n[a1-falsifier] === six-condition accuracy/macro-F1 summary ===", flush=True)
    summary = {
        "a_real_fresh": (stats_a["accuracy"], stats_a["macro_f1"]),
        "b_ws_zeroed": (stats_b["accuracy"], stats_b["macro_f1"]),
        "c_within_session_swap": (stats_c_within["accuracy"], stats_c_within["macro_f1"]),
        "c_cross_session_swap": (stats_c_cross["accuracy"], stats_c_cross["macro_f1"]),
        "e_random_matched_stats": (stats_e["accuracy"], stats_e["macro_f1"]),
        "f_dataset_mean": (stats_f["accuracy"], stats_f["macro_f1"]),
    }
    for k, (acc, f1) in summary.items():
        print(f"  {k:28s} acc={acc*100:.2f}%  macro_F1={f1*100:.2f}%", flush=True)
    results["six_condition_summary"] = {k: {"accuracy": v[0], "macro_f1": v[1]} for k, v in summary.items()}

    print("\n[a1-falsifier] === bootstrap significance (n_boot=10000) ===", flush=True)
    gap_ab = bootstrap_acc_gap(preds_a, preds_b, y, seed=1)
    gap_ac_within = bootstrap_acc_gap(preds_a, preds_c_within, y, seed=2)
    gap_ac_cross = bootstrap_acc_gap(preds_a, preds_c_cross, y, seed=5)
    gap_ae = bootstrap_acc_gap(preds_a, preds_e, y, seed=6)
    gap_af = bootstrap_acc_gap(preds_a, preds_f, y, seed=8)
    gap_be = bootstrap_acc_gap(preds_b, preds_e, y, seed=9)   # zeroed vs matched-random: is "off-manifold" itself the effect?
    print(f"acc(a)-acc(b): {json.dumps(gap_ab)}", flush=True)
    print(f"acc(a)-acc(c_within): {json.dumps(gap_ac_within)}", flush=True)
    print(f"acc(a)-acc(c_cross): {json.dumps(gap_ac_cross)}", flush=True)
    print(f"acc(a)-acc(e_random_matched): {json.dumps(gap_ae)}", flush=True)
    print(f"acc(a)-acc(f_dataset_mean): {json.dumps(gap_af)}", flush=True)
    print(f"acc(b)-acc(e_random_matched) [zeroed vs matched-random, both off-manifold-ish]: {json.dumps(gap_be)}", flush=True)
    results["bootstrap_gap_a_minus_b"] = gap_ab
    results["bootstrap_gap_a_minus_c_within_session"] = gap_ac_within
    results["bootstrap_gap_a_minus_c_cross_session"] = gap_ac_cross
    results["bootstrap_gap_a_minus_e_random_matched"] = gap_ae
    results["bootstrap_gap_a_minus_f_dataset_mean"] = gap_af
    results["bootstrap_gap_b_minus_e"] = gap_be

    # PASS is gated on the WITHIN-session swap (the real grounding number) --
    # cross-session, random-matched, and dataset-mean are reported for
    # contrast/diagnosis only, never used to gate.
    passed = gap_ab["excludes_zero"] and gap_ab["point_estimate"] > 0 and \
             gap_ac_within["excludes_zero"] and gap_ac_within["point_estimate"] > 0
    results["PASS"] = passed
    print(f"\n[a1-falsifier] PASS = {passed} "
          f"(a beats b AND within-session-swapped-c, both gaps' 95% CI exclude zero and are positive)", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[a1-falsifier] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
