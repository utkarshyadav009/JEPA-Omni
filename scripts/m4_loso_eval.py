"""scripts/m4_loso_eval.py — leave-one-speaker-out (LOSO) evaluation of the
M4 decision head on EasyCom turn-taking, to answer the flagged limitation:
the session-level split (checkpoints/m4_decision_head/gate_results.json,
speak_recall=0.994/silence_recall=0.976/accuracy=0.986) holds out
RECORDINGS, not VOICES -- all 5 speakers with usable close-mic audio
({3,4,5,6,7}) appear in both train and test sessions. This script instead
holds out a whole SPEAKER (all their sessions, all their ticks) and
reports the honest unseen-voice number.

Cheap re-use, not re-extraction: build_ticks() is deterministic (glob-
sorted), so re-calling it recovers the exact tick order used when
checkpoints/m4_decision_head/features_cache.pt's ec_train_sf/ec_test_sf
were extracted -- no need to re-run Whisper, just realign the cached
feature tensors with per-tick participant IDs (parsed from audio_path,
every tick -- speak or silence -- has one, since silence ticks are drawn
from a specific participant's own audio-gap).

VGGSound features/labels are reused unchanged from the cache in every fold
(no speaker concept there -- included so the missing-modality-as-zero
training recipe matches the original head exactly).

Usage:
    python scripts/m4_loso_eval.py --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
import torch.nn.functional as F

from models.m4_decision_head import SpeakSilenceHead, DecisionHeadConfig
from data.m4_easycom_turntaking import build_ticks

_PID_RE = re.compile(r"Participant_ID_(\d+)\.wav$")


def pid_of(tick) -> int:
    m = _PID_RE.search(tick.audio_path)
    return int(m.group(1))


def train_and_eval_fold(vgg_train_ws, vgg_train_y, vgg_test_ws, vgg_test_y,
                         ec_fold_train_sf, ec_fold_train_y, ec_heldout_sf, ec_heldout_y,
                         ws_dim, sf_dim, device, epochs, lr, weight_decay, batch_size, threshold, seed):
    torch.manual_seed(seed)

    def build_xy(ws, y_ws, sf, y_sf):
        n_ws, n_sf = ws.shape[0], sf.shape[0]
        zero_sf = torch.zeros(n_ws, sf_dim)
        zero_ws = torch.zeros(n_sf, ws_dim)
        X = torch.cat([torch.cat([ws, zero_sf], dim=1), torch.cat([zero_ws, sf], dim=1)], dim=0)
        y = torch.cat([y_ws, y_sf], dim=0)
        src = torch.cat([torch.zeros(n_ws, dtype=torch.long), torch.ones(n_sf, dtype=torch.long)], dim=0)
        return X, y, src

    X_train, y_train, _ = build_xy(vgg_train_ws, vgg_train_y, ec_fold_train_sf, ec_fold_train_y)
    X_test, y_test, src_test = build_xy(vgg_test_ws, vgg_test_y, ec_heldout_sf, ec_heldout_y)

    n_pos, n_neg = y_train.sum().item(), (1 - y_train).sum().item()
    pos_weight = torch.tensor(n_neg / max(1, n_pos)).to(device)

    cfg = DecisionHeadConfig(world_state_dim=ws_dim, speech_feat_dim=sf_dim)
    head = SpeakSilenceHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test, src_test = X_test.to(device), y_test.to(device), src_test.to(device)

    n_train = X_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            logits = head(X_train[idx, :ws_dim], X_train[idx, ws_dim:])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        logits = head(X_test[:, :ws_dim], X_test[:, ws_dim:])
        preds = (torch.sigmoid(logits) > threshold).float()
    out = {}
    for name, mask_val in [("vgg_mechanism_check", 0), ("easycom_heldout_speaker", 1)]:
        m = src_test == mask_val
        if m.sum() == 0:
            continue
        yy, pp = y_test[m], preds[m]
        n_pos_s, n_neg_s = yy.sum().item(), (1 - yy).sum().item()
        speak_recall = ((pp == 1) & (yy == 1)).sum().item() / max(1, n_pos_s)
        silence_recall = ((pp == 0) & (yy == 0)).sum().item() / max(1, n_neg_s)
        acc = (pp == yy).float().mean().item()
        out[name] = {"n": int(m.sum()), "gt_speak_rate": n_pos_s / max(1, m.sum().item()),
                     "speak_recall": speak_recall, "silence_recall": silence_recall, "accuracy": acc}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cache-path", default="checkpoints/m4_decision_head/features_cache.pt")
    p.add_argument("--out", default="checkpoints/m4_decision_head/loso_results.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[loso] loading cached features from {args.cache_path}...", flush=True)
    cache = torch.load(args.cache_path, weights_only=False)
    ws_dim = cache["vgg_train_ws"].shape[1]
    sf_dim = cache["ec_train_sf"].shape[1]

    print("[loso] rebuilding EasyCom tick order (deterministic, matches cache extraction order)...", flush=True)
    ec_train_ticks, ec_test_ticks = build_ticks()
    all_ticks = ec_train_ticks + ec_test_ticks
    all_sf = torch.cat([cache["ec_train_sf"], cache["ec_test_sf"]], dim=0)
    all_y = torch.cat([cache["ec_train_y"], cache["ec_test_y"]], dim=0)
    assert all_sf.shape[0] == len(all_ticks), "cache/tick-order mismatch -- cache was not built from build_ticks()"

    pids = torch.tensor([pid_of(t) for t in all_ticks], dtype=torch.long)
    speaker_ids = sorted(set(pids.tolist()))
    print(f"[loso] {len(all_ticks)} total EasyCom ticks across {len(speaker_ids)} speakers with usable "
          f"close-mic audio: {speaker_ids}", flush=True)
    for pid in speaker_ids:
        n = int((pids == pid).sum())
        print(f"[loso]   speaker {pid}: {n} ticks ({(pids==pid).sum().item()/len(pids):.1%} of total)", flush=True)

    results = {}
    for pid in speaker_ids:
        heldout_mask = pids == pid
        train_mask = ~heldout_mask
        print(f"\n[loso] === fold: hold out speaker {pid} ({int(heldout_mask.sum())} ticks) ===", flush=True)
        fold_out = train_and_eval_fold(
            cache["vgg_train_ws"], cache["vgg_train_y"], cache["vgg_test_ws"], cache["vgg_test_y"],
            all_sf[train_mask], all_y[train_mask], all_sf[heldout_mask], all_y[heldout_mask],
            ws_dim, sf_dim, device, args.epochs, args.lr, args.weight_decay, args.batch_size,
            args.threshold, args.seed)
        print(f"[loso] speaker {pid}: {json.dumps(fold_out)}", flush=True)
        results[f"speaker_{pid}"] = fold_out

    ec_results = [v["easycom_heldout_speaker"] for v in results.values() if "easycom_heldout_speaker" in v]
    macro_speak_recall = sum(r["speak_recall"] for r in ec_results) / len(ec_results)
    macro_silence_recall = sum(r["silence_recall"] for r in ec_results) / len(ec_results)
    macro_acc = sum(r["accuracy"] for r in ec_results) / len(ec_results)
    n_total = sum(r["n"] for r in ec_results)
    n_correct = sum(r["accuracy"] * r["n"] for r in ec_results)
    micro_acc = n_correct / n_total

    summary = {"macro_speak_recall": macro_speak_recall, "macro_silence_recall": macro_silence_recall,
               "macro_accuracy": macro_acc, "micro_accuracy": micro_acc, "n_total_heldout_ticks": n_total}
    print(f"\n[loso] === LOSO SUMMARY (5 folds, unseen-voice) ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[loso] vs SESSION-split headline (known voices, held-out recordings): "
          f"speak_recall=0.994 silence_recall=0.976 accuracy=0.986")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_speaker": results, "summary": summary}, f, indent=2)
    print(f"[loso] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
