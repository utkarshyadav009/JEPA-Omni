"""train_decision_head.py — train the M4 speak/silence decision head.

Cheap by design: no LLM forward/backward anywhere in this script. Extracts
World-State (VGGSound pseudo-timeline ticks, via frozen M2) and speech-
activity features (EasyCom turn-taking ticks, via frozen Whisper) ONCE,
caches them as plain tensors, then trains a small MLP (models/
m4_decision_head.py) on the cached features -- standard BCE + pos_weight
class balancing, a genuine 2-class problem unlike the LLM/LoRA route's
150K-way discrete token competition.

Usage:
    python train_decision_head.py --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_m3 import build_splits, _cap_ambient_len, CACHE_DIR
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m4_speech import WhisperSpeechEncoder
from models.m4_decision_head import SpeakSilenceHead, DecisionHeadConfig
from data.m4_pseudo_timeline import M4PseudoTimelineDataset, m4_collate_fn, SPEAK_FRACTION, SILENCE_FRACTIONS
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset

FIELD = "gpt_sound_acoustic"


@torch.no_grad()
def extract_vgg_features(predictor, tokenizer_pad_id, pairs, device, tag):
    """Returns (world_state (N,1024), label (N,) 1=speak/0=silence)."""
    ds = M4PseudoTimelineDataset(pairs, CACHE_DIR, _DummyTokenizer(tokenizer_pad_id), silence_token_id=0)
    ws_list, label_list = [], []
    for i in range(len(ds)):
        item = ds[i]
        batch = m4_collate_fn([item], tokenizer_pad_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            ws = predictor.encode_world_state(feats_f, tbins)
        ws_list.append(ws.float().cpu())
        label_list.append(1 if item["label"] == "speak" else 0)
        if (i + 1) % 200 == 0:
            print(f"[decision-head] {tag}: extracted {i+1}/{len(ds)}", flush=True)
    return torch.cat(ws_list, 0), torch.tensor(label_list, dtype=torch.float32)


class _DummyTokenizer:
    """M4PseudoTimelineDataset needs a tokenizer to build caption target_ids,
    which we don't use here (we only need feats/tbins/label) -- a minimal
    stand-in avoids loading the real 1.5B-param tokenizer/embedding table
    for a script that never touches the LLM."""
    eos_token_id = 0

    def __init__(self, pad_id):
        self.pad_token_id = pad_id

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0]}   # never actually used downstream here


@torch.no_grad()
def extract_easycom_features(whisper, ticks, device, tag):
    ds = EasyComTurnTakingDataset(ticks)
    feat_list, label_list = [], []
    for i in range(len(ds)):
        item = ds[i]
        hidden, valid_frames = whisper([item["waveform"]], [item["duration_sec"]], device)
        vf = int(valid_frames[0].item())
        pooled = hidden[0, :vf].float().mean(dim=0, keepdim=True)   # (1, whisper_hidden)
        feat_list.append(pooled.cpu())
        label_list.append(1 if item["is_speak"] else 0)
        if (i + 1) % 500 == 0:
            print(f"[decision-head] {tag}: extracted {i+1}/{len(ds)}", flush=True)
    return torch.cat(feat_list, 0), torch.tensor(label_list, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--limit-vgg", type=int, default=3000, help="cap VGGSound clips (x4 ticks each) for feature extraction cost")
    p.add_argument("--ckpt-dir", default="checkpoints/m4_decision_head")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-path", default="checkpoints/m4_decision_head/features_cache.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if os.path.isfile(args.cache_path):
        print(f"[decision-head] loading cached features from {args.cache_path}", flush=True)
        cache = torch.load(args.cache_path, weights_only=False)
    else:
        print("[decision-head] loading frozen M2 predictor...", flush=True)
        predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
        predictor = AVJepaPredictor(predictor_cfg).to(device)
        m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
        predictor.load_state_dict(m2ckpt["model"], strict=True)
        predictor.eval()

        print("[decision-head] loading frozen whisper encoder...", flush=True)
        whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

        vgg_train_pairs3, vgg_test_pairs3 = build_splits(FIELD)
        vgg_train_pairs = [(c, t) for c, _, t in vgg_train_pairs3][:args.limit_vgg]
        vgg_test_pairs = [(c, t) for c, _, t in vgg_test_pairs3][:max(300, args.limit_vgg // 5)]

        print(f"[decision-head] extracting VGGSound World-State features "
              f"(train_clips={len(vgg_train_pairs)} test_clips={len(vgg_test_pairs)}, x4 ticks each)...", flush=True)
        vgg_train_ws, vgg_train_y = extract_vgg_features(predictor, 0, vgg_train_pairs, device, "vgg-train")
        vgg_test_ws, vgg_test_y = extract_vgg_features(predictor, 0, vgg_test_pairs, device, "vgg-test")

        ec_train_ticks, ec_test_ticks = build_ticks()
        print(f"[decision-head] extracting EasyCom speech-activity features "
              f"(train_ticks={len(ec_train_ticks)} test_ticks={len(ec_test_ticks)})...", flush=True)
        ec_train_sf, ec_train_y = extract_easycom_features(whisper, ec_train_ticks, device, "easycom-train")
        ec_test_sf, ec_test_y = extract_easycom_features(whisper, ec_test_ticks, device, "easycom-test")

        cache = {
            "vgg_train_ws": vgg_train_ws, "vgg_train_y": vgg_train_y,
            "vgg_test_ws": vgg_test_ws, "vgg_test_y": vgg_test_y,
            "ec_train_sf": ec_train_sf, "ec_train_y": ec_train_y,
            "ec_test_sf": ec_test_sf, "ec_test_y": ec_test_y,
        }
        torch.save(cache, args.cache_path)
        print(f"[decision-head] cached features to {args.cache_path}", flush=True)

    ws_dim = cache["vgg_train_ws"].shape[1]
    sf_dim = cache["ec_train_sf"].shape[1]

    def build_xy(ws, y_ws, sf, y_sf):
        n_ws, n_sf = ws.shape[0], sf.shape[0]
        zero_sf = torch.zeros(n_ws, sf_dim)
        zero_ws = torch.zeros(n_sf, ws_dim)
        X_ws = torch.cat([ws, zero_sf], dim=1)
        X_sf = torch.cat([zero_ws, sf], dim=1)
        X = torch.cat([X_ws, X_sf], dim=0)
        y = torch.cat([y_ws, y_sf], dim=0)
        src = torch.cat([torch.zeros(n_ws, dtype=torch.long), torch.ones(n_sf, dtype=torch.long)], dim=0)
        return X, y, src

    X_train, y_train, src_train = build_xy(cache["vgg_train_ws"], cache["vgg_train_y"],
                                            cache["ec_train_sf"], cache["ec_train_y"])
    X_test, y_test, src_test = build_xy(cache["vgg_test_ws"], cache["vgg_test_y"],
                                         cache["ec_test_sf"], cache["ec_test_y"])
    print(f"[decision-head] train N={X_train.shape[0]} (speak={int(y_train.sum())}, "
          f"silence={int((1-y_train).sum())})  test N={X_test.shape[0]} "
          f"(speak={int(y_test.sum())}, silence={int((1-y_test).sum())})", flush=True)

    n_pos, n_neg = y_train.sum().item(), (1 - y_train).sum().item()
    pos_weight = torch.tensor(n_neg / max(1, n_pos))
    print(f"[decision-head] class balance: pos(speak)={int(n_pos)} neg(silence)={int(n_neg)}  "
          f"pos_weight={pos_weight.item():.3f}", flush=True)

    cfg = DecisionHeadConfig(world_state_dim=ws_dim, speech_feat_dim=sf_dim)
    head = SpeakSilenceHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X_train, y_train, src_train = X_train.to(device), y_train.to(device), src_train.to(device)
    X_test, y_test, src_test = X_test.to(device), y_test.to(device), src_test.to(device)
    pos_weight = pos_weight.to(device)

    def eval_split(X, y, src, threshold):
        head.eval()
        with torch.no_grad():
            logits = head(X[:, :ws_dim], X[:, ws_dim:])
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        out = {}
        for name, mask_val in [("vgg", 0), ("easycom", 1)]:
            m = src == mask_val
            if m.sum() == 0:
                continue
            yy, pp = y[m], preds[m]
            n_pos_s = yy.sum().item()
            n_neg_s = (1 - yy).sum().item()
            speak_recall = ((pp == 1) & (yy == 1)).sum().item() / max(1, n_pos_s)
            silence_recall = ((pp == 0) & (yy == 0)).sum().item() / max(1, n_neg_s)
            acc = (pp == yy).float().mean().item()
            out[name] = {"n": int(m.sum()), "gt_speak_rate": n_pos_s / max(1, m.sum().item()),
                         "speak_recall": speak_recall, "silence_recall": silence_recall, "accuracy": acc}
        head.train()
        return out

    n_train = X_train.shape[0]
    bs = args.batch_size
    print(f"[decision-head] training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = head(X_train[idx, :ws_dim], X_train[idx, ws_dim:])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_stats = eval_split(X_test, y_test, src_test, args.decision_threshold)
            print(f"[decision-head] epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n_train:.4f}  "
                  f"test: {json.dumps(test_stats)}", flush=True)

    final_stats = eval_split(X_test, y_test, src_test, args.decision_threshold)
    print(f"\n[decision-head] FINAL (threshold={args.decision_threshold}): {json.dumps(final_stats, indent=2)}", flush=True)

    torch.save({"state_dict": head.state_dict(), "cfg": cfg.__dict__, "threshold": args.decision_threshold},
               os.path.join(args.ckpt_dir, "best.pt"))
    with open(os.path.join(args.ckpt_dir, "gate_results.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"[decision-head] DONE. wrote {args.ckpt_dir}/best.pt and gate_results.json", flush=True)


if __name__ == "__main__":
    main()
