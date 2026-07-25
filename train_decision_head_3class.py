"""train_decision_head_3class.py — extend the M4 decision head to 3 classes:
SPEAK / SILENCE / BACKCHANNEL.

Reuses the cached VGGSound World-State features from train_decision_head.py
(checkpoints/m4_decision_head/features_cache.pt) unchanged -- VGGSound has
no backchannel concept, its examples stay 2-of-3-class (speak/silence only,
never backchannel), which is fine for standard cross-entropy (no masking
needed, the model simply never receives a backchannel gradient from those
examples). Re-extracts EasyCom features (cheap, ~2 min) with the new
label3 field (data/m4_easycom_turntaking.py).

Usage:
    python train_decision_head_3class.py --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from models.m4_speech import WhisperSpeechEncoder
from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX, IDX_TO_LABEL
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset


@torch.no_grad()
def extract_easycom_features_3class(whisper, ticks, device, tag):
    ds = EasyComTurnTakingDataset(ticks)
    feat_list, label_list = [], []
    for i in range(len(ds)):
        item = ds[i]
        hidden, valid_frames = whisper([item["waveform"]], [item["duration_sec"]], device)
        vf = int(valid_frames[0].item())
        pooled = hidden[0, :vf].float().mean(dim=0, keepdim=True)
        feat_list.append(pooled.cpu())
        label_list.append(LABEL_TO_IDX[item["label3"]])
        if (i + 1) % 500 == 0:
            print(f"[dh3] {tag}: extracted {i+1}/{len(ds)}", flush=True)
    return torch.cat(feat_list, 0), torch.tensor(label_list, dtype=torch.long)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ckpt-dir", default="checkpoints/m4_decision_head_3class")
    p.add_argument("--vgg-cache", default="checkpoints/m4_decision_head/features_cache.pt",
                    help="reuse the 2-class run's cached VGGSound World-State features")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[dh3] loading cached VGGSound features from {args.vgg_cache}...", flush=True)
    vgg_cache = torch.load(args.vgg_cache, weights_only=False)
    vgg_train_ws, vgg_train_y = vgg_cache["vgg_train_ws"], vgg_cache["vgg_train_y"]   # y: 1=speak/0=silence
    vgg_test_ws, vgg_test_y = vgg_cache["vgg_test_ws"], vgg_cache["vgg_test_y"]
    ws_dim = vgg_train_ws.shape[1]

    print("[dh3] loading frozen whisper encoder, re-extracting EasyCom features with label3...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    ec_train_ticks, ec_test_ticks = build_ticks()
    ec_train_sf, ec_train_y3 = extract_easycom_features_3class(whisper, ec_train_ticks, device, "easycom-train")
    ec_test_sf, ec_test_y3 = extract_easycom_features_3class(whisper, ec_test_ticks, device, "easycom-test")
    sf_dim = ec_train_sf.shape[1]

    torch.save({"ec_train_sf": ec_train_sf, "ec_train_y3": ec_train_y3,
                "ec_test_sf": ec_test_sf, "ec_test_y3": ec_test_y3},
               os.path.join(args.ckpt_dir, "easycom_3class_features_cache.pt"))

    def build_xy(ws, y_ws_binary, sf, y_sf_3class):
        # VGG: 1=speak/0=silence (LABEL_TO_IDX-compatible: speak=1, silence=0 -- matches!)
        n_ws, n_sf = ws.shape[0], sf.shape[0]
        zero_sf = torch.zeros(n_ws, sf_dim)
        zero_ws = torch.zeros(n_sf, ws_dim)
        X_ws = torch.cat([ws, zero_sf], dim=1)
        X_sf = torch.cat([zero_ws, sf], dim=1)
        X = torch.cat([X_ws, X_sf], dim=0)
        y = torch.cat([y_ws_binary.long(), y_sf_3class], dim=0)
        src = torch.cat([torch.zeros(n_ws, dtype=torch.long), torch.ones(n_sf, dtype=torch.long)], dim=0)
        return X, y, src

    X_train, y_train, src_train = build_xy(vgg_train_ws, vgg_train_y, ec_train_sf, ec_train_y3)
    X_test, y_test, src_test = build_xy(vgg_test_ws, vgg_test_y, ec_test_sf, ec_test_y3)
    print(f"[dh3] train N={X_train.shape[0]}  test N={X_test.shape[0]}", flush=True)

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weight = class_counts.sum() / (3 * class_counts.clamp(min=1))
    print(f"[dh3] class counts (silence/speak/backchannel): {class_counts.tolist()}  "
          f"class_weight: {class_weight.tolist()}", flush=True)

    cfg = DecisionHeadConfig(world_state_dim=ws_dim, speech_feat_dim=sf_dim)
    head = ThreeClassHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X_train, y_train, src_train = X_train.to(device), y_train.to(device), src_train.to(device)
    X_test, y_test, src_test = X_test.to(device), y_test.to(device), src_test.to(device)
    class_weight = class_weight.to(device)

    def eval_split(X, y, src):
        head.eval()
        with torch.no_grad():
            logits = head(X[:, :ws_dim], X[:, ws_dim:])
            preds = logits.argmax(dim=-1)
        out = {}
        for name, mask_val in [("vgg", 0), ("easycom", 1)]:
            m = src == mask_val
            if m.sum() == 0:
                continue
            yy, pp = y[m], preds[m]
            per_class = {}
            for cls_idx, cls_name in IDX_TO_LABEL.items():
                cls_mask = yy == cls_idx
                n_cls = cls_mask.sum().item()
                if n_cls == 0:
                    per_class[cls_name] = None
                    continue
                recall = (pp[cls_mask] == cls_idx).float().mean().item()
                per_class[cls_name] = {"n": n_cls, "recall": recall}
            acc = (pp == yy).float().mean().item()
            out[name] = {"n": int(m.sum()), "accuracy": acc, "per_class_recall": per_class}
        head.train()
        return out

    n_train = X_train.shape[0]
    bs = args.batch_size
    print(f"[dh3] training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = head(X_train[idx, :ws_dim], X_train[idx, ws_dim:])
            loss = F.cross_entropy(logits, y_train[idx], weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_stats = eval_split(X_test, y_test, src_test)
            print(f"[dh3] epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n_train:.4f}  "
                  f"test: {json.dumps(test_stats)}", flush=True)

    final_stats = eval_split(X_test, y_test, src_test)
    print(f"\n[dh3] FINAL: {json.dumps(final_stats, indent=2)}", flush=True)

    torch.save({"state_dict": head.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.ckpt_dir, "best.pt"))
    with open(os.path.join(args.ckpt_dir, "gate_results.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"[dh3] DONE. wrote {args.ckpt_dir}/best.pt and gate_results.json", flush=True)


if __name__ == "__main__":
    main()
