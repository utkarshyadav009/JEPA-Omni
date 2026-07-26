"""train_decision_head_3class_bothpresent.py — A1: retrain the 3-class
decision head on REAL both-modalities-present EasyCom AV (real World-State
from Video_Compressed footage + real Whisper speech-feat, same tick,
never one zeroed) -- the actual streaming-loop input regime, unlike
train_decision_head_3class.py which trains on VGGSound (real-WS/zero-SF)
UNION EasyCom (zero-WS/real-SF) rows and never both real together.

Deliberately does NOT mix in the old zero-one-modality rows (VGGSound or
zero-WS EasyCom) -- one variable per experiment (user's explicit rule):
this run isolates "does training on both-present real data change what
the head learns to do with World-State," nothing else. If this run
underperforms the original on some axis, that comparison is itself the
answer, not a reason to go mix regimes back in before reporting.

Input: checkpoints/m4_decision_head_3class_bothpresent/{train,test}_bothpresent_cache.pt
  (built by scripts/m5_bothpresent_extract.py)
Output: checkpoints/m4_decision_head_3class_bothpresent/best.pt (NEW path,
  does not touch the frozen checkpoints/m4_decision_head_3class/best.pt)

Usage:
    python train_decision_head_3class_bothpresent.py --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX, IDX_TO_LABEL


def load_cache(path):
    batch = torch.load(path, weights_only=False)
    ws = torch.stack([c["ws"] for c in batch], 0)
    sf = torch.stack([c["sf"] for c in batch], 0)
    y = torch.tensor([LABEL_TO_IDX[c["label3"]] for c in batch], dtype=torch.long)
    return ws, sf, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/m4_decision_head_3class_bothpresent")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_cache = os.path.join(args.ckpt_dir, "train_bothpresent_cache.pt")
    test_cache = os.path.join(args.ckpt_dir, "test_bothpresent_cache.pt")
    ws_train, sf_train, y_train = load_cache(train_cache)
    ws_test, sf_test, y_test = load_cache(test_cache)
    ws_dim, sf_dim = ws_train.shape[1], sf_train.shape[1]
    print(f"[dh3-bp] train N={ws_train.shape[0]}  test N={ws_test.shape[0]}  "
          f"ws_dim={ws_dim} sf_dim={sf_dim}", flush=True)

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weight = class_counts.sum() / (3 * class_counts.clamp(min=1))
    print(f"[dh3-bp] train class counts (silence/speak/backchannel): {class_counts.tolist()}  "
          f"class_weight: {class_weight.tolist()}", flush=True)

    cfg = DecisionHeadConfig(world_state_dim=ws_dim, speech_feat_dim=sf_dim)
    head = ThreeClassHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ws_train, sf_train, y_train = ws_train.to(device), sf_train.to(device), y_train.to(device)
    ws_test, sf_test, y_test = ws_test.to(device), sf_test.to(device), y_test.to(device)
    class_weight = class_weight.to(device)

    def eval_split():
        head.eval()
        with torch.no_grad():
            logits = head(ws_test, sf_test)
            preds = logits.argmax(dim=-1)
        acc = (preds == y_test).float().mean().item()
        per_class = {}
        for cls_idx, cls_name in IDX_TO_LABEL.items():
            m = y_test == cls_idx
            n = m.sum().item()
            per_class[cls_name] = {"n": n, "recall": (preds[m] == cls_idx).float().mean().item() if n else None}
        head.train()
        return {"accuracy": acc, "per_class_recall": per_class}

    n_train = ws_train.shape[0]
    bs = args.batch_size
    print(f"[dh3-bp] training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = head(ws_train[idx], sf_train[idx])
            loss = F.cross_entropy(logits, y_train[idx], weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_stats = eval_split()
            print(f"[dh3-bp] epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n_train:.4f}  "
                  f"test: {json.dumps(test_stats)}", flush=True)

    final_stats = eval_split()
    print(f"\n[dh3-bp] FINAL: {json.dumps(final_stats, indent=2)}", flush=True)

    torch.save({"state_dict": head.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.ckpt_dir, "best.pt"))
    with open(os.path.join(args.ckpt_dir, "gate_results.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"[dh3-bp] DONE. wrote {args.ckpt_dir}/best.pt and gate_results.json", flush=True)


if __name__ == "__main__":
    main()
