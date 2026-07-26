"""train_decision_head_3class_speechonly.py — A1 condition (g): a 3-class
head with the World-State input branch REMOVED entirely (not zeroed, not
replaced with noise -- the parameter that consumes it doesn't exist).

Motivation: the six-condition A1 falsifier showed the deployed head's
World-State input functions as a presence/scale slot, not an information
channel (a constant dataset-mean vector matched or beat the real, correctly
-paired World-State). If a speech-only head performs within noise of the
full head, that head -- not the WS-consuming one -- is the deployment
candidate, since shipping a live input that structurally receives a
constant is not something to build a real-time dependency around.

Does NOT modify models/m4_decision_head.py (protected) -- defines a
separate, smaller architecture here (same MLP shape, just no World-State
branch, so params shrink roughly in proportion to the removed input dim).

Reuses the SAME cached features as train_decision_head_3class_bothpresent.py
(checkpoints/m4_decision_head_3class_bothpresent/{train,test}_bothpresent_cache.pt)
-- only speech_feat is read, world_state is ignored, no new extraction.

Usage:
    python train_decision_head_3class_speechonly.py --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.m4_decision_head import LABEL_TO_IDX, IDX_TO_LABEL


class SpeechOnlyThreeClassHead(nn.Module):
    """Same MLP shape as models.m4_decision_head.ThreeClassHead, but the
    input is speech_feat ALONE -- no World-State branch, no concatenation,
    no parameter that could ever receive vision-derived input."""

    def __init__(self, speech_feat_dim: int = 1024, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.speech_feat_dim = speech_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(speech_feat_dim),
            nn.Linear(speech_feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        return self.net(speech_feat)


def load_cache(path):
    batch = torch.load(path, weights_only=False)
    sf = torch.stack([c["sf"] for c in batch], 0)
    y = torch.tensor([LABEL_TO_IDX[c["label3"]] for c in batch], dtype=torch.long)
    return sf, y


def eval_condition(head, sf, y, device):
    with torch.no_grad():
        logits = head(sf.to(device))
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/m4_decision_head_3class_bothpresent_v2")
    p.add_argument("--out-dir", default="checkpoints/m4_decision_head_3class_speechonly_v2")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    sf_train, y_train = load_cache(os.path.join(args.ckpt_dir, "train_bothpresent_v2_cache.pt"))
    sf_test, y_test = load_cache(os.path.join(args.ckpt_dir, "test_bothpresent_v2_cache.pt"))
    sf_dim = sf_train.shape[1]
    print(f"[dh3-speechonly] train N={sf_train.shape[0]} test N={sf_test.shape[0]} sf_dim={sf_dim}", flush=True)

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weight = class_counts.sum() / (3 * class_counts.clamp(min=1))

    head = SpeechOnlyThreeClassHead(speech_feat_dim=sf_dim).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[dh3-speechonly] params={n_params:,} ({n_params/1e6:.2f}M)", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    sf_train, y_train = sf_train.to(device), y_train.to(device)
    sf_test, y_test = sf_test.to(device), y_test.to(device)
    class_weight = class_weight.to(device)

    n_train = sf_train.shape[0]
    bs = args.batch_size
    print(f"[dh3-speechonly] training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = head(sf_train[idx])
            loss = F.cross_entropy(logits, y_train[idx], weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            stats, _ = eval_condition(head, sf_test, y_test.cpu(), device)
            print(f"[dh3-speechonly] epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n_train:.4f}  "
                  f"test_acc={stats['accuracy']:.4f}", flush=True)

    final_stats, final_preds = eval_condition(head, sf_test, y_test.cpu(), device)
    print(f"\n[dh3-speechonly] FINAL: {json.dumps(final_stats, indent=2)}", flush=True)

    torch.save({"state_dict": head.state_dict(), "sf_dim": sf_dim}, os.path.join(args.out_dir, "best.pt"))
    with open(os.path.join(args.out_dir, "gate_results.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"[dh3-speechonly] DONE. wrote {args.out_dir}/best.pt and gate_results.json", flush=True)


if __name__ == "__main__":
    main()
