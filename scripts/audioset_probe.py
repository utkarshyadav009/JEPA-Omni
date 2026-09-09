"""Attentive + linear probes on frozen features for AudioSet-527 (Part B).

ATTENTIVE PROBE ARCHITECTURE. MJEPA states that linear probing materially understates
frozen-encoder quality and re-evaluates baselines with an attentive probe, but the exact
architecture is not stated in the material available here, so this documents what WE
implemented rather than claiming to reproduce theirs:
    a single learned query token cross-attends the frozen token sequence
    (nn.MultiheadAttention, 8 heads, batch_first), LayerNorm, then nn.Linear(d, 527).
Trainable parameters are the query, the attention projections, the norm and the classifier;
the features are frozen and precomputed. The linear probe is nn.Linear(d, 527) on the mean
over the FULL token sequence (not the 32-step pooled one).

Every variant and every control uses the identical optimiser, LR, schedule, epochs, batch
size and seed -- no per-model tuning. Metrics: mAP (macro average precision), mAUC (macro
ROC-AUC), d-prime = sqrt(2) * z(AUC).
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

SEED, EPOCHS, BATCH, LR, WD = 0, 30, 256, 1e-3, 1e-4
NUM_CLASSES = 527


class Attentive(nn.Module):
    def __init__(self, d, n_cls=NUM_CLASSES, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        q = self.q.expand(x.shape[0], -1, -1)
        o, _ = self.attn(q, x, x, need_weights=False)
        return self.head(self.norm(o.squeeze(1)))


def metrics(y, p):
    keep = y.sum(0) > 0                      # AP is undefined for a class with no positives
    ap = average_precision_score(y[:, keep], p[:, keep], average=None)
    au = roc_auc_score(y[:, keep], p[:, keep], average=None)
    mauc = float(np.mean(au))
    return (float(np.mean(ap)) * 100, mauc * 100,
            float(np.sqrt(2) * norm.ppf(min(max(mauc, 1e-6), 1 - 1e-6))), int(keep.sum()))


def run(Xtr, Ytr, Xte, Yte, kind, log):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda")
    seq = Xtr.dim() == 3
    d = Xtr.shape[-1]
    model = (Attentive(d) if kind == "attentive" else nn.Linear(d, NUM_CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(dev).float(); yb = Ytr[idx].to(dev).float()
            if not seq and xb.dim() == 3:
                xb = xb.mean(1)
            opt.zero_grad()
            l = lossf(model(xb), yb)
            l.backward(); opt.step(); tot += float(l) * len(idx)
        sched.step()
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 512):
            xb = Xte[i:i + 512].to(dev).float()
            if not seq and xb.dim() == 3:
                xb = xb.mean(1)
            ps.append(torch.sigmoid(model(xb)).cpu().numpy())
    P = np.concatenate(ps)
    mAP, mAUC, dp, ncls = metrics(Yte.numpy(), P)
    log("    %-10s mAP=%6.2f  mAUC=%6.2f  d'=%.3f  (%d classes scored, final train loss %.4f)"
        % (kind, mAP, mAUC, dp, ncls, tot / n))
    return {"probe": kind, "mAP": mAP, "mAUC": mAUC, "dprime": dp,
            "n_classes_scored": ncls, "dim": int(d), "seq_len": int(Xtr.shape[1]) if seq else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--clean-eval-ids", default="/tmp/audioset_eval_clean_ids.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    tr = torch.load(a.train, map_location="cpu", weights_only=False)
    te = torch.load(a.eval, map_location="cpu", weights_only=False)
    mids = sorted({m for L in tr["labels"] + te["labels"] for m in L})
    # canonical 527-class index from AudioSet's own class_labels_indices.csv
    idx = {}
    if os.path.exists("/tmp/class_labels_indices.csv"):
        import csv
        for r in csv.DictReader(open("/tmp/class_labels_indices.csv")):
            idx[r["mid"]] = int(r["index"])
    if len(idx) != NUM_CLASSES:
        idx = {m: i for i, m in enumerate(mids)}
        log("  [warn] class_labels_indices.csv unusable; using %d observed MIDs" % len(idx))

    def onehot(labels):
        Y = torch.zeros(len(labels), NUM_CLASSES)
        for i, L in enumerate(labels):
            for m in L:
                if m in idx:
                    Y[i, idx[m]] = 1.0
        return Y

    clean = set(json.load(open(a.clean_eval_ids)))
    keep = [i for i, c in enumerate(te["ids"]) if c in clean]
    log("[probe] train n=%d   eval n=%d -> %d after leakage exclusion (%.2f%% removed)"
        % (len(tr["ids"]), len(te["ids"]), len(keep),
           100 * (1 - len(keep) / max(1, len(te["ids"])))))
    Ytr, Yte = onehot(tr["labels"]), onehot([te["labels"][i] for i in keep])
    log("[probe] label matrix: train %s  eval %s  (%.2f labels/clip eval)"
        % (tuple(Ytr.shape), tuple(Yte.shape), float(Yte.sum(1).mean())))

    res = []
    VARIANTS = [
        ("ambient (WavJEPA-base)",       "ambient_mean",       "ambient_tokens"),
        ("world_state (vision ZEROED)",  "world_state",        "world_state_tokens"),
    ]
    for name, mk, tk in VARIANTS:
        log("  %s" % name)
        for kind, key in (("linear", mk), ("attentive", tk)):
            X_tr = tr[key]; X_te = te[key][keep]
            r = run(X_tr, Ytr, X_te, Yte, kind, log)
            r["variant"] = name; res.append(r)

    # ---- CONTROLS on the world_state variant ----
    log("  CONTROL label-shuffled (world_state, train labels permuted)")
    g = torch.Generator().manual_seed(SEED)
    Ysh = Ytr[torch.randperm(Ytr.shape[0], generator=g)]
    for kind, key in (("linear", "world_state"), ("attentive", "world_state_tokens")):
        r = run(tr[key], Ysh, te[key][keep], Yte, kind, log)
        r["variant"] = "CONTROL label-shuffled"; res.append(r)

    log("  CONTROL matched-statistics random (world_state mean/std per dim)")
    W = tr["world_state"].float()
    mu, sd = W.mean(0, keepdim=True), W.std(0, keepdim=True)
    gg = torch.Generator().manual_seed(SEED)
    Rtr = (torch.randn(W.shape, generator=gg) * sd + mu).half()
    Rte = (torch.randn(te["world_state"][keep].shape, generator=gg) * sd + mu).half()
    r = run(Rtr, Ytr, Rte, Yte, "linear", log)
    r["variant"] = "CONTROL matched-stats random"; res.append(r)

    out = {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD,
           "optimiser": "AdamW + CosineAnnealingLR", "loss": "BCEWithLogitsLoss",
           "num_classes": NUM_CLASSES, "n_train": int(Ytr.shape[0]), "n_eval": int(Yte.shape[0]),
           "eval_leakage_excluded": int(len(te["ids"]) - len(keep)),
           "m2_ckpt": tr.get("m2_ckpt"), "vision_zeroed": True,
           "attentive_arch": "1 learned query -> MultiheadAttention(8 heads) -> LayerNorm -> Linear",
           "results": res}
    json.dump(out, open(a.out, "w"), indent=2)
    log("[probe] WROTE %s" % a.out)


if __name__ == "__main__":
    main()
