"""train_identity_head.py — Phase 2: trains models/jepa_identity_head.py on frozen
WavJEPA features, and measures it against the frozen baselines Phase 1 established.

PROTOCOL (the part that makes the number mean something):
  * **Speaker-disjoint train/test.** The head is trained on one set of speakers and
    evaluated on speakers it has NEVER seen. That is the open-set task BMO actually
    faces -- enrol a stranger today, recognise them tomorrow -- and it is why a
    closed-set accuracy on training identities would be worthless here.
  * **Cross-video enrol/query within the test speakers.** Enrol on one set of source
    YouTube recordings, query from held-out recordings, so "same session" leakage
    cannot inflate the result. Same protocol as Phase 1B, so the numbers compose.

THREE-WAY COMPARISON, so any gain is attributable rather than just "it went up":
  1. frozen mean-pooled      -- the Phase 1B baseline (250-way top-1 0.250)
  2. frozen stats-pooled     -- isolates what mean+std pooling alone buys
  3. trained head on stats   -- isolates what the head itself adds

Reported at the gallery sizes that matter for a household (N=2..10), not just the
250-way number, since that is the operating point BMO actually runs at.

Usage:
    python train_identity_head.py --in-dir /dev/shm/jepa_mem_p2voice
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from models.jepa_identity_head import IdentityHead, IdentityHeadConfig, AAMSoftmax


def load(in_dir: str):
    rows, stats = [], []
    for p in sorted(glob.glob(os.path.join(in_dir, "shard*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows.extend(d["rows"]); stats.append(d["stats"])
    return rows, torch.cat(stats, 0)


def nway_eval(z: torch.Tensor, rows: List[Dict], speakers: List[str], rng,
              sizes=(2, 3, 5, 10, 20, 50, 100, 250), trials: int = 300) -> Dict:
    """Cross-video enrol/query, averaged over random galleries of each size."""
    by_sv = defaultdict(list)
    for i, r in enumerate(rows):
        by_sv[(r["speaker"], r["video"])].append(i)
    enroll, query = defaultdict(list), defaultdict(list)
    for s in speakers:
        vids = sorted({v for (sp, v) in by_sv if sp == s})
        if len(vids) < 2:
            continue
        n_en = max(1, len(vids) - max(1, len(vids) // 2))
        for v in vids[:n_en]:
            enroll[s].extend(by_sv[(s, v)])
        for v in vids[n_en:]:
            query[s].extend(by_sv[(s, v)])
    usable = [s for s in speakers if enroll[s] and query[s]]
    zc = F.normalize(z.float(), dim=-1)

    out = {}
    for N in sizes:
        if N > len(usable):
            continue
        accs = []
        for _ in range(trials if N < len(usable) else 1):
            sel = list(rng.choice(usable, size=N, replace=False)) if N < len(usable) else usable
            cent = torch.stack([F.normalize(zc[enroll[s]].mean(0, keepdim=True), dim=-1)[0] for s in sel])
            qi, qt = [], []
            for j, s in enumerate(sel):
                qi.extend(query[s]); qt.extend([j] * len(query[s]))
            pred = (zc[qi] @ cent.T).argmax(1).numpy()
            accs.append((pred == np.array(qt)).mean())
        a = np.array(accs)
        out[N] = {"top1": float(a.mean()), "chance": 1.0 / N,
                  "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    # verification AUC on the full usable set
    sel = usable
    cent = torch.stack([F.normalize(zc[enroll[s]].mean(0, keepdim=True), dim=-1)[0] for s in sel])
    qi, qt = [], []
    for j, s in enumerate(sel):
        qi.extend(query[s]); qt.extend([j] * len(query[s]))
    sims = (zc[qi] @ cent.T).numpy(); qt = np.array(qt)
    gen = sims[np.arange(len(qt)), qt]
    m = np.ones_like(sims, bool); m[np.arange(len(qt)), qt] = False
    imp = sims[m]
    allv = np.concatenate([gen, imp]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    out["auc"] = float((ranks[:len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp)))
    out["tar_at_far1pct"] = float((gen >= np.quantile(imp, 0.99)).mean())
    out["n_test_speakers"] = len(usable)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/dev/shm/jepa_mem_p2voice")
    ap.add_argument("--out-dir", default="checkpoints/jepa_identity_head_voice")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--warmup-frac", type=float, default=0.3,
                    help="fraction of epochs with margin ramped 0->margin. AAM with a "
                         "full margin from step 0 is a known way to stall convergence "
                         "when the backbone is frozen and weak.")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, stats = load(args.in_dir)
    speakers = sorted({r["speaker"] for r in rows})
    perm = rng.permutation(len(speakers))
    n_te = int(round(args.test_frac * len(speakers)))
    te_spk = sorted({speakers[i] for i in perm[:n_te]})
    tr_spk = sorted({speakers[i] for i in perm[n_te:]})
    print(f"[idh] {len(rows)} clips, {len(speakers)} speakers "
          f"-> train {len(tr_spk)} / test {len(te_spk)} (SPEAKER-DISJOINT)", flush=True)

    tr_idx = [i for i, r in enumerate(rows) if r["speaker"] in set(tr_spk)]
    te_idx = [i for i, r in enumerate(rows) if r["speaker"] in set(te_spk)]
    lab_map = {s: i for i, s in enumerate(tr_spk)}
    y = torch.tensor([lab_map[rows[i]["speaker"]] for i in tr_idx])
    X = stats[tr_idx].float()

    # ── baselines on the TEST speakers, before any training ──
    te_rows = [rows[i] for i in te_idx]
    D = stats.shape[1] // 2
    base_mean = nway_eval(stats[te_idx][:, :D], te_rows, te_spk, rng)
    base_stats = nway_eval(stats[te_idx], te_rows, te_spk, rng)

    cfg = IdentityHeadConfig(in_dims={"audio": stats.shape[1]}, emb_dim=args.emb_dim)
    head = IdentityHead(cfg).to(device)
    crit = AAMSoftmax(cfg.emb_dim, len(tr_spk), margin=args.margin).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(crit.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xd, yd = X.to(device), y.to(device)
    n = len(y)
    n_warm = max(1, int(args.warmup_frac * args.epochs))
    for ep in range(args.epochs):
        crit.margin = args.margin * min(1.0, ep / n_warm)
        head.train()
        idx = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = idx[i:i + args.batch]
            loss = crit(head({"audio": Xd[b]}), yd[b])
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
            print(f"[idh] epoch {ep+1}/{args.epochs} loss={tot/n:.4f} "
                  f"margin={crit.margin:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        Z = head({"audio": stats[te_idx].float().to(device)}).cpu()
    trained = nway_eval(Z, te_rows, te_spk, rng)

    print(f"\n== Voice identity, SPEAKER-DISJOINT test set "
          f"({trained['n_test_speakers']} unseen speakers), cross-video enrol/query ==")
    print(f"{'gallery N':>10} {'frozen mean':>12} {'frozen stats':>13} {'TRAINED head':>13} {'chance':>8}")
    for N in (2, 3, 5, 10, 20, 50, 100, 250):
        if N not in trained:
            continue
        print(f"{N:10d} {base_mean[N]['top1']:12.3f} {base_stats[N]['top1']:13.3f} "
              f"{trained[N]['top1']:13.3f} {trained[N]['chance']:8.3f}")
    print(f"{'AUC':>10} {base_mean['auc']:12.3f} {base_stats['auc']:13.3f} {trained['auc']:13.3f}")
    print(f"{'TAR@FAR1%':>10} {base_mean['tar_at_far1pct']:12.3f} "
          f"{base_stats['tar_at_far1pct']:13.3f} {trained['tar_at_far1pct']:13.3f}")

    torch.save({"head": head.state_dict(), "cfg": vars(cfg) if hasattr(cfg, "__dict__") else None,
                "in_dim": stats.shape[1], "emb_dim": args.emb_dim,
                "train_speakers": tr_spk, "test_speakers": te_spk},
               os.path.join(args.out_dir, "best.pt"))
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"frozen_mean": base_mean, "frozen_stats": base_stats,
                   "trained_head": trained, "n_clips": len(rows),
                   "n_speakers": len(speakers)}, f, indent=2, default=float)
    print(f"\n[idh] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
