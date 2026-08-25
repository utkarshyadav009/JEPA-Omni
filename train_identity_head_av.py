"""train_identity_head_av.py — Phase 2c: the JOINT audio-visual identity head.

Phase 2a trained on voice alone because that was where clean data existed. Per direct
user push-back -- "I don't just identify people by their voice, most of the time it's
their face" -- this adds the vision branch and measures what it actually contributes.

DATA: VoxCeleb2 mp4 (face-crop video WITH its own audio), so vision and voice are the
same person at the same instant -- genuine joint AV identity, which is the north star.
33,798 clips / 1,400 speakers / median 24 clips each.

PROTOCOL
  * train/test is **SPEAKER-DISJOINT** -- the head never sees a test identity. This is
    the open-set task BMO faces and it is fully clean; it needs no video grouping.
  * enrol/query WITHIN a test speaker uses an **index-range split** (enrol on the low
    half of that speaker's file_name indices, query on the high half). The mirror
    flattened VoxCeleb2's source-video paths, but the index was verified to follow the
    original traversal order (a speaker's clips arrive in contiguous numeric clusters),
    so a range split is an APPROXIMATE video split -- contaminated only by the single
    source video straddling the midpoint. **Never quote this as a clean cross-session
    number**; the clean cross-session voice number comes from the wds mirror (Phase 2a),
    which preserves real paths.

Reports frozen vs trained for vision-only / audio-only / joint, so "what does the face
add over the voice" is answered by measurement rather than assumption.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from models.jepa_identity_head import IdentityHead, IdentityHeadConfig, AAMSoftmax


def load(in_dir: str):
    rows, V, A = [], [], []
    for p in sorted(glob.glob(os.path.join(in_dir, "shard*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows.extend(d["rows"]); V.append(d["vision"]); A.append(d["audio"])
    return rows, torch.cat(V, 0), torch.cat(A, 0)


def split_enroll_query(rows: List[Dict], speakers: List[str]):
    """Index-range split per speaker: low half enrol, high half query."""
    by_spk = defaultdict(list)
    for i, r in enumerate(rows):
        by_spk[r["speaker"]].append(i)
    enroll, query = {}, {}
    for s in speakers:
        idxs = sorted(by_spk[s], key=lambda i: rows[i]["idx"])
        if len(idxs) < 4:
            continue
        cut = len(idxs) // 2
        enroll[s], query[s] = idxs[:cut], idxs[cut:]
    return enroll, query


def nway(z: torch.Tensor, enroll, query, rng, sizes=(2, 3, 5, 10, 20, 50, 100, 280),
         trials: int = 300) -> Dict:
    zc = F.normalize(z.float(), dim=-1)
    usable = [s for s in enroll if enroll[s] and query[s]]
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
            accs.append(((zc[qi] @ cent.T).argmax(1).numpy() == np.array(qt)).mean())
        a = np.array(accs)
        out[N] = {"top1": float(a.mean()), "chance": 1.0 / N}
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
    return out


def train_head(Xs: Dict[str, torch.Tensor], y, n_cls, dims, args, device):
    cfg = IdentityHeadConfig(in_dims=dims, emb_dim=args.emb_dim)
    head = IdentityHead(cfg).to(device)
    crit = AAMSoftmax(cfg.emb_dim, n_cls, margin=args.margin).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(crit.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xd = {k: v.to(device) for k, v in Xs.items()}
    yd = y.to(device); n = len(y)
    n_warm = max(1, int(args.warmup_frac * args.epochs))
    for ep in range(args.epochs):
        crit.margin = args.margin * min(1.0, ep / n_warm)   # full margin from step 0 stalls
        head.train()
        idx = torch.randperm(n, device=device); tot = 0.0
        for i in range(0, n, args.batch):
            b = idx[i:i + args.batch]
            loss = crit(head({k: v[b] for k, v in Xd.items()}), yd[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        if (ep + 1) % max(1, args.epochs // 6) == 0:
            print(f"      epoch {ep+1}/{args.epochs} loss={tot/n:.4f}", flush=True)
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/dev/shm/jepa_mem_p2av")
    ap.add_argument("--out-dir", default="checkpoints/jepa_identity_head_av")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--warmup-frac", type=float, default=0.3)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, V, A = load(args.in_dir)
    speakers = sorted({r["speaker"] for r in rows})
    perm = rng.permutation(len(speakers))
    n_te = int(round(args.test_frac * len(speakers)))
    te_spk = sorted({speakers[i] for i in perm[:n_te]})
    tr_spk = sorted({speakers[i] for i in perm[n_te:]})
    print(f"[av] {len(rows)} clips, {len(speakers)} speakers -> "
          f"train {len(tr_spk)} / test {len(te_spk)} (SPEAKER-DISJOINT)", flush=True)

    tr_idx = [i for i, r in enumerate(rows) if r["speaker"] in set(tr_spk)]
    lab = {s: i for i, s in enumerate(tr_spk)}
    y = torch.tensor([lab[rows[i]["speaker"]] for i in tr_idx])
    enroll, query = split_enroll_query(rows, te_spk)
    n_usable = len([s for s in enroll if enroll[s] and query[s]])
    print(f"[av] test speakers usable for enrol/query: {n_usable}", flush=True)

    CONFIGS = {
        "vision": {"vision": V.shape[1]},
        "audio": {"audio": A.shape[1]},
        "joint": {"vision": V.shape[1], "audio": A.shape[1]},
    }
    FEATS = {"vision": V, "audio": A}
    results: Dict[str, Dict] = {}

    for name, dims in CONFIGS.items():
        print(f"\n[av] === {name} ===", flush=True)
        frozen = nway(torch.cat([FEATS[k] for k in dims], 1), enroll, query, rng)
        Xs = {k: FEATS[k][tr_idx].float() for k in dims}
        head = train_head(Xs, y, len(tr_spk), dims, args, device)
        with torch.no_grad():
            Z = head({k: FEATS[k].float().to(device) for k in dims}).cpu()
        trained = nway(Z, enroll, query, rng)
        results[name] = {"frozen": frozen, "trained": trained}
        torch.save({"head": head.state_dict(), "dims": dims, "emb_dim": args.emb_dim,
                    "train_speakers": tr_spk, "test_speakers": te_spk},
                   os.path.join(args.out_dir, f"head_{name}.pt"))

    print(f"\n== JOINT AV IDENTITY, speaker-disjoint test ({n_usable} unseen speakers) ==")
    print("   enrol/query = index-range split (APPROXIMATE cross-video, see docstring)\n")
    hdr = f"{'N':>5} " + " ".join(f"{n+'_'+m:>14}" for n in CONFIGS for m in ("froz", "head"))
    print(hdr)
    for N in (2, 5, 10, 50, 280):
        if N not in results["joint"]["trained"]:
            continue
        row = f"{N:5d} "
        for n in CONFIGS:
            row += f"{results[n]['frozen'][N]['top1']:14.3f} {results[n]['trained'][N]['top1']:14.3f}"
        print(row)
    for metric in ("auc", "tar_at_far1pct"):
        row = f"{metric:>5} "
        for n in CONFIGS:
            row += f"{results[n]['frozen'][metric]:14.3f} {results[n]['trained'][metric]:14.3f}"
        print(row)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"n_clips": len(rows), "n_speakers": len(speakers),
                   "n_test_usable": n_usable, "results": results}, f, indent=2, default=float)
    print(f"\n[av] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
