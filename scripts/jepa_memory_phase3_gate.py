"""scripts/jepa_memory_phase3_gate.py — the Phase 3 gate, on the REAL trained head.

Simulates BMO's actual deployment situation rather than a retrieval benchmark:
a small household is enrolled, and everyone else on earth is a stranger who must be
REJECTED. The pre-registered gate is G7:

    G7: false-accept rate on never-enrolled identities <= 5% at the operating threshold.

Confidently greeting a stranger by the maker's name is the failure mode that matters
most in a home, which is why the gate is on false-accepts rather than on accuracy.

PROTOCOL (every split disjoint, and the threshold is never fit on the test set):
  * only the 280 SPEAKER-DISJOINT test identities are used -- the head never saw any
    of them in training.
  * `household` : N identities enrolled from their low-index clips, queried with their
    high-index clips (the approximate cross-video split; see train_identity_head_av.py).
  * `calib`     : a disjoint pool of strangers used ONLY to pick the threshold.
  * `test`      : a further disjoint pool of strangers used ONLY to score false-accepts.
  Fitting the threshold on the same strangers it is scored against would make FAR
  meaningless, so calib and test never overlap.

Reported over many random household draws, because which N people you happen to enrol
matters a lot at small N (Phase 1C measured wide speaker-pair variance).
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models.jepa_identity_head import IdentityHead, IdentityHeadConfig
from models.jepa_memory import JepaMemory, MemoryConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="/dev/shm/jepa_mem_p2av")
    ap.add_argument("--head", default="checkpoints/jepa_identity_head_av/head_joint.pt")
    ap.add_argument("--household", type=int, default=5)
    ap.add_argument("--enroll-clips", type=int, default=8)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="checkpoints/JEPA_MEMORY_PHASE3_GATE.json")
    ap.add_argument("--mode", choices=["av", "voice"], default="av",
                    help="'av' = joint head over {vision,audio} stats with an index-range "
                         "enrol/query split; 'voice' = audio-only head with a real "
                         "CROSS-VIDEO split (different source recordings), which is the "
                         "cleaner protocol of the two.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows, feats_raw = [], []
    for p in sorted(glob.glob(os.path.join(args.feat_dir, "shard*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows.extend(d["rows"])
        feats_raw.append(d["stats"] if args.mode == "voice" else (d["vision"], d["audio"]))
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    if args.mode == "voice":
        A = torch.cat(feats_raw, 0); V = None
        dims = {"audio": ck["in_dim"]}
    else:
        V = torch.cat([f[0] for f in feats_raw], 0)
        A = torch.cat([f[1] for f in feats_raw], 0)
        dims = ck["dims"]
    head = IdentityHead(IdentityHeadConfig(in_dims=dims, emb_dim=ck["emb_dim"]))
    head.load_state_dict(ck["head"]); head.eval()
    test_spk = set(ck["test_speakers"])
    print(f"[g7] {len(rows)} clips; head dims={list(dims)}; "
          f"{len(test_spk)} speaker-disjoint test identities", flush=True)

    keep = [i for i, r in enumerate(rows) if r["speaker"] in test_spk]
    feats = {"audio": A[keep].float()}
    if V is not None:
        feats["vision"] = V[keep].float()
    with torch.no_grad():
        Z = head({k: feats[k] for k in dims})          # (n, emb) L2-normalised
    krows = [rows[i] for i in keep]

    by_spk: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(krows):
        by_spk[r["speaker"]].append(i)
    if args.mode == "voice":
        # real cross-video split: enrol on some SOURCE RECORDINGS, query on held-out ones
        vids_of = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(krows):
            vids_of[r["speaker"]][r["video"]].append(i)
        enroll_map, query_map = {}, {}
        for s, vd in vids_of.items():
            vids = sorted(vd)
            if len(vids) < 2:
                continue
            n_en = max(1, len(vids) - max(1, len(vids) // 2))
            enroll_map[s] = [i for v in vids[:n_en] for i in vd[v]]
            query_map[s] = [i for v in vids[n_en:] for i in vd[v]]
        by_spk = {s: enroll_map[s] + query_map[s] for s in enroll_map}
    else:
        for s in by_spk:
            by_spk[s].sort(key=lambda i: krows[i]["idx"])
        enroll_map = {s: by_spk[s][:len(by_spk[s]) // 2] for s in by_spk}
        query_map = {s: by_spk[s][len(by_spk[s]) // 2:] for s in by_spk}
    speakers = sorted([s for s in by_spk if len(by_spk[s]) >= 4
                       and enroll_map.get(s) and query_map.get(s)])

    res = {"correct": [], "unknown_household": [], "far": [], "thr": [], "tar": []}
    for _ in range(args.trials):
        perm = list(rng.permutation(speakers))
        house = perm[:args.household]
        rest = perm[args.household:]
        half = len(rest) // 2
        calib, strangers = rest[:half], rest[half:]

        mem = JepaMemory(MemoryConfig(threshold=0.0, margin=0.02))
        queries = []
        for s in house:
            for i in enroll_map[s][: args.enroll_clips]:
                mem.enroll(Z[i], s)
            queries += [(i, s) for i in query_map[s]]

        # threshold fit ONLY on calibration strangers + household genuine scores
        gen = np.array([max(sc for _, sc in mem.scores(Z[i])) for i, _ in queries])
        imp_cal = np.array([max(sc for _, sc in mem.scores(Z[i]))
                            for s in calib for i in query_map[s][:2]])
        info = mem.calibrate_threshold(gen, imp_cal, target_far=args.target_far)

        # score on HELD-OUT strangers, never used for calibration
        n_fa = n_str = 0
        for s in strangers:
            for i in query_map[s][:2]:
                lab, _, _ = mem.query(Z[i])
                n_str += 1; n_fa += int(lab is not None)
        n_ok = n_unk = 0
        for i, s in queries:
            lab, _, _ = mem.query(Z[i])
            n_ok += int(lab == s); n_unk += int(lab is None)

        res["correct"].append(n_ok / max(1, len(queries)))
        res["unknown_household"].append(n_unk / max(1, len(queries)))
        res["far"].append(n_fa / max(1, n_str))
        res["thr"].append(info["threshold"]); res["tar"].append(info["tar"])

    out = {k: {"mean": float(np.mean(v)),
               "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
           for k, v in res.items()}
    print(f"\n== PHASE 3 GATE: household={args.household}, "
          f"enroll={args.enroll_clips} clips, target FAR={args.target_far:.0%}, "
          f"{args.trials} random households ==")
    for k in ("correct", "unknown_household", "far", "thr"):
        m = out[k]
        print(f"  {k:20s} {m['mean']:.4f}  95% CI [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]")
    g7 = out["far"]["mean"] <= 0.05
    print(f"\n  G7 (false-accept on never-enrolled <= 5%): "
          f"{out['far']['mean']:.4f} -> {'PASS' if g7 else 'FAIL'}")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": out, "G7_pass": bool(g7)}, f, indent=2)
    print(f"[g7] wrote {args.out}")


if __name__ == "__main__":
    main()
