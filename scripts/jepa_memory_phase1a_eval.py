"""scripts/jepa_memory_phase1a_eval.py — scores the Phase 1A falsifier.

Consumes the shards written by scripts/jepa_memory_phase1a_extract.py and answers
the one question Phase 1A exists to answer: **does the frozen M2 representation
carry person identity?**

Two evals, deliberately reported separately because they are NOT the same claim:

  A1 — WITHIN-SESSION, CROSS-CHUNK, N-way identification (28 guest identities).
       Gallery = the first 60% of a session's chunks, query = the last 40%.
       This is an UPPER BOUND and must always be labelled as one: same day, same
       clothing, same seat, same lighting, so part of any score is appearance
       matching rather than identity. It is still the right FIRST test, because a
       representation that fails even here cannot possibly support the north-star.

  A2 — CROSS-SESSION verification for P1, the single genuinely recurring identity
       (verified: byte-identical Participant_Photo in all 12 sessions, and the
       crops confirm the same man). Leave-one-session-out: enroll P1 on 11
       sessions, test on the held-out session against that session's guests as
       impostors. This is the real, un-inflated cross-session number.

       **CHANNEL CONFOUND, load-bearing:** P1 has no Close_Microphone_Audio, so
       P1's audio comes from the glasses array mixdown while every guest's comes
       from a close mic. Any stream that consumes audio (wavjepa, moonshine,
       m2_world_state, m2_prepool_mean, z_p) can therefore separate P1 from
       guests by MICROPHONE CHANNEL alone, with zero identity information. Only
       `vitl_crop` is channel-clean in A2. The audio-bearing streams are still
       printed, but marked CONFOUNDED and excluded from the gate -- they are not
       evidence of anything about identity.

Controls (both evals): label-shuffled baselines and bootstrap CIs, matching this
repo's standing matched-random-control practice.

Usage:
    python scripts/jepa_memory_phase1a_eval.py --in-dir /dev/shm/jepa_mem_p1a
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# streams that consume the audio path -> channel-confounded in A2
AUDIO_BEARING = {"wavjepa", "moonshine", "m2_world_state", "m2_prepool_mean", "z_p"}
# the caption-aligned control (plan §4a / gate G6)
CAPTION_CONTROL = "z_p"


def load_shards(in_dir: str) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
    rows: List[Dict] = []
    embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(in_dir, "shard*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows.extend(d["rows"])
        for k, v in d["emb"].items():
            embs[k].append(v)
    return rows, {k: torch.cat(v, 0) for k, v in embs.items()}


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _boot_ci(vals: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ──────────────────────────────────────────────────────────────────────────
# A1 — within-session, cross-chunk, N-way identification (guests only)
# ──────────────────────────────────────────────────────────────────────────
def eval_a1(rows: List[Dict], z: torch.Tensor, seed: int = 0) -> Dict:
    idx_by_sess: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["is_cross_session_identity"]:
            continue                      # P1 handled in A2; keep A1 homogeneous
        idx_by_sess[r["session"]].append(i)

    zc = _norm(z)
    correct, chance, shuf_correct = [], [], []
    rng = np.random.default_rng(seed)
    n_sess_used = 0

    for sess, idxs in sorted(idx_by_sess.items()):
        chunks = sorted({rows[i]["chunk"] for i in idxs})
        if len(chunks) < 3:
            continue
        cut = max(1, int(round(0.6 * len(chunks))))
        gal_chunks, qry_chunks = set(chunks[:cut]), set(chunks[cut:])
        if not qry_chunks:
            continue

        gal: Dict[str, List[int]] = defaultdict(list)
        qry: List[int] = []
        for i in idxs:
            (gal[rows[i]["identity"]].append(i) if rows[i]["chunk"] in gal_chunks
             else qry.append(i))
        ids = sorted([k for k, v in gal.items() if len(v) >= 2])
        if len(ids) < 2:
            continue
        qry = [i for i in qry if rows[i]["identity"] in ids]
        if not qry:
            continue
        n_sess_used += 1

        cent = torch.stack([_norm(zc[gal[k]].mean(0, keepdim=True))[0] for k in ids])  # (K,D)
        sims = zc[qry] @ cent.T                                                        # (Q,K)
        pred = sims.argmax(1).numpy()
        true = np.array([ids.index(rows[i]["identity"]) for i in qry])
        correct.extend((pred == true).astype(float))
        chance.extend([1.0 / len(ids)] * len(qry))
        # control: permute which centroid carries which label
        perm = rng.permutation(len(ids))
        shuf_correct.extend((perm[pred] == true).astype(float))

    c = np.array(correct)
    lo, hi = _boot_ci(c, seed=seed)
    return {
        "n_queries": int(len(c)),
        "n_sessions": n_sess_used,
        "top1": float(c.mean()) if len(c) else float("nan"),
        "ci95": [lo, hi],
        "chance": float(np.mean(chance)) if chance else float("nan"),
        "shuffled_top1": float(np.mean(shuf_correct)) if shuf_correct else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────
# A2 — cross-session verification for P1 (leave-one-session-out)
# ──────────────────────────────────────────────────────────────────────────
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _tar_at_far(pos: np.ndarray, neg: np.ndarray, far: float = 0.01) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - far)
    return float((pos >= thr).mean())


def eval_a2(rows: List[Dict], z: torch.Tensor, seed: int = 0) -> Dict:
    zc = _norm(z)
    p1 = [i for i, r in enumerate(rows) if r["is_cross_session_identity"]]
    if not p1:
        return {"n_genuine": 0, "note": "no P1 windows extracted"}
    sessions = sorted({rows[i]["session"] for i in p1})

    pos_all, neg_all, shuf_pos, shuf_neg = [], [], [], []
    rng = np.random.default_rng(seed)
    guests_by_sess: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if not r["is_cross_session_identity"]:
            guests_by_sess[r["session"]].append(i)

    for held in sessions:
        enroll = [i for i in p1 if rows[i]["session"] != held]
        test_pos = [i for i in p1 if rows[i]["session"] == held]
        test_neg = guests_by_sess.get(held, [])
        if len(enroll) < 5 or not test_pos or not test_neg:
            continue
        cent = _norm(zc[enroll].mean(0, keepdim=True))          # (1,D)
        pos_all.append((zc[test_pos] @ cent.T).squeeze(1).numpy())
        neg_all.append((zc[test_neg] @ cent.T).squeeze(1).numpy())

        # control: enroll a RANDOM guest identity from the non-held sessions and
        # score the same held-out P1/guest windows against it. A real P1 model
        # should beat this; a channel/appearance shortcut will not.
        other = [i for i in range(len(rows))
                 if not rows[i]["is_cross_session_identity"] and rows[i]["session"] != held]
        if len(other) >= 5:
            pick_id = rows[other[rng.integers(0, len(other))]]["identity"]
            fake = [i for i in other if rows[i]["identity"] == pick_id]
            fcent = _norm(zc[fake].mean(0, keepdim=True))
            shuf_pos.append((zc[test_pos] @ fcent.T).squeeze(1).numpy())
            shuf_neg.append((zc[test_neg] @ fcent.T).squeeze(1).numpy())

    if not pos_all:
        return {"n_genuine": 0, "note": "insufficient P1 coverage for LOSO"}
    pos = np.concatenate(pos_all); neg = np.concatenate(neg_all)
    out = {
        "n_genuine": int(len(pos)), "n_impostor": int(len(neg)),
        "n_folds": len(pos_all),
        "auc": _auc(pos, neg),
        "tar_at_far1pct": _tar_at_far(pos, neg, 0.01),
        "genuine_cos_mean": float(pos.mean()), "impostor_cos_mean": float(neg.mean()),
    }
    if shuf_pos:
        out["control_auc_random_identity"] = _auc(np.concatenate(shuf_pos),
                                                  np.concatenate(shuf_neg))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/dev/shm/jepa_mem_p1a")
    ap.add_argument("--out", default="checkpoints/JEPA_MEMORY_PHASE1A_RESULTS.json")
    args = ap.parse_args()

    rows, embs = load_shards(args.in_dir)
    n_ids = len({r["identity"] for r in rows})
    print(f"[p1a-eval] {len(rows)} windows, {n_ids} identities, "
          f"{len({r['session'] for r in rows})} sessions, streams={sorted(embs)}\n")

    res = {"n_windows": len(rows), "n_identities": n_ids, "A1": {}, "A2": {}}

    print("== A1: within-session, cross-chunk, N-way identification (UPPER BOUND) ==")
    print(f"{'stream':<18} {'top1':>7} {'ci95':>16} {'chance':>7} {'shuffled':>9} {'n_qry':>6}")
    for k in sorted(embs):
        r = eval_a1(rows, embs[k])
        res["A1"][k] = r
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        tag = "  <- caption control" if k == CAPTION_CONTROL else ""
        print(f"{k:<18} {r['top1']:7.3f} {ci:>16} {r['chance']:7.3f} "
              f"{r['shuffled_top1']:9.3f} {r['n_queries']:6d}{tag}")

    print("\n== A2: CROSS-SESSION verification, P1 leave-one-session-out ==")
    print(f"{'stream':<18} {'AUC':>7} {'TAR@FAR1%':>10} {'ctrl_AUC':>9} {'gen_cos':>8} {'imp_cos':>8}")
    for k in sorted(embs):
        r = eval_a2(rows, embs[k])
        res["A2"][k] = r
        if r.get("n_genuine", 0) == 0:
            print(f"{k:<18}  {r.get('note','n/a')}"); continue
        flag = "  CONFOUNDED(channel)" if k in AUDIO_BEARING else ""
        print(f"{k:<18} {r['auc']:7.3f} {r['tar_at_far1pct']:10.3f} "
              f"{r.get('control_auc_random_identity', float('nan')):9.3f} "
              f"{r['genuine_cos_mean']:8.3f} {r['impostor_cos_mean']:8.3f}{flag}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[p1a-eval] wrote {args.out}")

    # ── gates ────────────────────────────────────────────────────────────
    non_ctrl = [k for k in embs if k != CAPTION_CONTROL]
    best_a1 = max(non_ctrl, key=lambda k: res["A1"][k]["top1"])
    b = res["A1"][best_a1]
    g1 = b["top1"] >= 0.80
    g2 = b["ci95"][0] > b["chance"] and b["shuffled_top1"] < b["ci95"][0]
    g6 = res["A1"][CAPTION_CONTROL]["top1"] < b["top1"] if CAPTION_CONTROL in embs else None
    a2v = res["A2"].get("vitl_crop", {})
    g_x = a2v.get("auc", float("nan")) >= 0.70

    print("\n== PRE-REGISTERED GATES ==")
    print(f"  G1-A  best non-control A1 top1 >= 0.80 : {b['top1']:.3f} ({best_a1})  -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2-A  CI excludes chance + shuffle low : ci_lo={b['ci95'][0]:.3f} chance={b['chance']:.3f} "
          f"shuf={b['shuffled_top1']:.3f}  -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G6    caption-space z_p worse than best: "
          f"{res['A1'].get(CAPTION_CONTROL,{}).get('top1',float('nan')):.3f} < {b['top1']:.3f}  -> "
          f"{'PASS (confirms plan 4a)' if g6 else 'FAIL (plan 4a wrong)'}")
    print(f"  A2    cross-session vitl_crop AUC>=0.70: {a2v.get('auc',float('nan')):.3f}  -> {'PASS' if g_x else 'FAIL'}")


if __name__ == "__main__":
    main()
