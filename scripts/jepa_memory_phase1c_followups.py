"""scripts/jepa_memory_phase1c_followups.py — answers three direct follow-up
questions with measurements instead of argument. Reuses the Phase 1A/1B embeddings
already on disk, so it costs seconds, not GPU-hours.

Q1 "M2 discards 16pp of identity -- what if we add it back?"
    -> FUSION: L2-normalise each stream and concatenate, then re-run the A1
       identification eval. Tests directly whether a raw-ViT-L side-channel
       alongside M2 recovers the loss, and whether M2 ADDS anything on top of
       raw ViT-L (i.e. is M2 carrying complementary information, or is it a
       strictly lossy view of its own input?).

Q2 "I trained the predictor on Action100M + rich VGGSound captions -- didn't that
    compensate?"
    -> Compare z_p (the caption-trained predictor output) against its own upstream
       inputs on the SAME identity task. If rich-caption training recovered
       identity-correlated information, z_p should beat the M2 world-state it is
       computed from. (Spoiler in the results table: it does.)

Q3 "When I say it remembers the voice, I mean the embedding produced when I speak
    -- can we do that?"
    -> The 250-way number is the wrong operating point for a household. Sweep the
       gallery size N (how many enrolled people) and the enrollment depth (how many
       clips per person), which is what actually determines whether BMO can do this.

Usage:
    python scripts/jepa_memory_phase1c_followups.py
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
from scripts.jepa_memory_phase1a_eval import load_shards, eval_a1   # noqa: E402


def fuse(embs: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    """L2-normalise each stream, then concatenate. Equal weight per stream, so a
    cosine on the result is the mean of the per-stream cosines -- the simplest
    honest fusion, no tuned weights to overfit."""
    return torch.cat([F.normalize(embs[k].float(), dim=-1) for k in keys], dim=-1)


def q1_q2_fusion(in_dir: str) -> Dict:
    rows, embs = load_shards(in_dir)
    combos = [
        (["vitl_crop"], "raw ViT-L only (identity head's proposed input)"),
        (["m2_world_state"], "M2 world-state only (what the pipeline exposes today)"),
        (["z_p"], "z_p only (caption-trained predictor output)"),
        (["vitl_crop", "m2_world_state"], "ViT-L + M2  <- 'add it back'"),
        (["vitl_crop", "z_p"], "ViT-L + z_p"),
        (["m2_world_state", "z_p"], "M2 + z_p (no raw ViT-L side-channel)"),
        (["vitl_crop", "m2_world_state", "z_p"], "ViT-L + M2 + z_p"),
        (["vitl_crop", "wavjepa"], "ViT-L + WavJEPA (joint audio-visual)"),
        (["vitl_crop", "m2_world_state", "z_p", "wavjepa", "moonshine"], "everything"),
    ]
    out = {}
    print("== Q1/Q2 — within-session identification (A1) under stream fusion ==")
    print(f"{'streams':<52} {'top1':>7} {'ci95':>16} {'vs ViT-L':>9}")
    base = None
    for keys, label in combos:
        if any(k not in embs for k in keys):
            continue
        r = eval_a1(rows, fuse(embs, keys))
        if base is None:
            base = r["top1"]
        delta = r["top1"] - base
        out[label] = r
        print(f"{label:<52} {r['top1']:7.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
              f"{delta:+9.3f}")
    print(f"   (chance = {r['chance']:.3f}, n_queries = {r['n_queries']})")
    return out


def q3_voice_operating_point(emb_path: str, seed: int = 0) -> Dict:
    d = torch.load(emb_path, map_location="cpu", weights_only=False)
    rows, E = d["rows"], d["emb"]
    rng = np.random.default_rng(seed)

    by_sv = defaultdict(list)
    for i, r in enumerate(rows):
        by_sv[(r["speaker"], r["video"])].append(i)
    spk_vids = defaultdict(list)
    for (s, v) in by_sv:
        spk_vids[s].append(v)

    # cross-video split, identical protocol to Phase 1B
    enroll, query = defaultdict(list), defaultdict(list)
    for s, vids in spk_vids.items():
        vids = sorted(vids)
        if len(vids) < 2:
            continue
        n_en = max(1, len(vids) - max(1, len(vids) // 2))
        for v in vids[:n_en]:
            enroll[s].extend(by_sv[(s, v)])
        for v in vids[n_en:]:
            query[s].extend(by_sv[(s, v)])
    speakers = [s for s in enroll if enroll[s] and query[s]]

    out = {}
    for stream in ("wavjepa", "moonshine"):
        z = F.normalize(E[stream].float(), dim=-1)
        out[stream] = {}
        print(f"\n== Q3 — '{stream}' cross-session voice ID vs gallery size "
              f"(how many people are enrolled) ==")
        print(f"{'N enrolled':>11} {'top-1':>8} {'95% CI':>16} {'chance':>8}")
        for N in (2, 3, 5, 10, 20, 50, 100, min(250, len(speakers))):
            if N > len(speakers):
                continue
            accs = []
            for _ in range(400):
                sel = rng.choice(speakers, size=N, replace=False)
                cent = torch.stack([F.normalize(z[enroll[s]].mean(0, keepdim=True), dim=-1)[0]
                                    for s in sel])
                qi, qt = [], []
                for j, s in enumerate(sel):
                    qi.extend(query[s]); qt.extend([j] * len(query[s]))
                pred = (z[qi] @ cent.T).argmax(1).numpy()
                accs.append((pred == np.array(qt)).mean())
            a = np.array(accs)
            lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
            out[stream][N] = {"top1": float(a.mean()), "ci95": [float(lo), float(hi)],
                              "chance": 1.0 / N}
            print(f"{N:11d} {a.mean():8.3f} [{lo:.3f},{hi:.3f}] {1.0/N:8.3f}")

    # enrollment depth at a household-scale gallery
    stream = "wavjepa"
    z = F.normalize(E[stream].float(), dim=-1)
    print(f"\n== Q3b — '{stream}', gallery N=5, vs how many clips you enrol per person ==")
    print(f"{'clips/person':>13} {'top-1':>8} {'95% CI':>16}")
    depth_out = {}
    for k in (1, 2, 3, 5, 8):
        accs = []
        for _ in range(400):
            sel = [s for s in rng.choice(speakers, size=5, replace=False)]
            if any(len(enroll[s]) < 1 for s in sel):
                continue
            cent = []
            for s in sel:
                pick = enroll[s][:k] if len(enroll[s]) >= k else enroll[s]
                cent.append(F.normalize(z[pick].mean(0, keepdim=True), dim=-1)[0])
            cent = torch.stack(cent)
            qi, qt = [], []
            for j, s in enumerate(sel):
                qi.extend(query[s]); qt.extend([j] * len(query[s]))
            pred = (z[qi] @ cent.T).argmax(1).numpy()
            accs.append((pred == np.array(qt)).mean())
        a = np.array(accs)
        depth_out[k] = {"top1": float(a.mean()),
                        "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
        print(f"{k:13d} {a.mean():8.3f} [{np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f}]")
    out["enrollment_depth_N5"] = depth_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1a-dir", default="/dev/shm/jepa_mem_p1a")
    ap.add_argument("--p1b-emb", default="/dev/shm/jepa_mem_p1b/voice_emb.pt")
    ap.add_argument("--out", default="checkpoints/JEPA_MEMORY_PHASE1C_FOLLOWUPS.json")
    args = ap.parse_args()

    res = {"fusion_A1": q1_q2_fusion(args.p1a_dir),
           "voice_operating_point": q3_voice_operating_point(args.p1b_emb)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n[p1c] wrote {args.out}")


if __name__ == "__main__":
    main()
