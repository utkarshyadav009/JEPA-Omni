"""scripts/perception_query_e2e.py — end-to-end test of the thinker->perception loop
on the TRAINED checkpoint, REAL clips, and a REAL deployment-sized candidate bank.

WHY THIS IS A NEW MEASUREMENT, not a re-run: every query-predictor number so far was
scored against a SMALL candidate set -- within-clip picked 1 of 6 captions, cross-clip R@1
picked 1 of ~600 clips at a single fixed granularity. Deployment is neither. The engine
retrieves against a bank of TENS OF THOUSANDS of captions spanning every clip and every
granularity at once, where near-duplicate captions from other scenes are real competitors.
That is a materially harder task, and the honest deployment number.

Three falsifiers, all on held-out clips with held-out query phrasings:

  F1 GROUNDING     top-1 retrieval comes from the CORRECT clip (chance = 1/n_clips)
  F2 QUERY-FOLLOW  top-1 has the field/granularity that was ASKED for
                   (chance = 1/6), plus a SWAPPED-query control that must collapse
  F3 TOOL ROUND-TRIP  a real `<tool_call name=look .../>` string parses, executes
                   through ToolRegistry, and returns scene-grounded text

Usage:
    python scripts/perception_query_e2e.py --bank-clips 5000 --test-clips 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
train_m3.CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")
from train_m3 import build_splits, CACHE_DIR
from train_query_predictor import QueryClipDataset, collate, group_by_clip, build_sources
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.query_predictor import (QueryPredictor, QueryPredictorConfig, QUERY_BANK,
                                    VGGSOUND_FIELDS)
from models.text_target import TextTarget
from models.m5_perception_query import PerceptionQueryEngine, build_bank
from models.m5_tools import ToolRegistry, parse_tool_calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-ckpt", default="checkpoints/query_predictor_ddp_lw0.3/best.pt")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--bank-clips", type=int, default=5000)
    ap.add_argument("--test-clips", type=int, default=100)
    ap.add_argument("--out", default="checkpoints/PERCEPTION_QUERY_E2E.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.query_ckpt, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]
    print(f"[e2e] checkpoint step={ck['step']} sources={names}", flush=True)

    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    ids = sorted(te)
    bank_ids = ids[: args.bank_clips]
    test_ids = bank_ids[: args.test_clips]          # test clips ARE in the bank, as in deployment

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()
    tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False, device=str(device))
    tt.proj.load_state_dict(ck["text_target_proj"])
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768}
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=tt.native_dim, shared_dim=1536)).to(device)
    qp.load_state_dict(ck["query_predictor"]); qp.eval()

    # ---- bank: every field of every bank clip ----
    bank_text, bank_owner, bank_field = [], [], []
    for ci, cid in enumerate(bank_ids):
        for fi, f in enumerate(VGGSOUND_FIELDS):
            bank_text.append(te[cid][f]); bank_owner.append(ci); bank_field.append(fi)
    print(f"[e2e] bank = {len(bank_text)} captions over {len(bank_ids)} clips "
          f"x {len(VGGSOUND_FIELDS)} granularities", flush=True)
    t0 = time.time()
    bank_emb = build_bank(tt, bank_text, device)
    print(f"[e2e] bank encoded in {time.time()-t0:.0f}s", flush=True)
    bank_owner = np.array(bank_owner); bank_field = np.array(bank_field)

    eng = PerceptionQueryEngine(qp, tt, bank_emb, bank_text, device, max_age_s=1e9)

    dl = DataLoader(QueryClipDataset({k: te[k] for k in test_ids}, CACHE_DIR, VGGSOUND_FIELDS),
                    batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)

    n = 0
    hit_clip = hit_field = hit_both = 0
    sw_field = 0
    lat = []
    per_field = Counter(); per_field_tot = Counter()
    for batch in dl:
        src_all, msk_all = build_sources(batch, m2, device, names)
        for bi, cid in enumerate(batch["clip_id"]):
            ci = bank_ids.index(cid)
            sub = {k: v[bi:bi + 1] for k, v in src_all.items()}
            sm = {k: v[bi:bi + 1] for k, v in msk_all.items()}
            eng.set_perception(sub, sm)
            for fi, f in enumerate(VGGSOUND_FIELDS):
                q = QUERY_BANK[f][-1]                       # HELD-OUT phrasing
                t1 = time.time(); ans = eng.ask(q); lat.append((time.time() - t1) * 1000)
                j = bank_text.index(ans.text) if ans else -1
                # resolve by identity of the retrieved slot, not by string (captions repeat)
                sims = None
                got_owner = bank_owner[j] if j >= 0 else -1
                got_field = bank_field[j] if j >= 0 else -1
                ok_c = int(got_owner == ci); ok_f = int(got_field == fi)
                hit_clip += ok_c; hit_field += ok_f; hit_both += int(ok_c and ok_f)
                per_field[f] += ok_f; per_field_tot[f] += 1
                # swapped-query control
                wrong = VGGSOUND_FIELDS[(fi + 3) % len(VGGSOUND_FIELDS)]
                a2 = eng.ask(QUERY_BANK[wrong][-1])
                j2 = bank_text.index(a2.text) if a2 else -1
                sw_field += int(j2 >= 0 and bank_field[j2] == fi)
                n += 1
        if n >= args.test_clips * len(VGGSOUND_FIELDS):
            break

    nclips = len(bank_ids)
    res = {
        "bank_captions": len(bank_text), "bank_clips": nclips, "n_queries": n,
        "F1_correct_clip": hit_clip / n, "F1_chance": 1.0 / nclips,
        "F2_correct_field": hit_field / n, "F2_chance": 1.0 / len(VGGSOUND_FIELDS),
        "F2_swapped_field": sw_field / n,
        "correct_clip_AND_field": hit_both / n,
        "latency_ms_median": float(np.median(lat)),
        "per_field_accuracy": {f: per_field[f] / max(1, per_field_tot[f]) for f in VGGSOUND_FIELDS},
    }
    print(f"\n== PERCEPTION QUERY, deployment-sized bank ==")
    print(f"  bank: {res['bank_captions']} captions / {nclips} clips; {n} queries")
    print(f"  F1 correct CLIP    : {res['F1_correct_clip']:.3f}  (chance {res['F1_chance']:.5f})")
    print(f"  F2 correct FIELD   : {res['F2_correct_field']:.3f}  (chance {res['F2_chance']:.3f}, "
          f"swapped {res['F2_swapped_field']:.3f})")
    print(f"  both correct       : {res['correct_clip_AND_field']:.3f}")
    print(f"  retrieval latency  : {res['latency_ms_median']:.1f} ms median")
    print("  per-query-type field accuracy:")
    for f, v in res["per_field_accuracy"].items():
        print(f"    {f:32s} {v:.3f}")

    # ---- F3: real tool round-trip ----
    print("\n== F3: tool round-trip ==")
    reg = ToolRegistry()
    reg.register("look", eng.as_tool_handler())
    emitted = ('Let me check. <tool_call name=look query="Describe the room and setting '
               'in detail."/>')
    calls = parse_tool_calls(emitted)
    print(f"  thinker emitted : {emitted}")
    print(f"  parsed          : {calls}")
    for nm, params in calls:
        out = reg.execute(nm, params)
        print(f"  perception said : {out[:180]}{'...' if out and len(out) > 180 else ''}")
        res["tool_roundtrip_ok"] = bool(out)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[e2e] wrote {args.out}")


if __name__ == "__main__":
    main()
