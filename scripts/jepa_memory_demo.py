"""scripts/jepa_memory_demo.py — end-to-end demonstration of both capabilities built
in this track, on REAL data and REAL trained checkpoints. CPU-only by default so it
never contends with training jobs.

PART 1 — "the thinker asks perception for more detail"
    One clip, several different questions, and the answer that comes back changes with
    the question. This is the capability the north star describes: not one fixed caption
    per tick, but a scene you can interrogate.

PART 2 — "BMO remembers who you are, and knows when it doesn't"
    Enrol a household from real VoxCeleb2 identity embeddings, then query with (a) a
    held-out clip of an enrolled person and (b) a person who was never enrolled. The
    stranger must come back as unknown, not as a confident wrong name.

Nothing here is synthetic: the clips, captions, identity features, and checkpoints are
all the ones the measured results in JEPA_MEMORY_PLAN.md were produced from.

Usage:
    python scripts/jepa_memory_demo.py --n-clips 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
from models.jepa_identity_head import IdentityHead, IdentityHeadConfig
from models.jepa_memory import JepaMemory, MemoryConfig
from models.query_predictor import QUERY_BANK, VGGSOUND_FIELDS


def part1(args, device):
    from torch.utils.data import DataLoader
    train_m3.CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")
    from train_m3 import build_splits, CACHE_DIR
    from train_query_predictor import QueryClipDataset, collate, group_by_clip, build_sources
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from models.query_predictor import QueryPredictor, QueryPredictorConfig
    from models.text_target import TextTarget

    ck_path = args.query_ckpt
    if not os.path.exists(ck_path):
        print(f"[demo] no query checkpoint at {ck_path}; skipping Part 1")
        return
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]

    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    ids = sorted(te)[: args.n_clips]
    te = {k: te[k] for k in ids}
    dl = DataLoader(QueryClipDataset(te, CACHE_DIR, VGGSOUND_FIELDS), batch_size=len(ids),
                    shuffle=False, num_workers=0, collate_fn=collate)

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

    batch = next(iter(dl))
    with torch.no_grad():
        src, msk = build_sources(batch, m2, device, names)
        # candidate answers = this clip's own 6 field captions
        for bi, cid in enumerate(batch["clip_id"]):
            caps = batch["caps"][bi]
            z_t = tt.encode_text(caps)
            print(f"\n{'='*74}\nCLIP {cid}")
            for f in VGGSOUND_FIELDS:
                q = QUERY_BANK[f][-1]          # held-out phrasing, never trained on
                qe = tt.encode_text_frozen_raw([q]).to(device)
                sub = {k: v[bi:bi + 1] for k, v in src.items()}
                sm = {k: v[bi:bi + 1] for k, v in msk.items()}
                z_q = qp(sub, qe, sm)
                sims = (z_q @ z_t.T)[0]
                pick = int(sims.argmax())
                mark = "OK " if VGGSOUND_FIELDS[pick] == f else "MISS"
                ans = caps[pick].replace("\n", " ")
                print(f"  [{mark}] Q: {q}")
                print(f"        A: {ans[:150]}{'...' if len(ans) > 150 else ''}")


def part2(args, device):
    if not os.path.exists(args.head):
        print(f"[demo] no identity head at {args.head}; skipping Part 2")
        return
    rows, V, A = [], [], []
    for p in sorted(glob.glob(os.path.join(args.feat_dir, "shard*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows.extend(d["rows"]); V.append(d["vision"]); A.append(d["audio"])
    V = torch.cat(V, 0); A = torch.cat(A, 0)
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    head = IdentityHead(IdentityHeadConfig(in_dims=ck["dims"], emb_dim=ck["emb_dim"]))
    head.load_state_dict(ck["head"]); head.eval()

    test = set(ck["test_speakers"])
    keep = [i for i, r in enumerate(rows) if r["speaker"] in test]
    with torch.no_grad():
        Z = head({"vision": V[keep].float(), "audio": A[keep].float()})
    krows = [rows[i] for i in keep]
    by = defaultdict(list)
    for i, r in enumerate(krows):
        by[r["speaker"]].append(i)
    for s in by:
        by[s].sort(key=lambda i: krows[i]["idx"])
    spk = sorted([s for s in by if len(by[s]) >= 8])

    rng = np.random.default_rng(args.seed)
    perm = list(rng.permutation(spk))
    house, calib, strangers = perm[:4], perm[4:120], perm[120:]

    mem = JepaMemory(MemoryConfig(margin=0.02))
    nick = ["maker", "housemate_A", "housemate_B", "housemate_C"]
    for s, name in zip(house, nick):
        for i in by[s][: len(by[s]) // 2][:8]:
            mem.enroll(Z[i], name)
    gen = np.array([max(sc for _, sc in mem.scores(Z[i]))
                    for s in house for i in by[s][len(by[s]) // 2:]])
    imp = np.array([max(sc for _, sc in mem.scores(Z[i])) for s in calib for i in by[s][:2]])
    info = mem.calibrate_threshold(gen, imp, target_far=args.target_far)
    print(f"\n{'='*74}\nMEMORY: {mem}")
    print(f"  enrolled: {', '.join(f'{n} (n={mem.entries[n].n})' for n in nick)}")
    print(f"  calibrated at target FAR={args.target_far:.0%} -> threshold={info['threshold']:.3f}, "
          f"measured FAR={info['far']:.3f}, TAR={info['tar']:.3f}\n")

    for s, name in zip(house, nick):
        i = by[s][len(by[s]) // 2]                 # held-out clip of an ENROLLED person
        lab, conf, why = mem.query(Z[i])
        got = lab if lab else f"UNKNOWN ({why})"
        print(f"  enrolled {name:12s} -> {got:28s} conf={conf:.3f} "
              f"{'CORRECT' if lab == name else ('rejected' if lab is None else 'WRONG NAME')}")
    n_fa = 0
    for s in strangers[:20]:
        lab, conf, why = mem.query(Z[by[s][0]])
        n_fa += int(lab is not None)
        if s == strangers[0]:
            got = lab if lab else f"UNKNOWN ({why})"
            print(f"  STRANGER {'(never enrolled)':12s} -> {got:28s} conf={conf:.3f}")
    print(f"\n  false-accepts over 20 strangers: {n_fa}/20")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-ckpt", default="checkpoints/query_predictor_unified/best.pt")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--head", default="checkpoints/jepa_identity_head_av/head_joint.pt")
    ap.add_argument("--feat-dir", default="/dev/shm/jepa_mem_p2av")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--target-far", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    print("PART 1 — the thinker asks perception for more detail")
    part1(args, device)
    print("\n\nPART 2 — BMO remembers who you are, and knows when it doesn't")
    part2(args, device)


if __name__ == "__main__":
    main()
