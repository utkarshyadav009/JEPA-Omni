"""scripts/encode_captions_siglip2.py — pre-encode every caption (and every query phrasing)
into SigLIP2's frozen text space, once.

WHY. The retrain moves the query predictor's TARGET space from EmbeddingGemma's learned
1536-d space to SigLIP2's native, frozen 768-d joint space (see
models/text_target.py::SigLIP2TextTarget). Because that space is frozen, the caption
embeddings are constants: encoding them inside the training loop re-computes the identical
vectors every epoch through a 282M-param tower for nothing.

Pre-encoding buys three things:
  1. training never loads the text tower at all (it is 538 MiB of the model's 716 MiB, and
     3x the vision tower, because of a 256k-token Gemma vocab embedding);
  2. the same vectors define the deployable bank, so the bank can no longer drift out of
     sync with the checkpoint -- the failure that produced a mismatched
     `perception_bank_max_fp16.pt`;
  3. on-device, only these vectors ship. The Jetson never runs a text encoder.

PADDING. SigLIP2 requires `padding="max_length"` at 64 tokens (handled inside
SigLIP2TextTarget). Captions here average 132 characters, so many are truncated -- that is
expected and is itself evidence for moving to short tag-like candidates, but the corpus
captions are what carry the query supervision, so they are what we train against.

Usage (4-way shard, one GPU each):
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/encode_captions_siglip2.py \
        --shard-idx $i --num-shards 4 &
    done; wait
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
from models.query_predictor import QUERY_BANK, VGGSOUND_FIELDS, ACTION100M_FIELDS


def collect_texts(captions_path: str) -> list:
    train_m3.CAPTIONS_PATH = captions_path
    from train_m3 import build_splits
    from train_m2_embed_predictor import load_action100m_splits
    from train_query_predictor import group_by_clip

    texts = set()
    vtr, vte = build_splits(VGGSOUND_FIELDS)
    for pairs in (vtr, vte):
        for cid, fld, txt in pairs:
            if txt and txt.strip():
                texts.add(txt.strip())
    print(f"[enc] VGGSound captions: {len(texts)} unique", flush=True)

    n0 = len(texts)
    for f in ACTION100M_FIELDS:
        tr_p, te_p = load_action100m_splits(f)
        for pairs in (tr_p, te_p):
            for cid, fld, txt in pairs:
                if txt and txt.strip():
                    texts.add(txt.strip())
    print(f"[enc] + Action100M: {len(texts) - n0} more -> {len(texts)} unique", flush=True)

    # every query phrasing, including the held-out ones used at eval
    nq = len(texts)
    for f, qs in QUERY_BANK.items():
        for q in qs:
            texts.add(q.strip())
    print(f"[enc] + {len(texts) - nq} query phrasings -> {len(texts)} total", flush=True)
    return sorted(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-path",
                    default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl"))
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out-dir", default="/dev/shm/siglip2_text_cache")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    texts = collect_texts(args.captions_path)
    mine = texts[args.shard_idx::args.num_shards]
    print(f"[enc:{args.shard_idx}] {len(mine)} texts on this shard", flush=True)

    from models.text_target import SigLIP2TextTarget
    tt = SigLIP2TextTarget(repo=args.siglip, device="cuda")
    print(f"[enc:{args.shard_idx}] {args.siglip} text tower loaded (dim {tt.native_dim})", flush=True)

    out = torch.empty(len(mine), tt.native_dim, dtype=torch.float16)
    t0 = time.time()
    for i in range(0, len(mine), args.batch):
        chunk = mine[i:i + args.batch]
        # RAW (pre-proj) on purpose: the cache must stay valid across every choice of
        # --siglip-shared-dim and every retrain. encode_text() would bake in whatever
        # projection happens to be attached, silently coupling the cache to one run.
        out[i:i + len(chunk)] = tt.encode_text_frozen_raw(chunk).cpu().to(torch.float16)
        if (i // args.batch) % 50 == 0:
            done = i + len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            print(f"[enc:{args.shard_idx}] {done}/{len(mine)} {rate:.0f}/s "
                  f"eta {(len(mine)-done)/max(1e-6,rate):.0f}s", flush=True)

    path = os.path.join(args.out_dir, f"text_shard{args.shard_idx}.pt")
    torch.save({"text": mine, "emb": out, "siglip": args.siglip}, path)
    print(f"[enc:{args.shard_idx}] DONE {len(mine)} -> {path} "
          f"({out.numel()*2/2**20:.0f} MiB, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
