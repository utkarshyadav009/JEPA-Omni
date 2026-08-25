"""scripts/quantize_embeddinggemma_eval.py — measure what int8 quantizing the QUERY
encoder actually costs in retrieval quality, before shipping it to the Jetson.

WHY THIS NEEDS MEASURING RATHER THAN ASSUMING: EmbeddingGemma is ~1000 MiB on the Jetson,
the single largest consumer in the describe pipeline, so quantizing it is the obvious way
to buy headroom. But it is not a passive component -- `encode_text_frozen_raw()` produces
the QueryPredictor's INPUT, and the predictor was trained on bf16 embeddings. Quantizing
shifts that input distribution, so "int8 is usually fine" is not evidence for THIS model in
THIS role. The metric below is the one that matters downstream: does the right caption still
come back.

CONFIGS COMPARED
  bf16                : baseline (TextTarget's default, what everything was trained/built on)
  int8_weight_only    : plain weight quantization, no activation information
  int8_dyn_act        : Int8DynamicActivationInt8WeightConfig -- activations are quantized
                        per-token at runtime, i.e. activation-AWARE at inference, and the
                        config most likely to be faster (int8 matmuls, not just smaller weights)
  awq                 : torchao.prototype.awq -- true activation-aware weight scaling, using
                        real calibration queries. The most faithful reading of "AWQ", and the
                        one that should lose the least quality if it applies cleanly here.

BANK CONSISTENCY, the subtle part: the bank is encoded with `encode_text` and the query with
`encode_text_frozen_raw`, both through the SAME base encoder. If the base is quantized for
queries but the bank was built in bf16, the two sides no longer agree. Both are therefore
measured: `bank_bf16` (ship the existing bank) and `bank_requantized` (rebuild the bank with
the quantized encoder). Reported separately so the deployment choice is evidence-based.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

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
from torch.utils.data import DataLoader

CALIB_QUERIES = [q for f in VGGSOUND_FIELDS for q in QUERY_BANK[f]] + [
    "What is in front of you?", "Describe what you can see.", "What is this place?",
    "Tell me about the room.", "What objects are around?", "Is anyone there?",
    "What is the person doing?", "Describe the background.", "What can you hear now?",
    "Give me a detailed description of everything visible.",
]


def quantize_base(base, mode: str, device, calib_fn=None):
    """Returns a quantized COPY of the encoder (the caller keeps the bf16 original)."""
    from torchao.quantization import (quantize_, Int8WeightOnlyConfig,
                                      Int8DynamicActivationInt8WeightConfig)
    m = copy.deepcopy(base)
    if mode == "int8_weight_only":
        quantize_(m, Int8WeightOnlyConfig())
    elif mode == "int8_dyn_act":
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
    elif mode == "awq":
        from torchao.prototype.awq import AWQConfig, AWQObservedLinear
        base_cfg = Int8DynamicActivationInt8WeightConfig()
        quantize_(m, AWQConfig(base_cfg, step="prepare"),
                  filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear))
        calib_fn(m)                                     # real queries through the observers
        quantize_(m, AWQConfig(base_cfg, step="convert"),
                  filter_fn=lambda mod, fqn: isinstance(mod, AWQObservedLinear))
    else:
        raise ValueError(mode)
    return m


@torch.no_grad()
def evaluate(tt, qp, m2, loader, bank_emb, bank_owner, bank_field, bank_ids, device,
             names) -> Dict[str, float]:
    hit_clip = hit_field = n = 0
    lat = []
    for batch in loader:
        src_all, msk_all = build_sources(batch, m2, device, names)
        for bi, cid in enumerate(batch["clip_id"]):
            ci = bank_ids.index(cid)
            sub = {k: v[bi:bi + 1] for k, v in src_all.items()}
            sm = {k: v[bi:bi + 1] for k, v in msk_all.items()}
            for fi, f in enumerate(VGGSOUND_FIELDS):
                q = QUERY_BANK[f][-1]                    # held-out phrasing
                t0 = time.time()
                qe = tt.encode_text_frozen_raw([q]).to(device)
                z = qp(sub, qe, sm)
                lat.append((time.time() - t0) * 1000)
                j = int((z @ bank_emb.T)[0].argmax())
                hit_clip += int(bank_owner[j] == ci)
                hit_field += int(bank_field[j] == fi)
                n += 1
    return {"correct_clip": hit_clip / n, "correct_field": hit_field / n,
            "query_ms_median": float(np.median(lat)), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-ckpt", default="checkpoints/query_predictor_ddp_lw0.3/best.pt")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--bank-clips", type=int, default=1000)
    ap.add_argument("--test-clips", type=int, default=40)
    ap.add_argument("--out", default="checkpoints/EMBEDDINGGEMMA_QUANT_EVAL.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.query_ckpt, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]

    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    ids = sorted(te)
    bank_ids = ids[: args.bank_clips]
    test_ids = bank_ids[: args.test_clips]

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

    bank_text, bank_owner, bank_field = [], [], []
    for ci, cid in enumerate(bank_ids):
        for fi, f in enumerate(VGGSOUND_FIELDS):
            bank_text.append(te[cid][f]); bank_owner.append(ci); bank_field.append(fi)
    bank_owner = np.array(bank_owner); bank_field = np.array(bank_field)

    dl = DataLoader(QueryClipDataset({k: te[k] for k in test_ids}, CACHE_DIR, VGGSOUND_FIELDS),
                    batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)

    @torch.no_grad()
    def encode_bank():
        out = []
        for i in range(0, len(bank_text), 256):
            out.append(tt.encode_text(bank_text[i:i + 256]).float().cpu())
        return F.normalize(torch.cat(out, 0), dim=-1).to(device)

    def calib(m):
        saved = tt.base
        tt.base = m
        with torch.no_grad():
            for i in range(0, len(CALIB_QUERIES), 8):
                tt.encode_text_frozen_raw(CALIB_QUERIES[i:i + 8])
        tt.base = saved

    results: Dict[str, Dict] = {}
    bf16_base = tt.base
    bank_bf16 = encode_bank()
    print(f"[q] bank {len(bank_text)} captions; baseline eval ...", flush=True)
    results["bf16"] = evaluate(tt, qp, m2, dl, bank_bf16, bank_owner, bank_field, bank_ids, device, names)
    results["bf16"]["params_MiB"] = sum(p.numel() * p.element_size() for p in bf16_base.parameters()) / 2**20
    print(f"  bf16: clip={results['bf16']['correct_clip']:.3f} field={results['bf16']['correct_field']:.3f} "
          f"{results['bf16']['query_ms_median']:.1f}ms {results['bf16']['params_MiB']:.0f}MiB", flush=True)

    for mode in ("int8_weight_only", "int8_dyn_act", "awq"):
        try:
            t0 = time.time()
            qbase = quantize_base(bf16_base, mode, device, calib_fn=calib)
            tt.base = qbase
            r_keep = evaluate(tt, qp, m2, dl, bank_bf16, bank_owner, bank_field, bank_ids, device, names)
            bank_q = encode_bank()
            r_re = evaluate(tt, qp, m2, dl, bank_q, bank_owner, bank_field, bank_ids, device, names)
            sz = sum(p.numel() * p.element_size() for p in qbase.parameters()) / 2**20
            results[mode] = {"bank_bf16": r_keep, "bank_requantized": r_re,
                             "params_MiB": sz, "quantize_s": time.time() - t0}
            print(f"  {mode}: bank_bf16 clip={r_keep['correct_clip']:.3f} field={r_keep['correct_field']:.3f} | "
                  f"bank_requant clip={r_re['correct_clip']:.3f} field={r_re['correct_field']:.3f} | "
                  f"{r_re['query_ms_median']:.1f}ms {sz:.0f}MiB", flush=True)
        except Exception as e:
            print(f"  {mode}: FAILED {type(e).__name__}: {str(e)[:180]}", flush=True)
            results[mode] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        finally:
            tt.base = bf16_base
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[q] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
