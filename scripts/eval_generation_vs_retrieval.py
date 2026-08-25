"""scripts/eval_generation_vs_retrieval.py — the decisive head-to-head.

The training metric (word-overlap F1 on VGGSound) measures IN-DISTRIBUTION caption matching.
That is not the problem the user actually raised. The problem is OUT-of-distribution: pointed
at a real bedroom, retrieval answered "a tuning fork being tapped", because a bank of
VGGSound sound-event captions contains nothing closer. So the question that matters is:

    on scenes the bank does NOT cover, does GENERATION degrade gracefully where
    RETRIEVAL fails hard?

Two evaluations, because either alone would mislead:

  A) IN-DISTRIBUTION (VGGSound held-out): retrieval should be strong here -- the bank
     literally contains the right caption. Generation only needs to be COMPETITIVE, not
     better. If generation wins here too, that is a bonus, not the point.

  B) OUT-OF-DISTRIBUTION: the same clips with the correct caption (and its whole clip)
     REMOVED from the bank -- a leave-one-out bank. This simulates exactly the live
     situation: the scene is real, the bank has never seen it. Retrieval is then forced to
     return its nearest wrong answer; generation is free to compose a new sentence.
     **This is the measurement that speaks to the user's actual complaint.**

  (+ optional) a REAL Jetson camera frame via --real-frame, scored qualitatively side by
     side, since no ground-truth caption exists for it.

Metrics: word-overlap F1 (comparable to M3's 0.317) and semantic cosine via the frozen
text encoder (which credits a correct paraphrase that shares no words).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

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
from models.perception_prefix import PerceptionPrefix, PerceptionPrefixConfig
from models.query_predictor import (QueryPredictor, QueryPredictorConfig, QUERY_BANK,
                                    VGGSOUND_FIELDS)
from models.text_target import TextTarget


def wf1(ref: str, hyp: str) -> float:
    r, h = set(ref.lower().split()), set(hyp.lower().split())
    if not r or not h:
        return 0.0
    i = len(r & h)
    p, rc = i / len(h), i / len(r)
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-ckpt", default="checkpoints/perception_prefix_thinker/best.pt")
    ap.add_argument("--query-ckpt", default="checkpoints/query_predictor_ddp_lw0.3/best.pt")
    ap.add_argument("--llm", default="checkpoints/bmo_thinker_qwen3_v3_merged")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--bank-clips", type=int, default=2000)
    ap.add_argument("--test-clips", type=int, default=40)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--out", default="checkpoints/GENERATION_VS_RETRIEVAL.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    ids = sorted(te)
    bank_ids, test_ids = ids[: args.bank_clips], ids[: args.test_clips]

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()

    qck = torch.load(args.query_ckpt, map_location="cpu", weights_only=False)
    names = [s.strip() for s in qck["cfg"]["token_sources"].split(",")]
    tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False, device=str(device))
    tt.proj.load_state_dict(qck["text_target_proj"])
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768}
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=tt.native_dim, shared_dim=1536)).to(device)
    qp.load_state_dict(qck["query_predictor"]); qp.eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device).eval()
    emb_layer = llm.get_input_embeddings()
    pck = torch.load(args.prefix_ckpt, map_location="cpu", weights_only=False)
    proj = PerceptionPrefix(PerceptionPrefixConfig(
        source_dims={s: SRC[s] for s in names}, llm_hidden=llm.config.hidden_size,
        n_prefix=pck["cfg"]["n_prefix"])).to(device)
    proj.load_state_dict(pck["projector"]); proj.eval()
    print(f"[hh] prefix ckpt step={pck['step']} trainF1={pck['metrics']['word_overlap_f1']:.4f}", flush=True)

    bank_text, bank_owner, bank_field = [], [], []
    for ci, cid in enumerate(bank_ids):
        for fi, f in enumerate(VGGSOUND_FIELDS):
            bank_text.append(te[cid][f]); bank_owner.append(ci); bank_field.append(fi)
    bank_owner = np.array(bank_owner); bank_field = np.array(bank_field)
    with torch.no_grad():
        be = []
        for i in range(0, len(bank_text), 256):
            be.append(tt.encode_text(bank_text[i:i + 256]).float().cpu())
        bank_emb = F.normalize(torch.cat(be, 0), dim=-1).to(device)
    print(f"[hh] bank {len(bank_text)} captions / {len(bank_ids)} clips", flush=True)

    dl = DataLoader(QueryClipDataset({k: te[k] for k in test_ids}, CACHE_DIR, VGGSOUND_FIELDS),
                    batch_size=4, shuffle=False, num_workers=4, collate_fn=collate)

    res = {k: {"f1": [], "cos": []} for k in
           ("retrieval_in", "generation_in", "retrieval_ood", "generation_ood")}
    samples: List[Dict] = []

    with torch.no_grad():
        for batch in dl:
            src, msk = build_sources(batch, m2, device, names)
            pfx = proj(src, msk).to(torch.bfloat16)
            for bi, cid in enumerate(batch["clip_id"]):
                ci = bank_ids.index(cid)
                sub = {k: v[bi:bi + 1] for k, v in src.items()}
                sm = {k: v[bi:bi + 1] for k, v in msk.items()}
                # OOD bank = this clip's own captions removed (leave-one-out)
                keep = torch.tensor(bank_owner != ci, device=device)
                for fi, f in enumerate(VGGSOUND_FIELDS):
                    q = QUERY_BANK[f][-1]
                    ref = batch["caps"][bi][fi]
                    ref_e = F.normalize(tt.encode_text([ref]).float(), dim=-1)

                    qe = tt.encode_text_frozen_raw([q]).to(device)
                    z = qp(sub, qe, sm)
                    sims = (z @ bank_emb.T)[0]
                    r_in = bank_text[int(sims.argmax())]
                    r_ood = bank_text[int(torch.where(keep, sims, torch.full_like(sims, -1e4)).argmax())]

                    ids_q = tok(f"Task: {q}\n", add_special_tokens=False)["input_ids"]
                    qemb = emb_layer(torch.tensor(ids_q, device=device)).to(pfx.dtype)
                    inp = torch.cat([pfx[bi:bi + 1], qemb.unsqueeze(0)], 1)
                    out = llm.generate(inputs_embeds=inp,
                                       attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=device),
                                       max_new_tokens=args.gen_tokens, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    gen = tok.batch_decode(out, skip_special_tokens=True)[0].strip()

                    for tag, hyp in (("retrieval_in", r_in), ("retrieval_ood", r_ood),
                                     ("generation_in", gen), ("generation_ood", gen)):
                        res[tag]["f1"].append(wf1(ref, hyp))
                        h = F.normalize(tt.encode_text([hyp]).float(), dim=-1)
                        res[tag]["cos"].append(float((ref_e @ h.T)[0, 0]))
                    if len(samples) < 12 and fi in (1, 3):
                        samples.append({"clip": cid, "question": q, "reference": ref[:160],
                                        "retrieval_ood": r_ood[:160], "generation": gen[:160]})

    out_json = {"n_bank": len(bank_text), "n_clips": len(test_ids), "samples": samples,
                "results": {k: {"word_overlap_f1": float(np.mean(v["f1"])),
                                "semantic_cos": float(np.mean(v["cos"])),
                                "n": len(v["f1"])} for k, v in res.items()}}
    print("\n== GENERATION vs RETRIEVAL ==")
    print(f"{'condition':<18} {'word-F1':>9} {'sem-cos':>9}")
    for k in ("retrieval_in", "generation_in", "retrieval_ood", "generation_ood"):
        r = out_json["results"][k]
        print(f"{k:<18} {r['word_overlap_f1']:9.4f} {r['semantic_cos']:9.4f}")
    print("\n(in = bank contains the right caption; ood = that clip removed from the bank)")
    print("\n-- samples (OOD) --")
    for s in samples[:4]:
        print(f"  Q: {s['question']}")
        print(f"    ref : {s['reference'][:110]}")
        print(f"    retr: {s['retrieval_ood'][:110]}")
        print(f"    gen : {s['generation'][:110]}")
    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[hh] wrote {args.out}")


if __name__ == "__main__":
    main()
