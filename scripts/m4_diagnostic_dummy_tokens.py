"""scripts/m4_diagnostic_dummy_tokens.py — distribution vs interference
diagnostic for the check-(c) collapse.

Re-runs the M3 swap-control (normal / swapped / zeroed on the M3 latents)
with N=62 DUMMY tokens concatenated where the real M4b speech segment was,
instead of real speech content:
  (a) zeroed dummy  -- all-zero embeddings, shape (62, llm_hidden)
  (b) random dummy  -- iid N(0,1) embeddings, same shape

Logic: if M3 still collapses with meaningless dummy tokens of the right
shape, the failure is about SEQUENCE SHAPE/LENGTH/POSITION (M3's connector
was trained on prompts of one fixed structure -- 32 latents + tag + text --
and never saw anything longer/differently-shaped, so ANY extra block breaks
it regardless of content). If M3 recovers with dummies, the failure is
about SEMANTIC interference from real, in-distribution-for-language
speech-derived content specifically (dummy noise doesn't compete for the
same representational territory the way real projected speech does).

Usage:
    python scripts/m4_diagnostic_dummy_tokens.py --n-clips 100 --n-dummy 62
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, m3_collate_fn, word_overlap_f1, _cap_ambient_len, CACHE_DIR, GRANULARITY_TAGS
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from data.av_cached_dataset import AVCachedDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELD = "gpt_sound_acoustic"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--m3-connector-ckpt", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--n-dummy", type=int, default=62)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4b/diagnostic_dummy_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print("[diag] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[diag] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)

    print("[diag] loading frozen M3 connector...", flush=True)
    m3ckpt = torch.load(args.m3_connector_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    llm_hidden = llm.config.hidden_size
    N = args.n_dummy
    zeroed_dummy = torch.zeros(1, N, llm_hidden, dtype=torch.bfloat16, device=device)
    random_dummy = torch.randn(1, N, llm_hidden, dtype=torch.bfloat16, device=device)
    print(f"[diag] dummy block shape=(1,{N},{llm_hidden})  "
          f"zeroed_norm={zeroed_dummy.float().norm().item():.2f}  "
          f"random_norm={random_dummy.float().norm().item():.2f}", flush=True)

    print("[diag] building VGGSound eval pool...", flush=True)
    _, vgg_test_pairs = build_splits(FIELD)
    rng.shuffle(vgg_test_pairs)
    vgg_eval_pairs = vgg_test_pairs[:args.n_clips]

    def prefix_embeds_for_field(field, B):
        ids = tokenizer(GRANULARITY_TAGS[field], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return llm.get_input_embeddings()(ids).expand(B, -1, -1)

    @torch.no_grad()
    def generate(inputs_embeds, attention_mask):
        gen_ids = llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=60,
                                do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    def m3_batch_pre_pool(pairs_batch):
        clip_ids = [c for c, _, _ in pairs_batch]
        ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
        items = [{"feats": ds[i]["feats"], "tbins": ds[i]["tbins"], "clip_id": ds[i]["clip_id"],
                  "prefix_ids": torch.zeros(0, dtype=torch.long), "caption_ids": torch.zeros(1, dtype=torch.long),
                  "caption_text": "", "field": None} for i in range(len(ds))]
        batch = m3_collate_fn(items, tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
        kpm = torch.cat([pad["vision"], pad["ambient"]], dim=1)
        return pre_pool.to(torch.bfloat16), kpm

    results = {}
    for dummy_name, dummy_block in [("zeroed_dummy", zeroed_dummy), ("random_dummy", random_dummy)]:
        print(f"\n[diag] === condition: {dummy_name} (N={N}) ===", flush=True)
        f1_normal, f1_swapped, f1_zeroed = [], [], []
        bs = args.batch_size
        for start in range(0, len(vgg_eval_pairs), bs):
            batch_pairs = vgg_eval_pairs[start:start + bs]
            if len(batch_pairs) < 2:
                continue
            gts = [g for _, _, g in batch_pairs]
            pre_pool, kpm = m3_batch_pre_pool(batch_pairs)
            pre_pool_swapped = torch.roll(pre_pool, shifts=1, dims=0)
            kpm_swapped = torch.roll(kpm, shifts=1, dims=0)
            pre_pool_zeroed = torch.zeros_like(pre_pool)
            B = pre_pool.shape[0]
            prefix = prefix_embeds_for_field(FIELD, B)
            prefix_attn = torch.ones(B, prefix.shape[1], dtype=torch.long, device=device)
            dummy = dummy_block.expand(B, -1, -1)
            dummy_attn = torch.ones(B, N, dtype=torch.long, device=device)

            for cond_name, pp, mask in [("normal", pre_pool, kpm), ("swapped", pre_pool_swapped, kpm_swapped),
                                         ("zeroed", pre_pool_zeroed, kpm)]:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    m3_latents = m3_connector(pp, mask)
                m3_attn = torch.ones(B, m3_latents.shape[1], dtype=torch.long, device=device)
                emb = torch.cat([m3_latents, dummy, prefix], dim=1)
                attn = torch.cat([m3_attn, dummy_attn, prefix_attn], dim=1)
                gen = generate(emb, attn)
                for i in range(B):
                    f1 = word_overlap_f1(gen[i], gts[i])
                    {"normal": f1_normal, "swapped": f1_swapped, "zeroed": f1_zeroed}[cond_name].append(f1)
            print(f"[diag] {dummy_name} processed {min(start+bs, len(vgg_eval_pairs))}/{len(vgg_eval_pairs)}", flush=True)

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0
        results[dummy_name] = {"n": len(f1_normal), "f1_normal": mean(f1_normal),
                                "f1_swapped": mean(f1_swapped), "f1_zeroed": mean(f1_zeroed)}
        print(f"[diag] {dummy_name}: normal={mean(f1_normal):.3f}  swapped={mean(f1_swapped):.3f}  "
              f"zeroed={mean(f1_zeroed):.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[diag] DONE. wrote {args.out}", flush=True)
    print("[diag] Compare against known baselines: without-any-extra-block normal=0.471/swapped=0.268/zeroed=0.274; "
          "with-REAL-speech normal=0.000/swapped=0.000/zeroed=0.001", flush=True)


if __name__ == "__main__":
    main()
