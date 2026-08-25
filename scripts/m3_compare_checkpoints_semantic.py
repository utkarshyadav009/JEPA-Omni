"""scripts/m3_compare_checkpoints_semantic.py — larger-sample semantic
comparison of the OLD M3 connector (checkpoints/m3_multigran_best, trained
on the pre-fix captions + old M2 checkpoint) vs the NEW M3 connector
(checkpoints/m3_multigran_richcaption_v2, trained on the corrected rich
captions + RUN-2 M2 checkpoint).

IMPORTANT CONFOUND, stated up front: this compares two checkpoints that
differ in TWO things at once -- (1) the caption fix and (2) the M2 backbone
(old run used checkpoints/m2_fusion_20k_best/step19000_peak.pt, new run
used the RUN-2 checkpoint checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/
step19000.pt, per train_m3.py's --m2-ckpt for each launch). This script
answers "is the NEW (current, real) M3 pipeline better than the OLD one"
-- it does NOT isolate the caption-fix effect alone. A caption-fix-only
ablation would need holding M2 fixed across both connector trainings.

Uses word-overlap F1 (known to penalize valid paraphrases/length
differences -- see scripts/m3_semantic_rescore.py's docstring) ALONGSIDE
a semantic cosine metric (sentence-transformers/all-MiniLM-L6-v2, mean-
pooled, L2-normalized) so the two can be read side by side, same approach
as the existing task-52 rescore script.

Held-out clips are drawn from the SAME test split (train_m3.py's
build_splits, VGGSound test.csv membership) so both checkpoints are scored
on identical clips. Ground truth for scoring is the CORRECTED (v2) caption
for both checkpoints -- a fair, single target for "which model's captions
are semantically closer to the accurate description."

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/m3_compare_checkpoints_semantic.py --n-clips 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                           AutoTokenizer)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
import train_m3 as m3mod
from train_m3 import (ALL_GRANULARITIES, GRANULARITY_TAGS, CACHE_DIR,
                       build_splits, m3_collate_fn, word_overlap_f1)

CHECKPOINTS = {
    "new_richcaption_v2": {
        "connector_ckpt": "checkpoints/m3_multigran_richcaption_v2/last.pt",
        "m2_ckpt": "checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
    },
    "new_richcaption_v2_align": {
        "connector_ckpt": "checkpoints/m3_multigran_richcaption_v2_align/last.pt",
        "m2_ckpt": "checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
    },
}
# Note: this pairing is a CLEAN single-variable ablation -- both share the identical
# M2 checkpoint and identical (corrected) captions; the only difference is whether
# train_m3.py's --lam-align (EmbeddingGemma alignment InfoNCE loss) was on or off.
CAPTIONS_PATH_V2 = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


@torch.no_grad()
def encode_semantic(texts, tokenizer, model, device, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        hidden = model(**tok).last_hidden_state
        pooled = mean_pool(hidden, tok["attention_mask"])
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def generate_for_checkpoint(tag, cfg, test_pairs, tokenizer, llm, device, rng):
    print(f"[compare] === generating with checkpoint '{tag}' ===", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(cfg["m2_ckpt"], map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()

    conn_ckpt = torch.load(cfg["connector_ckpt"], map_location=device, weights_only=False)
    connector_cfg = M3ConnectorConfig(**conn_ckpt["connector_cfg"])
    connector = M3Connector(connector_cfg).to(device)
    state = conn_ckpt.get("connector", conn_ckpt)
    connector.load_state_dict(state, strict=True)
    connector.eval()

    test_ds = m3mod.M3CaptionDataset(test_pairs, CACHE_DIR, tokenizer)
    print(f"[compare] '{tag}': {len(test_ds)} (clip, field) test rows cached", flush=True)

    results = []
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        batch = m3_collate_fn([item], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        m3mod._cap_ambient_len(feats, tbins, pad_mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
            key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)
            soft_prompt = connector(pre_pool.to(torch.bfloat16), key_padding_mask)
            field = item.get("field")
            prefix_text = GRANULARITY_TAGS.get(field, "")
            if prefix_text:
                prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                                        return_tensors="pt")["input_ids"].to(device)
                prefix_embeds = llm.get_input_embeddings()(prefix_ids)
                gen_inputs_embeds = torch.cat([soft_prompt, prefix_embeds], dim=1)
            else:
                gen_inputs_embeds = soft_prompt
            attn = torch.ones(gen_inputs_embeds.shape[:2], dtype=torch.long, device=device)
            gen_ids = llm.generate(inputs_embeds=gen_inputs_embeds, attention_mask=attn,
                                    max_new_tokens=60, do_sample=False, repetition_penalty=1.15,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        results.append({"clip_id": item["clip_id"], "field": field, "generated": gen_text})
        if (idx + 1) % 100 == 0:
            print(f"[compare] '{tag}': {idx + 1}/{len(test_ds)} generated", flush=True)

    del predictor, connector
    torch.cuda.empty_cache()
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=300, help="held-out clips sampled per field")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "checkpoints", "m3_checkpoint_compare_semantic.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    m3mod.CAPTIONS_PATH = CAPTIONS_PATH_V2
    print(f"[compare] using CORRECTED captions as ground truth for both checkpoints: {CAPTIONS_PATH_V2}", flush=True)
    _, test_pairs_all = build_splits(ALL_GRANULARITIES)

    by_field = {}
    for cid, field, text in test_pairs_all:
        by_field.setdefault(field, []).append((cid, field, text))
    sampled_pairs = []
    for field in ALL_GRANULARITIES:
        pool = by_field.get(field, [])
        rng.shuffle(pool)
        sampled_pairs.extend(pool[:args.n_clips])
    print(f"[compare] sampled {len(sampled_pairs)} (clip, field) rows across {len(ALL_GRANULARITIES)} fields "
          f"(target {args.n_clips}/field)", flush=True)

    gt_lookup = {(cid, field): text for cid, field, text in sampled_pairs}

    print(f"[compare] loading tokenizer + frozen LLM ({args.llm})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    gen_by_tag = {}
    for tag, cfg in CHECKPOINTS.items():
        gen_by_tag[tag] = generate_for_checkpoint(tag, cfg, sampled_pairs, tokenizer, llm, device, rng)

    print("[compare] loading sentence-transformers/all-MiniLM-L6-v2 for semantic scoring...", flush=True)
    sem_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    report = {}
    for field in ALL_GRANULARITIES:
        row_old = [r for r in gen_by_tag["new_richcaption_v2"] if r["field"] == field]
        row_new = [r for r in gen_by_tag["new_richcaption_v2_align"] if r["field"] == field]
        old_by_cid = {r["clip_id"]: r["generated"] for r in row_old}
        new_by_cid = {r["clip_id"]: r["generated"] for r in row_new}
        common_cids = [cid for cid in old_by_cid if cid in new_by_cid and (cid, field) in gt_lookup]
        if not common_cids:
            continue
        gts = [gt_lookup[(cid, field)] for cid in common_cids]
        old_gens = [old_by_cid[cid] for cid in common_cids]
        new_gens = [new_by_cid[cid] for cid in common_cids]

        z_gt = encode_semantic(gts, sem_tok, sem_model, device)
        z_old = encode_semantic(old_gens, sem_tok, sem_model, device)
        z_new = encode_semantic(new_gens, sem_tok, sem_model, device)

        cos_old = (z_old * z_gt).sum(-1).mean().item()
        cos_new = (z_new * z_gt).sum(-1).mean().item()
        f1_old = sum(word_overlap_f1(p, g) for p, g in zip(old_gens, gts)) / len(common_cids)
        f1_new = sum(word_overlap_f1(p, g) for p, g in zip(new_gens, gts)) / len(common_cids)

        report[field] = {
            "n": len(common_cids),
            "cos_old_vs_corrected_gt": cos_old,
            "cos_new_vs_corrected_gt": cos_new,
            "cos_delta_new_minus_old": cos_new - cos_old,
            "f1_old_vs_corrected_gt": f1_old,
            "f1_new_vs_corrected_gt": f1_new,
            "f1_delta_new_minus_old": f1_new - f1_old,
        }
        print(f"[compare] {field} (n={len(common_cids)}): "
              f"cos old={cos_old:.4f} new={cos_new:.4f} delta={cos_new - cos_old:+.4f}  ||  "
              f"F1 old={f1_old:.4f} new={f1_new:.4f} delta={f1_new - f1_old:+.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compare] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
