"""scripts/m3_multigran_falsifier.py — M3 stage-2 (multi-granularity) modality
balance test.

Extends the stage-1 grounding falsifier (m3_grounding_falsifier.py) to all
five caption granularities, on a FIXED pool of held-out clips that have all
five captions present, so every field is evaluated on the identical clip
set (apples-to-apples). For each field:
  - NORMAL:  connector output for clip i's own World-State + that field's tag
  - SWAPPED: connector output for clip j's World-State (rolled in-batch) +
             the SAME field's tag, target is still clip i's ground truth
  - prior baselines: most-common exact training caption for that field, and
             a random training caption for that field

Reports the per-granularity table (grouped VISUAL vs ACOUSTIC) and, for a
handful of clips, all five granularities' generations side by side so style
modulation (brief vs detailed vs acoustic) can be read directly.

Usage:
    python scripts/m3_multigran_falsifier.py --n-clips 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import (build_splits, m3_collate_fn, word_overlap_f1, _cap_ambient_len,
                       CACHE_DIR, ALL_GRANULARITIES, GRANULARITY_TAGS)
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from data.av_cached_dataset import AVCachedDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

VISUAL_FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief", "gpt_summary_detailed"]
ACOUSTIC_FIELDS = ["gpt_sound_acoustic"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--connector-ckpt", default="checkpoints/m3_multigran/last.pt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-side-by-side", type=int, default=10)
    p.add_argument("--out", default="checkpoints/m3_multigran/multigran_falsifier_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[mg-falsifier] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_config = AutoConfig.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for p_ in llm.parameters():
        p_.requires_grad_(False)

    print("[mg-falsifier] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for p_ in predictor.parameters():
        p_.requires_grad_(False)

    print(f"[mg-falsifier] loading connector from {args.connector_ckpt}...", flush=True)
    conn_ckpt = torch.load(args.connector_ckpt, map_location=device, weights_only=False)
    connector_cfg = M3ConnectorConfig(**conn_ckpt["connector_cfg"])
    connector = M3Connector(connector_cfg).to(device)
    connector.load_state_dict(conn_ckpt["connector"])
    connector.eval()
    print(f"[mg-falsifier] connector trained to step {conn_ckpt.get('step')}", flush=True)

    print("[mg-falsifier] building splits across all 5 granularities...", flush=True)
    train_pairs, test_pairs = build_splits(ALL_GRANULARITIES)

    # ── group by clip, keep only clips with ALL 5 fields present ───────────
    train_by_clip = defaultdict(dict)
    for cid, field, text in train_pairs:
        train_by_clip[cid][field] = text
    test_by_clip = defaultdict(dict)
    for cid, field, text in test_pairs:
        test_by_clip[cid][field] = text

    common_test_clips = [cid for cid, d in test_by_clip.items() if len(d) == len(ALL_GRANULARITIES)]
    rng.shuffle(common_test_clips)
    eval_clips = common_test_clips[:args.n_clips]
    print(f"[mg-falsifier] {len(common_test_clips)} test clips have all 5 fields; "
          f"evaluating on {len(eval_clips)}", flush=True)

    # ── prior baselines: per-field mode + random training caption ──────────
    train_captions_by_field = defaultdict(list)
    for cid, field, text in train_pairs:
        train_captions_by_field[field].append(text)
    mode_caption, mode_n = {}, {}
    for field in ALL_GRANULARITIES:
        counts = Counter(train_captions_by_field[field])
        mode_caption[field], mode_n[field] = counts.most_common(1)[0]
    rng2 = random.Random(args.seed + 1)

    @torch.no_grad()
    def get_pre_pool_for_clips(clip_ids):
        av_ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
        items = [av_ds[i] for i in range(len(av_ds))]
        batch = m3_collate_fn(
            [{"feats": it["feats"], "tbins": it["tbins"], "clip_id": it["clip_id"],
              "prefix_ids": torch.zeros(0, dtype=torch.long),
              "caption_ids": torch.zeros(1, dtype=torch.long), "caption_text": "", "field": None}
             for it in items], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad_mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
        key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)
        return pre_pool.to(torch.bfloat16), key_padding_mask, batch["clip_ids"]

    @torch.no_grad()
    def generate_batch(pre_pool, key_padding_mask, field):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = connector(pre_pool, key_padding_mask)   # (B, 32, H)
        B = soft_prompt.shape[0]
        prefix_ids = tokenizer(GRANULARITY_TAGS[field], add_special_tokens=False,
                                return_tensors="pt")["input_ids"].to(device).expand(B, -1)
        prefix_embeds = llm.get_input_embeddings()(prefix_ids)
        inputs_embeds = torch.cat([soft_prompt, prefix_embeds], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        gen_ids = llm.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                                max_new_tokens=80, do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    # ── main sweep: per field, normal + swapped, on the same clip pool ─────
    per_field = {f: {"f1_normal": [], "f1_swapped": []} for f in ALL_GRANULARITIES}
    side_by_side = []   # list of {clip_id, per-field: {ground_truth, normal}}
    all_clips = []      # full per-clip records for ALL eval_clips (not just side_by_side)
    bs = args.batch_size
    for start in range(0, len(eval_clips), bs):
        batch_clips = eval_clips[start:start + bs]
        if len(batch_clips) < 2:
            continue
        pre_pool, kpm, clip_ids = get_pre_pool_for_clips(batch_clips)
        pre_pool_swapped = torch.roll(pre_pool, shifts=1, dims=0)
        kpm_swapped = torch.roll(kpm, shifts=1, dims=0)

        row = {cid: {"clip_id": cid, "fields": {}} for cid in clip_ids}
        for field in ALL_GRANULARITIES:
            gen_normal = generate_batch(pre_pool, kpm, field)
            gen_swapped = generate_batch(pre_pool_swapped, kpm_swapped, field)
            for i, cid in enumerate(clip_ids):
                gt = test_by_clip[cid][field]
                f1n = word_overlap_f1(gen_normal[i], gt)
                f1s = word_overlap_f1(gen_swapped[i], gt)
                per_field[field]["f1_normal"].append(f1n)
                per_field[field]["f1_swapped"].append(f1s)
                row[cid]["fields"][field] = {"ground_truth": gt, "normal": gen_normal[i],
                                              "swapped": gen_swapped[i],
                                              "f1_normal": f1n, "f1_swapped": f1s}
        for cid in clip_ids:
            all_clips.append(row[cid])
            if len(side_by_side) < args.n_side_by_side:
                side_by_side.append(row[cid])
        print(f"[mg-falsifier] processed {min(start+bs, len(eval_clips))}/{len(eval_clips)}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    table = {}
    for field in ALL_GRANULARITIES:
        eval_gts = [test_by_clip[cid][field] for cid in eval_clips]
        f1_mode = mean([word_overlap_f1(mode_caption[field], gt) for gt in eval_gts])
        f1_rand = mean([word_overlap_f1(rng2.choice(train_captions_by_field[field]), gt) for gt in eval_gts])
        table[field] = {
            "group": "visual" if field in VISUAL_FIELDS else "acoustic",
            "n": len(per_field[field]["f1_normal"]),
            "f1_normal": mean(per_field[field]["f1_normal"]),
            "f1_swapped": mean(per_field[field]["f1_swapped"]),
            "f1_mode_baseline": f1_mode,
            "f1_random_train_baseline": f1_rand,
            "mode_caption": mode_caption[field],
            "mode_n_in_train": mode_n[field],
        }
        print(f"[mg-falsifier] {field:22s} ({table[field]['group']:8s})  "
              f"normal={table[field]['f1_normal']:.3f}  swapped={table[field]['f1_swapped']:.3f}  "
              f"mode={f1_mode:.3f}  random={f1_rand:.3f}", flush=True)

    results = {"per_field": table, "side_by_side": side_by_side, "all_clips": all_clips,
               "n_eval_clips": len(eval_clips)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mg-falsifier] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
