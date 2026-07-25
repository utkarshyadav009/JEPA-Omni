"""scripts/m3_grounding_falsifier.py — M3 grounding falsifier.

Tests whether the trained M3 connector+LLM is actually READING the World-
State conditioning, or just emitting corpus-level language priors that
happen to score reasonable word-overlap F1 (VGGSound captions cluster
heavily around a small set of sound-category phrasings, so a model that
ignores conditioning entirely could still score non-trivially by memorizing
what captions "generally sound like").

Three checks:
  1. World-State swap control: generate each of ~200 held-out clips' captions
     three ways -- NORMAL (own World-State), SWAPPED (a different clip's
     World-State, rolled across the batch), ZEROED (all-zero World-State).
     PASS = normal >> swapped ~= zeroed.
  2. Prior-only baselines: F1 of (a) the single most common exact training
     caption, (b) a random training caption, against the same 200 clips'
     ground truth -- calibrates how much of 0.519 is corpus regularity.
  3. Caption-noise check: for the specific clips flagged as wrong-category
     in the qualitative 30-sample review, cross-reference against every
     Streamlit human-review CSV to see if the GROUND-TRUTH caption itself
     was already flagged bad.

Usage:
    python scripts/m3_grounding_falsifier.py --n-clips 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import (build_splits, M3CaptionDataset, m3_collate_fn, word_overlap_f1,
                       _cap_ambient_len, CACHE_DIR)
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

WRONG_CATEGORY_CLIPS = [
    "hTAWbHXCJ2A_000060", "ywS3pG7yQfg_000161", "RyAJMDDxNBQ_000070",
    "6wEyfLFSmmI_000080", "XkyKliH1X3c_000025", "AVYuega54og_000188",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--connector-ckpt", default="checkpoints/m3_connector/last.pt")
    p.add_argument("--field", default="gpt_sound_acoustic")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-side-by-side", type=int, default=10)
    p.add_argument("--out", default="checkpoints/m3_connector/falsifier_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[falsifier] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_config = AutoConfig.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for p_ in llm.parameters():
        p_.requires_grad_(False)

    print("[falsifier] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for p_ in predictor.parameters():
        p_.requires_grad_(False)

    print(f"[falsifier] loading connector from {args.connector_ckpt}...", flush=True)
    conn_ckpt = torch.load(args.connector_ckpt, map_location=device, weights_only=False)
    connector_cfg = M3ConnectorConfig(**conn_ckpt["connector_cfg"])
    connector = M3Connector(connector_cfg).to(device)
    connector.load_state_dict(conn_ckpt["connector"])
    connector.eval()

    print(f"[falsifier] building splits (field={args.field})...", flush=True)
    train_pairs, test_pairs = build_splits(args.field)
    print(f"[falsifier] train={len(train_pairs)}  test={len(test_pairs)}", flush=True)

    rng.shuffle(test_pairs)
    eval_pairs = test_pairs[:args.n_clips]
    eval_ds = M3CaptionDataset(eval_pairs, CACHE_DIR, tokenizer)
    print(f"[falsifier] evaluating on {len(eval_ds)} held-out clips", flush=True)

    @torch.no_grad()
    def get_pre_pool_batch(items):
        batch = m3_collate_fn(items, tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad_mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
        key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)
        return pre_pool.to(torch.bfloat16), key_padding_mask, batch["clip_ids"], batch["caption_texts"]

    @torch.no_grad()
    def generate_batch(pre_pool, key_padding_mask):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = connector(pre_pool, key_padding_mask)   # (B, 32, H)
        B, n_lat, _ = soft_prompt.shape
        attn = torch.ones(B, n_lat, dtype=torch.long, device=device)
        gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn,
                                max_new_tokens=60, do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    # ── Part 1: World-State swap control ────────────────────────────────
    print("[falsifier] === Part 1: World-State swap control ===", flush=True)
    f1_normal, f1_swapped, f1_zeroed = [], [], []
    side_by_side = []
    bs = args.batch_size
    for start in range(0, len(eval_ds), bs):
        items = [eval_ds[i] for i in range(start, min(start + bs, len(eval_ds)))]
        if len(items) < 2:
            continue   # need >=2 to roll meaningfully
        pre_pool, kpm, clip_ids, gts = get_pre_pool_batch(items)

        gen_normal = generate_batch(pre_pool, kpm)
        pre_pool_swapped = torch.roll(pre_pool, shifts=1, dims=0)
        kpm_swapped = torch.roll(kpm, shifts=1, dims=0)
        gen_swapped = generate_batch(pre_pool_swapped, kpm_swapped)
        pre_pool_zeroed = torch.zeros_like(pre_pool)
        gen_zeroed = generate_batch(pre_pool_zeroed, kpm)

        for i in range(len(items)):
            f1_normal.append(word_overlap_f1(gen_normal[i], gts[i]))
            f1_swapped.append(word_overlap_f1(gen_swapped[i], gts[i]))
            f1_zeroed.append(word_overlap_f1(gen_zeroed[i], gts[i]))
            if len(side_by_side) < args.n_side_by_side:
                side_by_side.append({
                    "clip_id": clip_ids[i], "ground_truth": gts[i],
                    "normal": gen_normal[i], "swapped_from": clip_ids[(i - 1) % len(items)],
                    "swapped": gen_swapped[i],
                })
        print(f"[falsifier] processed {min(start+bs, len(eval_ds))}/{len(eval_ds)}", flush=True)

    mean_normal = sum(f1_normal) / len(f1_normal)
    mean_swapped = sum(f1_swapped) / len(f1_swapped)
    mean_zeroed = sum(f1_zeroed) / len(f1_zeroed)
    print(f"[falsifier] F1 -- normal={mean_normal:.3f}  swapped={mean_swapped:.3f}  "
          f"zeroed={mean_zeroed:.3f}  (n={len(f1_normal)})", flush=True)

    # ── Part 2: prior-only baselines ────────────────────────────────────
    print("[falsifier] === Part 2: prior-only baselines ===", flush=True)
    train_captions = [c for _, c in train_pairs]
    caption_counts = Counter(train_captions)
    most_common_caption, most_common_n = caption_counts.most_common(1)[0]
    print(f"[falsifier] most common training caption (n={most_common_n}/{len(train_captions)}): "
          f"\"{most_common_caption}\"", flush=True)

    eval_gts = [c for _, c in eval_pairs]
    f1_mode = [word_overlap_f1(most_common_caption, gt) for gt in eval_gts]
    mean_f1_mode = sum(f1_mode) / len(f1_mode)

    rng2 = random.Random(args.seed + 1)
    f1_random_train = [word_overlap_f1(rng2.choice(train_captions), gt) for gt in eval_gts]
    mean_f1_random_train = sum(f1_random_train) / len(f1_random_train)

    print(f"[falsifier] F1 -- most_common_caption={mean_f1_mode:.3f}  "
          f"random_train_caption={mean_f1_random_train:.3f}", flush=True)

    # ── Part 3: caption-noise check ─────────────────────────────────────
    print("[falsifier] === Part 3: caption-noise check ===", flush=True)
    review_files = [
        "scripts/caption_review_results.csv", "scripts/mg_pilot_review_results.csv",
        "scripts/mgv2_review_results.csv", "scripts/congruence_filter_review_results.csv",
    ]
    reviews_by_clip: dict = {}
    for rf in review_files:
        path = os.path.join(PROJECT_ROOT, rf)
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                reviews_by_clip.setdefault(row["clip_id"], []).append({"file": rf, "row": row})

    caption_by_id = {}
    with open(os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions.jsonl")) as f:
        wanted = set(WRONG_CATEGORY_CLIPS)
        for line in f:
            r = json.loads(line)
            if r["clip_id"] in wanted:
                caption_by_id[r["clip_id"]] = r

    noise_report = []
    for cid in WRONG_CATEGORY_CLIPS:
        rec = caption_by_id.get(cid, {})
        entry = {"clip_id": cid, "vggsound_label": rec.get("vggsound_label"),
                 "gpt_sound_acoustic": rec.get("gpt_sound_acoustic"),
                 "review_hits": reviews_by_clip.get(cid, [])}
        noise_report.append(entry)
        print(f"[falsifier] {cid}  [{rec.get('vggsound_label')}]  "
              f"reviewed_elsewhere={len(entry['review_hits'])>0}", flush=True)

    # ── Save + print full report ────────────────────────────────────────
    results = {
        "swap_control": {"n": len(f1_normal), "f1_normal": mean_normal,
                          "f1_swapped": mean_swapped, "f1_zeroed": mean_zeroed},
        "prior_baselines": {"most_common_caption": most_common_caption,
                             "most_common_n_in_train": most_common_n,
                             "f1_most_common": mean_f1_mode,
                             "f1_random_train": mean_f1_random_train},
        "side_by_side": side_by_side,
        "caption_noise_check": noise_report,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[falsifier] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
