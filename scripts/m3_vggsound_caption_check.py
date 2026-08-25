"""scripts/m3_vggsound_caption_check.py — sanity check requested directly:
does the locked M2+M3+Qwen stack produce sensible captions on VGGSound
TEST clips (in-distribution, real cached features, real ground-truth
captions to compare against) -- run on mercury (full precision, no
Jetson hardware/latency confound) to isolate "is the grounding itself any
good" from "is it fast enough on the Jetson."

Uses the SAME generation convention validated live on the Jetson this
session: the exact GRANULARITY_TAGS task-tag format M3 was trained with
(not a chat-style question), repetition_penalty=1.15, no_repeat_ngram_size=3,
matching scripts/m4_joint_eval.py's own validated generate() call.

Usage:
    python scripts/m3_vggsound_caption_check.py --n-clips 8 --field gpt_action_brief
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, CACHE_DIR, GRANULARITY_TAGS
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from data.av_cached_dataset import AVCachedDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--m3-ckpt", default="checkpoints/m3_multigran_richcaption_v2/last.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--field", default="gpt_action_brief", choices=list(GRANULARITY_TAGS.keys()),
                   help="same tag family as the live Jetson run that worked well")
    p.add_argument("--n-clips", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=14)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[caption-check] device={device}", flush=True)

    print("[caption-check] loading locked M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    print("[caption-check] loading locked M3 connector...", flush=True)
    m3ckpt = torch.load(args.m3_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    print("[caption-check] loading Qwen2.5-1.5B-Instruct...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()

    print(f"[caption-check] loading VGGSound test split, field={args.field!r}...", flush=True)
    _, test_pairs = build_splits(args.field)
    rng = random.Random(args.seed)
    rng.shuffle(test_pairs)
    sample = test_pairs[: args.n_clips]
    clip_ids = [c for c, _, _ in sample]
    gt_captions = {c: cap for c, _, cap in sample}

    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")

    prefix_text = GRANULARITY_TAGS[args.field]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prefix_embeds = llm.get_input_embeddings()(prefix_ids)

    print(f"\n[caption-check] === {len(ds)} VGGSound test clips, tag={prefix_text!r} ===\n", flush=True)
    n_match_ish = 0
    for i in range(len(ds)):
        item = ds[i]
        clip_id = item["clip_id"]
        feats = {k: v.unsqueeze(0).to(device) for k, v in item["feats"].items()}
        tbins = {k: v.unsqueeze(0).to(device) for k, v in item["tbins"].items()}

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feats_f = {k: v.float() for k, v in feats.items()}
                pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
                soft_prompt = m3_connector(pre_pool)
            inputs_embeds = torch.cat([soft_prompt.to(prefix_embeds.dtype), prefix_embeds], dim=1)
            attn = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)
            gen_ids = llm.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        gt = gt_captions.get(clip_id, "<no caption>")
        print(f"[{i+1}/{len(ds)}] clip_id={clip_id}", flush=True)
        print(f"    PRED: {pred!r}", flush=True)
        print(f"    GT:   {gt!r}\n", flush=True)

    print("[caption-check] DONE -- eyeball PRED vs GT above; this is a qualitative "
          "check (do predictions land in the same topical neighborhood as ground "
          "truth), not an automated metric.", flush=True)


if __name__ == "__main__":
    main()
