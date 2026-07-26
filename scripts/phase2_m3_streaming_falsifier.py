"""scripts/phase2_m3_streaming_falsifier.py — Phase 2.1: M3 grounding
falsifier through the STREAMING (world_state_builder) construction on
real VGGSound clips decoded fresh from source, not read from the cache.
Closes the item-0 divergence if it reproduces the cached-feature
reference (0.482 normal / 0.29 swapped / 0.15 zeroed).

Reuses scripts/m3_grounding_falsifier.py's exact word_overlap_f1 metric,
build_splits() for the held-out (clip_id, caption) pairs, and the same
connector/LLM generation call -- only the FEATURE CONSTRUCTION differs
(world_state_builder.build_world_state_features() on freshly-decoded
video+audio, instead of train_m3.M3CaptionDataset reading the cache).

Usage:
    python scripts/phase2_m3_streaming_falsifier.py --n-clips 50
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

from train_m3 import build_splits, word_overlap_f1, MAX_AMBIENT_T
from scripts.extract_features_av import _decode_video_raw, _decode_audio_raw, NUM_FRAMES, RESOLUTION, AUDIO_SR
from models.world_state_builder import build_world_state_features
from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

VIDEO_DIR = "/home/utkarsh/data/vggsound"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=50)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--connector-ckpt", default="checkpoints/m3_connector/last.pt")
    p.add_argument("--field", default="gpt_sound_acoustic")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/PHASE2_M3_STREAMING_FALSIFIER.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[phase2-m3] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()

    print("[phase2-m3] loading frozen M2 predictor + connector + real encoders...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()

    conn_ckpt = torch.load(args.connector_ckpt, map_location=device, weights_only=False)
    connector_cfg = M3ConnectorConfig(**conn_ckpt["connector_cfg"])
    connector = M3Connector(connector_cfg).to(device)
    connector.load_state_dict(conn_ckpt["connector"])
    connector.eval()

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    print(f"[phase2-m3] building splits (field={args.field})...", flush=True)
    _, test_pairs = build_splits(args.field)
    rng.shuffle(test_pairs)

    eval_pairs = []
    for vid, field, cap in test_pairs:
        video_path = os.path.join(VIDEO_DIR, vid + ".mp4")
        if os.path.isfile(video_path):
            eval_pairs.append((vid, cap, video_path))
        if len(eval_pairs) >= args.n_clips:
            break
    print(f"[phase2-m3] {len(eval_pairs)} real clips with source video available", flush=True)

    @torch.no_grad()
    def pre_pool_for(video_path):
        frames = _decode_video_raw(video_path, NUM_FRAMES, RESOLUTION)
        audio = _decode_audio_raw(video_path, AUDIO_SR)
        true_dur = audio.shape[0] / AUDIO_SR
        result = build_world_state_features(frames, audio, true_dur, vision_enc, base_enc, nat_enc,
                                             predictor_cfg.max_tdm_bins, device)
        feats, tbins = result.feats, result.tbins
        if feats["ambient"].shape[1] > MAX_AMBIENT_T:
            feats = dict(feats); tbins = dict(tbins)
            feats["ambient"] = feats["ambient"][:, :MAX_AMBIENT_T]
            tbins["ambient"] = tbins["ambient"][:, :MAX_AMBIENT_T]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pre_pool = predictor.encode_pre_pool_tokens(feats, tbins)
        return pre_pool.to(torch.bfloat16)   # (1, S, d)

    @torch.no_grad()
    def generate_one(pre_pool):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = connector(pre_pool, None)   # (1, n_lat, H)
        attn = torch.ones(soft_prompt.shape[0], soft_prompt.shape[1], dtype=torch.long, device=device)
        gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn,
                                max_new_tokens=60, do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    f1_normal, f1_swapped, f1_zeroed = [], [], []
    side_by_side = []
    pre_pools = []
    print("[phase2-m3] decoding + encoding all clips (fresh from source)...", flush=True)
    for i, (vid, cap, video_path) in enumerate(eval_pairs):
        try:
            pre_pools.append(pre_pool_for(video_path))
        except Exception as e:
            print(f"[phase2-m3] {vid} FAILED to build features: {e!r}", flush=True)
            pre_pools.append(None)
        if (i + 1) % 10 == 0:
            print(f"[phase2-m3] encoded {i+1}/{len(eval_pairs)}", flush=True)

    print("[phase2-m3] generating (normal / swapped / zeroed)...", flush=True)
    n = len(eval_pairs)
    for i in range(n):
        if pre_pools[i] is None:
            continue
        j = (i + 1) % n   # swap partner
        if pre_pools[j] is None:
            continue
        vid, cap, _ = eval_pairs[i]
        try:
            gen_normal = generate_one(pre_pools[i])
            gen_swapped = generate_one(pre_pools[j])
            gen_zeroed = generate_one(torch.zeros_like(pre_pools[i]))
        except Exception as e:
            print(f"[phase2-m3] {vid} FAILED to generate: {e!r}", flush=True)
            continue

        f1_normal.append(word_overlap_f1(gen_normal, cap))
        f1_swapped.append(word_overlap_f1(gen_swapped, cap))
        f1_zeroed.append(word_overlap_f1(gen_zeroed, cap))
        if len(side_by_side) < 10:
            side_by_side.append({"clip_id": vid, "ground_truth": cap, "normal": gen_normal,
                                  "swapped_from": eval_pairs[j][0], "swapped": gen_swapped,
                                  "zeroed": gen_zeroed})
        if (i + 1) % 10 == 0:
            print(f"[phase2-m3] generated {i+1}/{n}", flush=True)

    result = {
        "n_clips_attempted": len(eval_pairs),
        "n_clips_scored": len(f1_normal),
        "f1_normal_mean": sum(f1_normal) / len(f1_normal) if f1_normal else None,
        "f1_swapped_mean": sum(f1_swapped) / len(f1_swapped) if f1_swapped else None,
        "f1_zeroed_mean": sum(f1_zeroed) / len(f1_zeroed) if f1_zeroed else None,
        "reference_cached_feature_normal": 0.482,
        "reference_cached_feature_swapped": 0.29,
        "reference_cached_feature_zeroed": 0.15,
        "side_by_side_sample": side_by_side,
    }
    print("\n[phase2-m3] === RESULTS ===", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "side_by_side_sample"}, indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[phase2-m3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
