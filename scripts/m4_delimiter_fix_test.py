"""scripts/m4_delimiter_fix_test.py — cheap fix attempt for the check-(c)
collapse: explicit stream-delimiter text between the M3 soft prompt and the
M4b speech segment.

Uses literal text markers (tokenized normally through the frozen LLM's own
vocabulary, zero new parameters, works zero-shot) rather than new reserved
special tokens -- leverages the frozen LLM's existing, pretrained ability to
read natural-language section markers, instead of asking it to assign
meaning to an arbitrary unused token id it has never seen used this way.

Re-runs the same (c) check (M3 normal/swapped/zeroed with a FIXED real
M4b speech segment concatenated) with a delimiter phrase inserted between
the M3 latents and the speech block:
    [M3 32 latents] + delimiter_text + [M4b speech tokens] + tag prefix

Usage:
    python scripts/m4_delimiter_fix_test.py --n-clips 100
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
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.av_cached_dataset import AVCachedDataset
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELD = "gpt_sound_acoustic"
DELIMITER_TEXT = "\n\n[End of scene description. A separate audio clip transcript follows below, unrelated to the scene above.]\n\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--m3-connector-ckpt", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--m4b-projector-ckpt", default="checkpoints/m4b/best.pt")
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4b/delimiter_fix_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[delim] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[delim] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)

    print("[delim] loading frozen M3 connector...", flush=True)
    m3ckpt = torch.load(args.m3_connector_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    print("[delim] loading frozen whisper + trained M4b projector...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    m4bckpt = torch.load(args.m4b_projector_ckpt, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**m4bckpt["projector_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(m4bckpt["projector"])
    m4b_projector.eval()

    print("[delim] building VGGSound eval pool...", flush=True)
    _, vgg_test_pairs = build_splits(FIELD)
    rng.shuffle(vgg_test_pairs)
    vgg_eval_pairs = vgg_test_pairs[:args.n_clips]

    print("[delim] building fixed speech segment...", flush=True)
    _, easycom_test_segs = build_segments()
    rng.shuffle(easycom_test_segs)
    easycom_ds = EasyComSpeechDataset(easycom_test_segs[:1])
    fixed_item = easycom_ds[0]
    fixed_batch = m4b_collate_fn([fixed_item])
    fixed_hidden, fixed_valid_frames = whisper(fixed_batch["waveforms"], fixed_batch["durations_sec"], device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        FIXED_SPEECH_TOKENS, FIXED_SPEECH_MASK = m4b_projector(fixed_hidden.float(), fixed_valid_frames)
    print(f"[delim] fixed speech segment: {fixed_item['text'][:80]!r}  "
          f"n_speech_tokens={FIXED_SPEECH_TOKENS.shape[1]}", flush=True)

    delim_ids = tokenizer(DELIMITER_TEXT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    delim_embeds_single = llm.get_input_embeddings()(delim_ids)
    print(f"[delim] delimiter text tokenizes to {delim_embeds_single.shape[1]} tokens: {DELIMITER_TEXT!r}", flush=True)

    def prefix_embeds_for_field(field, B):
        ids = tokenizer(GRANULARITY_TAGS[field], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return llm.get_input_embeddings()(ids).expand(B, -1, -1)

    def expand_fixed(fixed_tensor, B):
        return fixed_tensor.expand(B, *fixed_tensor.shape[1:])

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

    f1_normal, f1_swapped, f1_zeroed = [], [], []
    side_by_side = []
    bs = args.batch_size
    for start in range(0, len(vgg_eval_pairs), bs):
        batch_pairs = vgg_eval_pairs[start:start + bs]
        if len(batch_pairs) < 2:
            continue
        gts = [g for _, _, g in batch_pairs]
        clip_ids = [c for c, _, _ in batch_pairs]
        pre_pool, kpm = m3_batch_pre_pool(batch_pairs)
        pre_pool_swapped = torch.roll(pre_pool, shifts=1, dims=0)
        kpm_swapped = torch.roll(kpm, shifts=1, dims=0)
        pre_pool_zeroed = torch.zeros_like(pre_pool)
        B = pre_pool.shape[0]
        prefix = prefix_embeds_for_field(FIELD, B)
        prefix_attn = torch.ones(B, prefix.shape[1], dtype=torch.long, device=device)
        delim_embeds = delim_embeds_single.expand(B, -1, -1)
        delim_attn = torch.ones(B, delim_embeds.shape[1], dtype=torch.long, device=device)
        speech_tokens = expand_fixed(FIXED_SPEECH_TOKENS, B)
        speech_mask = expand_fixed(FIXED_SPEECH_MASK, B)
        speech_attn = (~speech_mask).long()

        for cond_name, pp, mask in [("normal", pre_pool, kpm), ("swapped", pre_pool_swapped, kpm_swapped),
                                     ("zeroed", pre_pool_zeroed, kpm)]:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                m3_latents = m3_connector(pp, mask)
            m3_attn = torch.ones(B, m3_latents.shape[1], dtype=torch.long, device=device)
            emb = torch.cat([m3_latents, delim_embeds, speech_tokens, prefix], dim=1)
            attn = torch.cat([m3_attn, delim_attn, speech_attn, prefix_attn], dim=1)
            gen = generate(emb, attn)
            for i in range(B):
                f1 = word_overlap_f1(gen[i], gts[i])
                {"normal": f1_normal, "swapped": f1_swapped, "zeroed": f1_zeroed}[cond_name].append(f1)
                if cond_name == "normal" and len(side_by_side) < 8:
                    side_by_side.append({"clip_id": clip_ids[i], "ground_truth": gts[i], "generated": gen[i], "f1": f1})
        print(f"[delim] processed {min(start+bs, len(vgg_eval_pairs))}/{len(vgg_eval_pairs)}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    results = {"n": len(f1_normal), "f1_normal": mean(f1_normal), "f1_swapped": mean(f1_swapped),
               "f1_zeroed": mean(f1_zeroed), "side_by_side": side_by_side}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[delim] WITH DELIMITER + real speech: normal={results['f1_normal']:.3f}  "
          f"swapped={results['f1_swapped']:.3f}  zeroed={results['f1_zeroed']:.3f}", flush=True)
    print("[delim] compare: without-any-extra-block normal=0.471/swapped=0.268/zeroed=0.274; "
          "with-speech-NO-delimiter normal=0.000/swapped=0.000/zeroed=0.001", flush=True)
    print(f"[delim] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
