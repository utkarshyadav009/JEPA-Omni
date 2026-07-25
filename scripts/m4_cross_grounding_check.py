"""scripts/m4_cross_grounding_check.py — M4b gate checks (c) and (d).

The genuinely risky part of the M4b design: M3's connector and M4b's
projector were trained INDEPENDENTLY (M3 on VGGSound, frozen; M4b on
EasyCom, with M3 frozen throughout) but both write into the SAME shared LLM
input space. Nothing so far has tested whether they actually coexist
without interference when BOTH are present in one context at once.

(c) M3 grounding still holds WITH the speech stream concatenated: rerun the
    M3 swap-control (normal / swapped / zeroed on the M3 latents) on
    VGGSound clips, but with a FIXED (same for every trial) M4b speech
    segment ALSO concatenated into context. Compare the normal-vs-swapped/
    zeroed gap against the WITHOUT-speech baseline (re-measured here, not
    assumed from the earlier standalone falsifier, so both numbers come
    from the exact same run/conditions).

(d) The reverse: M4b's speech swap-control (normal vs swapped) WITH a FIXED
    (same for every trial) M3 32-latent segment (from a real VGGSound clip)
    ALSO concatenated into context. Compare against the WITHOUT-M3-latents
    baseline, measured here under the same conditions.

Sequence convention (fixed for both checks): [M3 32 latents] + [M4b speech
tokens] + [target text]. If either connector's own grounding degrades when
the other's stream is present, that is the failure this script is built to
catch.

Usage:
    python scripts/m4_cross_grounding_check.py --n-clips 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, m3_collate_fn, word_overlap_f1, _cap_ambient_len, CACHE_DIR, GRANULARITY_TAGS
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn
from train_m4b import word_error_rate

FIELD = "gpt_sound_acoustic"


@torch.no_grad()
def encode_semantic(texts, tokenizer, model, device, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        hidden = model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--m3-connector-ckpt", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--m4b-projector-ckpt", default="checkpoints/m4b/last.pt")
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4b/cross_grounding_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[cross-check] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[cross-check] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)

    print("[cross-check] loading frozen M3 connector...", flush=True)
    m3ckpt = torch.load(args.m3_connector_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    print("[cross-check] loading frozen whisper + trained M4b projector...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    m4bckpt = torch.load(args.m4b_projector_ckpt, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**m4bckpt["projector_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(m4bckpt["projector"])
    m4b_projector.eval()

    # ---- data pools ----
    print("[cross-check] building VGGSound eval pool (M3 side)...", flush=True)
    _, vgg_test_pairs = build_splits(FIELD)
    rng.shuffle(vgg_test_pairs)
    vgg_eval_pairs = vgg_test_pairs[:args.n_clips]

    print("[cross-check] building EasyCom eval pool (M4b side)...", flush=True)
    _, easycom_test_segs = build_segments()
    rng.shuffle(easycom_test_segs)
    easycom_eval_segs = easycom_test_segs[:args.n_clips]
    easycom_ds = EasyComSpeechDataset(easycom_eval_segs)

    print("[cross-check] loading MiniLM for semantic-cosine...", flush=True)
    sem_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    # ---- fixed cross-modal segments (held constant across all trials) ----
    fixed_easycom_item = easycom_ds[0]
    fixed_batch = m4b_collate_fn([fixed_easycom_item])
    fixed_hidden, fixed_valid_frames = whisper(fixed_batch["waveforms"], fixed_batch["durations_sec"], device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        FIXED_SPEECH_TOKENS, FIXED_SPEECH_MASK = m4b_projector(fixed_hidden.float(), fixed_valid_frames)
    print(f"[cross-check] fixed speech segment (held constant for check c): "
          f"{fixed_easycom_item['text'][:80]!r}", flush=True)

    from data.av_cached_dataset import AVCachedDataset
    fixed_vgg_clip_id, _, fixed_vgg_gt = vgg_eval_pairs[0]
    fixed_ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=[fixed_vgg_clip_id], max_tdm_bins=512, audio_mode="mean")
    fixed_vgg_batch = m3_collate_fn([{
        "feats": fixed_ds[0]["feats"], "tbins": fixed_ds[0]["tbins"], "clip_id": fixed_ds[0]["clip_id"],
        "prefix_ids": torch.zeros(0, dtype=torch.long), "caption_ids": torch.zeros(1, dtype=torch.long),
        "caption_text": "", "field": None,
    }], tokenizer.pad_token_id)
    fv_feats = {k: v.to(device) for k, v in fixed_vgg_batch["feats"].items()}
    fv_tbins = {k: v.to(device) for k, v in fixed_vgg_batch["tbins"].items()}
    fv_pad = {k: v.to(device) for k, v in fixed_vgg_batch["padding_mask"].items()}
    _cap_ambient_len(fv_feats, fv_tbins, fv_pad)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        fv_feats_f = {k: v.float() for k, v in fv_feats.items()}
        fv_pre_pool = predictor.encode_pre_pool_tokens(fv_feats_f, fv_tbins)
        fv_kpm = torch.cat([fv_pad["vision"], fv_pad["ambient"]], dim=1)
        FIXED_M3_LATENTS = m3_connector(fv_pre_pool.to(torch.bfloat16), fv_kpm)   # (1, 32, H)
    print(f"[cross-check] fixed M3 clip (held constant for check d): "
          f"{fixed_vgg_clip_id}  {fixed_vgg_gt[:80]!r}", flush=True)

    def expand_fixed(fixed_tensor, B):
        return fixed_tensor.expand(B, *fixed_tensor.shape[1:])

    @torch.no_grad()
    def generate(inputs_embeds, attention_mask):
        gen_ids = llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=60,
                                do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    def prefix_embeds_for_field(field, B):
        ids = tokenizer(GRANULARITY_TAGS[field], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return llm.get_input_embeddings()(ids).expand(B, -1, -1)

    # ============ (c) M3 grounding WITH / WITHOUT fixed speech stream ============
    print("\n[cross-check] === (c) M3 swap-control with speech stream present ===", flush=True)
    from data.av_cached_dataset import AVCachedDataset as _DS

    def m3_batch_pre_pool(pairs_batch):
        clip_ids = [c for c, _, _ in pairs_batch]
        ds = _DS(cache_dir=CACHE_DIR, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
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
        return pre_pool.to(torch.bfloat16), kpm, batch["clip_ids"]

    f1_normal_c, f1_swapped_c, f1_zeroed_c = [], [], []
    f1_normal_nospeech, f1_swapped_nospeech, f1_zeroed_nospeech = [], [], []
    bs = args.batch_size
    for start in range(0, len(vgg_eval_pairs), bs):
        batch_pairs = vgg_eval_pairs[start:start + bs]
        if len(batch_pairs) < 2:
            continue
        gts = [g for _, _, g in batch_pairs]
        pre_pool, kpm, _ = m3_batch_pre_pool(batch_pairs)
        pre_pool_swapped = torch.roll(pre_pool, shifts=1, dims=0)
        kpm_swapped = torch.roll(kpm, shifts=1, dims=0)
        pre_pool_zeroed = torch.zeros_like(pre_pool)
        B = pre_pool.shape[0]
        prefix = prefix_embeds_for_field(FIELD, B)

        for cond_name, pp, mask in [("normal", pre_pool, kpm), ("swapped", pre_pool_swapped, kpm_swapped),
                                     ("zeroed", pre_pool_zeroed, kpm)]:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                m3_latents = m3_connector(pp, mask)   # (B, 32, H)
            m3_attn = torch.ones(B, m3_latents.shape[1], dtype=torch.long, device=device)

            # WITHOUT speech stream (M3 alone + tag prefix)
            emb_no_speech = torch.cat([m3_latents, prefix], dim=1)
            attn_no_speech = torch.cat([m3_attn, torch.ones(B, prefix.shape[1], dtype=torch.long, device=device)], dim=1)
            gen_no_speech = generate(emb_no_speech, attn_no_speech)

            # WITH fixed speech stream (M3 + speech + tag prefix)
            speech_tokens = expand_fixed(FIXED_SPEECH_TOKENS, B)
            speech_mask = expand_fixed(FIXED_SPEECH_MASK, B)
            speech_attn = (~speech_mask).long()
            emb_with_speech = torch.cat([m3_latents, speech_tokens, prefix], dim=1)
            attn_with_speech = torch.cat([m3_attn, speech_attn,
                                           torch.ones(B, prefix.shape[1], dtype=torch.long, device=device)], dim=1)
            gen_with_speech = generate(emb_with_speech, attn_with_speech)

            for i in range(B):
                f1_ns = word_overlap_f1(gen_no_speech[i], gts[i])
                f1_ws = word_overlap_f1(gen_with_speech[i], gts[i])
                if cond_name == "normal":
                    f1_normal_nospeech.append(f1_ns); f1_normal_c.append(f1_ws)
                elif cond_name == "swapped":
                    f1_swapped_nospeech.append(f1_ns); f1_swapped_c.append(f1_ws)
                else:
                    f1_zeroed_nospeech.append(f1_ns); f1_zeroed_c.append(f1_ws)
        print(f"[cross-check] (c) processed {min(start+bs, len(vgg_eval_pairs))}/{len(vgg_eval_pairs)}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    result_c = {
        "n": len(f1_normal_c),
        "without_speech": {"f1_normal": mean(f1_normal_nospeech), "f1_swapped": mean(f1_swapped_nospeech), "f1_zeroed": mean(f1_zeroed_nospeech)},
        "with_speech": {"f1_normal": mean(f1_normal_c), "f1_swapped": mean(f1_swapped_c), "f1_zeroed": mean(f1_zeroed_c)},
    }
    print(f"\n[cross-check] (c) WITHOUT speech: normal={result_c['without_speech']['f1_normal']:.3f} "
          f"swapped={result_c['without_speech']['f1_swapped']:.3f} zeroed={result_c['without_speech']['f1_zeroed']:.3f}", flush=True)
    print(f"[cross-check] (c) WITH speech:    normal={result_c['with_speech']['f1_normal']:.3f} "
          f"swapped={result_c['with_speech']['f1_swapped']:.3f} zeroed={result_c['with_speech']['f1_zeroed']:.3f}", flush=True)

    # ============ (d) M4b swap-control WITH / WITHOUT fixed M3 latents ============
    print("\n[cross-check] === (d) M4b swap-control with M3 latents present ===", flush=True)
    cos_normal_nom3, cos_swapped_target_nom3, cos_swapped_donor_nom3 = [], [], []
    cos_normal_d, cos_swapped_target_d, cos_swapped_donor_d = [], [], []
    for start in range(0, len(easycom_ds), bs):
        items = [easycom_ds[i] for i in range(start, min(start + bs, len(easycom_ds)))]
        if len(items) < 2:
            continue
        batch = m4b_collate_fn(items)
        gts = batch["texts"]
        hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            speech_tokens, speech_mask = m4b_projector(hidden.float(), valid_frames)
        B = speech_tokens.shape[0]
        speech_tokens_swapped = torch.roll(speech_tokens, shifts=1, dims=0)
        speech_mask_swapped = torch.roll(speech_mask, shifts=1, dims=0)
        donor_texts = [gts[(i - 1) % B] for i in range(B)]

        for cond_name, stoks, smask in [("normal", speech_tokens, speech_mask), ("swapped", speech_tokens_swapped, speech_mask_swapped)]:
            sattn = (~smask).long()

            # WITHOUT M3 latents
            emb_no_m3 = stoks
            attn_no_m3 = sattn
            gen_no_m3 = generate(emb_no_m3, attn_no_m3)

            # WITH fixed M3 latents
            m3_lat = expand_fixed(FIXED_M3_LATENTS, B)
            m3_attn = torch.ones(B, m3_lat.shape[1], dtype=torch.long, device=device)
            emb_with_m3 = torch.cat([m3_lat, stoks], dim=1)
            attn_with_m3 = torch.cat([m3_attn, sattn], dim=1)
            gen_with_m3 = generate(emb_with_m3, attn_with_m3)

            z_gen_no_m3 = encode_semantic(gen_no_m3, sem_tok, sem_model, device)
            z_gen_with_m3 = encode_semantic(gen_with_m3, sem_tok, sem_model, device)
            z_gt = encode_semantic(gts, sem_tok, sem_model, device)
            z_donor = encode_semantic(donor_texts, sem_tok, sem_model, device)

            if cond_name == "normal":
                cos_normal_nom3.extend((z_gen_no_m3 * z_gt).sum(-1).tolist())
                cos_normal_d.extend((z_gen_with_m3 * z_gt).sum(-1).tolist())
            else:
                cos_swapped_target_nom3.extend((z_gen_no_m3 * z_gt).sum(-1).tolist())
                cos_swapped_donor_nom3.extend((z_gen_no_m3 * z_donor).sum(-1).tolist())
                cos_swapped_target_d.extend((z_gen_with_m3 * z_gt).sum(-1).tolist())
                cos_swapped_donor_d.extend((z_gen_with_m3 * z_donor).sum(-1).tolist())
        print(f"[cross-check] (d) processed {min(start+bs, len(easycom_ds))}/{len(easycom_ds)}", flush=True)

    result_d = {
        "n": len(cos_normal_nom3),
        "without_m3": {"cos_normal": mean(cos_normal_nom3), "cos_swapped_vs_target": mean(cos_swapped_target_nom3), "cos_swapped_vs_donor": mean(cos_swapped_donor_nom3)},
        "with_m3": {"cos_normal": mean(cos_normal_d), "cos_swapped_vs_target": mean(cos_swapped_target_d), "cos_swapped_vs_donor": mean(cos_swapped_donor_d)},
    }
    print(f"\n[cross-check] (d) WITHOUT M3 latents: normal={result_d['without_m3']['cos_normal']:.3f} "
          f"swapped_vs_target={result_d['without_m3']['cos_swapped_vs_target']:.3f} "
          f"swapped_vs_donor={result_d['without_m3']['cos_swapped_vs_donor']:.3f}", flush=True)
    print(f"[cross-check] (d) WITH M3 latents:    normal={result_d['with_m3']['cos_normal']:.3f} "
          f"swapped_vs_target={result_d['with_m3']['cos_swapped_vs_target']:.3f} "
          f"swapped_vs_donor={result_d['with_m3']['cos_swapped_vs_donor']:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"c_m3_with_speech": result_c, "d_m4b_with_m3": result_d}, f, indent=2)
    print(f"\n[cross-check] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
