"""scripts/m4_joint_eval.py — mandatory post-joint-training checks.

After train_m4_joint.py, per instruction, MUST verify:
  1. Standalone M3 grounding falsifier (normal/swapped/zeroed, no speech
     present) with the jointly-trained M3 connector -- catastrophic-
     forgetting check vs the original frozen-LLM M3 baseline.
  2. Standalone M4b speech swap-control (no M3 present) with the jointly-
     trained M4b projector -- same check for the speech side.
  3. Semantic-cosine of M3 captions vs the frozen-LLM M3 baseline
     (computed fresh here for both, so it's an apples-to-apples number).
  4. Gate (c) and (d) RE-RUN with the jointly-trained connectors -- did the
     composition failure actually get fixed?

Usage:
    python scripts/m4_joint_eval.py --n-clips 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, m3_collate_fn, word_overlap_f1, _cap_ambient_len, CACHE_DIR, GRANULARITY_TAGS
from train_m4b import word_error_rate
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.av_cached_dataset import AVCachedDataset
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn

FIELD = "gpt_sound_acoustic"


@torch.no_grad()
def encode_semantic(texts, tok, model, device, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        t = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        hidden = model(**t).last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--m3-baseline-ckpt", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--m4b-baseline-ckpt", default="checkpoints/m4b/best.pt")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4_joint/joint_eval_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[joint-eval] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[joint-eval] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)

    print("[joint-eval] loading whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[joint-eval] loading baseline M3 connector, baseline M4b projector, and joint checkpoint...", flush=True)
    m3_base_ckpt = torch.load(args.m3_baseline_ckpt, map_location=device, weights_only=False)
    m3_base_cfg = M3ConnectorConfig(**m3_base_ckpt["connector_cfg"])
    m3_baseline = M3Connector(m3_base_cfg).to(device)
    m3_baseline.load_state_dict(m3_base_ckpt["connector"])
    m3_baseline.eval()

    m4b_base_ckpt = torch.load(args.m4b_baseline_ckpt, map_location=device, weights_only=False)
    m4b_base_cfg = UltravoxProjectorConfig(**m4b_base_ckpt["projector_cfg"])
    m4b_baseline = UltravoxProjector(m4b_base_cfg).to(device)
    m4b_baseline.load_state_dict(m4b_base_ckpt["projector"])
    m4b_baseline.eval()

    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m3_joint_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_joint = M3Connector(m3_joint_cfg).to(device)
    m3_joint.load_state_dict(joint_ckpt["m3_connector"])
    m3_joint.eval()

    m4b_joint_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_joint = UltravoxProjector(m4b_joint_cfg).to(device)
    m4b_joint.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_joint.eval()
    print(f"[joint-eval] joint checkpoint from step {joint_ckpt.get('step')}", flush=True)

    print("[joint-eval] loading MiniLM for semantic-cosine...", flush=True)
    sem_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    print("[joint-eval] building eval pools...", flush=True)
    _, vgg_test_pairs = build_splits(FIELD)
    rng.shuffle(vgg_test_pairs)
    vgg_eval_pairs = vgg_test_pairs[:args.n_clips]

    _, easycom_test_segs = build_segments()
    rng.shuffle(easycom_test_segs)
    easycom_eval_segs = easycom_test_segs[:args.n_clips]
    easycom_ds = EasyComSpeechDataset(easycom_eval_segs)

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

    # ============ (1)/(3) standalone M3, baseline vs joint ============
    def eval_m3_standalone(m3_connector, name):
        print(f"\n[joint-eval] === standalone M3 falsifier: {name} ===", flush=True)
        f1_normal, f1_swapped, f1_zeroed = [], [], []
        gens_normal_all, gts_all = [], []
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
            for cond_name, pp, mask in [("normal", pre_pool, kpm), ("swapped", pre_pool_swapped, kpm_swapped),
                                         ("zeroed", pre_pool_zeroed, kpm)]:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    m3_lat = m3_connector(pp, mask)
                m3_attn = torch.ones(B, m3_lat.shape[1], dtype=torch.long, device=device)
                emb = torch.cat([m3_lat, prefix], dim=1)
                attn = torch.cat([m3_attn, prefix_attn], dim=1)
                gen = generate(emb, attn)
                for i in range(B):
                    f1 = word_overlap_f1(gen[i], gts[i])
                    {"normal": f1_normal, "swapped": f1_swapped, "zeroed": f1_zeroed}[cond_name].append(f1)
                if cond_name == "normal":
                    gens_normal_all.extend(gen); gts_all.extend(gts)
        z_gen = encode_semantic(gens_normal_all, sem_tok, sem_model, device)
        z_gt = encode_semantic(gts_all, sem_tok, sem_model, device)
        cos_normal = (z_gen * z_gt).sum(-1).mean().item()

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0
        result = {"n": len(f1_normal), "f1_normal": mean(f1_normal), "f1_swapped": mean(f1_swapped),
                  "f1_zeroed": mean(f1_zeroed), "cos_normal": cos_normal}
        print(f"[joint-eval] {name}: F1 normal={result['f1_normal']:.3f} swapped={result['f1_swapped']:.3f} "
              f"zeroed={result['f1_zeroed']:.3f}  cos_normal={result['cos_normal']:.3f}", flush=True)
        return result

    m3_baseline_result = eval_m3_standalone(m3_baseline, "BASELINE (frozen-LLM M3, original checkpoint)")
    m3_joint_result = eval_m3_standalone(m3_joint, "JOINT-TRAINED M3 (standalone, no speech)")

    # ============ (2) standalone M4b, baseline vs joint ============
    def eval_m4b_standalone(m4b_projector, name):
        print(f"\n[joint-eval] === standalone M4b swap-control: {name} ===", flush=True)
        cos_normal_l, cos_swapped_target_l, cos_swapped_donor_l = [], [], []
        bs = args.batch_size
        for start in range(0, len(easycom_ds), bs):
            items = [easycom_ds[i] for i in range(start, min(start + bs, len(easycom_ds)))]
            if len(items) < 2:
                continue
            batch = m4b_collate_fn(items)
            gts = batch["texts"]
            hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                stoks, smask = m4b_projector(hidden.float(), valid_frames)
            B = stoks.shape[0]
            sattn = (~smask).long()
            gen_normal = generate(stoks, sattn)
            stoks_sw = torch.roll(stoks, shifts=1, dims=0)
            smask_sw = torch.roll(smask, shifts=1, dims=0)
            sattn_sw = (~smask_sw).long()
            gen_swapped = generate(stoks_sw, sattn_sw)
            donor_texts = [gts[(i - 1) % B] for i in range(B)]

            z_gt = encode_semantic(gts, sem_tok, sem_model, device)
            z_donor = encode_semantic(donor_texts, sem_tok, sem_model, device)
            z_normal = encode_semantic(gen_normal, sem_tok, sem_model, device)
            z_swapped = encode_semantic(gen_swapped, sem_tok, sem_model, device)
            cos_normal_l.extend((z_normal * z_gt).sum(-1).tolist())
            cos_swapped_target_l.extend((z_swapped * z_gt).sum(-1).tolist())
            cos_swapped_donor_l.extend((z_swapped * z_donor).sum(-1).tolist())

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0
        result = {"n": len(cos_normal_l), "cos_normal": mean(cos_normal_l),
                  "cos_swapped_vs_target": mean(cos_swapped_target_l), "cos_swapped_vs_donor": mean(cos_swapped_donor_l)}
        print(f"[joint-eval] {name}: cos_normal={result['cos_normal']:.3f}  "
              f"swapped_vs_target={result['cos_swapped_vs_target']:.3f}  "
              f"swapped_vs_donor={result['cos_swapped_vs_donor']:.3f}", flush=True)
        return result

    m4b_baseline_result = eval_m4b_standalone(m4b_baseline, "BASELINE (m4b/best.pt, original)")
    m4b_joint_result = eval_m4b_standalone(m4b_joint, "JOINT-TRAINED M4b (standalone, no M3)")

    # ============ (4) gate (c)/(d) re-run with JOINT connectors ============
    print("\n[joint-eval] === (c) re-run: M3(joint) swap-control WITH M4b(joint) speech present ===", flush=True)
    fixed_item = easycom_ds[0]
    fixed_batch = m4b_collate_fn([fixed_item])
    fixed_hidden, fixed_valid_frames = whisper(fixed_batch["waveforms"], fixed_batch["durations_sec"], device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        FIXED_SPEECH_TOKENS, FIXED_SPEECH_MASK = m4b_joint(fixed_hidden.float(), fixed_valid_frames)
    print(f"[joint-eval] fixed speech segment: {fixed_item['text'][:80]!r}", flush=True)

    def expand_fixed(t, B):
        return t.expand(B, *t.shape[1:])

    f1_normal_c, f1_swapped_c, f1_zeroed_c = [], [], []
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
        speech_tokens = expand_fixed(FIXED_SPEECH_TOKENS, B)
        speech_mask = expand_fixed(FIXED_SPEECH_MASK, B)
        speech_attn = (~speech_mask).long()
        for cond_name, pp, mask in [("normal", pre_pool, kpm), ("swapped", pre_pool_swapped, kpm_swapped),
                                     ("zeroed", pre_pool_zeroed, kpm)]:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                m3_lat = m3_joint(pp, mask)
            m3_attn = torch.ones(B, m3_lat.shape[1], dtype=torch.long, device=device)
            emb = torch.cat([m3_lat, speech_tokens, prefix], dim=1)
            attn = torch.cat([m3_attn, speech_attn, prefix_attn], dim=1)
            gen = generate(emb, attn)
            for i in range(B):
                f1 = word_overlap_f1(gen[i], gts[i])
                {"normal": f1_normal_c, "swapped": f1_swapped_c, "zeroed": f1_zeroed_c}[cond_name].append(f1)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    result_c = {"n": len(f1_normal_c), "f1_normal": mean(f1_normal_c), "f1_swapped": mean(f1_swapped_c), "f1_zeroed": mean(f1_zeroed_c)}
    print(f"[joint-eval] (c) JOINT, WITH speech: normal={result_c['f1_normal']:.3f}  "
          f"swapped={result_c['f1_swapped']:.3f}  zeroed={result_c['f1_zeroed']:.3f}", flush=True)

    print("\n[joint-eval] === (d) re-run: M4b(joint) swap-control WITH M3(joint) latents present ===", flush=True)
    fixed_vgg_cid, _, fixed_vgg_gt = vgg_eval_pairs[0]
    fixed_ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=[fixed_vgg_cid], max_tdm_bins=512, audio_mode="mean")
    fixed_vgg_batch = m3_collate_fn([{
        "feats": fixed_ds[0]["feats"], "tbins": fixed_ds[0]["tbins"], "clip_id": fixed_ds[0]["clip_id"],
        "prefix_ids": torch.zeros(0, dtype=torch.long), "caption_ids": torch.zeros(1, dtype=torch.long),
        "caption_text": "", "field": None}], tokenizer.pad_token_id)
    fv_feats = {k: v.to(device) for k, v in fixed_vgg_batch["feats"].items()}
    fv_tbins = {k: v.to(device) for k, v in fixed_vgg_batch["tbins"].items()}
    fv_pad = {k: v.to(device) for k, v in fixed_vgg_batch["padding_mask"].items()}
    _cap_ambient_len(fv_feats, fv_tbins, fv_pad)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        fv_feats_f = {k: v.float() for k, v in fv_feats.items()}
        fv_pre_pool = predictor.encode_pre_pool_tokens(fv_feats_f, fv_tbins)
        fv_kpm = torch.cat([fv_pad["vision"], fv_pad["ambient"]], dim=1)
        FIXED_M3_LATENTS = m3_joint(fv_pre_pool.to(torch.bfloat16), fv_kpm)
    print(f"[joint-eval] fixed M3 clip: {fixed_vgg_cid}  {fixed_vgg_gt[:80]!r}", flush=True)

    cos_normal_d, cos_swapped_target_d, cos_swapped_donor_d = [], [], []
    for start in range(0, len(easycom_ds), bs):
        items = [easycom_ds[i] for i in range(start, min(start + bs, len(easycom_ds)))]
        if len(items) < 2:
            continue
        batch = m4b_collate_fn(items)
        gts = batch["texts"]
        hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stoks, smask = m4b_joint(hidden.float(), valid_frames)
        B = stoks.shape[0]
        sattn = (~smask).long()
        m3_lat = expand_fixed(FIXED_M3_LATENTS, B)
        m3_attn = torch.ones(B, m3_lat.shape[1], dtype=torch.long, device=device)

        emb_normal = torch.cat([m3_lat, stoks], dim=1)
        attn_normal = torch.cat([m3_attn, sattn], dim=1)
        gen_normal = generate(emb_normal, attn_normal)

        stoks_sw = torch.roll(stoks, shifts=1, dims=0)
        smask_sw = torch.roll(smask, shifts=1, dims=0)
        sattn_sw = (~smask_sw).long()
        emb_sw = torch.cat([m3_lat, stoks_sw], dim=1)
        attn_sw = torch.cat([m3_attn, sattn_sw], dim=1)
        gen_swapped = generate(emb_sw, attn_sw)
        donor_texts = [gts[(i - 1) % B] for i in range(B)]

        z_gt = encode_semantic(gts, sem_tok, sem_model, device)
        z_donor = encode_semantic(donor_texts, sem_tok, sem_model, device)
        z_normal = encode_semantic(gen_normal, sem_tok, sem_model, device)
        z_swapped = encode_semantic(gen_swapped, sem_tok, sem_model, device)
        cos_normal_d.extend((z_normal * z_gt).sum(-1).tolist())
        cos_swapped_target_d.extend((z_swapped * z_gt).sum(-1).tolist())
        cos_swapped_donor_d.extend((z_swapped * z_donor).sum(-1).tolist())

    result_d = {"n": len(cos_normal_d), "cos_normal": mean(cos_normal_d),
                "cos_swapped_vs_target": mean(cos_swapped_target_d), "cos_swapped_vs_donor": mean(cos_swapped_donor_d)}
    print(f"[joint-eval] (d) JOINT, WITH M3: normal={result_d['cos_normal']:.3f}  "
          f"swapped_vs_target={result_d['cos_swapped_vs_target']:.3f}  "
          f"swapped_vs_donor={result_d['cos_swapped_vs_donor']:.3f}", flush=True)

    all_results = {
        "m3_standalone_baseline": m3_baseline_result, "m3_standalone_joint": m3_joint_result,
        "m4b_standalone_baseline": m4b_baseline_result, "m4b_standalone_joint": m4b_joint_result,
        "c_m3_joint_with_speech": result_c, "d_m4b_joint_with_m3": result_d,
    }
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[joint-eval] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
