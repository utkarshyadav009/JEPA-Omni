"""scripts/m4a_lora_falsifier_regression.py — mandatory falsifier regression
for Phase 1a (LoRA-only run), per the standing rule in CLAUDE.md.

Re-runs the SAME standalone M3 and M4b falsifiers as
scripts/m4_joint_eval.py, but with the LLM now LoRA-wrapped (+ trained
control-token delta) instead of the plain frozen LLM -- connectors
(M3/M4b) are held at their post-joint-training state, UNCHANGED, so any
movement here is attributable to LoRA alone, isolating the one variable
this stage introduced.

Usage:
    python scripts/m4a_lora_falsifier_regression.py --n-clips 100
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
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from models.m4_control import wrap_lora, attach_control_token_embedding
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
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--lora-ckpt", default="checkpoints/m4a_lora/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4a_lora/falsifier_regression_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[1a-falsifier] loading tokenizer + base LLM, wrapping LoRA + control delta...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    base_llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    for prm in base_llm.parameters():
        prm.requires_grad_(False)

    lora_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
    llm = wrap_lora(base_llm, r=lora_ckpt["lora_r"], alpha=lora_ckpt["lora_alpha"], dropout=lora_ckpt["lora_dropout"])
    control_embed = attach_control_token_embedding(llm, tokenizer)
    missing, unexpected = llm.load_state_dict(lora_ckpt["lora_state_dict"], strict=False)
    control_embed.delta.data.copy_(lora_ckpt["control_delta"].to(device))
    llm.eval()
    print(f"[1a-falsifier] loaded LoRA from step {lora_ckpt.get('step')}  test_loss={lora_ckpt.get('test_loss')}", flush=True)

    print("[1a-falsifier] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for prm in predictor.parameters():
        prm.requires_grad_(False)

    print(f"[1a-falsifier] loading FROZEN M3/M4b from {args.joint_ckpt} (post-joint, unchanged)...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(joint_ckpt["m3_connector"])
    m3_connector.eval()

    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()

    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[1a-falsifier] loading MiniLM for semantic-cosine...", flush=True)
    sem_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    _, vgg_test_pairs = build_splits(FIELD)
    rng.shuffle(vgg_test_pairs)
    vgg_eval_pairs = vgg_test_pairs[:args.n_clips]

    _, easycom_test_segs = build_segments()
    rng.shuffle(easycom_test_segs)
    easycom_ds = EasyComSpeechDataset(easycom_test_segs[:args.n_clips])

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

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # ---- standalone M3 ----
    print("\n[1a-falsifier] === standalone M3 falsifier (LoRA'd LLM) ===", flush=True)
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
        print(f"[1a-falsifier] M3 processed {min(start+bs, len(vgg_eval_pairs))}/{len(vgg_eval_pairs)}", flush=True)
    z_gen = encode_semantic(gens_normal_all, sem_tok, sem_model, device)
    z_gt = encode_semantic(gts_all, sem_tok, sem_model, device)
    cos_normal = (z_gen * z_gt).sum(-1).mean().item()
    m3_result = {"n": len(f1_normal), "f1_normal": mean(f1_normal), "f1_swapped": mean(f1_swapped),
                 "f1_zeroed": mean(f1_zeroed), "cos_normal": cos_normal}
    print(f"[1a-falsifier] M3 (LoRA'd LLM): F1 normal={m3_result['f1_normal']:.3f} "
          f"swapped={m3_result['f1_swapped']:.3f} zeroed={m3_result['f1_zeroed']:.3f}  "
          f"cos_normal={m3_result['cos_normal']:.3f}", flush=True)

    # ---- standalone M4b ----
    print("\n[1a-falsifier] === standalone M4b swap-control (LoRA'd LLM) ===", flush=True)
    cos_normal_l, cos_swapped_target_l, cos_swapped_donor_l = [], [], []
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
        print(f"[1a-falsifier] M4b processed {min(start+bs, len(easycom_ds))}/{len(easycom_ds)}", flush=True)

    m4b_result = {"n": len(cos_normal_l), "cos_normal": mean(cos_normal_l),
                  "cos_swapped_vs_target": mean(cos_swapped_target_l), "cos_swapped_vs_donor": mean(cos_swapped_donor_l)}
    print(f"[1a-falsifier] M4b (LoRA'd LLM): cos_normal={m4b_result['cos_normal']:.3f}  "
          f"swapped_vs_target={m4b_result['cos_swapped_vs_target']:.3f}  "
          f"swapped_vs_donor={m4b_result['cos_swapped_vs_donor']:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"m3_standalone_with_lora": m3_result, "m4b_standalone_with_lora": m4b_result}, f, indent=2)
    print(f"\n[1a-falsifier] DONE. wrote {args.out}", flush=True)
    print("[1a-falsifier] compare vs post-joint (pre-LoRA): M3 0.430/0.270/0.249 cos=0.714; "
          "M4b cos_normal=0.419 swap_target=0.133 swap_donor=0.419", flush=True)


if __name__ == "__main__":
    main()
