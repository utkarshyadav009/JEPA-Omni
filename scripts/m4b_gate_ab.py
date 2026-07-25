"""scripts/m4b_gate_ab.py — M4b gate checks (a) and (b).

(a) WER + semantic-cosine of LLM output vs ground-truth transcription, on
    the held-out SESSIONS (10, 11, 12 -- never seen during training).
(b) Speech swap-control, same falsifier discipline as M3: swap in a
    DIFFERENT utterance's speech tokens (rolled across the batch) and check
    whether the output follows the swapped utterance's content, not the
    target's -- plus prior-only baselines (most-common training
    transcription, random training transcription) to calibrate how much of
    the "normal" score is genuine grounding vs corpus regularity.

Usage:
    python scripts/m4b_gate_ab.py --n-clips 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoTokenizer as HFTok

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn
from train_m4b import word_overlap_f1, word_error_rate


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
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--projector-ckpt", default="checkpoints/m4b/last.pt")
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-side-by-side", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="checkpoints/m4b/gate_ab_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[gate-ab] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[gate-ab] loading frozen whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print(f"[gate-ab] loading trained projector from {args.projector_ckpt}...", flush=True)
    ckpt = torch.load(args.projector_ckpt, map_location=device, weights_only=False)
    proj_cfg = UltravoxProjectorConfig(**ckpt["projector_cfg"])
    projector = UltravoxProjector(proj_cfg).to(device)
    projector.load_state_dict(ckpt["projector"])
    projector.eval()
    print(f"[gate-ab] projector trained to step {ckpt.get('step')}", flush=True)

    train_segs, test_segs = build_segments()
    rng.shuffle(test_segs)
    eval_segs = test_segs[:args.n_clips]
    eval_ds = EasyComSpeechDataset(eval_segs)
    print(f"[gate-ab] evaluating on {len(eval_ds)} held-out segments "
          f"(sessions {sorted(set(s.session for s in eval_segs))})", flush=True)

    print("[gate-ab] loading MiniLM for semantic-cosine metric...", flush=True)
    sem_tok = HFTok.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    @torch.no_grad()
    def generate_batch(hidden, valid_frames):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt, key_padding_mask = projector(hidden.float(), valid_frames)
        attn = (~key_padding_mask).long()
        gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn, max_new_tokens=60,
                                do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    bs = args.batch_size
    normal_gens, swapped_gens, gts, swap_from_texts = [], [], [], []
    side_by_side = []
    for start in range(0, len(eval_ds), bs):
        items = [eval_ds[i] for i in range(start, min(start + bs, len(eval_ds)))]
        if len(items) < 2:
            continue
        batch = m4b_collate_fn(items)
        hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)

        gen_normal = generate_batch(hidden, valid_frames)
        hidden_swapped = torch.roll(hidden, shifts=1, dims=0)
        valid_frames_swapped = torch.roll(valid_frames, shifts=1, dims=0)
        gen_swapped = generate_batch(hidden_swapped, valid_frames_swapped)

        for i in range(len(items)):
            normal_gens.append(gen_normal[i]); swapped_gens.append(gen_swapped[i])
            gts.append(batch["texts"][i])
            donor_idx = (i - 1) % len(items)
            swap_from_texts.append(batch["texts"][donor_idx])
            if len(side_by_side) < args.n_side_by_side:
                side_by_side.append({
                    "ground_truth": batch["texts"][i], "normal": gen_normal[i],
                    "swapped_from_text": batch["texts"][donor_idx], "swapped": gen_swapped[i],
                })
        print(f"[gate-ab] processed {min(start+bs, len(eval_ds))}/{len(eval_ds)}", flush=True)

    # (a) WER + semantic-cosine, normal condition
    wer_normal = [word_error_rate(p, g) for p, g in zip(normal_gens, gts)]
    f1_normal = [word_overlap_f1(p, g) for p, g in zip(normal_gens, gts)]
    z_gt = encode_semantic(gts, sem_tok, sem_model, device)
    z_normal = encode_semantic(normal_gens, sem_tok, sem_model, device)
    z_swapped = encode_semantic(swapped_gens, sem_tok, sem_model, device)
    z_swap_from = encode_semantic(swap_from_texts, sem_tok, sem_model, device)
    cos_normal = (z_normal * z_gt).sum(-1)
    cos_swapped_vs_target = (z_swapped * z_gt).sum(-1)
    cos_swapped_vs_donor = (z_swapped * z_swap_from).sum(-1)

    # (b) swap-control F1/WER against target (should be bad) vs donor (should be good)
    f1_swapped_vs_target = [word_overlap_f1(p, g) for p, g in zip(swapped_gens, gts)]
    f1_swapped_vs_donor = [word_overlap_f1(p, g) for p, g in zip(swapped_gens, swap_from_texts)]

    # prior baselines
    train_texts = [s.transcription for s in train_segs]
    mode_text, mode_n = Counter(train_texts).most_common(1)[0]
    rng2 = random.Random(args.seed + 1)
    f1_mode = [word_overlap_f1(mode_text, g) for g in gts]
    f1_random = [word_overlap_f1(rng2.choice(train_texts), g) for g in gts]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    results = {
        "n": len(gts),
        "a_wer_normal": mean(wer_normal),
        "a_f1_normal": mean(f1_normal),
        "a_cos_normal": mean(cos_normal.tolist()),
        "b_cos_swapped_vs_target": mean(cos_swapped_vs_target.tolist()),
        "b_cos_swapped_vs_donor": mean(cos_swapped_vs_donor.tolist()),
        "b_f1_swapped_vs_target": mean(f1_swapped_vs_target),
        "b_f1_swapped_vs_donor": mean(f1_swapped_vs_donor),
        "prior_f1_mode": mean(f1_mode), "mode_text": mode_text, "mode_n_in_train": mode_n,
        "prior_f1_random_train": mean(f1_random),
        "side_by_side": side_by_side,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[gate-ab] === (a) held-out transcription quality (n={results['n']}) ===", flush=True)
    print(f"  WER={results['a_wer_normal']:.3f}  F1={results['a_f1_normal']:.3f}  cos={results['a_cos_normal']:.3f}", flush=True)
    print(f"\n[gate-ab] === (b) speech swap-control ===", flush=True)
    print(f"  cos(swapped, TARGET gt)  = {results['b_cos_swapped_vs_target']:.3f}  (want LOW)", flush=True)
    print(f"  cos(swapped, DONOR text) = {results['b_cos_swapped_vs_donor']:.3f}  (want HIGH, near cos_normal={results['a_cos_normal']:.3f})", flush=True)
    print(f"  F1(swapped, TARGET gt)   = {results['b_f1_swapped_vs_target']:.3f}", flush=True)
    print(f"  F1(swapped, DONOR text)  = {results['b_f1_swapped_vs_donor']:.3f}", flush=True)
    print(f"  prior baselines: mode_F1={results['prior_f1_mode']:.3f}  random_train_F1={results['prior_f1_random_train']:.3f}", flush=True)
    print(f"[gate-ab] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
