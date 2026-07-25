"""scripts/m4_backchannel_production_test.py — PRODUCTION-side backchannel
test: verify the robot can emit a SHORT backchannel instead of a full
utterance, using the existing frozen-LLM + M4b projector pipeline
UNMODIFIED (no new capability trained into the LLM -- consistent with the
whole M4c policy of keeping the LLM frozen after the LoRA silence-collapse
lesson).

Policy: the 3-class decision head's output gates max_new_tokens, not the
LLM's behavior. SPEAK -> normal generation budget (40 tokens, matches
m4c_gate_eval). BACKCHANNEL -> a tight budget (8 tokens) so any full-turn
continuation gets truncated into something backchannel-shaped. This is
deliberately the cheapest and least risky "production" mechanism available:
it reuses the exact soft-prompt + generation path already falsifier-verified
in M4c, and does not require touching the frozen LLM or connectors at all.

Usage:
    python scripts/m4_backchannel_production_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX, IDX_TO_LABEL
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset

BACKCHANNEL_MAX_NEW_TOKENS = 8
SPEAK_MAX_NEW_TOKENS = 40


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--decision-head-ckpt", default="checkpoints/m4_decision_head_3class/best.pt")
    p.add_argument("--n-per-class", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="checkpoints/m4_decision_head_3class/production_test_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print("[bc-prod] loading frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[bc-prod] loading M4b projector (post-joint, frozen) + whisper...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[bc-prod] loading 3-class decision head...", flush=True)
    dh_ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = ThreeClassHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    _, test_ticks = build_ticks()
    speak_ticks = [t for t in test_ticks if t.label3 == "speak"]
    bc_ticks = [t for t in test_ticks if t.label3 == "backchannel"]
    rng.shuffle(speak_ticks)
    rng.shuffle(bc_ticks)
    sample_ticks = speak_ticks[:args.n_per_class] + bc_ticks[:args.n_per_class]
    ds = EasyComTurnTakingDataset(sample_ticks)

    zero_ws = torch.zeros(1, dh_cfg.world_state_dim, device=device)

    def generate(stoks, smask, max_new_tokens):
        sattn = (~smask).long()
        gen_ids = llm.generate(inputs_embeds=stoks, attention_mask=sattn, max_new_tokens=max_new_tokens,
                                do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    records = []
    n_correctly_routed = 0
    for i in range(len(ds)):
        item = ds[i]
        gt_label = item["label3"]
        with torch.no_grad():
            hidden, valid_frames = whisper([item["waveform"]], [item["duration_sec"]], device)
            pooled = hidden[0, :int(valid_frames[0].item())].float().mean(dim=0, keepdim=True)
            logits = decision_head(zero_ws, pooled)
            pred_idx = logits.argmax(dim=-1).item()
            pred_label = IDX_TO_LABEL[pred_idx]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                stoks, smask = m4b_projector(hidden.float(), valid_frames)
            budget = BACKCHANNEL_MAX_NEW_TOKENS if pred_label == "backchannel" else SPEAK_MAX_NEW_TOKENS
            gen_text = generate(stoks, smask, budget)

        n_gen_tokens = len(tokenizer(gen_text, add_special_tokens=False)["input_ids"])
        correctly_routed = (pred_label == "backchannel") == (gt_label == "backchannel")
        n_correctly_routed += int(correctly_routed)
        records.append({"gt_label3": gt_label, "gt_text": item["text"], "pred_label3": pred_label,
                         "budget_used": budget, "gen_text": gen_text, "n_gen_tokens": n_gen_tokens,
                         "correctly_routed": correctly_routed})
        print(f"[bc-prod] gt={gt_label:12s} pred={pred_label:12s} budget={budget:2d}  "
              f"gen_tokens={n_gen_tokens:2d}  gen=\"{gen_text}\"  (real_transcript=\"{item['text']}\")", flush=True)

    bc_records = [r for r in records if r["gt_label3"] == "backchannel"]
    speak_records = [r for r in records if r["gt_label3"] == "speak"]
    mean_len_bc = sum(r["n_gen_tokens"] for r in bc_records) / max(1, len(bc_records))
    mean_len_speak = sum(r["n_gen_tokens"] for r in speak_records) / max(1, len(speak_records))

    results = {
        "n": len(records),
        "routing_accuracy": n_correctly_routed / len(records),
        "mean_gen_tokens_on_backchannel_ground_truth": mean_len_bc,
        "mean_gen_tokens_on_speak_ground_truth": mean_len_speak,
        "records": records,
    }
    print(f"\n[bc-prod] routing accuracy (3-class head correctly identifies backchannel vs not): "
          f"{results['routing_accuracy']:.3f}", flush=True)
    print(f"[bc-prod] mean generated length on backchannel GT: {mean_len_bc:.1f} tokens", flush=True)
    print(f"[bc-prod] mean generated length on real-speak GT:  {mean_len_speak:.1f} tokens", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bc-prod] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
