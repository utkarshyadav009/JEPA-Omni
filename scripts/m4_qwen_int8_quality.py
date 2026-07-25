"""scripts/m4_qwen_int8_quality.py — M5 int8-Qwen quality check: does
torchao Int8WeightOnlyConfig(version=2) weight-only quantization of
Qwen2.5-1.5B-Instruct cost anything on real EasyCom captioning output,
measured via the same semantic-cosine methodology already established in
this project (scripts/m4_joint_eval.py's encode_semantic /
sentence-transformers/all-MiniLM-L6-v2).

Run on mercury (fast GPU, quality is hardware-independent -- the memory
RECLAIMED number is measured separately on the Jetson itself).

Usage:
    python scripts/m4_qwen_int8_quality.py --n 100
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

from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.m4_speech_dataset import build_segments, m4b_collate_fn, EasyComSpeechDataset


def encode_semantic(texts, tok, model, device, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(pooled, dim=-1))
    return torch.cat(embs, 0)


def bootstrap_se(x, n_boot=200, seed=0):
    rng = random.Random(seed)
    n = len(x)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(x[i] for i in idx) / n)
    m = sum(means) / n_boot
    var = sum((v - m) ** 2 for v in means) / (n_boot - 1)
    return var ** 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m5_jetson/qwen_int8_quality.json")
    args = p.parse_args()

    device = torch.device("cuda")
    rng = random.Random(args.seed)

    print("[int8-quality] loading whisper + M4b projector (frozen, unmodified)...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()

    print("[int8-quality] loading EasyCom test segments...", flush=True)
    _, test_segs = build_segments()
    rng.shuffle(test_segs)
    test_segs = test_segs[:args.n]
    ds = EasyComSpeechDataset(test_segs)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    def build_stoks():
        items = [ds[i] for i in range(len(ds))]
        batch = m4b_collate_fn(items)
        gts = batch["texts"]
        hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stoks, smask = m4b_projector(hidden.float(), valid_frames)
        return stoks, smask, gts

    stoks, smask, gts = build_stoks()
    sattn = (~smask).long()

    def generate_with(llm, tag):
        print(f"[int8-quality] generating with {tag}...", flush=True)
        gen_ids = llm.generate(inputs_embeds=stoks, attention_mask=sattn, max_new_tokens=60,
                                do_sample=False, repetition_penalty=1.15,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]

    print("[int8-quality] loading Qwen2.5-1.5B-Instruct bf16 baseline...", flush=True)
    llm_bf16 = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm_bf16.eval()
    for prm in llm_bf16.parameters():
        prm.requires_grad_(False)
    gens_bf16 = generate_with(llm_bf16, "bf16")
    del llm_bf16
    torch.cuda.empty_cache()

    print("[int8-quality] loading Qwen2.5-1.5B-Instruct, quantizing int8 (torchao v2)...", flush=True)
    llm_int8 = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)
    llm_int8.eval()
    for prm in llm_int8.parameters():
        prm.requires_grad_(False)
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    quantize_(llm_int8, Int8WeightOnlyConfig(version=2))
    llm_int8 = llm_int8.to(device)
    gens_int8 = generate_with(llm_int8, "int8(v2)")
    del llm_int8
    torch.cuda.empty_cache()

    print("[int8-quality] scoring semantic cosine vs ground truth...", flush=True)
    sem_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sem_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    sem_model.eval()

    z_gt = encode_semantic(gts, sem_tok, sem_model, device)
    z_bf16 = encode_semantic(gens_bf16, sem_tok, sem_model, device)
    z_int8 = encode_semantic(gens_int8, sem_tok, sem_model, device)

    cos_bf16_list = (z_bf16 * z_gt).sum(-1).tolist()
    cos_int8_list = (z_int8 * z_gt).sum(-1).tolist()
    # also compare int8's output directly against bf16's own output --
    # the tightest test of "did quantization change what it says"
    cos_bf16_vs_int8_list = (z_bf16 * z_int8).sum(-1).tolist()

    cos_bf16 = sum(cos_bf16_list) / len(cos_bf16_list)
    cos_int8 = sum(cos_int8_list) / len(cos_int8_list)
    cos_bf16_vs_int8 = sum(cos_bf16_vs_int8_list) / len(cos_bf16_vs_int8_list)
    se_bf16 = bootstrap_se(cos_bf16_list)
    exact_match_rate = sum(1 for a, b in zip(gens_bf16, gens_int8) if a == b) / len(gens_bf16)

    results = {
        "n": len(gts), "cos_vs_gt_bf16": cos_bf16, "cos_vs_gt_int8": cos_int8,
        "delta": cos_int8 - cos_bf16, "bootstrap_se_bf16": se_bf16,
        "within_noise": abs(cos_int8 - cos_bf16) < 2 * se_bf16,
        "cos_bf16_output_vs_int8_output": cos_bf16_vs_int8,
        "exact_text_match_rate": exact_match_rate,
        "examples": [{"gt": gt, "bf16": b, "int8": i}
                     for gt, b, i in list(zip(gts, gens_bf16, gens_int8))[:8]],
    }
    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, indent=2))
    print(f"\ncos vs GT:  bf16={cos_bf16:.4f}  int8={cos_int8:.4f}  delta={cos_int8-cos_bf16:+.4f}  "
          f"(bootstrap SE on bf16={se_bf16:.4f}, so 2*SE={2*se_bf16:.4f})")
    print(f"within noise (|delta| < 2*SE): {results['within_noise']}")
    print(f"exact text match rate bf16 vs int8: {exact_match_rate:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
