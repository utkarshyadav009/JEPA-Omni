"""scripts/eval_stt_projector.py -- real inference sanity check for the
Ultravox-style STT projector (models/m4d_stt_projector.py): feeds held-out
LibriSpeech test-clean audio (never seen in training) through the trained
Whisper encoder + projector, then generates FREELY from the LLM (no
teacher-forcing) -- a genuinely out-of-distribution inference mode vs. how
the projector was trained, which is the honest test of whether it learned
real semantic content rather than just minimizing training loss.

Real bug found and fixed here (2026-08-05 night, first noticed then never
addressed): greedy decoding (do_sample=False) with no repeat_penalty
produced a real repetition-loop artifact on one held-out case ("the french
the french the french..."). Fixed by adding repeat_penalty to the
generate() call, standard mitigation for this well-known greedy-decoding
failure mode.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor

from models.m4d_stt_projector import AudioEncoderProjector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-path", default="/home/utkarsh/hf_models/LFM2-700M")
    ap.add_argument("--whisper-repo", default="openai/whisper-base")
    ap.add_argument("--projector-ckpt", default="checkpoints/bmo_stt_projector_base/projector.pt")
    ap.add_argument("--test-clean-glob",
                     default="/home/utkarsh/data/librispeech_clean100/all/test.clean/*.parquet")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--repeat-penalty", type=float, default=1.3)
    args = ap.parse_args()

    device = "cuda"
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.llm_path)

    fe = WhisperFeatureExtractor.from_pretrained(args.whisper_repo)
    audio_model = AudioEncoderProjector(args.whisper_repo, llm.config.hidden_size, device=device)
    audio_model.projector = audio_model.projector.to(torch.bfloat16)
    audio_model.projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
    audio_model.projector.eval()

    ds = load_dataset("parquet", data_files=args.test_clean_glob, split="train")

    n_exact = 0
    for i in range(args.n_samples):
        sample = ds[i]
        audio = sample["audio"]["array"]
        true_text = sample["text"].lower()

        feats = fe([audio], sampling_rate=16000, return_tensors="pt").input_features.to(device).to(torch.bfloat16)
        with torch.no_grad():
            audio_embeds = audio_model(feats)
            out = llm.generate(
                inputs_embeds=audio_embeds, max_new_tokens=args.max_new_tokens,
                do_sample=False, repetition_penalty=args.repeat_penalty,
            )
        generated = tok.decode(out[0], skip_special_tokens=True)

        exact = generated.strip().lower().startswith(true_text.strip().lower()[:20])
        n_exact += int(exact)
        print(f"[{i}] TRUE : {true_text!r}", flush=True)
        print(f"[{i}] MODEL: {generated!r}{'  (near-exact prefix match)' if exact else ''}", flush=True)
        print(flush=True)

    print(f"DONE: {n_exact}/{args.n_samples} near-exact prefix matches", flush=True)


if __name__ == "__main__":
    main()
