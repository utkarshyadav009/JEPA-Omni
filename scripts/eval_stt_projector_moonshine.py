"""scripts/eval_stt_projector_moonshine.py -- honest inference check for the
MOONSHINE Ultravox projector: feeds held-out LibriSpeech test-clean audio
(never seen in training) through the frozen Moonshine encoder + trained
projector, then generates FREELY from LFM2.5-350M (no teacher-forcing) via
inputs_embeds. If the projected audio embeddings carry real speech content,
the LLM should transcribe/paraphrase the utterance.

This is the offline proof that the no-text speech->LLM path works. (The live
Jetson loop can't use this directly yet: the deployed fast tier is a llama.cpp
GGUF, which takes token IDs, not inputs_embeds -- feeding embeddings needs an
HF-runtime fast tier or llama.cpp's embd-input API. Flagged for review.)
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from models.m4d_stt_projector_moonshine import MoonshineEncoderProjector


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-path", default="/home/utkarsh/hf_models/LFM2.5-350M")
    ap.add_argument("--moonshine-repo", default="UsefulSensors/moonshine-base")
    ap.add_argument("--projector-ckpt", default="checkpoints/bmo_stt_projector_moonshine/projector.pt")
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--proj-arch", default="stack", choices=["stack", "conv", "perceiver"])
    ap.add_argument("--n-latents", type=int, default=64)
    ap.add_argument("--total-stride", type=int, default=8)
    ap.add_argument("--llm-adapter", default=None, help="path to an LLM LoRA adapter to load")
    ap.add_argument("--test-clean-glob",
                    default="/home/utkarsh/data/librispeech_clean100/all/test.clean/*.parquet")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)     # upper bound; per-sample cap ties to audio length
    ap.add_argument("--repeat-penalty", type=float, default=1.2)  # 1.15-1.2 range to curb runaway generation
    ap.add_argument("--bos", action="store_true", help="append BOS embed after audio (match training only if tokenizer adds BOS)")
    args = ap.parse_args()

    device = "cuda"
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.llm_path)
    llm_dim = llm.config.hidden_size
    if args.llm_adapter:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.llm_adapter).to(device).eval()

    proc = AutoProcessor.from_pretrained(args.moonshine_repo)
    audio_model = MoonshineEncoderProjector(
        args.moonshine_repo, llm_dim=llm_dim, stack=args.stack, arch=args.proj_arch,
        n_latents=args.n_latents, total_stride=args.total_stride, device=device)
    audio_model.projector = audio_model.projector.to(torch.bfloat16)
    audio_model.projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
    audio_model.projector.eval()

    ds = load_dataset("parquet", data_files=args.test_clean_glob, split="train")

    wers = []
    for i in range(args.n_samples):
        sample = ds[i]
        audio = sample["audio"]["array"]
        true_text = sample["text"].lower().strip()

        feats = proc([audio], sampling_rate=16000, return_tensors="pt", padding=True)
        iv = feats["input_values"].to(device, torch.bfloat16)
        am = feats.get("attention_mask")
        am = am.to(device) if am is not None else None
        # Hard token cap tied to audio length (~speaking rate) -- truncates any
        # runaway generation; the EOS the model now learns should stop it earlier.
        dur = len(audio) / 16000.0
        cap = min(args.max_new_tokens, int(dur * 7) + 6)
        with torch.no_grad():
            audio_embeds = audio_model(iv, attention_mask=am)
            if args.bos and tok.bos_token_id is not None:
                bos = torch.tensor([[tok.bos_token_id]], device=device)
                bos_emb = llm.get_input_embeddings()(bos).to(audio_embeds.dtype)
                audio_embeds = torch.cat([audio_embeds, bos_emb], dim=1)
            out = llm.generate(
                inputs_embeds=audio_embeds, max_new_tokens=cap,
                do_sample=False, repetition_penalty=args.repeat_penalty,
                no_repeat_ngram_size=3,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        generated = tok.decode(out[0], skip_special_tokens=True).strip().lower()
        w = wer(true_text, generated)
        wers.append(w)
        print(f"[{i}] WER={w:.2f}\n    TRUE : {true_text!r}\n    MODEL: {generated!r}\n", flush=True)

    print(f"DONE: mean WER over {len(wers)} held-out utterances = {sum(wers)/max(1,len(wers)):.3f}", flush=True)


if __name__ == "__main__":
    main()
