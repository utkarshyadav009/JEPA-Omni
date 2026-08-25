"""scripts/train_stt_projector.py -- trains the Ultravox-style audio
projector (models/m4d_stt_projector.py) on real paired speech-transcript
data (LibriSpeech train-clean-100). Both the Whisper encoder and the LLM
(LFM2-700M) are frozen; only the projector's weights are updated.

Training objective, the real Ultravox-style approach (not text-generation
from audio "as if reading captions" -- direct embedding substitution):
for each (audio, transcript) pair, run the LLM twice on the same target
continuation --
  1. TEXT path (teacher, no grad): embed the transcript normally, get the
     LLM's per-token output distribution.
  2. AUDIO path (student, projector gets gradient): replace the transcript's
     token embeddings with the projected audio embeddings, feed those in as
     a prefix, then have the LLM predict the SAME transcript continuation
     as next tokens.
The projector is trained so path 2's next-token predictions match path 1's
target tokens (real cross-entropy against the known transcript) -- this is
what makes the projected audio embeddings "look like" the corresponding
text to the frozen LLM, without ever needing to touch the LLM's own weights.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor

from models.m4d_stt_projector import AudioEncoderProjector


def collate(batch, feature_extractor, tokenizer, device):
    audios = [b["audio"]["array"] for b in batch]
    texts = [b["text"].lower().strip() + "." for b in batch]
    feats = feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    return feats, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="/home/utkarsh/data/librispeech_clean100")
    ap.add_argument("--llm-path", default="/home/utkarsh/hf_models/LFM2-700M")
    ap.add_argument("--whisper-repo", default="openai/whisper-tiny")
    ap.add_argument("--out", default="checkpoints/bmo_stt_projector/projector.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16).to(device).eval()
    for p in llm.parameters():
        p.requires_grad = False

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_repo)
    audio_model = AudioEncoderProjector(args.whisper_repo, llm.config.hidden_size, device=device)
    audio_model.projector = audio_model.projector.to(torch.bfloat16)

    ds = load_dataset("parquet", data_files=f"{args.dataset_dir}/all/train.clean.100/*.parquet", split="train")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, feature_extractor, tokenizer, device),
    )

    opt = torch.optim.AdamW(audio_model.projector.parameters(), lr=args.lr)
    embed_layer = llm.get_input_embeddings()

    step = 0
    for feats, tok in loader:
        if step >= args.max_steps:
            break
        feats = feats.to(torch.bfloat16)
        audio_embeds = audio_model(feats)  # (B, T_audio, hidden)

        text_embeds = embed_layer(tok["input_ids"])  # (B, T_text, hidden)
        inputs_embeds = torch.cat([audio_embeds, text_embeds[:, :-1, :]], dim=1)
        targets = tok["input_ids"][:, 1:]

        out = llm(inputs_embeds=inputs_embeds)
        logits = out.logits[:, audio_embeds.shape[1]:, :]
        logits = logits[:, : targets.shape[1], :]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            print(f"[step {step}] loss={loss.item():.4f}", flush=True)
        step += 1

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(audio_model.projector.state_dict(), args.out)
    print(f"DONE: projector saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
