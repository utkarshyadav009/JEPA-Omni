"""scripts/train_stt_projector_moonshine.py -- trains the Moonshine-based
Ultravox projector (models/m4d_stt_projector_moonshine.py) on paired
speech->transcript data (LibriSpeech train-clean-100). Both the Moonshine
encoder and the LLM (LFM2.5-350M, the deployed fast tier) are FROZEN; only the
projector's weights update.

Objective = Ultravox's real recipe (not plain "read the caption" CE):
  TEXT teacher (no grad): embed the transcript, run the LLM, get its per-token
    next-token distribution over the continuation.
  AUDIO student (projector gets grad): replace the transcript's token embeddings
    with a prefix of projected Moonshine audio embeddings, then have the LLM
    predict the SAME continuation.
  loss = CE(student, transcript) + kl_weight * KL(teacher || student)
The CE anchors the projector to the literal transcript; the KL distills the
frozen LLM's full distribution (the Ultravox "your speech LLM can be a text
LLM" term), which transfers more than argmax tokens alone.

The projector targets LFM2.5-350M's INPUT-EMBEDDING space, which LoRA
fine-tunes do NOT change (LoRA adapts attention/MLP, not embed_tokens), so a
projector trained against base LFM2.5-350M transfers to the BMO fine-tune of
it -- i.e. this can train in parallel with the LLM retrain, no dependency.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from models.m4d_stt_projector_moonshine import MoonshineEncoderProjector


def cycle(loader, max_steps):
    """Yield batches, re-iterating the loader (new shuffle each pass) until
    max_steps -- so max_steps can exceed one epoch (~3568 batches on clean-100)."""
    s = 0
    while s < max_steps:
        for b in loader:
            yield b
            s += 1
            if s >= max_steps:
                return


def collate(batch, processor, tokenizer):
    audios = [b["audio"]["array"] for b in batch]
    texts = [b["text"].lower().strip() + "." for b in batch]
    feats = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    # Explicitly APPEND EOS to every target so the model is trained to STOP after
    # the transcript (without this the 350M rambles into high-WER insertions).
    enc = tokenizer(texts, truncation=True, max_length=63, add_special_tokens=True)["input_ids"]
    eos = tokenizer.eos_token_id
    seqs = [ids + [eos] for ids in enc]
    maxlen = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return feats, {"input_ids": input_ids, "attention_mask": attn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="/home/utkarsh/data/librispeech_clean100")
    ap.add_argument("--train-globs", default=None,
                    help="comma-separated parquet globs; default = dataset-dir clean-100")
    ap.add_argument("--llm-path", default="/home/utkarsh/hf_models/LFM2.5-350M")
    ap.add_argument("--moonshine-repo", default="UsefulSensors/moonshine-base")
    ap.add_argument("--out", default="checkpoints/bmo_stt_projector_moonshine/projector.pt")
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--proj-arch", default="stack", choices=["stack", "conv", "perceiver"])
    ap.add_argument("--n-latents", type=int, default=64)     # perceiver: fixed soft-token count
    ap.add_argument("--total-stride", type=int, default=8)   # conv: frame compression factor
    ap.add_argument("--llm-lora", action="store_true")       # unfreeze LLM via LoRA (Ultravox Stage-2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=None)   # separate (lower) LR for LoRA -> avoids collapse
    ap.add_argument("--init-projector", default=None)        # warm-start projector (staged training)
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16).to(device).eval()
    for p in llm.parameters():
        p.requires_grad = False
    llm_dim = llm.config.hidden_size
    embed_layer = llm.get_input_embeddings()

    if args.llm_lora:
        # LoRA on q/v/k/out projections: give the frozen text network capacity to
        # "give up token real estate" for acoustic inputs. Teacher forward disables
        # the adapter (base text distribution) so KL preserves text ability.
        from peft import LoraConfig, get_peft_model, TaskType
        llm = get_peft_model(llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
            lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]))
        llm.print_trainable_parameters()
        llm.train()
        embed_layer = llm.get_input_embeddings()

    processor = AutoProcessor.from_pretrained(args.moonshine_repo)
    audio_model = MoonshineEncoderProjector(
        args.moonshine_repo, llm_dim=llm_dim, stack=args.stack, arch=args.proj_arch,
        n_latents=args.n_latents, total_stride=args.total_stride, device=device)
    audio_model.projector = audio_model.projector.to(torch.bfloat16)
    if args.init_projector:
        audio_model.projector.load_state_dict(torch.load(args.init_projector, map_location=device))
        print(f"[init] warm-started projector from {args.init_projector}", flush=True)
    print(f"[moonshine-proj] enc_dim={audio_model.enc_dim} llm_dim={llm_dim} "
          f"arch={args.proj_arch} n_latents={args.n_latents} stride={args.total_stride} "
          f"llm_lora={args.llm_lora}", flush=True)

    globs = (args.train_globs.split(",") if args.train_globs
             else [f"{args.dataset_dir}/all/train.clean.100/*.parquet"])
    ds = load_dataset("parquet", data_files=globs, split="train")
    print(f"[data] {len(ds)} utterances from {len(globs)} glob(s)", flush=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, processor, tokenizer),
    )

    proj_params = list(audio_model.projector.parameters())
    if args.llm_lora:
        lora_params = [p for p in llm.parameters() if p.requires_grad]
        lora_lr = args.lora_lr if args.lora_lr is not None else args.lr
        opt = torch.optim.AdamW([
            {"params": proj_params, "lr": args.lr},
            {"params": lora_params, "lr": lora_lr},
        ])
        print(f"[opt] projector lr={args.lr}  lora lr={lora_lr}", flush=True)
    else:
        opt = torch.optim.AdamW(proj_params, lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    step, t0 = 0, time.time()
    for feats, tok in cycle(loader, args.max_steps):
        if step >= args.max_steps:
            break
        input_values = feats["input_values"].to(device, torch.bfloat16)
        audio_attn = feats.get("attention_mask")
        audio_attn = audio_attn.to(device) if audio_attn is not None else None
        input_ids = tok["input_ids"].to(device)
        text_attn = tok["attention_mask"].to(device)
        B = input_ids.shape[0]

        audio_embeds = audio_model(input_values, attention_mask=audio_attn)  # (B, A, H) bf16
        A = audio_embeds.shape[1]

        text_embeds = embed_layer(input_ids)                 # (B, L, H)
        targets = input_ids[:, 1:].clone()                   # (B, L-1)
        # pad_token == eos_token in LFM2.5, so mask by POSITION (attention mask),
        # never by id -- id-masking would also ignore the real EOS target and the
        # model would never be supervised to stop.
        tgt_mask = text_attn[:, 1:]                           # 1 = real (incl EOS), 0 = pad
        targets[tgt_mask == 0] = -100
        keep = (tgt_mask == 1)

        # teacher: text-only, frozen BASE distribution (disable LoRA if present),
        # no grad -> the target the audio student is distilled toward.
        with torch.no_grad():
            if args.llm_lora:
                with llm.disable_adapter():
                    tlogits = llm(inputs_embeds=text_embeds, attention_mask=text_attn).logits[:, :-1, :]
            else:
                tlogits = llm(inputs_embeds=text_embeds, attention_mask=text_attn).logits[:, :-1, :]

        # student: audio prefix + teacher-forced text[:-1] -> same continuation
        inp = torch.cat([audio_embeds, text_embeds[:, :-1, :]], dim=1)
        s_attn = torch.cat([torch.ones(B, A, device=device, dtype=text_attn.dtype),
                            text_attn[:, :-1]], dim=1)
        slogits = llm(inputs_embeds=inp, attention_mask=s_attn).logits[:, A:, :]
        slogits = slogits[:, : targets.shape[1], :]

        ce = F.cross_entropy(
            slogits.reshape(-1, slogits.shape[-1]).float(),
            targets.reshape(-1), ignore_index=-100)

        T = args.kl_temp
        kl_tok = F.kl_div(
            F.log_softmax(slogits.float() / T, dim=-1),
            F.log_softmax(tlogits.float() / T, dim=-1),
            reduction="none", log_target=True).sum(-1)          # (B, L-1)
        kl = (kl_tok * keep).sum() / keep.sum().clamp(min=1) * (T * T)

        loss = ce + args.kl_weight * kl
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            print(f"[step {step}] loss={loss.item():.4f} ce={ce.item():.4f} kl={kl.item():.4f} "
                  f"({(time.time()-t0)/max(1,step):.2f}s/step)", flush=True)
        if step > 0 and step % args.save_every == 0:
            ckpt = args.out.replace(".pt", f"_step{step}.pt")
            torch.save(audio_model.projector.state_dict(), ckpt)   # distinct, for SWA / WER-vs-step
            torch.save(audio_model.projector.state_dict(), args.out)
            print(f"[ckpt] saved {ckpt}", flush=True)
        step += 1

    torch.save(audio_model.projector.state_dict(), args.out)
    if args.llm_lora:
        adapter_dir = args.out.replace(".pt", "_llmlora")
        llm.save_pretrained(adapter_dir)
        print(f"[llm-lora] adapter saved to {adapter_dir}", flush=True)
    print(f"DONE: Moonshine projector saved to {args.out} after {step} steps", flush=True)


if __name__ == "__main__":
    main()
