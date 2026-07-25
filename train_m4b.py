"""train_m4b.py — M4b speech-path training.

Frozen whisper-medium encoder -> trainable UltravoxProjector -> frozen
Qwen2.5-1.5B-Instruct, trained as a soft-prompt ASR-alignment task
(cross-entropy on transcription tokens only) against EasyCom
Close_Microphone_Audio <-> Transcription pairs.

Only the projector is trainable. Whisper and the LLM are both frozen.
M3's connector is NOT part of this training loop at all (EasyCom has no
matched M2/VGGSound feature cache) -- it only enters at gate time (checks
c/d), where a FIXED stream from the other modality is concatenated to test
whether the two independently-trained connectors interfere with each other
in the shared LLM input space. No LoRA here -- that's M4a/M4c.

Usage:
    python train_m4b.py --max-steps 3000 --batch-size 8 \\
        --eval-every 300 --save-every 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def word_overlap_f1(pred: str, gold: str) -> float:
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    p_set, g_set = set(p), set(g)
    tp = len(p_set & g_set)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p_set), tp / len(g_set)
    return 2 * precision * recall / (precision + recall)


def word_error_rate(pred: str, gold: str) -> float:
    """Standard WER via Levenshtein distance on word sequences."""
    p, g = pred.lower().split(), gold.lower().split()
    if not g:
        return 0.0 if not p else 1.0
    dp = list(range(len(p) + 1))
    for i in range(1, len(g) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(p) + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (g[i - 1] != p[j - 1]))
            prev = tmp
    return dp[len(p)] / len(g)


def build_scheduler(opt, warmup: int, total: int):
    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step) / max(1, warmup)
        prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("[m4b] building EasyCom session-split segments...", flush=True)
    train_segs, test_segs = build_segments()
    train_sessions = sorted(set(s.session for s in train_segs))
    test_sessions = sorted(set(s.session for s in test_segs))
    print(f"[m4b] SESSION SPLIT: train_sessions={train_sessions}  test_sessions={test_sessions}", flush=True)
    print(f"[m4b] segments: train={len(train_segs)}  test={len(test_segs)}", flush=True)
    assert not (set(train_sessions) & set(test_sessions)), "session leakage between train/test"

    if args.limit_train:
        rng.shuffle(train_segs)
        train_segs = train_segs[:args.limit_train]
    if args.limit_test:
        test_segs = test_segs[:args.limit_test]

    print(f"[m4b] loading tokenizer + frozen LLM ({args.llm})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print(f"[m4b] loading frozen whisper encoder ({args.whisper})...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    proj_cfg = UltravoxProjectorConfig(whisper_hidden=whisper.hidden_size, stack_factor=args.stack_factor,
                                        llm_hidden=llm.config.hidden_size, mlp_dim=args.mlp_dim)
    projector = UltravoxProjector(proj_cfg).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"[m4b] projector params={n_params:,} ({n_params/1e6:.1f}M)  stack_factor={args.stack_factor}", flush=True)

    train_ds = EasyComSpeechDataset(train_segs)
    test_ds = EasyComSpeechDataset(test_segs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=m4b_collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=m4b_collate_fn, drop_last=False)

    def batches_forever():
        while True:
            for b in train_loader:
                yield b
    batches = batches_forever()

    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_steps, args.max_steps)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_f = open(os.path.join(args.ckpt_dir, "train_log.jsonl"), "a")

    def compute_batch_loss(batch, train_mode: bool):
        hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt, key_padding_mask = projector(hidden.float(), valid_frames)   # (B, T, H), (B, T) True=pad
            text_ids_list = [tokenizer(t, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                              for t in batch["texts"]]
            max_L = max(len(t) for t in text_ids_list)
            B = len(text_ids_list)
            text_ids = torch.full((B, max_L), tokenizer.pad_token_id, dtype=torch.long, device=device)
            text_mask = torch.zeros(B, max_L, dtype=torch.long, device=device)
            for i, ids in enumerate(text_ids_list):
                text_ids[i, :len(ids)] = torch.tensor(ids, device=device)
                text_mask[i, :len(ids)] = 1

            text_embeds = llm.get_input_embeddings()(text_ids)
            inputs_embeds = torch.cat([soft_prompt, text_embeds], dim=1)
            prompt_attn = (~key_padding_mask).long()
            attention_mask = torch.cat([prompt_attn, text_mask], dim=1)

            labels = text_ids.masked_fill(text_mask == 0, -100)
            prompt_labels = torch.full((B, soft_prompt.shape[1]), -100, dtype=torch.long, device=device)
            labels = torch.cat([prompt_labels, labels], dim=1)

            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def eval_loss(n_batches: int = 20) -> float:
        projector.eval()
        losses = []
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            losses.append(compute_batch_loss(batch, train_mode=False).item())
        projector.train()
        return sum(losses) / max(1, len(losses))

    @torch.no_grad()
    def generate_samples(n: int, out_path: str):
        projector.eval()
        idxs = list(range(len(test_ds)))
        rng.shuffle(idxs)
        samples = []
        for idx in idxs:
            if len(samples) >= n:
                break
            item = test_ds[idx]
            batch = m4b_collate_fn([item])
            hidden, valid_frames = whisper(batch["waveforms"], batch["durations_sec"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                soft_prompt, key_padding_mask = projector(hidden.float(), valid_frames)
            attn = (~key_padding_mask).long()
            gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn, max_new_tokens=60,
                                    do_sample=False, repetition_penalty=1.15,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            wer = word_error_rate(gen_text, item["text"])
            f1 = word_overlap_f1(gen_text, item["text"])
            samples.append({"session": item["session"], "chunk": item["chunk"],
                             "participant_id": item["participant_id"], "ground_truth": item["text"],
                             "generated": gen_text, "wer": wer, "word_overlap_f1": f1})
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        mean_wer = sum(s["wer"] for s in samples) / max(1, len(samples))
        mean_f1 = sum(s["word_overlap_f1"] for s in samples) / max(1, len(samples))
        print(f"[m4b] wrote {len(samples)} samples to {out_path}  mean_WER={mean_wer:.3f}  mean_F1={mean_f1:.3f}", flush=True)
        projector.train()

    print(f"[m4b] starting {'SMOKE TEST' if args.smoke_test else 'training'}: max_steps={args.max_steps} "
          f"batch_size={args.batch_size} lr={args.lr}", flush=True)
    projector.train()
    t_start = time.time()
    loss_ema = None
    best_test_loss = float("inf")
    best_step = -1
    for step in range(args.max_steps):
        batch = next(batches)
        opt.zero_grad()
        loss = compute_batch_loss(batch, train_mode=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        opt.step()
        sched.step()

        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"[m4b] step {step:6d}/{args.max_steps}  loss={loss.item():.4f}  loss_ema={loss_ema:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s", flush=True)
            log_f.write(json.dumps({"step": step, "loss": loss.item(), "loss_ema": loss_ema, "split": "train"}) + "\n")
            log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            tl = eval_loss()
            is_best = tl < best_test_loss
            print(f"[m4b]   eval @ step {step}: test_loss={tl:.4f}"
                  f"{'  (new best)' if is_best else f'  (best={best_test_loss:.4f}@{best_step})'}", flush=True)
            log_f.write(json.dumps({"step": step, "test_loss": tl, "split": "test"}) + "\n")
            log_f.flush()
            if is_best and not args.smoke_test:
                best_test_loss, best_step = tl, step
                torch.save({"step": step, "test_loss": tl, "projector": projector.state_dict(),
                            "projector_cfg": proj_cfg.__dict__}, os.path.join(args.ckpt_dir, "best.pt"))

        if not args.smoke_test and step > 0 and step % args.save_every == 0:
            torch.save({"step": step, "projector": projector.state_dict(), "projector_cfg": proj_cfg.__dict__},
                       os.path.join(args.ckpt_dir, "last.pt"))

    if args.smoke_test:
        print("[m4b] SMOKE TEST DONE. Mechanism validated end to end.", flush=True)
        return

    final_test_loss = eval_loss(n_batches=50)
    print(f"[m4b] FINAL (step {args.max_steps}) test_loss={final_test_loss:.4f}  "
          f"BEST test_loss={best_test_loss:.4f} @ step {best_step}", flush=True)
    torch.save({"step": args.max_steps, "test_loss": final_test_loss, "projector": projector.state_dict(),
                "projector_cfg": proj_cfg.__dict__}, os.path.join(args.ckpt_dir, "last.pt"))
    if final_test_loss < best_test_loss:
        torch.save({"step": args.max_steps, "test_loss": final_test_loss, "projector": projector.state_dict(),
                    "projector_cfg": proj_cfg.__dict__}, os.path.join(args.ckpt_dir, "best.pt"))
        best_step = args.max_steps

    print(f"[m4b] reloading BEST checkpoint (step {best_step}) for sample generation, "
          f"not the final (possibly overfit) state", flush=True)
    best_ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pt"), map_location=device, weights_only=False)
    projector.load_state_dict(best_ckpt["projector"])
    generate_samples(args.n_samples, os.path.join(args.ckpt_dir, "sample_generations.jsonl"))
    log_f.close()
    print("[m4b] DONE.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--stack-factor", type=int, default=8)
    p.add_argument("--mlp-dim", type=int, default=4096)
    p.add_argument("--ckpt-dir", default="checkpoints/m4b")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--save-every", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
