"""train_m4_joint.py — short joint-exposure fine-tune of M3's connector +
M4b's projector, together, on sequences containing BOTH streams.

Why: gates (c)/(d) showed two independently-trained connectors do not
compose safely by naive concatenation -- neither connector, nor the frozen
LLM, was ever exposed to an input containing both a perceptual soft-prompt
AND a speech soft-prompt at once. The dummy-token diagnostic
(scripts/m4_diagnostic_dummy_tokens.py) and the delimiter fix
(scripts/m4_delimiter_fix_test.py) both failed to recover it, so per plan
this is the last resort before touching LoRA.

DATA CONSTRUCTION -- FLAGGED ARTIFICIALITY: there is no corpus with a
VGGSound-style scene and EasyCom speech co-occurring for real. Each
training step independently samples ONE VGGSound (clip, caption) pair and
ONE EasyCom (audio, transcription) segment -- unrelated, i.i.d., no
temporal or causal correspondence between them. This teaches the model to
correctly ROUTE ATTENTION to whichever stream a task-tag asks for while
the other stream is present as (irrelevant) context, which is exactly what
checks (c)/(d) probe -- but it does NOT teach genuine reasoning over two
streams that are actually causally related in time, which is what a real
M4c duplex system will eventually need. This is a real limitation of this
fix, not a hidden one.

Training: for each step, with 50% probability the target is the VGGSound
caption (using the existing GRANULARITY_TAGS[field] tag) and the model
must attend to the M3 latents while treating the (present but irrelevant)
speech block as noise; the other 50% the target is the EasyCom
transcription (using a new SPEECH_TASK_TAG) and the model must attend to
the speech block while ignoring the (present but irrelevant) M3 latents.

Trainable: M3Connector (re-initialized from checkpoints/m3_multigran_best)
+ UltravoxProjector (re-initialized from checkpoints/m4b/best.pt), BOTH
unfrozen, trained together. Frozen: M2, Whisper, LLM. NO LoRA in this run
-- one variable, per instruction.

Usage:
    python train_m4_joint.py --max-steps 1500 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_m3 import build_splits, m3_collate_fn, word_overlap_f1, _cap_ambient_len, CACHE_DIR, GRANULARITY_TAGS
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from data.av_cached_dataset import AVCachedDataset
from data.m4_speech_dataset import build_segments, EasyComSpeechDataset, m4b_collate_fn
from train_m4b import word_error_rate

FIELD = "gpt_sound_acoustic"
SPEECH_TASK_TAG = "Task: transcribe the speech segment, ignoring any unrelated scene description.\n"
VISION_TASK_TAG_SUFFIX = " (ignore any unrelated speech transcript segment.)\n"


def build_scheduler(opt, warmup, total):
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

    print("[joint] loading tokenizer + frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print("[joint] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    print(f"[joint] loading M3 connector (trainable, init from {args.m3_init})...", flush=True)
    m3init = torch.load(args.m3_init, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3init["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3init["connector"])

    print("[joint] loading frozen whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print(f"[joint] loading M4b projector (trainable, init from {args.m4b_init})...", flush=True)
    m4binit = torch.load(args.m4b_init, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**m4binit["projector_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(m4binit["projector"])

    n_m3 = sum(p.numel() for p in m3_connector.parameters())
    n_m4b = sum(p.numel() for p in m4b_projector.parameters())
    print(f"[joint] trainable: M3 connector={n_m3:,}  M4b projector={n_m4b:,}  total={n_m3+n_m4b:,}", flush=True)

    print("[joint] building VGGSound + EasyCom pools...", flush=True)
    vgg_train_pairs, vgg_test_pairs = build_splits(FIELD)
    easycom_train_segs, easycom_test_segs = build_segments()
    if args.limit_train:
        rng.shuffle(vgg_train_pairs); vgg_train_pairs = vgg_train_pairs[:args.limit_train]
        rng.shuffle(easycom_train_segs); easycom_train_segs = easycom_train_segs[:args.limit_train]
    vgg_test_pairs_eval = vgg_test_pairs[:args.limit_test] if args.limit_test else vgg_test_pairs[:300]
    easycom_test_segs_eval = easycom_test_segs[:args.limit_test] if args.limit_test else easycom_test_segs[:300]
    print(f"[joint] VGGSound train={len(vgg_train_pairs)} test={len(vgg_test_pairs_eval)}  "
          f"EasyCom train={len(easycom_train_segs)} test={len(easycom_test_segs_eval)}", flush=True)

    easycom_train_ds = EasyComSpeechDataset(easycom_train_segs)
    easycom_test_ds = EasyComSpeechDataset(easycom_test_segs_eval)

    trainable_params = list(m3_connector.parameters()) + list(m4b_projector.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_steps, args.max_steps)

    speech_tag_ids = tokenizer(SPEECH_TASK_TAG, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    def get_m3_latents(clip_id, train_mode):
        ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=[clip_id], max_tdm_bins=512, audio_mode="mean")
        item = ds[0]
        batch = m3_collate_fn([{"feats": item["feats"], "tbins": item["tbins"], "clip_id": item["clip_id"],
                                 "prefix_ids": torch.zeros(0, dtype=torch.long), "caption_ids": torch.zeros(1, dtype=torch.long),
                                 "caption_text": "", "field": None}], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
        kpm = torch.cat([pad["vision"], pad["ambient"]], dim=1)
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            m3_lat = m3_connector(pre_pool.to(torch.bfloat16), kpm)
        return m3_lat   # (1, 32, H)

    def get_speech_tokens(easycom_item, train_mode):
        b = m4b_collate_fn([easycom_item])
        with torch.no_grad():
            hidden, valid_frames = whisper(b["waveforms"], b["durations_sec"], device)
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stoks, smask = m4b_projector(hidden.float(), valid_frames)
        return stoks, smask   # (1, T, H), (1, T) True=pad

    def make_step_example(vgg_pairs, easycom_ds_local, easycom_segs, train_mode):
        # AVCachedDataset's own missing-file fallback picks a random OTHER
        # clip from its clip_ids list -- but we instantiate it with a
        # single-clip list each call, so a missing .pt file has nowhere to
        # fall back to except itself (infinite recursion -> stack overflow).
        # Retry with a different sampled clip at this level instead.
        for _ in range(10):
            vgg_cid, _, vgg_gt = rng.choice(vgg_pairs)
            try:
                m3_lat = get_m3_latents(vgg_cid, train_mode)
                break
            except (FileNotFoundError, RuntimeError, RecursionError):
                continue
        else:
            raise RuntimeError("could not find a loadable VGGSound clip after 10 retries")
        e_idx = rng.randrange(len(easycom_ds_local))
        e_item = easycom_ds_local[e_idx]
        stoks, smask = get_speech_tokens(e_item, train_mode)

        target_is_vision = rng.random() < 0.5
        if target_is_vision:
            tag_ids = tokenizer(GRANULARITY_TAGS[FIELD].rstrip("\n") + VISION_TASK_TAG_SUFFIX,
                                 add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            target_text = vgg_gt
        else:
            tag_ids = speech_tag_ids
            target_text = e_item["text"]

        tag_embeds = llm.get_input_embeddings()(tag_ids)   # (1, Lp, H)
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        target_ids_t = torch.tensor([target_ids], dtype=torch.long, device=device)
        target_embeds = llm.get_input_embeddings()(target_ids_t)

        sattn = (~smask).long()
        m3_attn = torch.ones(1, m3_lat.shape[1], dtype=torch.long, device=device)
        tag_attn = torch.ones(1, tag_embeds.shape[1], dtype=torch.long, device=device)
        tgt_attn = torch.ones(1, target_embeds.shape[1], dtype=torch.long, device=device)

        inputs_embeds = torch.cat([m3_lat, stoks, tag_embeds, target_embeds], dim=1)
        attention_mask = torch.cat([m3_attn, sattn, tag_attn, tgt_attn], dim=1)
        n_prefix = m3_lat.shape[1] + stoks.shape[1] + tag_embeds.shape[1]
        labels = torch.cat([torch.full((1, n_prefix), -100, dtype=torch.long, device=device), target_ids_t], dim=1)
        return inputs_embeds, attention_mask, labels, target_is_vision, target_text

    def compute_loss(train_mode):
        pool = vgg_train_pairs if train_mode else vgg_test_pairs_eval
        eds = easycom_train_ds if train_mode else easycom_test_ds
        segs = easycom_train_segs if train_mode else easycom_test_segs_eval
        inputs_embeds, attention_mask, labels, _, _ = make_step_example(pool, eds, segs, train_mode)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def eval_loss(n: int = 40):
        m3_connector.eval(); m4b_projector.eval()
        losses = [compute_loss(train_mode=False).item() for _ in range(n)]
        m3_connector.train(); m4b_projector.train()
        return sum(losses) / len(losses)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_f = open(os.path.join(args.ckpt_dir, "train_log.jsonl"), "a")

    print(f"[joint] starting training: max_steps={args.max_steps} lr={args.lr}", flush=True)
    m3_connector.train(); m4b_projector.train()
    t_start = time.time()
    loss_ema = None
    best_test_loss = float("inf")
    best_step = -1
    for step in range(args.max_steps):
        opt.zero_grad()
        loss = compute_loss(train_mode=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"[joint] step {step:5d}/{args.max_steps}  loss={loss.item():.4f}  loss_ema={loss_ema:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s", flush=True)
            log_f.write(json.dumps({"step": step, "loss": loss.item(), "loss_ema": loss_ema, "split": "train"}) + "\n")
            log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            tl = eval_loss()
            is_best = tl < best_test_loss
            print(f"[joint]   eval @ step {step}: test_loss={tl:.4f}"
                  f"{'  (new best)' if is_best else f'  (best={best_test_loss:.4f}@{best_step})'}", flush=True)
            log_f.write(json.dumps({"step": step, "test_loss": tl, "split": "test"}) + "\n")
            log_f.flush()
            if is_best:
                best_test_loss, best_step = tl, step
                torch.save({"step": step, "m3_connector": m3_connector.state_dict(), "m3_cfg": m3_cfg.__dict__,
                            "m4b_projector": m4b_projector.state_dict(), "m4b_cfg": m4b_cfg.__dict__},
                           os.path.join(args.ckpt_dir, "best.pt"))

    final_tl = eval_loss(n=60)
    print(f"[joint] FINAL test_loss={final_tl:.4f}  BEST={best_test_loss:.4f}@{best_step}", flush=True)
    torch.save({"step": args.max_steps, "m3_connector": m3_connector.state_dict(), "m3_cfg": m3_cfg.__dict__,
                "m4b_projector": m4b_projector.state_dict(), "m4b_cfg": m4b_cfg.__dict__},
               os.path.join(args.ckpt_dir, "last.pt"))
    if final_tl < best_test_loss:
        torch.save({"step": args.max_steps, "m3_connector": m3_connector.state_dict(), "m3_cfg": m3_cfg.__dict__,
                    "m4b_projector": m4b_projector.state_dict(), "m4b_cfg": m4b_cfg.__dict__},
                   os.path.join(args.ckpt_dir, "best.pt"))
    log_f.close()
    print("[joint] DONE.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--m3-init", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--m4b-init", default="checkpoints/m4b/best.pt")
    p.add_argument("--ckpt-dir", default="checkpoints/m4_joint")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=1)   # per-example loop (variable seq len); see note
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
