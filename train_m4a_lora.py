"""train_m4a_lora.py — Phase 1a: LoRA-only run. Isolates ONE question: can
the LLM learn to emit <silence>/<stop_interruption> at the right times at
all, via LoRA + the control-token delta -- WITHOUT wiring the full duplex
loop (no example in this run contains both M3 and M4b streams at once;
that's Phase 1b).

Mixed single-stream training, alternated per step:
  (a) M4a pseudo-timeline (data/m4_pseudo_timeline.py): M3-only conditioning
      (VGGSound pre-pool -> frozen M3 connector), speak=caption / silence=
      <silence> token, using the scene-continuity tick fractions already
      built for M4a.
  (b) EasyCom turn-taking (data/m4_easycom_turntaking.py): M4b-only
      conditioning (Whisper -> frozen M4b projector), speak=transcript /
      silence=<silence> token, using REAL audio gaps between utterances
      (not synthetic).

Frozen (unchanged from their post-joint-training state, checkpoints/
m4_joint/best.pt): M2 predictor, M3 connector, whisper-medium, M4b
projector, and the LLM's own base weights. Trainable: LoRA adapters +
the 2-row control-token delta only. Isolating this from any further
connector drift is deliberate -- the standing rule's whole point is to
attribute falsifier movement to ONE variable per stage, and LoRA is that
stage's variable here.

Usage:
    python train_m4a_lora.py --max-steps 2000 --eval-every 200
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

from train_m3 import build_splits, m3_collate_fn, _cap_ambient_len, CACHE_DIR
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from models.m4_control import get_control_token_ids, wrap_lora, attach_control_token_embedding
from data.av_cached_dataset import AVCachedDataset
from data.m4_pseudo_timeline import M4PseudoTimelineDataset, m4_collate_fn, SPEAK_FRACTION, SILENCE_FRACTIONS
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset
from train_m4b import word_error_rate

FIELD = "gpt_sound_acoustic"


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

    print("[1a] loading tokenizer + LLM (base frozen, LoRA on top)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    base_llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    for p in base_llm.parameters():
        p.requires_grad_(False)
    target_modules = (["q_proj", "k_proj", "v_proj", "o_proj"] if args.lora_target_modules == "attn_only"
                       else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    llm = wrap_lora(base_llm, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                     target_modules=target_modules)
    control_embed = attach_control_token_embedding(llm, tokenizer)
    control_ids = get_control_token_ids(tokenizer)
    n_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    print(f"[1a] LoRA r={args.lora_r} target_modules={args.lora_target_modules} ({target_modules})  "
          f"trainable params={n_trainable:,}  control_ids={control_ids}  silence_weight={args.silence_weight}",
          flush=True)

    print("[1a] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    print(f"[1a] loading FROZEN M3 connector + M4b projector from {args.joint_ckpt} (post-joint state)...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(joint_ckpt["m3_connector"])
    m3_connector.eval()
    for p in m3_connector.parameters():
        p.requires_grad_(False)

    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()
    for p in m4b_projector.parameters():
        p.requires_grad_(False)

    print("[1a] loading frozen whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[1a] building datasets...", flush=True)
    vgg_train_pairs3, vgg_test_pairs3 = build_splits(FIELD)
    vgg_train_pairs = [(c, t) for c, _, t in vgg_train_pairs3]
    vgg_test_pairs = [(c, t) for c, _, t in vgg_test_pairs3]
    vgg_train_ds = M4PseudoTimelineDataset(vgg_train_pairs[:args.limit_vgg] if args.limit_vgg else vgg_train_pairs,
                                            CACHE_DIR, tokenizer, control_ids["silence"])
    vgg_test_ds = M4PseudoTimelineDataset(vgg_test_pairs[:300], CACHE_DIR, tokenizer, control_ids["silence"])

    ec_train_ticks, ec_test_ticks = build_ticks()
    ec_train_ds = EasyComTurnTakingDataset(ec_train_ticks)
    ec_test_ds = EasyComTurnTakingDataset(ec_test_ticks)
    print(f"[1a] VGGSound pseudo-timeline: train={len(vgg_train_ds)} test={len(vgg_test_ds)}  "
          f"EasyCom turn-taking: train={len(ec_train_ds)} test={len(ec_test_ds)}", flush=True)

    trainable_params = [p for p in llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_steps, args.max_steps)

    def m3_soft_prompt(vgg_item, train_mode):
        batch = m4_collate_fn([vgg_item], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
            kpm = torch.cat([pad["vision"], pad["ambient"]], dim=1)
            m3_lat = m3_connector(pre_pool.to(torch.bfloat16), kpm)
        return m3_lat, torch.zeros(1, m3_lat.shape[1], dtype=torch.bool, device=device)   # no padding

    def m4b_soft_prompt(ec_item, train_mode):
        b = {"waveforms": [ec_item["waveform"]], "durations_sec": [ec_item["duration_sec"]]}
        with torch.no_grad():
            hidden, valid_frames = whisper(b["waveforms"], b["durations_sec"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                stoks, smask = m4b_projector(hidden.float(), valid_frames)
        return stoks, smask

    def compute_loss(train_mode):
        use_vgg = rng.random() < 0.5
        if use_vgg:
            ds = vgg_train_ds if train_mode else vgg_test_ds
            item = ds[rng.randrange(len(ds))]
            soft_prompt, kpm = m3_soft_prompt(item, train_mode)
            target_ids_t = item["target_ids"].unsqueeze(0).to(device)
            is_speak = item["label"] == "speak"
        else:
            ds = ec_train_ds if train_mode else ec_test_ds
            item = ds[rng.randrange(len(ds))]
            soft_prompt, kpm = m4b_soft_prompt(item, train_mode)
            if item["is_speak"]:
                ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            else:
                ids = [control_ids["silence"], tokenizer.eos_token_id]
            target_ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            is_speak = item["is_speak"]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            target_embeds = llm.get_input_embeddings()(target_ids_t)
            inputs_embeds = torch.cat([soft_prompt, target_embeds], dim=1)
            sp_attn = torch.ones(1, soft_prompt.shape[1], dtype=torch.long, device=device)
            tgt_attn = torch.ones(1, target_embeds.shape[1], dtype=torch.long, device=device)
            attention_mask = torch.cat([sp_attn, tgt_attn], dim=1)
            labels = torch.cat([torch.full((1, soft_prompt.shape[1]), -100, dtype=torch.long, device=device),
                                 target_ids_t], dim=1)
            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss, use_vgg, is_speak

    def weighted_loss(train_mode):
        """FIX 1a: <silence> targets are only 2 tokens (silence_id, eos) vs
        speak targets' many tokens -- HF's default per-example loss is
        already token-MEAN-reduced (not summed), so raw loss magnitude is
        roughly comparable per example regardless of length; the real
        imbalance is that the rare "override the fluent-text prior" signal
        is thin and easily swamped, across many steps, by the abundant,
        easy-to-fit "predict the specific right word" signal sharing the
        same low-rank LoRA capacity. Explicitly upweight silence examples'
        loss (not speak examples') so their gradient contribution is
        proportionally larger over the course of training."""
        loss, use_vgg, is_speak = compute_loss(train_mode)
        w = 1.0 if is_speak else args.silence_weight
        return loss * w, loss, use_vgg, is_speak

    @torch.no_grad()
    def eval_loss(n=40):
        llm.eval()
        losses = []
        for _ in range(n):
            l, _, _ = compute_loss(train_mode=False)
            losses.append(l.item())
        llm.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def eval_silence_rate_and_recall(n=200):
        """Per task source, separately: silence-rate DISTRIBUTION + speak/silence recall."""
        llm.eval()
        stats = {"vgg": {"n_speak": 0, "n_silence": 0, "speak_correct": 0, "silence_correct": 0},
                 "easycom": {"n_speak": 0, "n_silence": 0, "speak_correct": 0, "silence_correct": 0}}
        for i in range(n):
            use_vgg = i % 2 == 0
            if use_vgg:
                item = vgg_test_ds[rng.randrange(len(vgg_test_ds))]
                soft_prompt, kpm = m3_soft_prompt(item, False)
                is_speak = item["label"] == "speak"
                key = "vgg"
            else:
                item = ec_test_ds[rng.randrange(len(ec_test_ds))]
                soft_prompt, kpm = m4b_soft_prompt(item, False)
                is_speak = item["is_speak"]
                key = "easycom"
            attn = torch.ones(1, soft_prompt.shape[1], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn, max_new_tokens=1,
                                        do_sample=False, pad_token_id=tokenizer.pad_token_id)
            pred_silence = gen_ids[0, 0].item() == control_ids["silence"]
            if is_speak:
                stats[key]["n_speak"] += 1
                stats[key]["speak_correct"] += int(not pred_silence)
            else:
                stats[key]["n_silence"] += 1
                stats[key]["silence_correct"] += int(pred_silence)
        llm.train()
        out = {}
        for key, s in stats.items():
            total = s["n_speak"] + s["n_silence"]
            out[key] = {
                "n": total,
                "gt_silence_rate": s["n_silence"] / max(1, total),
                "gt_speak_rate": s["n_speak"] / max(1, total),
                "silence_recall": s["silence_correct"] / max(1, s["n_silence"]),
                "speak_recall": s["speak_correct"] / max(1, s["n_speak"]),
            }
        return out

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_f = open(os.path.join(args.ckpt_dir, "train_log.jsonl"), "a")

    print(f"[1a] starting training: max_steps={args.max_steps} lr={args.lr}", flush=True)
    llm.train()
    t_start = time.time()
    loss_ema = None
    best_test_loss = float("inf")
    best_step = -1
    def save_ckpt(step, tl, path):
        lora_state = {k: v for k, v in llm.state_dict().items() if "lora_" in k}
        torch.save({"step": step, "test_loss": tl, "lora_state_dict": lora_state,
                    "control_delta": control_embed.delta.detach().cpu(),
                    "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules},
                   path)

    best_silence_score = -1.0   # min(silence_recall, speak_recall) across both sources -- avoids the always-X trap
    best_silence_step = -1
    for step in range(args.max_steps):
        opt.zero_grad()
        wloss, raw_loss, use_vgg, is_speak = weighted_loss(train_mode=True)
        wloss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        loss_ema = raw_loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * raw_loss.item()
        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"[1a] step {step:5d}/{args.max_steps}  loss={raw_loss.item():.4f}  weighted={wloss.item():.4f}  "
                  f"loss_ema={loss_ema:.4f}  src={'vgg' if use_vgg else 'easycom'}  speak={is_speak}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s", flush=True)
            log_f.write(json.dumps({"step": step, "loss": raw_loss.item(), "loss_ema": loss_ema, "split": "train"}) + "\n")
            log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            tl = eval_loss()
            is_best = tl < best_test_loss
            print(f"[1a]   eval @ step {step}: test_loss={tl:.4f}"
                  f"{'  (new best)' if is_best else f'  (best={best_test_loss:.4f}@{best_step})'}", flush=True)
            log_f.write(json.dumps({"step": step, "test_loss": tl, "split": "test"}) + "\n")
            log_f.flush()
            if is_best:
                best_test_loss, best_step = tl, step
                save_ckpt(step, tl, os.path.join(args.ckpt_dir, "best.pt"))

            # FIX 1b: log the silence-rate TRAJECTORY, not just the endpoint,
            # so slow-convergence vs stuck-at-zero can be told apart.
            sr = eval_silence_rate_and_recall(n=args.silence_eval_n_during_training)
            combined_silence_recall = (sr["vgg"]["silence_recall"] + sr["easycom"]["silence_recall"]) / 2
            combined_speak_recall = (sr["vgg"]["speak_recall"] + sr["easycom"]["speak_recall"]) / 2
            score = min(combined_silence_recall, combined_speak_recall)   # penalizes always-X collapse either direction
            print(f"[1a]   silence-rate @ step {step}: vgg(sil_recall={sr['vgg']['silence_recall']:.3f} "
                  f"speak_recall={sr['vgg']['speak_recall']:.3f})  easycom(sil_recall={sr['easycom']['silence_recall']:.3f} "
                  f"speak_recall={sr['easycom']['speak_recall']:.3f})  balance_score={score:.3f}", flush=True)
            log_f.write(json.dumps({"step": step, "silence_rate_trajectory": sr, "balance_score": score}) + "\n")
            log_f.flush()
            if score > best_silence_score:
                best_silence_score, best_silence_step = score, step
                save_ckpt(step, tl, os.path.join(args.ckpt_dir, "best_silence_balance.pt"))

    final_tl = eval_loss(n=60)
    print(f"[1a] FINAL test_loss={final_tl:.4f}  BEST(by test_loss)={best_test_loss:.4f}@{best_step}  "
          f"BEST(by silence-balance)={best_silence_score:.4f}@{best_silence_step}", flush=True)
    save_ckpt(args.max_steps, final_tl, os.path.join(args.ckpt_dir, "last.pt"))

    print("[1a] running FINAL silence-rate / recall eval (per source, using best_silence_balance.pt)...", flush=True)
    if os.path.isfile(os.path.join(args.ckpt_dir, "best_silence_balance.pt")):
        bs_ckpt = torch.load(os.path.join(args.ckpt_dir, "best_silence_balance.pt"), map_location=device, weights_only=False)
        llm.load_state_dict(bs_ckpt["lora_state_dict"], strict=False)
        control_embed.delta.data.copy_(bs_ckpt["control_delta"].to(device))
    sr_stats = eval_silence_rate_and_recall(n=args.silence_eval_n)
    print(f"[1a] SILENCE-RATE REPORT (best_silence_balance.pt, step {best_silence_step}): "
          f"{json.dumps(sr_stats, indent=2)}", flush=True)
    with open(os.path.join(args.ckpt_dir, "silence_rate_report.json"), "w") as f:
        json.dump(sr_stats, f, indent=2)

    log_f.close()
    print("[1a] DONE.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", choices=["full", "attn_only"], default="full",
                    help="FIX 2b: 'attn_only' drops FFN (gate/up/down_proj) from LoRA targets.")
    p.add_argument("--silence-weight", type=float, default=1.0,
                    help="FIX 1a: loss multiplier for <silence>-labeled examples (speak examples always 1.0).")
    p.add_argument("--ckpt-dir", default="checkpoints/m4a_lora")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--silence-eval-n", type=int, default=300)
    p.add_argument("--silence-eval-n-during-training", type=int, default=60,
                    help="cheaper eval size used for the mid-training trajectory log")
    p.add_argument("--limit-vgg", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
