"""train_m4.py — M4 tick-grid full-duplex loop, Ego4D-independent build.

Per the M4 design proposal: the LLM needs LoRA now (frozen-through-M3 was
correct for translation-only captioning; M4 needs the model to learn a
genuinely new behavior -- emitting <silence> instead of always completing a
fluent response -- which a frozen decoder cannot pick up from soft-prompt
conditioning alone). Base LLM weights stay frozen; LoRA adapters + the two
repurposed control-token embedding rows (see models/m4_control.py) are
trainable, along with the M3 connector (re-initialized from the frozen
multi-granularity checkpoint, not from scratch, then allowed to adapt to
the tick-truncated/streaming input distribution this task actually sees).

Trains on the Ego4D-independent synthetic pseudo-timeline
(data/m4_pseudo_timeline.py): predicts either the clip's caption (first
tick of a new scene) or the <silence> control token (later ticks of the
same, unchanged scene). This validates the full mechanism -- control
tokens, tick-conditioned soft-prompt refresh, LoRA wiring, loss computation
-- end to end, so that when better turn-taking data (Ego4D or a fallback)
lands, only the DATA needs to change, not the code.

STATUS: plumbing/mechanism only. NOT a real training run -- per the current
instruction, this script is validated with --smoke-test (a handful of
steps) and reported before any full LoRA training job is launched.

Usage (smoke test only):
    python train_m4.py --smoke-test --limit-train 80 --limit-test 40 --max-steps 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_control import get_control_token_ids, wrap_lora, attach_control_token_embedding
from data.m4_pseudo_timeline import M4PseudoTimelineDataset, m4_collate_fn
from train_m3 import load_vgg_split_ids, word_overlap_f1, _cap_ambient_len, CACHE_DIR, CAPTIONS_PATH

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_pseudo_pairs(field: str = "gpt_sound_acoustic") -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_ids = load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_ids = load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    train_pairs, test_pairs = [], []
    with open(CAPTIONS_PATH) as f:
        import json as _json
        for line in f:
            if not line.strip():
                continue
            r = _json.loads(line)
            text = r.get(field)
            if not text:
                continue
            cid = r["clip_id"]
            if cid in train_ids:
                train_pairs.append((cid, text))
            elif cid in test_ids:
                test_pairs.append((cid, text))
    return train_pairs, test_pairs


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("[m4] building Ego4D-independent pseudo-timeline pairs...", flush=True)
    train_pairs, test_pairs = build_pseudo_pairs(args.field)
    if args.limit_train:
        rng.shuffle(train_pairs)
        train_pairs = train_pairs[:args.limit_train]
    if args.limit_test:
        test_pairs = test_pairs[:args.limit_test]
    print(f"[m4] clips: train={len(train_pairs)}  test={len(test_pairs)}  "
          f"(x4 ticks/clip -> {len(train_pairs)*4} / {len(test_pairs)*4} tick examples)", flush=True)

    print(f"[m4] loading tokenizer + LLM ({args.llm})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_config = AutoConfig.from_pretrained(args.llm)
    control_ids = get_control_token_ids(tokenizer)
    print(f"[m4] control token ids: {control_ids}", flush=True)

    base_llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    for p in base_llm.parameters():
        p.requires_grad_(False)
    llm = wrap_lora(base_llm, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    control_embed = attach_control_token_embedding(llm, tokenizer)
    n_ctrl = sum(p.numel() for p in control_embed.parameters() if p.requires_grad)   # just the (2, hidden) delta
    n_total = sum(p.numel() for p in llm.parameters() if p.requires_grad)             # LoRA adapters + delta (delta is a submodule, already included)
    print(f"[m4] LoRA-wrapped LLM: {n_total:,} trainable params total "
          f"({n_ctrl:,} of those are the control-token delta; the base embedding/LM-head "
          f"matrix -- 233M params -- stays fully frozen, no optimizer state allocated for it)", flush=True)

    print(f"[m4] loading frozen M2 predictor from {args.m2_ckpt}...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    print(f"[m4] loading M3 connector (init) from {args.connector_ckpt}...", flush=True)
    conn_ckpt = torch.load(args.connector_ckpt, map_location=device, weights_only=False)
    connector_cfg = M3ConnectorConfig(**conn_ckpt["connector_cfg"])
    connector = M3Connector(connector_cfg).to(device)
    connector.load_state_dict(conn_ckpt["connector"])
    n_conn = sum(p.numel() for p in connector.parameters())
    print(f"[m4] connector params={n_conn:,} (trainable, initialized from frozen M3 multigran ckpt)", flush=True)

    train_ds = M4PseudoTimelineDataset(train_pairs, CACHE_DIR, tokenizer, control_ids["silence"])
    test_ds = M4PseudoTimelineDataset(test_pairs, CACHE_DIR, tokenizer, control_ids["silence"])
    print(f"[m4] tick examples: train={len(train_ds)}  test={len(test_ds)}", flush=True)

    collate = lambda b: m4_collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=False)

    def batches_forever():
        while True:
            for b in train_loader:
                yield b
    batches = batches_forever()

    trainable_params = [p for p in llm.parameters() if p.requires_grad] + list(connector.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    def compute_batch_loss(batch, train_mode: bool):
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad_mask)
        tgt_ids = batch["target_ids"].to(device)
        tgt_mask = batch["target_mask"].to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
        key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = connector(pre_pool.to(torch.bfloat16), key_padding_mask)
            text_embeds = llm.get_input_embeddings()(tgt_ids)
            inputs_embeds = torch.cat([soft_prompt, text_embeds], dim=1)

            B, n_lat, _ = soft_prompt.shape
            prompt_attn = torch.ones(B, n_lat, dtype=torch.long, device=device)
            attention_mask = torch.cat([prompt_attn, tgt_mask], dim=1)

            labels = tgt_ids.masked_fill(tgt_mask == 0, -100)
            prompt_labels = torch.full((B, n_lat), -100, dtype=torch.long, device=device)
            labels = torch.cat([prompt_labels, labels], dim=1)

            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def eval_silence_rate(n_batches: int = 10) -> Dict:
        """Per Correction 3: report the SILENCE RATE distribution, not just
        accuracy -- an always-silent or never-silent model can look good on
        accuracy alone if the label distribution is imbalanced (here it's
        75% silence / 25% speak by construction)."""
        llm.eval(); connector.eval()
        n_speak_gt = n_silence_gt = 0
        n_speak_pred_correct = n_silence_pred_correct = 0
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            feats = {k: v.to(device) for k, v in batch["feats"].items()}
            tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
            pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
            _cap_ambient_len(feats, tbins, pad_mask)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feats_f = {k: v.float() for k, v in feats.items()}
                pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)
                key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)
                soft_prompt = connector(pre_pool.to(torch.bfloat16), key_padding_mask)
                B, n_lat, _ = soft_prompt.shape
                attn = torch.ones(B, n_lat, dtype=torch.long, device=device)
                gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn, max_new_tokens=1,
                                        do_sample=False, pad_token_id=tokenizer.pad_token_id)
            first_tok = gen_ids[:, 0].tolist()
            for j, label in enumerate(batch["labels"]):
                pred_silence = (first_tok[j] == control_ids["silence"])
                if label == "silence":
                    n_silence_gt += 1
                    n_silence_pred_correct += int(pred_silence)
                else:
                    n_speak_gt += 1
                    n_speak_pred_correct += int(not pred_silence)
        llm.train(); connector.train()
        total = n_speak_gt + n_silence_gt
        return {
            "n_examples": total,
            "gt_silence_rate": n_silence_gt / max(1, total),
            "gt_speak_rate": n_speak_gt / max(1, total),
            "silence_recall": n_silence_pred_correct / max(1, n_silence_gt),
            "speak_recall": n_speak_pred_correct / max(1, n_speak_gt),
        }

    print(f"[m4] starting {'SMOKE TEST' if args.smoke_test else 'training'}: "
          f"max_steps={args.max_steps} batch_size={args.batch_size} lr={args.lr}", flush=True)
    llm.train(); connector.train()
    t_start = time.time()
    for step in range(args.max_steps):
        batch = next(batches)
        opt.zero_grad()
        loss = compute_batch_loss(batch, train_mode=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        if step % max(1, args.log_every) == 0:
            elapsed = time.time() - t_start
            print(f"[m4] step {step:4d}/{args.max_steps}  loss={loss.item():.4f}  "
                  f"labels_this_batch={Counter(batch['labels'])}  elapsed={elapsed:.0f}s", flush=True)

    print("[m4] running silence-rate / recall eval...", flush=True)
    stats = eval_silence_rate(n_batches=args.eval_batches)
    print(f"[m4] SILENCE-RATE REPORT: {json.dumps(stats, indent=2)}", flush=True)

    if args.smoke_test:
        print("[m4] SMOKE TEST DONE. Mechanism validated: control tokens, tick-conditioned "
              "soft-prompt, LoRA forward/backward, silence-rate eval all ran end to end. "
              "No checkpoint saved (smoke test only).", flush=True)
    else:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save({
            "step": args.max_steps,
            "connector": connector.state_dict(), "connector_cfg": connector_cfg.__dict__,
            "lora_state_dict": {k: v for k, v in llm.state_dict().items() if "lora_" in k},
            "control_token_ids": control_ids,
        }, os.path.join(args.ckpt_dir, "last.pt"))
        print(f"[m4] saved checkpoint to {args.ckpt_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="gpt_sound_acoustic")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--connector-ckpt", default="checkpoints/m3_multigran_best/connector.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--ckpt-dir", default="checkpoints/m4")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true",
                    help="Validate the mechanism only; no checkpoint saved. Use small --limit-train/"
                         "--limit-test/--max-steps. Do NOT set this off for a real run without explicit go-ahead.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
