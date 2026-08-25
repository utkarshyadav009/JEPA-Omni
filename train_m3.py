"""train_m3.py — M3 connector training (stage 1).

Frozen M2 predictor (encode_pre_pool_tokens) -> trainable M3Connector
(Perceiver-style, 32 latents) -> frozen Qwen2.5-1.5B-Instruct, trained as a
soft-prompt caption-generation task (cross-entropy on caption tokens only).

Only the connector is trainable. M2 and the LLM are both frozen throughout
this stage -- no LoRA yet. Pre-pool tokens are computed ON THE FLY each step
from the already-cached raw AV features (NOT cached to disk themselves --
measured at 1.89ms/clip for the frozen M2 forward pass, cheaper than storing
and streaming the ~594GB a full pre-pool token cache would need at this
corpus size; see train_m3 report for the timing that justified this).

Dataset: scripts/qwen_omni_full_captions.jsonl (188,657 verified multi-
granularity captions), split by the ORIGINAL VGGSound train.csv/test.csv
membership (not a fresh arbitrary split) so M3's held-out set is
principled and reusable. Stage 1 trains on gpt_sound_acoustic only.

Usage:
    python train_m3.py --max-steps 5000 --batch-size 32 \\
        --eval-every 500 --save-every 500
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.text_target import TextTarget
from models.losses import info_nce

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions.jsonl")
CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_vgg51k"
_CAPTIONS_PATH_OVERRIDE = None  # set via --captions-path in main()
MAX_AMBIENT_T = 1024   # same memory-safety cap as train_m2.py, independent reasoning: same cause

ALL_GRANULARITIES = [
    "gpt_action_brief", "gpt_action_detailed",
    "gpt_summary_brief", "gpt_summary_detailed",
    "gpt_sound_acoustic",
]

# Text prefix prepended (as real LLM tokens, after the connector's 32 soft-
# prompt embeddings) to tell the frozen LLM which style to emit for THIS
# example. Plain text, no new params -- reuses the frozen embedding table,
# same mechanism the soft-prompt already relies on. Masked out of the loss
# (label=-100) exactly like the soft-prompt positions: the model is never
# trained to predict its own instruction, only the caption that follows.
GRANULARITY_TAGS = {
    "gpt_action_brief": "Task: state the main physical action in one short sentence.\n",
    "gpt_action_detailed": "Task: describe the sequence of actions in detail.\n",
    "gpt_summary_brief": "Task: summarize the whole scene in one short sentence.\n",
    "gpt_summary_detailed": "Task: summarize the scene, setting, and context in a short paragraph.\n",
    "gpt_sound_acoustic": "Task: describe the sound itself and its likely source.\n",
}


# ── Dataset ──────────────────────────────────────────────────────────────
def load_vgg_split_ids(csv_path: str) -> set:
    ids = set()
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                ids.add(os.path.splitext(row[0].strip())[0])
    return ids


def build_splits(fields) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Returns (train_pairs, test_pairs) of (clip_id, field, caption_text),
    split by the ORIGINAL VGGSound data/train.csv vs data/test.csv membership
    (both disjoint by construction). `fields` may be a single field name
    (str, back-compat) or a list of field names -- one example is emitted
    per (clip, field) pair with a non-null caption, so multi-granularity
    training sees each clip up to len(fields) times, once per granularity."""
    if isinstance(fields, str):
        fields = [fields]
    train_ids = load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_ids = load_vgg_split_ids(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    assert not (train_ids & test_ids), "VGGSound train/test csvs are not disjoint -- unexpected"

    train_pairs, test_pairs = [], []
    with open(CAPTIONS_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["clip_id"]
            if cid in train_ids:
                bucket = train_pairs
            elif cid in test_ids:
                bucket = test_pairs
            else:
                continue
            for field in fields:
                text = r.get(field)
                if not text:
                    continue
                bucket.append((cid, field, text))
    return train_pairs, test_pairs


class M3CaptionDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], cache_dir: str, tokenizer, max_tdm_bins: int = 512):
        self.pairs = pairs
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_tdm_bins = max_tdm_bins
        # local import to reuse the exact same feature-loading logic/format
        # as every other stage in this project (av_cached_dataset.py)
        from data.av_cached_dataset import AVCachedDataset
        # One dataset ROW per (clip, field) example -- clip_ids repeats for
        # clips with multiple granularities, AVCachedDataset indexes
        # positionally so this is safe (just re-reads the .pt per row).
        clip_ids = [cid for cid, _, _ in pairs]
        self.av_ds = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                                      max_tdm_bins=max_tdm_bins, audio_mode="mean")
        self.fields = [field for _, field, _ in pairs]
        self.captions = [text for _, _, text in pairs]

    def __len__(self) -> int:
        return len(self.av_ds)

    def __getitem__(self, idx: int) -> Dict:
        item = self.av_ds[idx]
        field = self.fields[idx]
        text = self.captions[idx]
        prefix = GRANULARITY_TAGS[field]
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        cap_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        cap_ids = cap_ids + [self.tokenizer.eos_token_id]   # teach the model to stop
        item["prefix_ids"] = torch.tensor(prefix_ids, dtype=torch.long)
        item["caption_ids"] = torch.tensor(cap_ids, dtype=torch.long)
        item["caption_text"] = text
        item["field"] = field
        return item


def m3_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    B = len(batch)
    max_Tv = max(s["feats"]["vision"].shape[0] for s in batch)
    max_Ta = max(s["feats"]["ambient"].shape[0] for s in batch)
    D_v = batch[0]["feats"]["vision"].shape[1]
    D_a = batch[0]["feats"]["ambient"].shape[1]

    vis_feats = torch.zeros(B, max_Tv, D_v, dtype=torch.bfloat16)
    aud_feats = torch.zeros(B, max_Ta, D_a, dtype=torch.bfloat16)
    vis_bins = torch.zeros(B, max_Tv, dtype=torch.long)
    aud_bins = torch.zeros(B, max_Ta, dtype=torch.long)
    vis_pad = torch.ones(B, max_Tv, dtype=torch.bool)   # True = padding
    aud_pad = torch.ones(B, max_Ta, dtype=torch.bool)

    # Text sequence per sample = prefix (granularity tag, masked out of loss)
    # followed by the caption itself. Concatenated once here so the LM sees
    # one contiguous text stream after the soft-prompt; label_start records
    # where each sample's real (loss-bearing) caption tokens begin.
    has_prefix = "prefix_ids" in batch[0]
    full_ids_list = []
    label_start_list = []
    for s in batch:
        prefix_ids = s["prefix_ids"] if has_prefix else torch.zeros(0, dtype=torch.long)
        full_ids_list.append(torch.cat([prefix_ids, s["caption_ids"]], dim=0))
        label_start_list.append(prefix_ids.shape[0])

    max_L = max(t.shape[0] for t in full_ids_list)
    cap_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long)
    cap_mask = torch.zeros(B, max_L, dtype=torch.long)
    label_start = torch.tensor(label_start_list, dtype=torch.long)

    clip_ids, texts, fields = [], [], []
    for i, s in enumerate(batch):
        Tv, Ta = s["feats"]["vision"].shape[0], s["feats"]["ambient"].shape[0]
        vis_feats[i, :Tv] = s["feats"]["vision"]; aud_feats[i, :Ta] = s["feats"]["ambient"]
        vis_bins[i, :Tv] = s["tbins"]["vision"]; aud_bins[i, :Ta] = s["tbins"]["ambient"]
        vis_pad[i, :Tv] = False; aud_pad[i, :Ta] = False
        L = full_ids_list[i].shape[0]
        cap_ids[i, :L] = full_ids_list[i]; cap_mask[i, :L] = 1
        clip_ids.append(s["clip_id"]); texts.append(s["caption_text"])
        fields.append(s.get("field"))

    return {
        "feats": {"vision": vis_feats, "ambient": aud_feats},
        "tbins": {"vision": vis_bins, "ambient": aud_bins},
        "padding_mask": {"vision": vis_pad, "ambient": aud_pad},
        "caption_ids": cap_ids, "caption_mask": cap_mask, "label_start": label_start,
        "clip_ids": clip_ids, "caption_texts": texts, "fields": fields,
    }


def _cap_ambient_len(feats, tbins, pad_mask, max_t=MAX_AMBIENT_T):
    if feats["ambient"].shape[1] > max_t:
        feats["ambient"] = feats["ambient"][:, :max_t]
        tbins["ambient"] = tbins["ambient"][:, :max_t]
        pad_mask["ambient"] = pad_mask["ambient"][:, :max_t]


# ── Simple automatic metric ─────────────────────────────────────────────
def word_overlap_f1(pred: str, gold: str) -> float:
    """Unigram (lowercased word) F1 -- deliberately simple, no external deps."""
    p = pred.lower().split()
    g = gold.lower().split()
    if not p or not g:
        return 0.0
    p_set, g_set = set(p), set(g)
    tp = len(p_set & g_set)
    if tp == 0:
        return 0.0
    precision = tp / len(p_set)
    recall = tp / len(g_set)
    return 2 * precision * recall / (precision + recall)


# ── Training ─────────────────────────────────────────────────────────────
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

    fields = args.fields.split(",") if args.fields else [args.field]
    print(f"[m3] building splits from fields={fields}...", flush=True)
    train_pairs, test_pairs = build_splits(fields)
    train_ids = {c for c, _, _ in train_pairs}
    test_ids = {c for c, _, _ in test_pairs}
    overlap = train_ids & test_ids
    print(f"[m3] split sizes: train={len(train_pairs)} examples / {len(train_ids)} clips  "
          f"test={len(test_pairs)} examples / {len(test_ids)} clips  "
          f"overlap={len(overlap)} clips (must be 0)", flush=True)
    assert len(overlap) == 0, "train/test split overlap -- must not happen"

    if args.limit_train:
        rng.shuffle(train_pairs)
        train_pairs = train_pairs[:args.limit_train]
    if args.limit_test:
        test_pairs = test_pairs[:args.limit_test]

    print(f"[m3] loading tokenizer + frozen LLM ({args.llm})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_config = AutoConfig.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    print(f"[m3] LLM hidden_size={llm_config.hidden_size} (confirmed from config, not hardcoded)", flush=True)

    print(f"[m3] loading frozen M2 predictor from {args.m2_ckpt}...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    connector_cfg = M3ConnectorConfig(d_model=1024, n_latents=32, n_layers=args.connector_layers,
                                       n_heads=8, llm_hidden=llm_config.hidden_size)
    connector = M3Connector(connector_cfg).to(device)
    n_params = sum(p.numel() for p in connector.parameters())
    print(f"[m3] connector params={n_params:,} ({n_params/1e6:.1f}M)  "
          f"layers={args.connector_layers}  n_latents=32", flush=True)

    # Optional auxiliary alignment loss (BLIP-2-stage-1-style): pool the connector's
    # soft-prompt vectors -> compare directly against a frozen text encoder's embedding
    # of the ground-truth caption via InfoNCE. This gives the connector a short, direct
    # gradient path that does not have to survive backprop through the full frozen LLM
    # -- the captioning cross-entropy loss alone is the ONLY signal without this (the
    # thing BLIP-2's own ablation shows causes weak/plateauing connector training).
    # TextTarget's projection head (native EmbeddingGemma dim -> llm_hidden) is
    # trainable and added to the same optimizer as the connector; the EmbeddingGemma
    # base itself stays frozen (unfreeze_base=False, the default).
    text_target = None
    if args.lam_align > 0:
        print(f"[m3] loading TextTarget (backbone={args.align_backbone}) for alignment loss "
              f"(lam_align={args.lam_align}, temp={args.align_temp})...", flush=True)
        text_target = TextTarget(backbone=args.align_backbone, shared_dim=llm_config.hidden_size,
                                  device=str(device), dtype=torch.bfloat16)
        n_align_params = sum(p.numel() for p in text_target.proj.parameters())
        print(f"[m3] TextTarget native_dim={text_target.native_dim} -> shared_dim={text_target.shared_dim}  "
              f"trainable proj params={n_align_params:,}", flush=True)

    train_ds = M3CaptionDataset(train_pairs, CACHE_DIR, tokenizer)
    test_ds = M3CaptionDataset(test_pairs, CACHE_DIR, tokenizer)
    print(f"[m3] cached clips found: train={len(train_ds)}  test={len(test_ds)}", flush=True)

    collate = lambda b: m3_collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=False)

    def batches_forever():
        while True:
            for b in train_loader:
                yield b
    batches = batches_forever()

    trainable_params = list(connector.parameters())
    if text_target is not None:
        trainable_params += list(text_target.proj.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_steps, args.max_steps)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    def compute_batch_loss(batch, train_mode: bool):
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad_mask = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad_mask)
        cap_ids = batch["caption_ids"].to(device)
        cap_mask = batch["caption_mask"].to(device)
        label_start = batch["label_start"].to(device)   # (B,) -- prefix length per sample

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            pre_pool = predictor.encode_pre_pool_tokens(feats_f, tbins)   # (B, S, 1024)
        key_padding_mask = torch.cat([pad_mask["vision"], pad_mask["ambient"]], dim=1)

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = connector(pre_pool.to(torch.bfloat16), key_padding_mask)  # (B, 32, H)
            text_embeds = llm.get_input_embeddings()(cap_ids)                        # (B, L, H)
            inputs_embeds = torch.cat([soft_prompt, text_embeds], dim=1)

            B, n_lat, _ = soft_prompt.shape
            prompt_attn = torch.ones(B, n_lat, dtype=torch.long, device=device)
            attention_mask = torch.cat([prompt_attn, cap_mask], dim=1)

            labels = cap_ids.masked_fill(cap_mask == 0, -100)
            # Mask out the granularity-tag prefix positions too (per sample,
            # since prefix length varies with the tag string) -- the model
            # is trained to predict the caption, never its own instruction.
            L = cap_ids.shape[1]
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            labels = labels.masked_fill(pos < label_start.unsqueeze(1), -100)
            prompt_labels = torch.full((B, n_lat), -100, dtype=torch.long, device=device)
            labels = torch.cat([prompt_labels, labels], dim=1)

            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            ce_loss = out.loss

            align_loss = None
            align_metrics = {}
            if text_target is not None:
                pooled_prompt = F.normalize(soft_prompt.mean(dim=1).float(), dim=-1)  # (B, H)
                z_text = text_target.encode_text(batch["caption_texts"])              # (B, H), fp32 normalized
                align_loss, align_metrics = info_nce(pooled_prompt, z_text, temperature=args.align_temp)

        total_loss = ce_loss if align_loss is None else ce_loss + args.lam_align * align_loss
        metrics = {"ce_loss": ce_loss.item(), "total_loss": total_loss.item()}
        if align_loss is not None:
            metrics["align_loss"] = align_loss.item()
            metrics["align_acc_v2t"] = align_metrics["acc_v2t"]
        return total_loss, metrics

    @torch.no_grad()
    def eval_loss(n_batches: int = 20) -> Dict[str, float]:
        connector.eval()
        agg: Dict[str, List[float]] = {}
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            _, metrics = compute_batch_loss(batch, train_mode=False)
            for k, v in metrics.items():
                agg.setdefault(k, []).append(v)
        connector.train()
        return {k: sum(v) / max(1, len(v)) for k, v in agg.items()}

    @torch.no_grad()
    def generate_samples(n: int, out_path: str) -> None:
        connector.eval()
        samples = []
        idxs = list(range(len(test_ds)))
        rng.shuffle(idxs)
        for idx in idxs:
            if len(samples) >= n:
                break
            item = test_ds[idx]
            batch = m3_collate_fn([item], tokenizer.pad_token_id)
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
                prefix_text = GRANULARITY_TAGS.get(item.get("field"), "")
                if prefix_text:
                    prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                                            return_tensors="pt")["input_ids"].to(device)
                    prefix_embeds = llm.get_input_embeddings()(prefix_ids)
                    gen_inputs_embeds = torch.cat([soft_prompt, prefix_embeds], dim=1)
                else:
                    gen_inputs_embeds = soft_prompt
                attn = torch.ones(gen_inputs_embeds.shape[:2], dtype=torch.long, device=device)
                gen_ids = llm.generate(inputs_embeds=gen_inputs_embeds, attention_mask=attn,
                                        max_new_tokens=60, do_sample=False,
                                        repetition_penalty=1.15,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            f1 = word_overlap_f1(gen_text, item["caption_text"])
            samples.append({"clip_id": item["clip_id"], "field": item.get("field"),
                             "ground_truth": item["caption_text"],
                             "generated": gen_text, "word_overlap_f1": f1})
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        mean_f1 = sum(s["word_overlap_f1"] for s in samples) / max(1, len(samples))
        print(f"[m3] wrote {len(samples)} samples to {out_path}  mean_word_overlap_f1={mean_f1:.3f}", flush=True)
        connector.train()

    print(f"[m3] starting training: max_steps={args.max_steps} batch_size={args.batch_size} "
          f"lr={args.lr} fields={fields}", flush=True)
    connector.train()
    t_start = time.time()
    loss_ema = None
    def save_ckpt(step: int) -> None:
        ckpt_obj = {"step": step, "connector": connector.state_dict(),
                    "connector_cfg": connector_cfg.__dict__}
        if text_target is not None:
            ckpt_obj["text_target_proj"] = text_target.proj.state_dict()
            ckpt_obj["align_backbone"] = args.align_backbone
        torch.save(ckpt_obj, os.path.join(args.ckpt_dir, "last.pt"))

    for step in range(args.max_steps):
        batch = next(batches)
        opt.zero_grad()
        loss, metrics = compute_batch_loss(batch, train_mode=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            lr_now = sched.get_last_lr()[0]
            elapsed = time.time() - t_start
            align_str = (f"  ce={metrics['ce_loss']:.4f}  align={metrics['align_loss']:.4f}  "
                         f"align_acc={metrics['align_acc_v2t']:.3f}" if text_target is not None else "")
            print(f"[m3] step {step:6d}/{args.max_steps}  loss={loss.item():.4f}  "
                  f"loss_ema={loss_ema:.4f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s{align_str}", flush=True)
            log_row = {"step": step, "loss": loss.item(), "loss_ema": loss_ema,
                       "lr": lr_now, "split": "train"}
            log_row.update(metrics)
            log_f.write(json.dumps(log_row) + "\n")
            log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            tm = eval_loss()
            print(f"[m3]   eval @ step {step}: test_loss={tm['total_loss']:.4f}"
                  + (f"  ce={tm['ce_loss']:.4f}  align={tm.get('align_loss', float('nan')):.4f}"
                     if text_target is not None else ""), flush=True)
            log_row = {"step": step, "split": "test"}
            log_row.update({f"test_{k}": v for k, v in tm.items()})
            log_f.write(json.dumps(log_row) + "\n")
            log_f.flush()

        if step > 0 and step % args.save_every == 0:
            save_ckpt(step)

    final_metrics = eval_loss(n_batches=50)
    print(f"[m3] FINAL test_loss={final_metrics['total_loss']:.4f}"
          + (f"  ce={final_metrics['ce_loss']:.4f}  align={final_metrics.get('align_loss', float('nan')):.4f}"
             if text_target is not None else ""), flush=True)
    save_ckpt(args.max_steps)

    sample_path = os.path.join(args.ckpt_dir, "sample_generations.jsonl")
    generate_samples(args.n_samples, sample_path)
    log_f.close()
    print(f"[m3] DONE.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="gpt_sound_acoustic",
                    help="Which single caption granularity to train on (stage 1 back-compat).")
    p.add_argument("--fields", default=None,
                    help="Comma-separated list of granularities to train on jointly "
                         "(stage 2 scaling). Overrides --field if set. "
                         f"Available: {','.join(ALL_GRANULARITIES)}")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--captions-path", default=None,
                    help="Override CAPTIONS_PATH (default: scripts/qwen_omni_full_captions.jsonl).")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--connector-layers", type=int, default=3)
    p.add_argument("--lam-align", type=float, default=0.0,
                    help="Weight for the auxiliary EmbeddingGemma alignment InfoNCE loss "
                         "(pooled soft-prompt vs frozen text encoding of the ground-truth "
                         "caption). 0.0 (default) = off, exact old behavior.")
    p.add_argument("--align-backbone", default="embeddinggemma",
                    help="TextTarget backbone for the alignment loss (see models/text_target.py).")
    p.add_argument("--align-temp", type=float, default=0.07, help="InfoNCE temperature for the alignment loss.")
    p.add_argument("--ckpt-dir", default="checkpoints/m3_connector")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--limit-train", type=int, default=None, help="cap train clips (smoke tests)")
    p.add_argument("--limit-test", type=int, default=None, help="cap test clips (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.captions_path:
        global CAPTIONS_PATH
        CAPTIONS_PATH = args.captions_path
        print(f"[m3] CAPTIONS_PATH overridden -> {CAPTIONS_PATH}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
