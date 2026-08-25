"""scripts/finetune_bmo_minicpm5_lora.py — LoRA fine-tune of MiniCPM5-1B on
real BMO dialogue (971 lines, BMO_SpeechDataset) + a small, EXPLICITLY
FLAGGED-AS-DRAFT synthetic functional-category set (data/
bmo_synthetic_functional.jsonl, 41 lines written by Claude to match observed
style patterns, NOT independently verified as authentically BMO-voiced --
review before trusting as real signal).

Per R5 (this session's research, 2026-08-04): LoRA not full fine-tune,
matches both the small dataset size (37min/971 real lines -- far too
little for a full 1B-param fine-tune without overfitting) and the
BMO-Project README's own roadmap ("SFT voice fine-tuning (LoRA + TIES
merge)").

Trains the model to continue a fixed BMO-persona prompt prefix with
BMO-style short lines -- matches EXACTLY the prompt format
models/m4_cognitive_core.py's FastTier already uses at inference
(apply_chat_template + enable_thinking=False), so the fine-tuned adapter
is a drop-in swap for FastTier's base model, no inference-code changes
needed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

BMO_PERSONA_PROMPT = "You are BMO, a small friendly companion robot. Speak briefly, warmly, and a little playfully, the way BMO does."


def load_real_lines(metadata_csv: str) -> list[str]:
    lines = []
    with open(metadata_csv) as f:
        next(f)  # header
        for row in f:
            parts = row.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            text = parts[1].strip()
            # skip pure non-verbal tags ([cry], [laugh], [scream], [sing]) --
            # not text-generation targets, those are audio-only events
            if text.startswith("[") and text.endswith("]"):
                continue
            if text:
                lines.append(text)
    return lines


def load_synthetic_examples(jsonl_path: str) -> list[tuple[str, dict, str | None]]:
    """Returns (text, state_dict, prompt) triples -- state_dict is {} for real
    lines (no state info exists) or the real {"energy":.., "mood":..} the
    synthetic line was written for. Training on the actual state prefix
    (not a fixed generic prompt) is required for the model to have ANY
    chance of learning to condition its response on injected state -- the
    first fine-tune run trained on a fixed prompt with no state signal at
    all and, unsurprisingly when tested, ignored injected state entirely.

    `prompt` is the real user utterance BMO is responding to (e.g. for the
    "open_question" category), or None for categories that are just
    mood-expression lines with no specific trigger utterance. Real bug found
    and fixed here: until this field existed, EVERY training example used
    the literal fixed string "Say something." as the user turn regardless of
    category, while real inference (models/m4_cognitive_core.py) always puts
    the actual ASR transcript there -- a genuine train/inference mismatch
    that meant the model had structurally never seen "respond to this
    specific question" during training, no matter how much question-style
    data existed in the corpus."""
    examples = []
    with open(jsonl_path) as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            r = json.loads(row)
            examples.append((r["text"], r.get("state", {}), r.get("prompt")))
    return examples


def _state_prefix(state: dict) -> str:
    """Mirrors models/m4_cognitive_core.py's _state_prefix EXACTLY -- must
    match the real inference-time format or training the association is
    pointless."""
    if not state:
        return ""
    parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in state.items()]
    return "[" + " ".join(parts) + "] "


class BmoLineDataset(Dataset):
    def __init__(self, examples: list[tuple[str, dict, str | None]], tokenizer, max_len: int = 96):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        target, state, prompt = self.examples[idx]
        prefix = _state_prefix(state)
        msgs = [{"role": "system", "content": BMO_PERSONA_PROMPT},
                {"role": "user", "content": prefix + (prompt if prompt else "Say something.")}]
        prompt_ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = prompt_ids[0]

        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        eos = torch.tensor([self.tokenizer.eos_token_id])
        input_ids = torch.cat([prompt_ids, target_ids, eos])[: self.max_len]
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100  # only supervise the BMO line itself, not the prompt
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openbmb/MiniCPM5-1B")
    p.add_argument("--real-metadata", default=os.path.expanduser(
        "~/../home/bmo/BMO-LabelData/Final_Dataset/BMO_SpeechDataset/metadata.csv"))
    p.add_argument("--synthetic-jsonl", default="data/bmo_synthetic_functional.jsonl")
    p.add_argument("--out-dir", default="checkpoints/bmo_minicpm5_lora")
    p.add_argument("--lora-r", type=int, default=16)
    # Real finding: LFM2's hybrid conv+GQA architecture uses different leaf
    # module names than MiniCPM5's standard Llama-style attention (checked
    # via named_modules(), not assumed) -- no o_proj/gate_proj/up_proj/
    # down_proj at all; attention uses q_proj/k_proj/v_proj + out_proj
    # (shared name with the conv blocks' own in_proj/out_proj), FFN uses
    # w1/w2/w3. Applying MiniCPM's target_modules list to LFM2 would
    # silently adapt few or zero real layers.
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    # Upsample factor for the synthetic (state+prompt-conditioned) examples.
    # 8x suited the old 41-line seed; for the ~940-pair v9 conversational
    # corpus a much smaller factor keeps real:synth balanced (2x -> synth ~2:1
    # over the 971 real lines, so the conversational pairs -- the fix -- lead
    # without drowning the real BMO voice).
    p.add_argument("--synth-upsample", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[bmo-lora] loading real lines from {args.real_metadata}", flush=True)
    real_lines = load_real_lines(args.real_metadata)
    real_examples = [(t, {}, None) for t in real_lines]   # no state/prompt info exists for real lines
    synth_examples = load_synthetic_examples(args.synthetic_jsonl) if os.path.exists(args.synthetic_jsonl) else []
    print(f"[bmo-lora] real={len(real_examples)} synthetic(DRAFT, unverified, state-conditioned)={len(synth_examples)}", flush=True)

    # Upsample the synthetic (state-conditioned) examples so the model sees
    # the state-prefix -> content association enough times to learn it,
    # not drowned out by 916 unconditioned real lines. Real ratio choice,
    # not tuned against a held-out state-following metric (none exists
    # yet) -- flagged as a reasonable starting point, not validated.
    synth_upsampled = synth_examples * args.synth_upsample
    all_examples = real_examples + synth_upsampled
    random.shuffle(all_examples)
    n_val = max(10, int(len(all_examples) * args.val_frac))
    val_examples, train_examples = all_examples[:n_val], all_examples[n_val:]
    print(f"[bmo-lora] train={len(train_examples)} val={len(val_examples)} "
          f"(synthetic upsampled {args.synth_upsample}x = {len(synth_upsampled)} of the pool)", flush=True)

    print(f"[bmo-lora] loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).to(device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, target_modules=args.lora_target_modules.split(","),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = BmoLineDataset(train_examples, tokenizer)
    val_ds = BmoLineDataset(val_examples, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    n_steps = len(train_loader) * args.epochs
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=max(5, n_steps // 20), num_training_steps=n_steps)

    print(f"[bmo-lora] training: {args.epochs} epochs x {len(train_loader)} steps/epoch = {n_steps} total steps", flush=True)
    step = 0
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"[bmo-lora] step {step}/{n_steps} epoch={epoch} loss={loss.item():.4f} "
                      f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(model(**batch).loss.item())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        is_best = val_loss < best_val_loss
        print(f"[bmo-lora] === epoch {epoch} done, val_loss={val_loss:.4f}"
              f"{' (new best)' if is_best else ''} ===", flush=True)
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_pretrained(os.path.join(args.out_dir, "best"))
            tokenizer.save_pretrained(os.path.join(args.out_dir, "best"))

    print(f"[bmo-lora] saving FINAL (last-epoch) adapter too, for comparison...", flush=True)
    model.save_pretrained(os.path.join(args.out_dir, "last"))
    tokenizer.save_pretrained(os.path.join(args.out_dir, "last"))
    print(f"[bmo-lora] DONE. best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} "
          f"-> {args.out_dir}/best (use this one, not /last, unless you've verified /last "
          f"generalizes despite the higher val_loss)", flush=True)

    print(f"\n[bmo-lora] reloading BEST checkpoint (epoch {best_epoch}) for sample generations "
          f"(not the final/possibly-overfit epoch)...", flush=True)
    from peft import PeftModel
    del model
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(base_model, os.path.join(args.out_dir, "best")).to(device)

    # Real qualitative check: sample a few generations, print them so quality
    # is visible immediately, not just a loss number
    print("\n[bmo-lora] === SAMPLE GENERATIONS (post-fine-tune, testing REAL state-conditioning) ===", flush=True)
    model.eval()
    # Test the ACTUAL fix: (state, user-utterance) pairs -- especially hostility
    # (should be hurt/firm, NOT chirpy), emotional support (empathetic), and
    # general conversation (specific, on-topic), plus the unprompted mood path.
    test_cases = [
        ({}, "Say something."),
        ({"energy": 0.8, "mood": "happy"}, "Say something."),
        ({"energy": 0.35, "mood": "stressed"}, "You are so stupid and useless."),
        ({"energy": 0.35, "mood": "stressed"}, "Shut up, nobody likes you."),
        ({"energy": 0.4, "mood": "anxious"}, "I hate you, you're the worst."),
        ({"energy": 0.5, "mood": "concerned"}, "I had the worst day of my life."),
        ({"energy": 0.45, "mood": "lonely"}, "I feel like nobody actually cares about me."),
        ({"energy": 0.6, "mood": "content"}, "What should we watch tonight?"),
        ({"energy": 0.6, "mood": "curious"}, "What's the weather like tomorrow?"),
        ({"energy": 0.78, "mood": "happy"}, "I love you BMO."),
    ]
    with torch.no_grad():
        for state, user in test_cases:
            prefix = _state_prefix(state)
            msgs = [{"role": "system", "content": BMO_PERSONA_PROMPT},
                    {"role": "user", "content": prefix + user}]
            prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
            if hasattr(prompt_ids, "keys"):
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = prompt_ids.to(device)
            out = model.generate(prompt_ids, max_new_tokens=40, do_sample=True, temperature=0.9,
                                  top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            print(f"  [{state.get('mood','-'):9s}] USER {user!r}\n             BMO  {text!r}", flush=True)


if __name__ == "__main__":
    main()
