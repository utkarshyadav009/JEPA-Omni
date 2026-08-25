"""scripts/finetune_bmo_emotion_neutts.py -- add emotion control tokens to the
deployed BMO NeuTTS voice.

Design (2026-08-07):
  - RESTORE FROM the deployed BMO voice (checkpoints/bmo_neutts_finetune_v5/best,
    the exact source of the production bmo_neutts_v5 GGUF), NOT raw neutts-air --
    so we keep BMO's timbre and learn emotion as a small delta on top.
  - Add 12 special control tokens: <|NEUTRAL|> + one per homeostatic mood
    (<|EXCITED|> <|HAPPY|> <|CONTENT|> <|SURPRISED|> <|STRESSED|> <|ANXIOUS|>
    <|CONCERNED|> <|LONELY|> <|TIRED|> <|BORED|> <|CURIOUS|>). These match the
    mood strings homeostatic_to_mood_state() already emits, so the SAME state
    that shapes the LLM's words shapes the voice: emotion=mood, zero translation.
  - Training format prepends the control token right where inference does:
        assistant:<|EXCITED|><|SPEECH_GENERATION_START|>{codes}<|SPEECH_GENERATION_END|>
    The control token is a CONDITION we supply at inference, so it is masked
    from the loss (labels start at <|SPEECH_GENERATION_START|>, same as base).
  - Data mix = Fish emotion clips (mood token) + the ORIGINAL neutral BMO
    recordings tagged <|NEUTRAL|>. Anchoring neutral on the real recordings
    keeps the production default (neutral) voice IDENTICAL while the emotion
    tokens pull toward the Fish-rendered emotional deliveries. This is the
    timbre-drift guard from TRAINING.md, made concrete.
  - mean_resizing initializes the 12 new embeddings near the embedding mean
    (transformers default) -- far better than random for few-shot token learning.

Eval-tracked (best-by-val-loss), same as finetune_bmo_neutts.py -- the
MiniCPM5 LoRA overfit once from blind step-count training; not repeating it.
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import partial

import phonemizer
import torch
from datasets import concatenate_datasets, load_from_disk

sys.path.insert(0, ".")
from models.m5_streaming_voice import normalize_bmo_text  # noqa: E402
from loguru import logger as LOGGER
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")

MOODS = ["neutral", "excited", "happy", "content", "surprised", "stressed",
         "anxious", "concerned", "lonely", "tired", "bored", "curious"]
EMO_TOKENS = [f"<|{m.upper()}|>" for m in MOODS]


def data_filter(sample):
    text = sample["text"]
    if len(text) == 0:
        return False
    if re.search(r"\d", text):
        return False
    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False
    if text[-1] not in ".,?!":
        return False
    if "£" in text or "$" in text:
        return False
    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    ignore_index = -100

    vq_codes = sample["codes"]
    text = sample["text"]
    mood = sample.get("mood", "neutral")
    if mood not in MOODS:
        mood = "neutral"
    emo_tok = f"<|{mood.upper()}|>"

    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        LOGGER.warning(f"Empty phonemization: {sample.get('__key__')} text={text}")
        return None
    phones = " ".join(phones[0].split())

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = (
        f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\n"
        f"assistant:{emo_tok}<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
    )
    ids = tokenizer.encode(chat)
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, ignore_index)
    idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        labels[idx[0]:] = input_ids[idx[0]:]  # emotion token stays masked (condition)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion-dataset", default="data/bmo_emotion_neutts_dataset")
    ap.add_argument("--neutral-dataset", default="data/bmo_neutts_dataset")
    ap.add_argument("--restore-from", default="checkpoints/bmo_neutts_finetune_v5/best")
    ap.add_argument("--out-dir", default="checkpoints/bmo_neutts_emotion")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-5)  # slightly lower than base 4e-5: we start from an already-tuned voice
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2400)  # ~2270 clips/bs2 ~= 1135 steps/epoch; ~2 epochs
    ap.add_argument("--eval-steps", type=int, default=150)
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.restore_from}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.restore_from)
    model = AutoModelForCausalLM.from_pretrained(args.restore_from, dtype="auto")

    # add the emotion control tokens (only the ones not already present)
    missing = [t for t in EMO_TOKENS if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id
               or t not in tokenizer.get_vocab()]
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": missing}) if missing else 0
    if n_added:
        model.resize_token_embeddings(len(tokenizer))  # mean_resizing=True default -> new embeds ~ mean
    print(f"Added {n_added} emotion control tokens: {missing}", flush=True)

    g2p = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True,
        words_mismatch="ignore", language_switch="remove-flags",
    )
    partial_preprocess = partial(preprocess_sample, tokenizer=tokenizer,
                                 max_len=args.max_seq_len, g2p=g2p)

    emo = load_from_disk(args.emotion_dataset)
    neu = load_from_disk(args.neutral_dataset)
    if "mood" not in neu.column_names:
        neu = neu.add_column("mood", ["neutral"] * len(neu))
    # keep only the shared columns before concat
    keep = ["text", "codes", "__key__", "mood"]
    emo = emo.remove_columns([c for c in emo.column_names if c not in keep])
    neu = neu.remove_columns([c for c in neu.column_names if c not in keep])
    ds = concatenate_datasets([emo, neu]).shuffle(seed=1337)
    print(f"emotion={len(emo)} neutral={len(neu)} total={len(ds)}", flush=True)

    # BMO->Beemo on BOTH sources before filter+phonemize: the emotion dataset is
    # already normalized by the prep, but the neutral anchor (bmo_neutts_dataset)
    # still has raw "BMO" -- normalizing here recovers its BMO lines from the
    # acronym filter and makes the whole mix pronounce "Beemo" consistently.
    ds = ds.map(lambda x: {"text": normalize_bmo_text(x["text"])})
    # drop over-long clips (>320 codes / ~6.4s) so no emotion token learns a
    # long-output distribution that suppresses EOS on short lines (Task 185).
    n_before = len(ds)
    ds = ds.filter(lambda x: len(x["codes"]) <= 320)
    print(f"length filter: {n_before} -> {len(ds)} (dropped {n_before - len(ds)} clips >320 codes)", flush=True)
    ds = ds.filter(data_filter).map(partial_preprocess, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None)

    split = ds.train_test_split(test_size=args.val_frac, seed=1337)
    train_ds, val_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        do_train=True, do_eval=True,
        learning_rate=args.lr, max_steps=args.max_steps, bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        warmup_ratio=0.03,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_steps=args.save_steps, logging_steps=20,
        save_strategy="steps", save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, ignore_data_skip=True,
        dataloader_drop_last=True, remove_unused_columns=False,
        dataloader_num_workers=8,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(args.out_dir + "/best")
    tokenizer.save_pretrained(args.out_dir + "/best")
    print(f"DONE: best (by eval_loss) -> {args.out_dir}/best", flush=True)


if __name__ == "__main__":
    main()
