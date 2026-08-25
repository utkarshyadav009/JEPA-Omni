"""scripts/finetune_bmo_neutts.py -- fine-tune NeuTTS-Air on the real BMO
speech dataset. Adapted from neuphonic/neutts's examples/finetune.py (same
preprocessing/chat-format logic, verified against the real repo), with two
real changes:

1. Loads our local `data/bmo_neutts_dataset` (from
   scripts/prep_bmo_neutts_dataset.py) via `load_from_disk` instead of
   `neuphonic/emilia-yodas-english-neucodec` from the HF hub.
2. Adds a train/val split + eval loop + best-checkpoint-by-val-loss tracking.
   The reference finetune.py has NEITHER -- it just trains blindly for
   max_steps and saves once at the end. This project's own MiniCPM5-1B LoRA
   run overfit badly the first time specifically because that same
   no-eval-tracking mistake was made once already (see checkpoints/
   bmo_minicpm5_lora's best/last split, added after the fact) -- not
   repeating it here. Our real dataset is ~37min (851 clips) vs. the
   10hr-scale reference config's `max_steps: 10000`, so step count is also
   scaled down and left as an explicit, reviewable default rather than a
   blind carryover.
"""
from __future__ import annotations

import argparse
import re
from functools import partial

import phonemizer
import torch
from datasets import load_from_disk
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

    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        LOGGER.warning(f"Empty phonemization for sample: {sample['__key__']} text={text}")
        return None
    phones = " ".join(phones[0].split())

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = (
        f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\n"
        f"assistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
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
        labels[idx[0]:] = input_ids[idx[0]:]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/bmo_neutts_dataset")
    ap.add_argument("--restore-from", default="neuphonic/neutts-air")
    ap.add_argument("--out-dir", default="checkpoints/bmo_neutts_finetune")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    # 851 real clips / batch 2 = ~425 steps/epoch. 4 epochs ~= 1700 steps --
    # scaled down from the reference config's 10000 (built for the ~10hr
    # Emilia-YODAS scale), with eval tracking to catch overfitting instead
    # of guessing the "right" step count blindly.
    ap.add_argument("--max-steps", type=int, default=1700)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.restore_from}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.restore_from)
    model = AutoModelForCausalLM.from_pretrained(args.restore_from, dtype="auto")

    g2p = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True,
        words_mismatch="ignore", language_switch="remove-flags",
    )
    partial_preprocess = partial(preprocess_sample, tokenizer=tokenizer, max_len=args.max_seq_len, g2p=g2p)

    ds = load_from_disk(args.dataset)
    ds = ds.filter(data_filter).map(partial_preprocess, remove_columns=["text", "codes"])
    ds = ds.filter(lambda x: x is not None)

    split = ds.train_test_split(test_size=args.val_frac, seed=1337)
    train_ds, val_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        do_train=True,
        do_eval=True,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        warmup_ratio=0.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=20,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(args.out_dir + "/best")
    tokenizer.save_pretrained(args.out_dir + "/best")
    print(f"DONE: best checkpoint (by eval_loss) saved to {args.out_dir}/best", flush=True)


if __name__ == "__main__":
    main()
