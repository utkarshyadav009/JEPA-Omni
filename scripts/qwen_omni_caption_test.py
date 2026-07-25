"""scripts/qwen_omni_caption_test.py — Track B smoke test: generate rich
audio-visual captions for N VGGSound clips using Qwen2.5-Omni-7B (joint
video+audio input, TMRoPE time-alignment). Text-only output (talker/audio
generation disabled). For manual human review -- not training data yet.

Usage:
    python scripts/qwen_omni_caption_test.py --n-clips 50 --split test \\
        --video-dir /home/utkarsh/data/vggsound --out captions_test50.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prompt(label: str) -> str:
    """Inject the VGGSound ground-truth label so the caption is forced to be
    audio-grounded -- the first pass (no label, generic prompt) produced
    sound-blind, frame-only captions (e.g. a 'sea lion barking' clip
    captioned purely from a visual frame with no mention of any bark)."""
    return (f"This clip's sound is \"{label}\". In one or two sentences, describe "
            f"what happens visually AND the sound and its source, as a single "
            f"natural caption.")


def load_split_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append((row[0].strip(), row[1].strip()))  # (filename, label)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=50)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--video-dir", default="/home/utkarsh/data/vggsound")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_captions_test50.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-ids", default=None,
                    help="Comma-separated clip_ids to caption instead of a random sample "
                         "(for targeted re-validation of specific clips).")
    args = p.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, "data", f"{args.split}.csv")
    rows = load_split_rows(csv_path)
    if args.clip_ids:
        wanted = set(args.clip_ids.split(","))
        rows = [(fname, label) for fname, label in rows
                if os.path.splitext(fname)[0] in wanted]
    else:
        import random
        random.Random(args.seed).shuffle(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[qwen-omni-caption] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[qwen-omni-caption] model loaded.", flush=True)

    n_written = 0
    t0 = time.time()
    with open(args.out, "w") as fout:
        for fname, label in rows:
            if n_written >= args.n_clips:
                break
            video_path = os.path.join(args.video_dir, fname)
            if not os.path.isfile(video_path):
                continue

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": build_prompt(label)},
                ]},
            ]
            USE_AUDIO_IN_VIDEO = True
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)

            with torch.no_grad():
                # do_sample=False kept for reproducibility; repetition_penalty +
                # no_repeat_ngram_size are the actual fix for the greedy
                # repetition spiral seen in the first pass (e.g. "wet sidewalk /
                # puddle" repeated ~40x) -- greedy decoding alone doesn't
                # prevent it once the model locks onto a locally-confident loop.
                out_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                          return_audio=False, max_new_tokens=90,
                                          do_sample=False, repetition_penalty=1.15,
                                          no_repeat_ngram_size=3)
            gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()

            rec = {"clip_id": os.path.splitext(fname)[0], "vggsound_label": label, "caption": response}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_written += 1
            print(f"[qwen-omni-caption] {n_written}/{args.n_clips}  "
                  f"({time.time()-t0:.0f}s)  {rec['clip_id']}  [{label}] -> {response[:80]}", flush=True)

    print(f"[qwen-omni-caption] DONE. wrote {n_written} captions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
