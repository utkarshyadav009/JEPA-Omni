"""scripts/qwen_omni_multigranularity_pilot.py — Track B pilot: VL-JEPA-style
multi-granularity captions + our audio axis, on a small clip sample.

Per clip, generates a single JSON object with five fields in ONE video/audio
pass (five separate generate() calls would re-encode the same video/audio
five times -- wasteful and irrelevant to the actual ask, which is distinct
TEXT granularities, not five independent model queries):

  gpt_action_brief     -- one short sentence, the main physical action (visual)
  gpt_action_detailed  -- 2-3 sentences, the action sequence in detail (visual)
  gpt_summary_brief    -- one short sentence, whole-scene summary (visual)
  gpt_summary_detailed -- short paragraph, whole-scene summary (visual)
  gpt_sound_acoustic   -- 1-2 sentences, THE SOUND ONLY: label-injected,
                          must describe the sound and its source, not visuals.
                          This is the non-negotiable audio-grounded field.

Same anti-repetition generation config as the fixed single-caption pipeline
(repetition_penalty=1.15, no_repeat_ngram_size=3), just a larger token cap
to fit five fields.

Usage:
    python scripts/qwen_omni_multigranularity_pilot.py --n-clips 100 --split test \\
        --out scripts/qwen_omni_multigranularity_pilot100.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief",
          "gpt_summary_detailed", "gpt_sound_acoustic"]


def build_prompt(label: str) -> str:
    return (
        f"This clip's sound is \"{label}\". Analyze the video, using BOTH what you see "
        f"and what you hear, and output ONLY a JSON object (no markdown fences, no other "
        f"text) with exactly these five string fields:\n"
        f'- "gpt_action_brief": one short sentence naming the main physical action.\n'
        f'- "gpt_action_detailed": 2-3 sentences describing the sequence of actions/movements in detail.\n'
        f'- "gpt_summary_brief": one short sentence summarizing the whole scene (setting, subject, action).\n'
        f'- "gpt_summary_detailed": a short paragraph (3-4 sentences) summarizing the scene, setting, subjects, and context.\n'
        f'- "gpt_sound_acoustic": 1-2 sentences describing ONLY the sound itself and its source '
        f'(what the "{label}" sound actually sounds like -- tone, rhythm, texture -- and what is making it). '
        f'Do NOT describe visuals in this field; it must explicitly reflect the given sound label.\n'
        f"Output strictly valid JSON."
    )


def load_split_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def parse_json_response(text: str) -> dict:
    """Best-effort JSON parse: strip markdown fences, find the outermost {...}
    if there's leading/trailing chatter, and never silently drop data -- on
    failure, return the raw text under a 'parse_error' field for review."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"parse_error": True, "raw_text": text}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=100)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--video-dir", default="/home/utkarsh/data/vggsound")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_multigranularity_pilot100.jsonl"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--exclude-csv", default=os.path.join(PROJECT_ROOT, "scripts", "caption_review_results.csv"),
                    help="Exclude clip_ids already reviewed under the single-caption format, for fresh coverage.")
    p.add_argument("--max-new-tokens", type=int, default=350)
    args = p.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, "data", f"{args.split}.csv")
    rows = load_split_rows(csv_path)
    random.Random(args.seed).shuffle(rows)

    excluded = set()
    if args.exclude_csv and os.path.isfile(args.exclude_csv):
        with open(args.exclude_csv, newline="") as f:
            excluded = {r["clip_id"] for r in csv.DictReader(f)}
    rows = [(fname, label) for fname, label in rows
            if os.path.splitext(fname)[0] not in excluded]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mg-pilot] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[mg-pilot] model loaded.", flush=True)

    n_written = 0
    n_parse_errors = 0
    per_clip_times = []
    t_start = time.time()
    with open(args.out, "w") as fout:
        for fname, label in rows:
            if n_written >= args.n_clips:
                break
            video_path = os.path.join(args.video_dir, fname)
            if not os.path.isfile(video_path):
                continue

            t0 = time.time()
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
                # NOTE: no_repeat_ngram_size is deliberately NOT used here (unlike
                # the single-caption pipeline). It's a hard block on any repeated
                # 3-token sequence anywhere in the output -- but this is structured
                # JSON with five keys sharing common prefixes/words ("gpt_",
                # "action_", "summary_", "brief", "detailed"). Smoke-testing with
                # no_repeat_ngram_size=3 here corrupted the key spelling itself
                # (model garbled repeated schema tokens to dodge the constraint,
                # e.g. "gpt_action_brief" -> "gpi_action_brie" on the 2nd/3rd use).
                # repetition_penalty alone (a soft downweight, not a hard ban)
                # still guards against the free-text runaway-loop failure mode
                # without forcing the schema tokens to mutate.
                out_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                          return_audio=False, max_new_tokens=args.max_new_tokens,
                                          do_sample=False, repetition_penalty=1.15)
            gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()
            elapsed = time.time() - t0
            per_clip_times.append(elapsed)

            parsed = parse_json_response(response)
            if "parse_error" in parsed:
                n_parse_errors += 1

            rec = {"clip_id": os.path.splitext(fname)[0], "vggsound_label": label,
                   "gen_seconds": round(elapsed, 2)}
            for field in FIELDS:
                rec[field] = parsed.get(field, None)
            if "parse_error" in parsed:
                rec["parse_error_raw"] = parsed["raw_text"]

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_written += 1
            ok = "OK" if "parse_error" not in parsed else "PARSE-FAIL"
            print(f"[mg-pilot] {n_written}/{args.n_clips}  ({elapsed:.1f}s, {ok})  "
                  f"{rec['clip_id']}  [{label}]", flush=True)

    total_elapsed = time.time() - t_start
    mean_t = sum(per_clip_times) / len(per_clip_times) if per_clip_times else 0.0
    print(f"[mg-pilot] DONE. wrote {n_written} clips to {args.out}  "
          f"parse_errors={n_parse_errors}  mean_gen_seconds={mean_t:.2f}  "
          f"total_wall_seconds={total_elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
