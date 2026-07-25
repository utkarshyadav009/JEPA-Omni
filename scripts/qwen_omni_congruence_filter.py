"""scripts/qwen_omni_congruence_filter.py — cheap pre-pass QC gate before rich
caption generation. Two SEPARATE model calls per clip, deliberately kept apart
so tuning one doesn't leak into the other (v1 single-shot prompting showed
this leakage: loosening the match/mismatch criterion also made the model less
skeptical about realism, causing it to miss obvious cartoon/game content it
had previously caught correctly):

  STAGE 1 (strict, label-independent): is this real-world footage, or
  cartoon/animated/video-game/CGI content? This check never sees the label
  and is never loosened -- it's a pure realism judgment.

  STAGE 2 (generalist, only run if Stage 1 says real): does the label's
  GENERAL sound category plausibly match what's audible? Deliberately does
  NOT require species/sub-type-level acoustic confirmation (e.g. a specific
  bird species, "engine KNOCKING" vs. just engine noise) -- human review
  showed demanding that level of precision produces a flood of false
  positives on fine-grained VGGSound labels that neither a human nor a model
  can reliably verify from a short clip. Uses BOTH video and audio: a
  plausible visible source + a general-category audio match counts.

Final categories: REAL_MATCH / REAL_MISMATCH / SYNTHETIC / UNCERTAIN.
Only REAL_MATCH clips should proceed to the multi-granularity captioning
pass. This script does NOT decide that automatically -- validate against
known human judgments first (see --clip-ids-from-csv).

Usage:
    python scripts/qwen_omni_congruence_filter.py --n-clips 20 \\
        --out scripts/congruence_filter_20.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = ["REAL_MATCH", "REAL_MISMATCH", "SYNTHETIC", "UNCERTAIN"]

STAGE1_PROMPT = (
    "Look at this clip carefully. Is it real-world footage (an actual camera recording of "
    "the real world), or is it cartoon, animated, video-game, or CGI content? Judge purely "
    "from what you see and hear -- do not assume anything about what it's supposed to depict.\n"
    'Output ONLY a JSON object: {"is_real": true or false, "reason": "<one short sentence>"}'
)


def build_stage2_prompt(label: str) -> str:
    return (
        f"This clip's dataset label claims the sound is \"{label}\". You have already "
        f"confirmed this is real-world footage -- this check is ONLY about whether the "
        f"label's general sound category is genuinely audible, not about realism.\n"
        f"IMPORTANT -- be a GENERALIST, not a specialist: many labels are fine-grained or "
        f"species/sub-type-specific (e.g. a specific bird species, \"car engine KNOCKING\" vs. "
        f"just engine noise, \"BELLY laughing\" vs. just laughing). You are NOT expected to "
        f"acoustically verify the exact species or sub-type -- nobody, human or model, can "
        f"reliably do that from a short clip. Instead: if the video shows a plausible source for "
        f"the label (e.g. a bird, the claimed animal, the claimed instrument, the claimed "
        f"machine) AND you hear a sound in the GENERAL category the label describes (e.g. some "
        f"bird call, some engine/mechanical noise, some laughing, some percussive rhythm), that "
        f"counts as a match -- even faint or brief, and even if you can't confirm the exact "
        f"species/sub-type wording.\n"
        f"Only call it a mismatch when the GENERAL category itself is wrong -- e.g. the "
        f"claimed object/animal isn't there at all, the claimed device is visible but clearly not "
        f"running, or a completely different kind of sound dominates (e.g. only music, only "
        f"speech, only silence) with no trace of the claimed category.\n"
        f'Output ONLY a JSON object: {{"match": "MATCH" or "MISMATCH" or "UNCERTAIN", '
        f'"reason": "<one short sentence>"}}'
    )


def parse_json_response(text: str) -> dict:
    cleaned = re.sub(r"^```(json)?", "", text.strip()).strip()
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


def load_split_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def run_generate(model, processor, video_path: str, prompt_text: str) -> str:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        out_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                  return_audio=False, max_new_tokens=60,
                                  do_sample=False, repetition_penalty=1.15)
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=20)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--video-dir", default="/home/utkarsh/data/vggsound")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "scripts", "congruence_filter_20.jsonl"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--clip-ids-from-csv", default=None,
                    help="Reuse the exact clip set from a prior review CSV (validates against known human tags).")
    p.add_argument("--exclude-clips-from", default=None,
                    help="Comma-separated list of prior jsonl/csv files whose clip_ids to "
                         "exclude, for fresh unbiased coverage.")
    args = p.parse_args()

    if args.clip_ids_from_csv:
        with open(args.clip_ids_from_csv, newline="") as f:
            rows = [(r["clip_id"] + ".mp4", r["vggsound_label"]) for r in csv.DictReader(f)]
    else:
        csv_path = os.path.join(PROJECT_ROOT, "data", f"{args.split}.csv")
        rows = load_split_rows(csv_path)
        import random
        random.Random(args.seed).shuffle(rows)

        if args.exclude_clips_from:
            excluded = set()
            for path in args.exclude_clips_from.split(","):
                path = path.strip()
                if path.endswith(".csv"):
                    with open(path, newline="") as f:
                        excluded |= {r["clip_id"] for r in csv.DictReader(f)}
                elif path.endswith(".jsonl"):
                    with open(path) as f:
                        excluded |= {json.loads(l)["clip_id"] for l in f if l.strip()}
            rows = [(fname, label) for fname, label in rows
                    if os.path.splitext(fname)[0] not in excluded]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[congruence] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[congruence] model loaded.", flush=True)

    n_written = 0
    t_start = time.time()
    with open(args.out, "w") as fout:
        for fname, label in rows:
            if n_written >= args.n_clips:
                break
            video_path = os.path.join(args.video_dir, fname)
            if not os.path.isfile(video_path):
                continue
            clip_id = os.path.splitext(fname)[0]
            t0 = time.time()

            # ── Stage 1: strict, label-independent realism check ───────────
            resp1 = run_generate(model, processor, video_path, STAGE1_PROMPT)
            parsed1 = parse_json_response(resp1)

            rec = {"clip_id": clip_id, "vggsound_label": label}
            if "parse_error" in parsed1:
                rec.update({"category": None, "reason": None,
                            "stage1_parse_error_raw": parsed1["raw_text"]})
            elif parsed1.get("is_real") is False:
                rec.update({"category": "SYNTHETIC", "reason": parsed1.get("reason")})
            else:
                # ── Stage 2: generalist match check, only for real footage ─
                resp2 = run_generate(model, processor, video_path, build_stage2_prompt(label))
                parsed2 = parse_json_response(resp2)
                if "parse_error" in parsed2:
                    rec.update({"category": None, "reason": None,
                                "stage2_parse_error_raw": parsed2["raw_text"]})
                else:
                    match = parsed2.get("match")
                    category = {"MATCH": "REAL_MATCH", "MISMATCH": "REAL_MISMATCH",
                                "UNCERTAIN": "UNCERTAIN"}.get(match, None)
                    rec.update({"category": category, "reason": parsed2.get("reason")})

            elapsed = time.time() - t0
            rec["gen_seconds"] = round(elapsed, 2)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_written += 1
            print(f"[congruence] {n_written}/{args.n_clips}  ({elapsed:.1f}s)  {clip_id}  "
                  f"[{label}] -> {rec['category']}  ({rec['reason']})", flush=True)

    print(f"[congruence] DONE. wrote {n_written} clips to {args.out}  "
          f"total_wall_seconds={time.time()-t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
