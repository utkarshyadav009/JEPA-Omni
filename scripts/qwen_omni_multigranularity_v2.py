"""scripts/qwen_omni_multigranularity_v2.py — Track B pipeline, root-cause fix.

Two changes vs. qwen_omni_multigranularity_pilot.py:

1. SYNTHETIC pre-filter (Stage 1 from qwen_omni_congruence_filter.py, the one
   part of the verifier experiments that proved reliable across every test):
   drop cartoon/animated/video-game/CGI clips before spending caption budget
   on them at all.

2. gpt_sound_acoustic is generated AUDIO-FIRST and label-free, in its OWN
   separate model call: "describe the sound and its likely source" -- no
   VGGSound label mentioned anywhere in this prompt. The prior design forced
   the label into the prompt as ground truth ("this clip's sound is X,
   describe X's sound"), which meant a wrong label (e.g. "tractor" on a
   faint-birds clip) got faithfully described as if true -- the label was
   POISONING the caption, not grounding it. Trusting the model's own hearing
   instead means label noise shows up as a visible mismatch between
   `vggsound_label` and `gpt_sound_acoustic` in the review app, rather than
   being silently baked into the caption.

   The four visual fields (gpt_action_*, gpt_summary_*) are UNCHANGED --
   label-injected, one combined JSON call, same as before. They were never
   the problem; only the acoustic field was.

Same anti-repetition generation config throughout: repetition_penalty=1.15,
no repeat_ngram_size (breaks structured JSON key names, see the pilot
script's docstring), same max_new_tokens cap.

STOPPING RULE: this is NOT trying to perfect audio-label matching -- that's
an open research problem, not a prompt-engineering exercise. This script's
only jobs are (a) drop synthetic content, (b) stop lying to the model about
what the label says. Whatever residual mismatch rate remains after that is
believed to be the same background label-noise level the M2 trunk already
trained through to 52% R@1 / 54% linear-probe.

Usage:
    python scripts/qwen_omni_multigranularity_v2.py --n-clips 50 --split test \\
        --out scripts/qwen_omni_multigranularity_v2_50.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_omni_congruence_filter import STAGE1_PROMPT, parse_json_response, run_generate  # noqa: E402

VISUAL_FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief", "gpt_summary_detailed"]

SOUND_PROMPT = (
    "Listen to this clip's audio. In 1-2 sentences, describe the sound itself (its tone, "
    "rhythm, texture) and its most likely source, based ONLY on what you actually hear. "
    "Do not guess based on the video alone if you can't actually hear a corresponding sound. "
    "Output plain text, not JSON -- just the description."
)


def build_visual_prompt(label: str) -> str:
    return (
        f"This clip's sound is labeled \"{label}\". Analyze the video, using BOTH what you see "
        f"and what you hear, and output ONLY a JSON object (no markdown fences, no other "
        f"text) with exactly these four string fields:\n"
        f'- "gpt_action_brief": one short sentence naming the main physical action.\n'
        f'- "gpt_action_detailed": 2-3 sentences describing the sequence of actions/movements in detail.\n'
        f'- "gpt_summary_brief": one short sentence summarizing the whole scene (setting, subject, action).\n'
        f'- "gpt_summary_detailed": a short paragraph (3-4 sentences) summarizing the scene, setting, subjects, and context.\n'
        f"Output strictly valid JSON."
    )


def load_split_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=None,
                    help="Cap on REAL (non-synthetic) clips to collect. Default: unlimited "
                         "(process the whole shard).")
    p.add_argument("--split", default="test", choices=["train", "test", "all"],
                    help="'all' combines train.csv + test.csv for full-corpus runs.")
    p.add_argument("--video-dir", default="/home/utkarsh/data/vggsound")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_multigranularity_v2_50.jsonl"))
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--exclude-clips-from", default=None,
                    help="Comma-separated jsonl/csv files whose clip_ids to exclude, for fresh coverage.")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--shard-idx", type=int, default=0, help="This process's shard index, for multi-GPU parallelism.")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards (e.g. one per GPU).")
    p.add_argument("--sub-idx", type=int, default=0,
                    help="Further index-based split of THIS shard's row list, for adding a "
                         "second GPU to help a lagging shard catch up mid-run. Pure list-"
                         "position split (deterministic, no race condition) -- pair with "
                         "--exclude-clips-from pointing at the original shard's --out file.")
    p.add_argument("--sub-of", type=int, default=1)
    args = p.parse_args()

    if args.split == "all":
        rows = (load_split_rows(os.path.join(PROJECT_ROOT, "data", "train.csv"))
                + load_split_rows(os.path.join(PROJECT_ROOT, "data", "test.csv")))
    else:
        rows = load_split_rows(os.path.join(PROJECT_ROOT, "data", f"{args.split}.csv"))
    random.Random(args.seed).shuffle(rows)

    if args.num_shards > 1:
        rows = rows[args.shard_idx::args.num_shards]

    if args.sub_of > 1:
        rows = rows[args.sub_idx::args.sub_of]

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
    print(f"[mg-v2] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[mg-v2] model loaded.", flush=True)

    # Resume support: a multi-day full-corpus run WILL get interrupted at some
    # point (crash, reboot, preemption) -- skip clip_ids already written to
    # --out from a prior invocation rather than reprocessing/duplicating them.
    already_done = set()
    if os.path.isfile(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    try:
                        already_done.add(json.loads(line)["clip_id"])
                    except json.JSONDecodeError:
                        pass
    if already_done:
        print(f"[mg-v2] resuming: {len(already_done)} clips already done in {args.out}", flush=True)

    n_written = 0
    n_synthetic_dropped = 0
    n_scanned = 0
    n_parse_errors = 0
    n_decode_errors = 0
    t_start = time.time()
    with open(args.out, "a") as fout:
        for fname, label in rows:
            if args.n_clips is not None and n_written >= args.n_clips:
                break
            clip_id = os.path.splitext(fname)[0]
            if clip_id in already_done:
                continue
            video_path = os.path.join(args.video_dir, fname)
            if not os.path.isfile(video_path):
                continue
            n_scanned += 1
            t0 = time.time()

            # At ~199k YouTube-scraped clips, some fraction WILL be corrupt/
            # truncated/undecodable -- both the primary (torchcodec) and
            # fallback (torchvision) decoders can fail on the same bad file.
            # A single bad clip must not take down an entire multi-day shard:
            # log it and move on, don't write a record (so it's naturally
            # retried on a future resume, e.g. if a decoder library upgrade
            # fixes it -- cheap either way if it's permanently unreadable).
            try:
                # ── Stage 1: SYNTHETIC pre-filter (label-independent) ───────
                resp1 = run_generate(model, processor, video_path, STAGE1_PROMPT)
                parsed1 = parse_json_response(resp1)
                if parsed1.get("is_real") is False:
                    n_synthetic_dropped += 1
                    print(f"[mg-v2] SKIP (synthetic): {clip_id}  [{label}]  ({parsed1.get('reason')})", flush=True)
                    continue

                # ── Visual fields: label-injected, unchanged ────────────────
                resp_visual = run_generate(model, processor, video_path, build_visual_prompt(label))
                parsed_visual = parse_json_response(resp_visual)

                # ── Sound field: audio-first, label-free, SEPARATE call ─────
                sound_text = run_generate(model, processor, video_path, SOUND_PROMPT)
            except Exception as e:
                n_decode_errors += 1
                print(f"[mg-v2] SKIP (error): {clip_id}  [{label}]  ({type(e).__name__}: {e})", flush=True)
                continue

            elapsed = time.time() - t0
            rec = {"clip_id": clip_id, "vggsound_label": label, "gen_seconds": round(elapsed, 2)}
            for field in VISUAL_FIELDS:
                rec[field] = parsed_visual.get(field, None)
            rec["gpt_sound_acoustic"] = sound_text
            if "parse_error" in parsed_visual:
                rec["visual_parse_error_raw"] = parsed_visual["raw_text"]
                n_parse_errors += 1

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_written += 1
            target = args.n_clips if args.n_clips is not None else len(rows)
            print(f"[mg-v2] {n_written}/{target}  ({elapsed:.1f}s)  {clip_id}  [{label}]  "
                  f"sound=\"{sound_text[:70]}\"", flush=True)

    total_elapsed = time.time() - t_start
    print(f"[mg-v2] DONE. wrote {n_written} clips this run "
          f"({len(already_done)} resumed from before), "
          f"scanned {n_scanned}, dropped {n_synthetic_dropped} synthetic, "
          f"decode_errors={n_decode_errors}, parse_errors={n_parse_errors}, "
          f"total_wall_seconds={total_elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
