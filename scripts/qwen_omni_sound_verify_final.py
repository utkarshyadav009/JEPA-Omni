"""scripts/qwen_omni_sound_verify_final.py — final mop-up pass on the 420 clips
that survived qwen_omni_sound_verify_cleanup.py's tightened prompt still hedging
(e.g. "there is no mention of a mandolin being played", "there is no sound
described here") or still parse_error. These resisted one round of "don't
negate" instruction, so this pass adds: (1) an explicit banned-word list
checked programmatically, not just requested, and (2) up to 2 in-process
retries with an escalating correction message when a banned word is detected,
before giving up and flagging for manual review.

Usage (GPU 0 excluded on purpose):
    CUDA_VISIBLE_DEVICES=1 python scripts/qwen_omni_sound_verify_final.py --shard-idx 0 --num-shards 3
    CUDA_VISIBLE_DEVICES=2 python scripts/qwen_omni_sound_verify_final.py --shard-idx 1 --num-shards 3
    CUDA_VISIBLE_DEVICES=3 python scripts/qwen_omni_sound_verify_final.py --shard-idx 2 --num-shards 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions.jsonl")
TARGET_IDS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_sound_final_target_ids.txt")
VIDEO_DIR = "/home/utkarsh/data/vggsound"

BANNED_WORD_PAT = re.compile(
    r"\bno\b|\bnot\b|\bisn'?t\b|\bdoesn'?t\b|\bdon'?t\b|\bcannot\b|\bcan'?t\b|"
    r"\bunable\b|\bincorrect\b|\bmismatch\b|\bmention\b|\bindication\b|\bevidence\b|"
    r"\bdescribed incorrectly\b",
    re.IGNORECASE,
)


def build_prompt(label: str, rec: dict, retry_note: str = "") -> str:
    base = (
        f"This clip's sound is labeled \"{label}\". Here is what is visually happening in this "
        f"clip, already verified correct:\n"
        f"- Action: {rec.get('gpt_action_detailed', '')}\n"
        f"- Scene: {rec.get('gpt_summary_detailed', '')}\n\n"
        f"Listen to the actual audio in this clip carefully and describe the ONE real sound you "
        f"hear most clearly, in 1-2 sentences.\n\n"
        f"STRICT FORMAT RULE: your answer must be a confident, concrete, POSITIVE description of "
        f"a real-world sound and its likely source only -- e.g. \"a metallic clanging as pots are "
        f"stacked together\" or \"a soft continuous hiss of running water\". "
        f"Your answer must NOT contain any of these words or ideas anywhere: no, not, isn't, "
        f"doesn't, don't, cannot, can't, unable, incorrect, mismatch, mention, indication, "
        f"evidence. Do not discuss whether a label or prior caption was right or wrong. Do not "
        f"describe what is absent. Only describe the actual sound you hear, as a plain positive "
        f"statement, even if you are not fully certain -- commit to your single best guess.\n"
        f"{retry_note}\n"
        f"Output ONLY a JSON object (no markdown fences, no other text) with exactly one field:\n"
        f'- "sound_acoustic": your 1-2 sentence positive sound description.\n'
        f"Output strictly valid JSON."
    )
    return base


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"parse_error": True, "raw_text": text}


def extract_sound(parsed: dict) -> str:
    v = (parsed.get("sound_acoustic") or parsed.get("sound_description")
         or parsed.get("sound") or parsed.get("description") or parsed.get("sound_audible"))
    if not v:
        for k, val in parsed.items():
            if k in ("consistent", "parse_error", "raw_text"):
                continue
            if isinstance(val, str) and val.strip():
                v = val
                break
    return v or ""


def run_generate(model, processor, video_path: str, prompt_text: str, max_new_tokens: int = 150) -> str:
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
                                  return_audio=False, max_new_tokens=max_new_tokens,
                                  do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=3)
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0].strip()


def generate_clean(model, processor, video_path: str, label: str, rec: dict, max_retries: int = 2):
    retry_note = ""
    last_sound, last_parsed = "", {}
    for attempt in range(max_retries + 1):
        prompt = build_prompt(label, rec, retry_note)
        resp = run_generate(model, processor, video_path, prompt)
        parsed = parse_json_response(resp)
        sound = extract_sound(parsed)
        last_sound, last_parsed = sound, parsed
        if sound and not BANNED_WORD_PAT.search(sound):
            return sound, parsed, attempt
        retry_note = (
            f"\nYour previous answer was: \"{sound}\" -- this contained a forbidden hedging "
            f"word or was empty. Rewrite from scratch using ONLY a concrete positive description "
            f"of an actual sound. No hedging, no negation, no meta-commentary."
        )
    return last_sound, last_parsed, max_retries + 1  # exhausted retries, still bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--video-dir", default=VIDEO_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    out_path = args.out or os.path.join(PROJECT_ROOT, "scripts", f"qwen_omni_sound_final_shard{args.shard_idx}.jsonl")

    with open(TARGET_IDS_PATH) as f:
        target_ids = set(l.strip() for l in f if l.strip())
    with open(IN_PATH) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    all_recs = [r for r in all_recs if r["clip_id"] in target_ids]
    my_recs = all_recs[args.shard_idx::args.num_shards]
    if args.limit is not None:
        my_recs = my_recs[:args.limit]
    print(f"[sound-final] shard {args.shard_idx}/{args.num_shards}: {len(my_recs)}/{len(all_recs)} "
          f"target clips (of {len(target_ids)} total target)", flush=True)

    already_done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        already_done.add(json.loads(line)["clip_id"])
                    except json.JSONDecodeError:
                        pass
    if already_done:
        print(f"[sound-final] resuming: {len(already_done)} already done", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sound-final] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[sound-final] model loaded.", flush=True)

    n_written, n_fixed, n_still_bad, n_missing_video, n_errors = 0, 0, 0, 0, 0
    t_start = time.time()
    with open(out_path, "a") as fout:
        for rec in my_recs:
            clip_id = rec["clip_id"]
            if clip_id in already_done:
                continue
            video_path = os.path.join(args.video_dir, clip_id + ".mp4")
            if not os.path.isfile(video_path):
                n_missing_video += 1
                continue
            try:
                sound, parsed, attempts = generate_clean(
                    model, processor, video_path, rec.get("vggsound_label", ""), rec)
            except Exception as e:
                n_errors += 1
                print(f"[sound-final] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            still_bad = (not sound) or bool(BANNED_WORD_PAT.search(sound))
            if still_bad:
                n_still_bad += 1
            else:
                n_fixed += 1

            out_rec = {
                "clip_id": clip_id,
                "vggsound_label": rec.get("vggsound_label"),
                "gpt_action_brief": rec.get("gpt_action_brief"),
                "gpt_action_detailed": rec.get("gpt_action_detailed"),
                "gpt_summary_brief": rec.get("gpt_summary_brief"),
                "gpt_summary_detailed": rec.get("gpt_summary_detailed"),
                "gpt_sound_acoustic_v1": rec.get("gpt_sound_acoustic"),
                "gpt_sound_acoustic_v2": sound if sound else rec.get("gpt_sound_acoustic"),
                "sound_verify_verdict": "still_hedge_needs_manual_review" if still_bad else "corrected",
                "retry_attempts": attempts,
            }
            fout.write(json.dumps(out_rec) + "\n")
            fout.flush()
            n_written += 1
            if n_written % 25 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed
                eta_min = (len(my_recs) - len(already_done) - n_written) / rate / 60 if rate > 0 else float("nan")
                print(f"[sound-final] shard {args.shard_idx}: {n_written}/{len(my_recs)-len(already_done)} "
                      f"fixed={n_fixed} still_bad={n_still_bad} errors={n_errors} "
                      f"rate={rate:.2f}/s eta={eta_min:.0f}min", flush=True)

    print(f"[sound-final] shard {args.shard_idx} DONE: written={n_written} fixed={n_fixed} "
          f"still_bad={n_still_bad} errors={n_errors} missing_video={n_missing_video}", flush=True)


if __name__ == "__main__":
    main()
