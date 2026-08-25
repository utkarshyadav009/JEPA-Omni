"""scripts/qwen_omni_sound_verify_fix.py — vision-anchored verification and
correction pass over the existing gpt_sound_acoustic captions in
scripts/qwen_omni_full_captions.jsonl.

Root cause this fixes: the original sound-caption call
(qwen_omni_congruence_filter.py's run_generate) already receives the FULL
video + audio (USE_AUDIO_IN_VIDEO=True) for every call, including the sound
call -- the earlier "isolated audio-only" premise was checked directly
against the code and found FALSE. The prompt-level instruction ("describe
ONLY what you hear") does not prevent the model from genuinely mishearing
or hallucinating despite having vision available (confirmed example:
a welding clip's gpt_sound_acoustic described "an electric toothbrush").

Fix: re-present the clip to Qwen2.5-Omni-7B ALONGSIDE its own already-
generated (and independently found reliable) visual captions + the
VGGSound label, and explicitly ask it to check the existing sound caption
for physical-plausibility consistency with the visible scene, re-listening
and correcting if it's wrong -- rather than generating from scratch blind.
One model call per clip (not two), reusing the already-good visual fields
as-is (untouched).

Output: same clip_id, all 4 original visual fields UNCHANGED, plus:
  - gpt_sound_acoustic_v2 (the corrected/confirmed caption)
  - sound_verify_verdict ("consistent" | "corrected")
  - gpt_sound_acoustic_v1 (the original, kept for diffing/audit)
Never overwrites the original file -- writes to a new output path.

Usage (per-GPU shard):
    CUDA_VISIBLE_DEVICES=0 python scripts/qwen_omni_sound_verify_fix.py --shard-idx 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions.jsonl")
VIDEO_DIR = "/home/utkarsh/data/vggsound"


def build_verify_prompt(label: str, rec: dict) -> str:
    return (
        f"This clip's sound is labeled \"{label}\". Here is what is visually happening in this "
        f"clip, already verified correct:\n"
        f"- Action: {rec.get('gpt_action_detailed', '')}\n"
        f"- Scene: {rec.get('gpt_summary_detailed', '')}\n\n"
        f"A separate pass produced this description of the clip's SOUND: "
        f"\"{rec.get('gpt_sound_acoustic', '')}\"\n\n"
        f"Listen to the actual audio in this clip carefully. Check: is that sound description "
        f"physically consistent with what is visibly happening (a plausible real-world sound "
        f"a viewer would actually hear from this visible scene/action)? If it is consistent, "
        f"output it again unchanged. If it is WRONG (describes a different sound than what you "
        f"actually hear, or a sound that doesn't match a visible cause), listen again and output "
        f"a corrected 1-2 sentence description of the sound itself and its likely source, based "
        f"on what you actually hear AND what is visibly happening. "
        f"Output ONLY a JSON object (no markdown fences, no other text) with exactly these fields:\n"
        f'- "consistent": true or false (was the original sound description correct?)\n'
        f'- "sound_acoustic": the final (unchanged or corrected) 1-2 sentence sound description.\n'
        f"Output strictly valid JSON."
    )


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


def run_generate(model, processor, video_path: str, prompt_text: str, max_new_tokens: int = 200) -> str:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--video-dir", default=VIDEO_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    out_path = args.out or os.path.join(PROJECT_ROOT, "scripts", f"qwen_omni_sound_verify_fix_shard{args.shard_idx}.jsonl")

    with open(IN_PATH) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    my_recs = all_recs[args.shard_idx::args.num_shards]
    if args.limit is not None:
        my_recs = my_recs[:args.limit]
    print(f"[sound-verify] shard {args.shard_idx}/{args.num_shards}: {len(my_recs)}/{len(all_recs)} clips", flush=True)

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
        print(f"[sound-verify] resuming: {len(already_done)} already done", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sound-verify] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[sound-verify] model loaded.", flush=True)

    n_written, n_corrected, n_missing_video, n_errors = 0, 0, 0, 0
    t_start = time.time()
    with open(out_path, "a") as fout:
        for i, rec in enumerate(my_recs):
            clip_id = rec["clip_id"]
            if clip_id in already_done:
                continue
            video_path = os.path.join(args.video_dir, clip_id + ".mp4")
            if not os.path.isfile(video_path):
                n_missing_video += 1
                continue
            try:
                resp = run_generate(model, processor, video_path,
                                     build_verify_prompt(rec.get("vggsound_label", ""), rec))
                parsed = parse_json_response(resp)
            except Exception as e:
                n_errors += 1
                print(f"[sound-verify] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            # Model doesn't always follow the exact requested key name (observed
            # "sound_description" in place of "sound_acoustic" during testing) --
            # accept the common variants rather than losing a real, correct answer.
            new_sound = (parsed.get("sound_acoustic") or parsed.get("sound_description")
                         or parsed.get("sound") or parsed.get("description"))
            consistent = parsed.get("consistent")
            verdict = "consistent" if consistent is True else ("corrected" if new_sound else "parse_error")
            if verdict == "corrected":
                n_corrected += 1

            out_rec = {
                "clip_id": clip_id,
                "vggsound_label": rec.get("vggsound_label"),
                "gpt_action_brief": rec.get("gpt_action_brief"),
                "gpt_action_detailed": rec.get("gpt_action_detailed"),
                "gpt_summary_brief": rec.get("gpt_summary_brief"),
                "gpt_summary_detailed": rec.get("gpt_summary_detailed"),
                "gpt_sound_acoustic_v1": rec.get("gpt_sound_acoustic"),
                "gpt_sound_acoustic_v2": new_sound if new_sound else rec.get("gpt_sound_acoustic"),
                "sound_verify_verdict": verdict,
            }
            if "parse_error" in parsed:
                out_rec["parse_error_raw"] = parsed.get("raw_text")
            fout.write(json.dumps(out_rec) + "\n")
            fout.flush()
            n_written += 1
            if n_written % 50 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed
                eta_min = (len(my_recs) - len(already_done) - n_written) / rate / 60 if rate > 0 else float("nan")
                print(f"[sound-verify] shard {args.shard_idx}: {n_written}/{len(my_recs)-len(already_done)} "
                      f"corrected={n_corrected} errors={n_errors} missing_video={n_missing_video} "
                      f"rate={rate:.2f}/s eta={eta_min:.0f}min", flush=True)

    print(f"[sound-verify] shard {args.shard_idx} DONE: written={n_written} corrected={n_corrected} "
          f"errors={n_errors} missing_video={n_missing_video}", flush=True)


if __name__ == "__main__":
    main()
