"""scripts/qwen_omni_sound_verify_cleanup.py — targeted re-run of
qwen_omni_sound_verify_fix.py's failure mode: ~8.1% (15,302/188,657) of clips
either (a) hit a JSON parse_error (verification never actually completed, still
carrying the unverified original v1 caption) or (b) got a "corrected" caption
that is pure negation/hedge with no actual description of the sound (e.g.
"The sound described does not match the visual content. There is no indication
of an 'air horn' being blown.") -- useless as an M3 training caption.

Fix: same vision-anchored verification setup as qwen_omni_sound_verify_fix.py,
but the prompt now explicitly FORBIDS a negation-only answer -- the model must
always commit to a concrete best-guess description of the actual audible sound,
even under uncertainty. Restricted to the exact target clip_id list (not the
full corpus) via --clip-ids-file.

Usage (per-GPU shard, GPU 0 excluded on purpose -- kept free):
    CUDA_VISIBLE_DEVICES=1 python scripts/qwen_omni_sound_verify_cleanup.py --shard-idx 0 --num-shards 3
    CUDA_VISIBLE_DEVICES=2 python scripts/qwen_omni_sound_verify_cleanup.py --shard-idx 1 --num-shards 3
    CUDA_VISIBLE_DEVICES=3 python scripts/qwen_omni_sound_verify_cleanup.py --shard-idx 2 --num-shards 3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions.jsonl")
TARGET_IDS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_sound_cleanup_target_ids.txt")
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
        f"output it again unchanged. If it is WRONG, listen again and output a corrected 1-2 "
        f"sentence description of the sound itself and its likely source.\n\n"
        f"CRITICAL RULE: your \"sound_acoustic\" field must ALWAYS be a concrete, positive "
        f"description of an actual sound you hear -- e.g. \"a low mechanical hum with periodic "
        f"clicking\" or \"footsteps crunching on gravel\". NEVER write a negation-only answer "
        f"like \"this does not match the visual content\" or \"there is no indication of X\" or "
        f"\"the audio does not contain Y\" with nothing else -- that is not a description and is "
        f"useless. Even if you are uncertain, or the sound seems to mismatch the scene, you must "
        f"still commit to your single best concrete guess at what the actual audible sound is.\n\n"
        f"Output ONLY a JSON object (no markdown fences, no other text) with exactly these fields:\n"
        f'- "consistent": true or false (was the original sound description correct?)\n'
        f'- "sound_acoustic": the final concrete (unchanged or corrected) 1-2 sentence sound '
        f"description. Never a negation-only sentence.\n"
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
    out_path = args.out or os.path.join(PROJECT_ROOT, "scripts", f"qwen_omni_sound_cleanup_shard{args.shard_idx}.jsonl")

    with open(TARGET_IDS_PATH) as f:
        target_ids = set(l.strip() for l in f if l.strip())

    with open(IN_PATH) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    all_recs = [r for r in all_recs if r["clip_id"] in target_ids]
    my_recs = all_recs[args.shard_idx::args.num_shards]
    if args.limit is not None:
        my_recs = my_recs[:args.limit]
    print(f"[sound-cleanup] shard {args.shard_idx}/{args.num_shards}: {len(my_recs)}/{len(all_recs)} "
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
        print(f"[sound-cleanup] resuming: {len(already_done)} already done", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sound-cleanup] loading Qwen2.5-Omni-7B on {device}...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[sound-cleanup] model loaded.", flush=True)

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
                print(f"[sound-cleanup] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            new_sound = (parsed.get("sound_acoustic") or parsed.get("sound_description")
                         or parsed.get("sound") or parsed.get("description")
                         or parsed.get("sound_audible"))
            if not new_sound:
                # model key-naming drifts (observed: sound_audible, others likely) --
                # fall back to any other string-valued field besides the known non-caption keys.
                for k, v in parsed.items():
                    if k in ("consistent", "parse_error", "raw_text"):
                        continue
                    if isinstance(v, str) and v.strip():
                        new_sound = v
                        break
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
                print(f"[sound-cleanup] shard {args.shard_idx}: {n_written}/{len(my_recs)-len(already_done)} "
                      f"corrected={n_corrected} errors={n_errors} missing_video={n_missing_video} "
                      f"rate={rate:.2f}/s eta={eta_min:.0f}min", flush=True)

    print(f"[sound-cleanup] shard {args.shard_idx} DONE: written={n_written} corrected={n_corrected} "
          f"errors={n_errors} missing_video={n_missing_video}", flush=True)


if __name__ == "__main__":
    main()
