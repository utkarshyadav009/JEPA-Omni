"""scripts/easycom_caption_probe.py — ONE-SHOT verification that EasyCom
video+audio is usable by our existing Qwen-Omni captioning pipeline, and to
produce one REAL example of what a genuinely-paired (scene caption, speech
segment) example looks like. Reuses run_generate/SOUND_PROMPT unmodified
from the existing pipeline -- this is a feasibility probe, not a new
dataset-construction pipeline.

Usage:
    python scripts/easycom_caption_probe.py --session 1 --chunk 00-00-000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from qwen_omni_congruence_filter import run_generate  # noqa: E402
from qwen_omni_multigranularity_v2 import SOUND_PROMPT  # noqa: E402

EASYCOM_ROOT = "/home/utkarsh/raid2-data/easycom/extracted/Main"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--chunk", default="00-00-000")
    args = p.parse_args()

    video_path = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{args.session}", f"{args.chunk}.mp4")
    trans_path = os.path.join(EASYCOM_ROOT, "Speech_Transcriptions", f"Session_{args.session}", f"{args.chunk}.json")
    assert os.path.isfile(video_path), video_path
    assert os.path.isfile(trans_path), trans_path

    try:
        with open(trans_path, encoding="utf-8") as f:
            segments = json.load(f)
    except UnicodeDecodeError:
        with open(trans_path, encoding="latin-1") as f:
            segments = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[probe] loading Qwen2.5-Omni-7B...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[probe] model loaded, running scene caption on the full 60s chunk...", flush=True)

    scene_caption = run_generate(model, processor, video_path, SOUND_PROMPT)

    print(f"\n=== EasyCom Session_{args.session} / {args.chunk}.mp4 (60s chunk) ===")
    print(f"SCENE CAPTION (from video+array audio, existing pipeline, unmodified):")
    print(f"  {scene_caption}")
    print(f"\n{len(segments)} speech segments in this chunk. First 5 paired examples:")
    for seg in segments[:5]:
        print(f"  [{seg.get('Start_Frame')}-{seg.get('End_Frame')} frames, "
              f"Participant_ID={seg.get('Participant_ID')}] {seg.get('Transcription', '')!r}")

    out = {"session": args.session, "chunk": args.chunk, "scene_caption": scene_caption,
           "n_segments": len(segments), "segments": segments}
    out_path = "checkpoints/m4_joint/easycom_caption_probe_example.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
