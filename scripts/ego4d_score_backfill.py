"""scripts/ego4d_score_backfill.py — score additional Ego4D candidates
(beyond the original cap=50 pool) to backfill the 42k budget after
category exclusion (acoustic-environment, wearer-produced) + Music cap
removed ~13% of the original pool. Reuses the exact same scoring logic
as scripts/ego4d_av_relevance_filter.py.

Usage:
    python scripts/ego4d_score_backfill.py --n 10000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ego4d_av_relevance_filter import extract_audio, energy_cov, SPEECH_IDX, NOISE_LIKE_IDX


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--candidates", default="/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad/ego4d_new_candidates.json")
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/ego4d_backfill_scores.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backfill] device={device}", flush=True)

    candidates = json.load(open(args.candidates))
    rng = random.Random(0)
    rng.shuffle(candidates)
    candidates = candidates[: args.n]
    print(f"[backfill] scoring {len(candidates)} new candidates", flush=True)

    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    get_speech_timestamps = vad_utils[0]

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_fe = AutoFeatureExtractor.from_pretrained(ast_name)
    ast_model = AutoModelForAudioClassification.from_pretrained(ast_name).to(device).eval()

    scored = []
    if os.path.isfile(args.out):
        scored = json.load(open(args.out))
        print(f"[backfill] RESUMING: {len(scored)} already scored", flush=True)

    t_start = time.time()
    for i, c in enumerate(candidates):
        if i < len(scored):
            continue
        try:
            audio = extract_audio(c["path"], c["start_sec"])
            wav_t = torch.from_numpy(audio)
            speech_ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=16000)
            speech_samples = sum(t["end"] - t["start"] for t in speech_ts)
            vad_speech_frac = speech_samples / audio.shape[0]
            inputs = ast_fe(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = ast_model(**{k: v.to(device) for k, v in inputs.items()}).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            mask = np.ones(probs.shape[0], dtype=bool)
            mask[SPEECH_IDX] = False
            mask[NOISE_LIKE_IDX] = False
            top_event_idx = int(np.argmax(probs * mask))
            top_event_prob = float(probs[top_event_idx])
            speech_prob = float(probs[SPEECH_IDX].sum())
            cov = energy_cov(audio)
            cov_norm = min(1.0, cov / 2.0)
            score = top_event_prob * (1.0 - vad_speech_frac) * cov_norm
            scored.append({**c, "score": score, "top_event_idx": top_event_idx,
                            "top_event_prob": top_event_prob, "vad_speech_frac": vad_speech_frac,
                            "speech_prob": speech_prob, "energy_cov": cov})
        except Exception as e:
            print(f"[backfill] candidate {i} FAILED: {e}", flush=True)
            continue
        if (i + 1) % 500 == 0 or i == len(candidates) - 1:
            with open(args.out, "w") as f:
                json.dump(scored, f)
            print(f"[backfill] {len(scored)}/{len(candidates)} scored, "
                  f"elapsed={(time.time()-t_start)/60:.1f}min", flush=True)

    with open(args.out, "w") as f:
        json.dump(scored, f)
    print(f"[backfill] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
