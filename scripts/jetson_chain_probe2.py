"""scripts/jetson_chain_probe2.py — probe 1 falsified the "speaker ignores the thinker"
story. Isolate what actually breaks the chain.

WHAT PROBE 1 SETTLED. Feeding scene+thought in ONE user turn (the current format) is the
only shape that grounds this model; system/assistant history turns are ignored outright.
So the speaker prompt is NOT the bug.

WHAT THE LIVE RUN ACTUALLY SHOWED, re-read:
  round 1: perception "a person lying down" -> thinker GROUNDED -> speaker generic
  round 2: perception "a person sitting"    -> thinker GENERIC  -> speaker echoed it faithfully
  round 3: identical to round 2 (greedy decoding, identical input)
So rounds 2-3 are a THINKER failure, not a speaker failure. Two suspects remain:

  H1  TRUNCATION. The test passes `think[:200]`, and the thought is 213 chars, so the plan
      clause gets cut mid-sentence. Probe 1 passed the FULL thought and got a grounded line.
  H2  PERCEPTION IS TOO THIN. `ask()` returns the single top-1 tag -- "a person sitting" is
      the entire world model handed to a reasoning model. `ask_topk` already exists; a
      handful of tags ("a person sitting, a desk, an office chair, dim lighting") is a
      scene, whereas one tag is a bare fact with nothing to reason about.

H2 is the interesting one because it is a REGRESSION INTRODUCED BY MOVING TO TAGS: a
121k-caption bank returned a whole descriptive sentence, while a tag vocabulary returns one
short phrase. Tags won on precision (they do not hallucinate clarinet players) but a single
tag may carry less for the LLM to work with than a full caption did. That trade was never
measured -- this measures it.

Run on the Jetson (fast: no perception stack, no camera):
    python3 scripts/jetson_chain_probe2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

FULL_THOUGHT = ("I see a person lying down, and I'm curious about them. Maybe they're resting "
                "or just want to be quiet - I should ask if they need anything while I can "
                "keep the mood bright for Finn")

# top-1 vs top-k, as ask() vs ask_topk() would return them
SCENES = {
    "top1_sitting":  "a person sitting",
    "top1_lying":    "a person lying down",
    "topk_tags":     "a person sitting, a desk, an office chair, dim lighting, a cluttered room",
    "caption_style": ("a person seated on an office chair in a dimly lit room with a desk and "
                      "clutter around them"),
}

GROUND_KEYS = {"sit", "sitting", "lying", "rest", "resting", "quiet", "desk", "chair",
               "dim", "dark", "light", "clutter", "room", "comfortable", "need", "okay",
               "tired", "work", "working"}


def grounded(text: str) -> bool:
    return any(k in text.lower() for k in GROUND_KEYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-gguf", default=f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf")
    ap.add_argument("--fast-tok", default=f"{PROD}/tokenizers/lfm25_350m_tok")
    ap.add_argument("--think-gguf", default=f"{PROD}/models_gguf/bmo_thinker_qwen3_v3_Q8_0.gguf")
    ap.add_argument("--think-tok", default=f"{PROD}/tokenizers/qwen3_thinker_tok")
    ap.add_argument("--out", default=os.path.expanduser("~/chain_probe2.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, GGUFReasoningTier
    fast = GGUFFastTier(args.fast_gguf, AutoTokenizer.from_pretrained(args.fast_tok),
                        max_new_tokens=48, n_gpu_layers=-1)
    think = GGUFReasoningTier(args.think_gguf, AutoTokenizer.from_pretrained(args.think_tok),
                              max_new_tokens=320, n_gpu_layers=-1)
    print("[probe2] both tiers loaded\n", flush=True)
    state = {"energy": 0.6, "mood": "curious"}
    R = {"H1_truncation": [], "H2_scene_richness": []}

    # ---- H1: does truncating the thought at 200 chars break the speaker? ----
    print("=== H1: speaker, FULL vs TRUNCATED thought (same scene) ===", flush=True)
    for tag, th in (("full", FULL_THOUGHT), ("trunc200", FULL_THOUGHT[:200])):
        p = (f"You see: a person sitting Your thought: {th} Say one short friendly line "
             f"to the person.")
        r = fast.generate(p, state)
        t = getattr(r, "text", str(r))
        print(f"── {tag:9s} grounded={grounded(t)}\n   {t}\n", flush=True)
        R["H1_truncation"].append({"thought": tag, "text": t, "grounded": grounded(t)})

    # ---- H2: how much does scene richness change the THINKER? ----
    print("=== H2: thinker, top-1 tag vs top-k tags vs caption ===", flush=True)
    for name, scene in SCENES.items():
        t0 = time.time()
        tr = think.generate(f"You can see: {scene} The user just walked in. What should you "
                            f"say and why?", state)
        th = getattr(tr, "text", str(tr))
        ms = (time.time() - t0) * 1000
        fr = fast.generate(f"You see: {scene} Your thought: {th} Say one short friendly "
                           f"line to the person.", state)
        sd = getattr(fr, "text", str(fr))
        print(f"── {name}  (thinker {ms:.0f} ms)")
        print(f"   SCENE   {scene}")
        print(f"   THINKER [grounded={grounded(th)}] {th[:200]}")
        print(f"   SAYS    [grounded={grounded(sd)}] {sd}\n", flush=True)
        R["H2_scene_richness"].append({"scene_kind": name, "scene": scene, "thinker": th,
                                       "said": sd, "thinker_grounded": grounded(th),
                                       "said_grounded": grounded(sd), "thinker_ms": ms})

    with open(args.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"[probe2] wrote {args.out}")


if __name__ == "__main__":
    main()
