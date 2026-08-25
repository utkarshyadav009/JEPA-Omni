"""scripts/jetson_speaker_prompt_probe.py — why does the speaker ignore the thinker?

OBSERVED (jetson_core_pipeline_test, 2026-08-15): perception said "a person sitting", the
thinker reasoned correctly ("I see a person lying down... maybe they're resting... I should
ask if they need anything"), and then the fast tier said "Finn, Jake - just how you like
it!" -- generic BMO flavour with zero grounding. Perception -> thinker works; thinker ->
speaker does not.

HYPOTHESIS. It is a PROMPT-FORMAT mismatch, not a capability limit. `GGUFFastTier.
_build_prompt_text` puts everything into ONE user turn:

    [energy=0.60 mood=curious] You see: <scene> Your thought: <reasoning> Say one short line.

But the fast tier is LFM2.5-350M LoRA'd on the v9 companion corpus, which is 92.7%
**(user utterance -> state-conditioned response) PAIRS**. The user turn is supposed to be
*something a person said*. Handing it a scene dump plus an instruction is out of
distribution, so it falls back to the personality prior it learned -- which is exactly the
"generic reply" failure mode the v9 corpus was built to fix in the first place.

This probes prompt SHAPES against the real deployed GGUF and prints what each produces. No
retraining, no perception stack, no camera -- so it loads in seconds and can be iterated on.

Run on the Jetson:
    python3 scripts/jetson_speaker_prompt_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

# Real material captured from the failing run, so this probes the actual case.
SCENE = "a person sitting"
THOUGHT = ("I see a person lying down, and I'm curious about them. Maybe they're resting or "
           "just want to be quiet - I should ask if they need anything while I can keep the "
           "mood bright for Finn")
UTTERANCE = "Hey BMO."


def variants(scene: str, thought: str, utt: str):
    """Each entry: (name, transcript, history). `history` rides the chat template as prior
    turns, which is the mechanism the corpus format actually supports."""
    plan = thought.split("I should", 1)[-1].strip() if "I should" in thought else thought
    return [
        # what the failing test did
        ("A_current_all_in_user",
         f"You see: {scene} Your thought: {thought} Say one short friendly line to the person.",
         None),
        # keep the user turn a real utterance; put grounding in a system turn
        ("B_system_grounding",
         utt,
         [{"role": "system", "content": f"You can see: {scene}. You are thinking: {thought}"}]),
        # grounding as the assistant's own prior thought, user turn stays an utterance
        ("C_assistant_thought",
         utt,
         [{"role": "assistant", "content": f"(thinking) {thought}"}]),
        # no scene dump: just ask it to voice the PLAN it already decided on
        ("D_voice_the_plan",
         f"Say this out loud in your own words, one short line: {plan}",
         None),
        # minimal grounding, no instruction verbs
        ("E_scene_only_utterance",
         utt,
         [{"role": "system", "content": f"You can see: {scene}."}]),
        # control: no grounding at all -- if this matches A, A carried no information
        ("F_control_no_grounding", utt, None),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf")
    ap.add_argument("--tok", default=f"{PROD}/tokenizers/lfm25_350m_tok")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=os.path.expanduser("~/speaker_prompt_probe.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier
    tok = AutoTokenizer.from_pretrained(args.tok)
    fast = GGUFFastTier(args.gguf, tok, max_new_tokens=args.max_new_tokens, n_gpu_layers=-1)
    print(f"[probe] fast tier loaded: {os.path.basename(args.gguf)}\n", flush=True)

    state = {"energy": 0.6, "mood": "curious"}
    out = []
    for name, transcript, history in variants(SCENE, THOUGHT, UTTERANCE):
        t0 = time.time()
        try:
            r = fast.generate(transcript, state, history=history)
            text = getattr(r, "text", str(r))
        except Exception as e:                       # history may be unsupported by template
            text = f"(ERROR: {type(e).__name__}: {e})"
        ms = (time.time() - t0) * 1000
        # crude grounding check: does the line reference anything the scene/plan mentioned?
        keys = {"sit", "sitting", "lying", "rest", "resting", "quiet", "person", "you okay",
                "need", "anything", "comfortable"}
        grounded = any(k in text.lower() for k in keys)
        print(f"── {name}  ({ms:.0f} ms)  grounded={grounded}")
        print(f"   {text}\n", flush=True)
        out.append({"variant": name, "text": text, "ms": ms, "grounded": grounded,
                    "transcript": transcript, "history": history})

    with open(args.out, "w") as f:
        json.dump({"scene": SCENE, "thought": THOUGHT, "results": out}, f, indent=2)
    print(f"[probe] wrote {args.out}")
    print("\nIf F (no grounding) reads like A, then A's grounding carried no information at "
          "all and the format -- not the model -- is the problem.")


if __name__ == "__main__":
    main()
