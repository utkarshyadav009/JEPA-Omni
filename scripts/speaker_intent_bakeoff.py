"""scripts/speaker_intent_bakeoff.py — does the speaker follow the thinker, and did v5 regress?

THE CLAIM UNDER TEST (user, 2026-08-16): *"the issue is the speaker being too stupid... I swear
v2 and v1 were better."* On-device the thinker reasons coherently and the speaker replies with a
non-sequitur:

    reasoning: ...The person in front of me has glasses, so maybe they're trying to stay quiet...
    answer:    Hey there! I'm listening to your screen - what's on?
    SPEAKER:   "Your glasses are shining like a secret menu!"

TWO COMPETING EXPLANATIONS, and they demand opposite fixes:

  (A) REGRESSION. v3->v5 corpus work (expand_name_stranger, v10c/d/e, per-scenario balancing)
      optimised idiom density and traded away instruction-following. Fix = go back / retrain.
  (B) FORMAT GAP. The speaker was NEVER trained to follow an instruction, so no version can,
      and we have been sending it something wildly out of distribution. Fix = corpus format,
      and rolling back would achieve nothing.

Corpus evidence already favours (B) and this script exists to confirm it on the real models:
  * median speaker prompt is **38 chars**, p90 66, and the LONGEST prompt in the entire 3,774-row
    corpus is **120**. The demo sends **453**.
  * only 126/3774 rows (3.3%) contain a scene at all, all `perception_grounded`, and every one
    of them is `short scene + a REAL USER UTTERANCE` ("You can see: two people, a couch, a living
    room. Let's play something."). The scene is context; the user's line is what BMO answers.
  * **zero rows** contain a thinker instruction, under any key, in any version.

So the design is: every version x BOTH prompt formats x the same fixtures. That factorisation is
the point -- comparing versions alone cannot separate (A) from (B).

  LONG  = what the pipeline sends today (full scene dump + 260 chars of CoT prose)
  SHORT = a compact directive in the shape the corpus actually uses

If every version fails LONG and every version improves on SHORT, the cause is (B) and no amount
of retraining on the current format would have helped. If v1/v2 beat v5 on BOTH, it is (A).

Scoring is rule-based per fixture, not vibes: each fixture declares what a compliant line MUST
contain and what it must NOT. Rules are deliberately generous on phrasing (an earlier verifier in
this repo falsely rejected 93 valid rows by demanding exact wording -- see expand_name_stranger.py)
and only check that the INTENT survived.

Usage (on the Jetson):
    python3 scripts/speaker_intent_bakeoff.py --out ~/speaker_bakeoff.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

SCENE = ("wearing: a person wearing headphones; doing: someone is watching a screen; "
         "who: a person wearing glasses; where: a home office; lighting: dim lighting")

# (name, thinker CoT prose as the real thinker emits it, compact directive, MUST, MUST_NOT)
FIXTURES = [
    (
        "ask_name",
        "I do not recognise this person and nobody is enrolled in memory, so this is a first "
        "meeting. The most useful thing I can do is find out who they are before anything else, "
        "otherwise I cannot remember them next time.",
        "You have never met them. Ask what their name is.",
        re.compile(r"(your name|who are you|what should i call you|what do (they|people) call you"
                   r"|call you\?|name\?|got a name|go by|introduce)", re.I),
        None,
    ),
    (
        "dont_interrupt",
        "They are concentrating hard on the screen and have headphones on. Interrupting with a "
        "game or a question would be intrusive. The right move is a brief, quiet acknowledgement "
        "that does not ask them for anything at all.",
        "They are focused and busy. Say something quiet. Do not ask a question.",
        None,
        re.compile(r"\?"),          # any question at all violates the instruction
    ),
    (
        "notice_jumper",
        "They are wearing a red jumper today, which is new. Noticing what someone is wearing is a "
        "warm, low-cost thing to do, and it shows I am actually paying attention to them.",
        "Warmly compliment their red jumper.",
        re.compile(r"(jumper|sweater|red)", re.I),
        None,
    ),
    (
        "suggest_break",
        "They have been at the screen a long time in a dim room and they look tired. Rather than "
        "starting something new, the caring move is to suggest they rest for a moment.",
        "They look tired. Gently suggest they take a break.",
        re.compile(r"(break|rest|pause|stretch|breathe|step away|sit back|lie down|nap|sleep)", re.I),
        None,
    ),
    (
        "greet_known",
        "I recognise this person as Sam with high confidence, and we last spoke yesterday evening. "
        "Greeting them by name is the natural thing and confirms I remember them.",
        "You recognise Sam. Greet them by name.",
        re.compile(r"\bsam\b", re.I),
        None,
    ),
    (
        "ask_what_doing",
        "I can see they are looking at a screen but I genuinely cannot tell what they are working "
        "on, and guessing would risk being wrong. Better to ask them directly what it is.",
        "You cannot tell what they are working on. Ask them.",
        re.compile(r"\?"),
        re.compile(r"\b(i can see you are (coding|gaming|writing)|you're definitely)\b", re.I),
    ),
]

LONG_TMPL = ("You can see: {scene}. Your private thinking: {cot} "
             "Now say one short line out loud to them.")
SHORT_TMPL = "{directive}"


def score(text: str, must, must_not) -> bool:
    if must is not None and not must.search(text):
        return False
    if must_not is not None and must_not.search(text):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="v1,v2,v3,v5")
    ap.add_argument("--tok", default=f"{PROD}/tokenizers/lfm25_350m_tok")
    ap.add_argument("--out", default=os.path.expanduser("~/speaker_bakeoff.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier

    tok = AutoTokenizer.from_pretrained(args.tok)
    results, table = [], {}

    for v in args.versions.split(","):
        path = f"{PROD}/models_gguf/bmo_lfm25_350m_{v}_Q8_0.gguf"
        if not os.path.exists(path):
            print(f"[bakeoff] {v}: MISSING {path}", flush=True)
            continue
        # n_ctx=512 matches production; the LONG prompt is ~130 tokens so it fits, but note
        # that it is ~4x anything the model saw in training regardless of context size.
        spk = GGUFFastTier(path, tok, max_new_tokens=48, n_gpu_layers=-1, n_ctx=512)
        print(f"\n===== {v} =====", flush=True)

        for fmt in ("LONG", "SHORT"):
            passed = 0
            for name, cot, directive, must, must_not in FIXTURES:
                prompt = (LONG_TMPL.format(scene=SCENE, cot=cot) if fmt == "LONG"
                          else SHORT_TMPL.format(directive=directive))
                t = time.time()
                r = spk.generate(prompt, {"energy": 0.6, "mood": "curious"})
                ok = score(r.text, must, must_not)
                passed += ok
                print(f"  [{fmt:5s}] {name:16s} {'PASS' if ok else 'FAIL'}  "
                      f"({(time.time()-t)*1000:.0f}ms)  {r.text[:96]}", flush=True)
                results.append({"version": v, "format": fmt, "fixture": name,
                                "pass": bool(ok), "text": r.text,
                                "prompt_chars": len(prompt)})
            table[f"{v}/{fmt}"] = f"{passed}/{len(FIXTURES)}"
            print(f"  [{fmt:5s}] TOTAL {passed}/{len(FIXTURES)}", flush=True)

        del spk
        gc.collect()

    print("\n" + "=" * 62)
    print(f"{'model/format':<18}{'intent adherence':>20}")
    for k in sorted(table):
        print(f"{k:<18}{table[k]:>20}")
    json.dump({"table": table, "rows": results}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
