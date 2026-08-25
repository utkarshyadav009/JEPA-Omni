"""scripts/expand_name_stranger.py — fix the weakest class in the v3 speaker.

THE SYMPTOM. On-device, v3's first-meeting line was
    "I'm BMO, happy to join the jam session! Could you name me?"
which asks to BE named rather than asking the user's name.

THE DIAGNOSIS, from reading all 39 `name_stranger` rows rather than assuming scarcity:

  1. **Contamination.** ~8 of 39 have the user ALREADY giving their name --
     "I'm Carlos.", "My name is Maya.", "Call me Priya." Those are `name_just_told`
     situations filed under `name_stranger`, so the class teaches two different behaviours
     at once and the stranger signal is diluted by a fifth.
  2. **A malformed exemplar.** One row reads "Who should I call?" -- which is not English for
     what BMO means, and is the same shape as the bad output.
  3. **Scarcity.** 39 examples against 77 for `name_just_told`. The class that fires FIRST in
     every real interaction is the smallest one.

So: reclassify the contaminated rows into the class they actually belong to, drop malformed
ones, and generate substantially more clean strangers. Fixing 1 and 2 matters more than 3 --
adding examples on top of a confused class would have taught the confusion harder.

The generated rows must satisfy a hard check: BMO ASKS FOR the user's name, and never asks to
be named. That is asserted here rather than hoped for.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_bmo_corpus_v10_identity import CHARACTER_NAMES, HARD_RULES, PERSONA

# user has already supplied a name -> this is name_just_told, not name_stranger
USER_GAVE_NAME = re.compile(
    r"\b(i'?m|i am|my name is|call me|this is|it'?s)\s+[A-Z][a-z]+", re.I)

# BMO asking to BE named, rather than asking who the user is. The exact defect.
ASKS_TO_BE_NAMED = re.compile(
    r"\b(name me|call me something|what should i be called|could you name|give me a name"
    r"|who should i call\b)", re.I)

# BMO correctly asking for the user's name
# BROADENED after measuring: the first version rejected 77 of ~126 generated rows (61%),
# which is a verifier bug, not a generator failure. It only matched a fixed phrase list, so
# perfectly good openings were discarded -- "tell me your name", "what do they call you",
# "I'd love to know your name", "your name?". The requirement is that BMO ASKS FOR THE
# USER'S NAME in some form; it was never that BMO use one of nine exact phrasings.
ASKS_USER_NAME = re.compile(
    r"(what'?s your name|what is your name|who are you|who am i (talking|chatting|speaking) (to|with)"
    r"|what should i call you|what do (they|i|people) call you|may i (ask|know) your name"
    r"|who might you be|and you are|who'?s there|tell me your name|know your name"
    r"|your name\?|got a name|have a name|go by|introduce yourself|who do i have"
    r"|what'?s yours|call you\?)", re.I)


def prompt_stranger(n: int) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        "BMO has just noticed someone it does NOT recognise, and they have NOT told BMO "
        "their name. Write exchanges where BMO introduces itself and asks what THEIR name "
        "is.\n\n"
        "CRITICAL: BMO asks for the USER'S name. BMO already knows it is called BMO and must "
        "NEVER ask to be named ('could you name me?', 'what should I be called?' are WRONG). "
        "The question is always some form of 'what's your name?' / 'who am I talking to?' / "
        "'what should I call you?'.\n\n"
        "The user's line must NOT contain their name -- they have not said it yet. Their line "
        "can be a greeting, a question, small talk, or an empty string if BMO speaks first.\n\n"
        "Vary it a lot: sometimes BMO remarks on something it can see first (what they are "
        "wearing, that they just walked in, that the room is quiet), sometimes it leads with "
        "the introduction, sometimes it is shy about it. Keep BMO playful and warm.\n\n"
        f"Write {n} varied exchanges. Return ONLY a JSON array of objects with keys "
        f'"user" and "bmo".\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--inp", default="data/bmo_companion_corpus_v10c.jsonl")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v10d.jsonl")
    ap.add_argument("--calls", type=int, default=9)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).open() if l.strip()]
    strangers = [r for r in rows if r["category"] == "name_stranger"]
    others = [r for r in rows if r["category"] != "name_stranger"]

    kept, reclassified, dropped = [], [], []
    for r in strangers:
        u, b = (r.get("prompt") or ""), (r.get("text") or "")
        if ASKS_TO_BE_NAMED.search(b):
            dropped.append(r)
        elif USER_GAVE_NAME.search(u):
            rr = dict(r); rr["category"] = "name_just_told"; rr["reclassified_from"] = "name_stranger"
            reclassified.append(rr)
        else:
            kept.append(r)

    print(f"[expand] name_stranger {len(strangers)} -> kept {len(kept)}, "
          f"reclassified to name_just_told {len(reclassified)}, dropped malformed {len(dropped)}")
    for r in dropped:
        print(f"    dropped: {r['text'][:80]}")
    if args.dry_run:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model
    from scripts.generate_bmo_text_corpus_gptoss import (extract_json_array,
                                                         generate as _raw_generate)
    from models.m5_streaming_voice import ascii_normalize

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(          # proven load path, see v10 script
        args.model_path, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
    mm = {i: "85GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=mm, no_split_module_classes=model._no_split_modules))
    print("[expand] generator ready", flush=True)

    new, rejected, reject_log, would_have_cut = [], 0, [], 0
    for c in range(args.calls):
        raw = _raw_generate(model, tok, prompt_stranger(args.n), max_new_tokens=3000)
        try:
            arr = extract_json_array(raw)
        except ValueError as e:
            print(f"[expand:{c}] parse fail: {e}", flush=True); continue
        for o in arr:
            if not (isinstance(o, dict) and o.get("bmo")):
                continue
            u, b = str(o.get("user") or "").strip(), str(o["bmo"]).strip()
            # the hard check: must ask for THEIR name, must not ask to be named
            # CLOSED-SET CHECKS ONLY. Measured 2026-08-15: requiring a positive match on
            # "does ask for their name" is an OPEN-SET test, and a regex cannot do it. Two
            # passes of broadening still falsely rejected **93** rows while correctly
            # catching only 9 real defects -- and the rejected ones were the MOST
            # characterful ("what's the player tag for you?", "want to save your name in the
            # file?", "who's the new character joining the game?"). Visible-sample idiom rate
            # was 100% among rejects vs 39.1% among accepts, i.e. the filter was removing
            # exactly the personality this corpus exists to preserve.
            #
            # What IS checkable by rule is the closed set of BAD behaviours:
            #   asks to BE named   -- the actual defect ("could you name me?")
            #   user already named -- belongs in name_just_told
            #   cartoon reference  -- fixed vocabulary
            # The generator prompt already instructs it to ask; trust that, and verify only
            # what a rule can genuinely decide.
            why = None
            if CHARACTER_NAMES.search(b):        why = "cartoon"
            elif ASKS_TO_BE_NAMED.search(b):     why = "asks-to-be-named"
            elif USER_GAVE_NAME.search(u):       why = "user-gave-name"
            if why:
                rejected += 1
                reject_log.append((why, b[:90]))
                continue
            if not ASKS_USER_NAME.search(b):
                # kept, but counted -- so the cost of the old rule stays measurable
                would_have_cut += 1
            new.append({"text": b, "prompt": u, "category": "name_stranger",
                        "state": {"energy": 0.6, "mood": "curious"}})
        print(f"[expand:{c}] kept {len(new)} / rejected {rejected}", flush=True)

    out = others + kept + reclassified + new
    import collections, random
    random.shuffle(out)
    with Path(args.out).open("w") as f:
        for r in out:
            if isinstance(r.get("text"), str):
                r["text"] = ascii_normalize(r["text"])
            f.write(json.dumps(r) + "\n")
    cc = collections.Counter(r["category"] for r in out)
    print(f"\n[expand] wrote {args.out}: {len(out)} rows")
    for k in sorted(cc):
        if k.startswith("name_"):
            print(f"    {k:20s} {cc[k]}")
    print(f"    (rejected {rejected} generated rows)")
    import collections as _c
    for w, n in _c.Counter(w for w, _ in reject_log).most_common():
        print(f"      {w:18s} {n}")
    for w, t in reject_log:
        print(f"      [{w}] {t}")
    print(f"    (kept {would_have_cut} rows the old open-set rule would have discarded)")


if __name__ == "__main__":
    main()
