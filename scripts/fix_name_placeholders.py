"""scripts/fix_name_placeholders.py — the `{name}` leak, and why naive substitution is worse.

FOUND 2026-08-16 by the speaker intent bake-off: v3 and v5 both emitted lines containing a
LITERAL curly-brace placeholder --

    v3/LONG  greet_known   "Hi there, {name}! Welcome to the house of fun."
    v5/LONG  greet_known   "Gentle welcome, {name} - ready for a quick quiz?"
    v5/SHORT ask_name      "I haven't met them, but I can suggest a name: {name}."

Grepping the corpora: **54 rows in every v10 variant** (v10, v10c, v10d, v10e) carry an
unsubstituted `{name}`. It survived five corpus regenerations because nobody grepped for it --
every regeneration reshuffled and re-scored the rows, and none of the scorers looked for
template syntax. The model dutifully learned to say the braces out loud.

THE OBVIOUS FIX IS THE DANGEROUS ONE. The tempting move is `{name}` -> a random real name. Do
not. Breakdown of the 54 by whether the USER'S OWN LINE supplies a name:

    41  name_just_told   only 13 of 54 rows have a name anywhere in the prompt
     7  name_recognised
     6  name_unsure

So for the majority the prompt is *"Thanks!"* or *"Play some music."* or *"I'm feeling sleepy."*
and BMO replies *"You're welcome, {name}!"*. Substituting "Maya" there teaches BMO to **address
a stranger by an invented name** -- which is precisely the failure the user hit head-on
("*I am not alice, I just picked alice as a random name, why does it think I am alice*"). The
placeholder bug and the invented-identity bug are the same bug wearing two hats.

THE RULE THIS SCRIPT APPLIES:
  * prompt DOES supply a name  -> substitute that exact name. BMO is repeating what it was told.
  * prompt does NOT supply one -> REMOVE the address entirely, keeping the sentence natural.
    BMO has no name, so BMO uses no name. If removing it leaves a degenerate sentence
    ("Are you ?"), drop that sentence rather than emit a stump.

Nothing here invents a name, ever.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

PH = re.compile(r"\{name\}")

# "I'm Carlos." / "my name is Maya" / "call me Priya" / "this is Sam"
NAME_IN_PROMPT = re.compile(
    r"\b(?:i'?m|i am|my name is|call me|this is|it'?s)\s+([A-Z][a-z]{1,15})\b")

# Vocative forms, longest first so the greedy ones match before the bare token.
_STRIP = [
    (re.compile(r",\s*\{name\}(?=\s*[!?.,])"), ""),        # "Thanks, {name}!"  -> "Thanks!"
    (re.compile(r",\s*\{name\}\b"), ""),                    # "Got it, {name} I'll..."
    (re.compile(r"\{name\},\s*"), ""),                      # "{name}, ready?"  -> "Ready?"
    (re.compile(r"\s+\{name\}(?=\s*[!?.])"), ""),           # "Welcome {name}!" -> "Welcome!"
    (re.compile(r"\s*\{name\}\s*"), " "),                   # anything left
]

# A sentence that only made sense WITH a name is deleted rather than stumped.
_DEGENERATE = re.compile(r"^\s*(are you|is it|did we meet before\? are you|hi there|hello)\s*[?!.]*\s*$",
                         re.I)


def substitute(text: str, name: str) -> str:
    return PH.sub(name, text)


def dename(text: str) -> str:
    """Remove the vocative and repair punctuation/casing."""
    out = text
    for pat, rep in _STRIP:
        out = pat.sub(rep, out)
    # split into sentences, drop any that became meaningless without the name
    parts = re.split(r"(?<=[.!?])\s+", out)
    kept = [p for p in parts if p.strip() and not _DEGENERATE.match(p)]
    out = " ".join(kept) if kept else ""
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/bmo_companion_corpus_v10e.jsonl")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v11.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).open() if l.strip()]
    subbed, denamed, dropped = 0, 0, 0
    stats = collections.Counter()
    out_rows = []

    # Names used ONLY to repair the USER's own line. Safe in a way substituting into BMO's line
    # is not: here the human is the one saying the name, so BMO is never taught to invent one.
    # Caught by the output assert, not by inspection -- two `name_unsure` rows had the
    # placeholder in `prompt` ("No, I'm not {name}.") rather than in `text`.
    USER_NAMES = ["Maya", "Carlos", "Priya", "Sam", "Noor", "Theo", "Amara", "Jonas"]
    n_prompt_fixed = 0

    for i, r in enumerate(rows):
        blob = json.dumps(r)
        if not PH.search(blob):
            out_rows.append(r)
            continue
        if PH.search(r.get("prompt") or ""):
            r["prompt"] = substitute(r["prompt"], USER_NAMES[i % len(USER_NAMES)])
            r["name_fix"] = "prompt-substituted (user speaks the name, BMO does not invent it)"
            n_prompt_fixed += 1
            if not PH.search(json.dumps(r)):
                out_rows.append(r)
                continue
        prompt = r.get("prompt") or ""
        m = NAME_IN_PROMPT.search(prompt)
        before = r.get("text", "")
        if m:
            r["text"] = substitute(before, m.group(1))
            r["name_fix"] = f"substituted:{m.group(1)}"
            subbed += 1
        else:
            fixed = dename(before)
            if not fixed:
                dropped += 1
                stats[r.get("category", "?")] += 1
                continue                     # nothing survived; the row was only a vocative
            r["text"] = fixed
            r["name_fix"] = "removed (prompt supplied no name)"
            denamed += 1
        out_rows.append(r)

    print(f"[names] {len(rows)} rows in")
    print(f"[names]   substituted from prompt : {subbed}")
    print(f"[names]   address removed         : {denamed}   <- BMO has no name, so BMO uses none")
    print(f"[names]   rows dropped (degenerate): {dropped} {dict(stats)}")
    print(f"[names]   user-line placeholders   : {n_prompt_fixed}")
    print(f"[names] {len(out_rows)} rows out")

    # HARD ASSERT: nothing may leave here with a placeholder still in it. This is the check that
    # was missing for five regenerations.
    left = [r for r in out_rows if PH.search(json.dumps(r))]
    assert not left, f"{len(left)} rows still contain a placeholder: {left[:2]}"

    print("\n[names] sample of the removals:")
    for r in out_rows:
        if r.get("name_fix", "").startswith("removed"):
            print(f"    prompt={(r.get('prompt') or '')[:44]!r}  ->  {r['text'][:64]!r}")
            if r is not out_rows[-1] and sum(1 for x in out_rows[:out_rows.index(r) + 1]
                                             if x.get("name_fix", "").startswith("removed")) >= 6:
                break

    if args.dry_run:
        return
    with Path(args.out).open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[names] wrote {args.out}")


if __name__ == "__main__":
    main()
