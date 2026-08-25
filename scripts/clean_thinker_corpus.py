"""scripts/clean_thinker_corpus.py — repair a thinker corpus that carries the two v6c defects.

Exists because `thinker_v7` was already 3 hours into generation when both defects were found,
with the old generator loaded in memory. Killing the run would have destroyed 3h of GPU work for
a corpus that is still mostly good; the generator is fixed for FUTURE runs
(`generate_thinker_corpus_gptoss.py`) and this repairs the one already in flight.

## Defect 1 — restraint scenarios that offer a game anyway

Measured in v6c: of 99 rows whose scenario calls for NOT intruding (someone concentrating, an
empty room, "whether saying anything at all is the right move"), **61% still offer a game,
jingle, music or "press start" in the answer**:

    "Okay, I'll be quiet for now. If you want to talk or need anything, just press start."
    "Hey there! It's dark, but I'm still feeling the beat - let's turn on a soft glow and play
     a happy jingle together."

The cause is upstream and was not obvious: the generator's own rule *"a flat, formal or generic
line is a FAILED example"* -- written to protect BMO's personality -- makes the model bolt an
activity offer onto a line that should end quietly. Personality and restraint were in conflict
and personality always won.

On-device this produced a thinker that offered a game EVERY round regardless of context and
failed the `respects_focus` behavioural case 0/4 (`scripts/thinker_behaviour_gate.py`).

**These rows are DROPPED, not rewritten.** Editing the answer to strip the offer would leave
reasoning that still argues for it, and a row whose reasoning and answer disagree teaches the
disagreement. A smaller consistent corpus beats a larger contradictory one -- the same
conclusion `expand_name_stranger.py` reached about contaminated `name_stranger` rows.

## Defect 2 — one name becomes the prior for "recognition"

v6c carried **"Alice" in 27 prompts and 24 answers** because three scenarios hardcoded it. A
single repeated name does not stay an example; it becomes what the model reaches for whenever it
recognises anyone. The user had picked that name at random and then met a robot convinced they
were her. Names are rotated here across a pool, consistently WITHIN each row so prompt and
answer never disagree about who is present.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# Scenario shapes where the right move is to offer nothing.
RESTRAINT = re.compile(
    r"not interrupt|absorbed|focused work|empty room|saying anything at all|without intruding"
    r"|nobody is there|asleep|sleeping|do not disturb|wants? quiet|leave them alone"
    r"|whether to say anything", re.I)

# An activity offer. "level"/"press start" included: measured, they are how the offer is
# actually phrased in this corpus, not incidental metaphor.
OFFER = re.compile(
    r"\b(play|game|jingle|quiz|puzzle|press start|mini-?game|sing|song|music|dance|adventure"
    r"|quest|level up|new level)\b", re.I)

NAME_POOL = ["Priya", "Theo", "Amara", "Jonas", "Noor", "Mateo", "Ines", "Kofi"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/bmo_thinker_corpus_v7.jsonl")
    ap.add_argument("--out", default="data/bmo_thinker_corpus_v7_clean.jsonl")
    ap.add_argument("--names", default="Alice,Bob",
                    help="over-used example names to rotate out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        raise SystemExit(f"{src} does not exist yet — run this after the generator finishes")
    rows = [json.loads(l) for l in src.open() if l.strip()]

    overused = [n.strip() for n in args.names.split(",") if n.strip()]
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in overused) + r")\b")

    kept, dropped = [], []
    renamed = 0
    stats = collections.Counter()

    for i, r in enumerate(rows):
        prompt = r.get("prompt") or ""
        answer = r.get("answer") or ""
        reasoning = r.get("reasoning") or ""

        # Defect 1: restraint scenario whose answer still offers an activity -> drop whole row.
        if RESTRAINT.search(prompt) and OFFER.search(answer):
            dropped.append(r)
            stats["restraint_offers_activity"] += 1
            continue

        # Defect 2: rotate the over-used name, consistently across all three fields so the row
        # never ends up naming two different people.
        if pat.search(prompt) or pat.search(answer) or pat.search(reasoning):
            new = NAME_POOL[i % len(NAME_POOL)]
            r["prompt"] = pat.sub(new, prompt)
            r["answer"] = pat.sub(new, answer)
            r["reasoning"] = pat.sub(new, reasoning)
            # "Alice ... her" -> keep it pronoun-neutral rather than guess a new person's gender
            for f in ("prompt", "answer", "reasoning"):
                r[f] = re.sub(r"\b(her|his)\b", "their", r[f])
                r[f] = re.sub(r"\b(she|he) (is|was|has|had|does|did)\b", r"they \2", r[f])
            r["name_rotated"] = new
            renamed += 1
            stats["name_rotated"] += 1
        kept.append(r)

    print(f"[clean] {len(rows)} rows in")
    print(f"[clean]   dropped, restraint scenario that still offers an activity : "
          f"{stats['restraint_offers_activity']}")
    print(f"[clean]   example name rotated out of {overused}                    : {renamed}")
    print(f"[clean] {len(kept)} rows out")

    n_restraint = sum(1 for r in rows if RESTRAINT.search(r.get("prompt") or ""))
    if n_restraint:
        pct = 100 * stats["restraint_offers_activity"] / n_restraint
        print(f"[clean]   ({stats['restraint_offers_activity']}/{n_restraint} = {pct:.0f}% of "
              f"restraint rows were contradictory; v6c measured 61%)")

    print("\n[clean] sample of what was dropped:")
    for r in dropped[:4]:
        print(f"    {(r.get('answer') or '')[:96]}")

    if args.dry_run:
        return

    # The checks that were missing for five regenerations: assert the defects are actually gone.
    leftover = [r for r in kept if pat.search(json.dumps(r))]
    assert not leftover, f"{len(leftover)} rows still carry an over-used name"
    bad = [r for r in kept
           if RESTRAINT.search(r.get("prompt") or "") and OFFER.search(r.get("answer") or "")]
    assert not bad, f"{len(bad)} contradictory restraint rows survived"

    with Path(args.out).open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n[clean] wrote {args.out}")
    print("[clean] asserts passed: no over-used names, no contradictory restraint rows")


if __name__ == "__main__":
    main()
