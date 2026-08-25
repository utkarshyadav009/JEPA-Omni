"""scripts/generate_thinker_directive_rows.py — teach the THINKER to emit a DIRECTIVE.

This is the inverse of scripts/generate_speaker_directive_rows.py. That script taught the
speaker to OBEY a directive; it works (v6 scores 8/8 on rule-checked directive fixtures,
against 5/8 for the un-finetuned base). Nothing teaches the thinker to PRODUCE one.

THE MEASUREMENT THAT MOTIVATES THIS (2026-08-23, real GGUFs on the Jetson):

    approach                                   result
    free-form "give ONE instruction"           returns finished BMO lines
    + verb anchor ("start with Ask/Offer/..")  right FORM, near-identical CONTENT --
                                               "Ask me what you'd like to do next?" for
                                               four unrelated inputs
    closed-set選択 over 13 options, with CoT    3/8  (picked #12 five times)
    closed-set selection, without CoT          1/8  (picked #1 EIGHT times out of eight)
    speaker v6 (350M) as the selector          1/8  (picked #7 five times)

Every prompting route collapses to a mode. The cause is visible in the thinker's own corpus
(data/bmo_thinker_corpus_v7c.jsonl, 987 rows): its schema is `prompt -> reasoning -> answer`
and **`answer` is always a finished BMO utterance**. There is no directive anywhere in it, so
asked for an instruction the model emits a spoken line wearing an imperative hat. Prompting
cannot add a capability the corpus never contained -- the same finding, in the same shape, as
the speaker's own directive gap. It has to be trained.

WHAT THIS TRAINS

    prompt   : "You can see: <six-category scene>. The person said: '<utterance>'"
               (+ optional "Your state is [energy=.. mood=..].")
    reasoning: BMO's private chain of thought about THIS situation
    directive: ONE imperative instruction for the speaker, matching the exact vocabulary the
               speaker was trained on (imperative + a `because` clause)

DESIGN CHOICE THAT MAKES THE ROWS DIVERSE. The naive prompt ("here is a situation, what should
BMO do?") reproduces the collapse in the training data itself -- the teacher also gravitates to
a few safe directives. So the generation is INVERTED: the directive is FIXED per request and
the teacher invents user utterances for which that directive is the right response. Sampling is
then controlled by us, so coverage across directives is balanced by construction rather than by
luck, and every row is correctly conditioned by design.

PAIRING IS NOT OPTIONAL. This project already paid for this once: 197 of 395 rows in the first
speaker directive attempt paired a RANDOM chain-of-thought with a line written for a different
directive, which trains the model to IGNORE the instruction -- worse than the defect being
fixed, and invisible to every count-based gate. Here `utterance`, `reasoning` and `directive`
are emitted together as one JSON object and carry a `paired` flag the training chain can gate
on. Nothing is ever substituted after the fact.

The verifier only checks what a rule can genuinely decide -- imperative form, first-person
leakage, placeholder syntax, show-setting leaks, length. It never tries to judge "is this a
good directive", which is an open-set question a regex loses (the lesson from
expand_name_stranger.py, where an open-set rule produced 93 false rejects against 9 real ones).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_bmo_corpus_v10_identity import CHARACTER_NAMES, HARD_RULES, PERSONA
from scripts.generate_speaker_directive_rows import (SCENES, SETTING_LEAK,
                                                     ALL_DIRECTIVES)

# The speaker's 372 directive rows contain only 13 DISTINCT directives (~29 examples each).
# 13 is far too narrow for open conversation, and the gaps are exactly where injected
# directives were measured to HURT: hostility and withdrawal. Widen here, then widen the
# speaker slice to match -- the two must share one vocabulary or the interface breaks.
# The canonical 26-directive vocabulary is defined in
# scripts/generate_speaker_directive_rows.py and imported above -- ONE source of
# truth, so the speaker and thinker can never drift apart.


STATES = [
    "[energy=0.8 mood=curious]", "[energy=0.5 mood=content]", "[energy=0.2 mood=tired]",
    "[energy=0.6 mood=playful]", "[energy=0.3 mood=concerned]", "[energy=0.9 mood=excited]",
]

# ---- closed-set rejection rules: only what a rule can genuinely decide --------------------
PLACEHOLDER = re.compile(r"(\{[a-z_]+\}|\[[a-z_]+\]|<[a-z_]+>)", re.I)
# a directive must be an INSTRUCTION, so it starts with an imperative verb
# A leading ADVERB is still an imperative ("gently suggest they take a break"). The first
# version anchored on the verb alone and false-rejected 3 of the 26 canonical directives,
# which then received ZERO training rows -- found by running the verifier over its own
# vocabulary rather than trusting it.
IMPERATIVE = re.compile(
    r"^(gently|quietly|warmly|briefly|firmly|calmly|kindly|simply|politely)?\s*"
    r"(ask|tell|offer|acknowledge|suggest|remind|encourage|greet|reassure|invite|comfort|"
    r"answer|explain|check|avoid|keep|stay|mention|remark|celebrate|hold|give|match|thank|"
    r"own|admit|apologise|apologize|react|say|do not|don't)\b", re.I)
# First person = a spoken LINE, not an instruction -- the exact failure mode measured on the
# deployed thinker ("Ask me what you'd like to do next?").
# `your`/`you're` REMOVED: they appear in perfectly good directives ("tell them YOUR battery is
# low", "hold YOUR ground gently") and cost those two every row. The degenerate case is still
# caught -- "Ask me what you'd like to do next?" trips on `me` and on the trailing question
# mark -- so nothing is lost by dropping them.
FIRST_PERSON = re.compile(r"\b(i|i'm|i am|i'll|let's|we|we're|my|me|shall we)\b", re.I)
LABEL_PREFIX = re.compile(r"^\s*(directive|instruction|answer|output|note|thinking)\s*:", re.I)


def reject_directive(d: str) -> str | None:
    t = (d or "").strip()
    if not t:
        return "empty"
    if len(t) < 12 or len(t) > 130:
        return "length"
    if PLACEHOLDER.search(t):
        return "placeholder"
    if LABEL_PREFIX.match(t):
        return "label_prefix"
    if not IMPERATIVE.match(t):
        return "not_imperative"
    if FIRST_PERSON.search(t):
        return "first_person"          # it is a spoken line, not an instruction
    if t.endswith("?"):
        return "question"              # a directive instructs; it does not ask the user
    if CHARACTER_NAMES.search(t) or SETTING_LEAK.search(t):
        return "setting_leak"
    return None


def reject_utterance(u: str) -> str | None:
    t = (u or "").strip()
    if not t:
        return "empty"
    if len(t) < 4 or len(t) > 160:
        return "length"
    if PLACEHOLDER.search(t):
        return "placeholder"
    if CHARACTER_NAMES.search(t) or SETTING_LEAK.search(t):
        return "setting_leak"
    if re.match(r"^\s*(bmo|beemo)\s*[:,]", t, re.I):
        return "bmo_speaking"          # this field is the PERSON's line, not BMO's
    return None


def reject_reasoning(r: str) -> str | None:
    t = (r or "").strip()
    if not t:
        return "empty"
    if len(t) < 40 or len(t) > 460:
        return "length"
    if PLACEHOLDER.search(t):
        return "placeholder"
    if CHARACTER_NAMES.search(t) or SETTING_LEAK.search(t):
        return "setting_leak"
    return None


def prompt_for(scene: str, directive: str, n: int) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        f"BMO can see this right now:\n  {scene}\n\n"
        f"BMO has privately decided what to do next. The decision is EXACTLY this:\n"
        f"  {directive}\n\n"
        f"Invent {n} DIFFERENT things a real person might have just said out loud that would "
        f"make that decision the right one. Vary them a lot: some short, some long, some "
        f"emotional, some flat, some a question, some not. They must be things a normal person "
        f"says in a normal room -- never anything from a cartoon.\n\n"
        f"For each one, also write BMO's private thinking: two or three sentences, first "
        f"person, about THIS person and THIS moment, that lead naturally to the decision above. "
        f"The thinking must be about the situation, NOT a restatement of the decision.\n\n"
        f"Return ONLY a JSON array of {n} objects, each with exactly these keys:\n"
        f'  "said"     - what the PERSON said out loud (never BMO\'s words)\n'
        f'  "thinking" - BMO\'s private reasoning, first person, 2-3 sentences\n\n'
        f"Do not include the decision itself in either field. Do not add any other keys, "
        f"commentary or markdown."
    )


def salvage_pairs(raw: str) -> list:
    """Recover objects one at a time when json.loads fails on the whole array.

    48% of generations died on `json.loads` in the speaker run because BMO's text is full of
    apostrophes and inner quotation, and ONE bad entry discarded the other three. Extracting
    each key independently took yield 82 -> 395. Two things that matters for here: the "said"
    and "thinking" of the SAME object must stay together (an unpaired fragment is exactly the
    197-row bug), and a salvaged value can stop at an unescaped inner quote and yield half a
    sentence -- worse than dropping it -- hence the terminal-punctuation check.
    """
    out = []
    saids = [(m.start(), m.group(1)) for m in re.finditer(r'"said"\s*:\s*"(.*?)"\s*[,}]', raw, re.S)]
    thinks = [(m.start(), m.group(1)) for m in re.finditer(r'"thinking"\s*:\s*"(.*?)"\s*[,}]', raw, re.S)]
    for pos, said in saids:
        after = [t for p, t in thinks if p > pos]
        if not after:
            continue
        think = after[0]
        if not said.strip() or not think.strip():
            continue
        if not think.strip()[-1] in ".!?":      # truncated mid-sentence -> drop, do not train
            continue
        out.append({"said": said.strip(), "thinking": think.strip(), "_salvaged": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bmo_thinker_directive_rows.jsonl")
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--per-combo", type=int, default=6)
    ap.add_argument("--scenes-per-directive", type=int, default=8)
    ap.add_argument("--min-rows", type=int, default=900)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--only", nargs="*", default=None,
                    help="generate ONLY these directives (top-up mode); appends to --out")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)

    kinds = args.only if args.only else ALL_DIRECTIVES
    combos = []
    for d in kinds:
        for s in random.sample(SCENES, min(args.scenes_per_directive, len(SCENES))):
            combos.append((s, d))
    random.shuffle(combos)
    target = len(combos) * args.per_combo
    print(f"[thinker-dir] {len(kinds)} directives x {args.scenes_per_directive} scenes "
          f"= {len(combos)} combos x {args.per_combo} = ~{target} rows (min gate {args.min_rows})",
          flush=True)

    if args.dry_run:
        print(prompt_for(*combos[0], args.per_combo))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model
    from scripts.generate_bmo_text_corpus_gptoss import (extract_json_array,
                                                         generate as _raw_generate)
    from models.m5_streaming_voice import ascii_normalize

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(      # device_map="auto" reproducibly hits an
        args.model_path, dtype=torch.bfloat16,         # illegal memory access on this box;
        device_map=None, low_cpu_mem_usage=True)       # dispatch explicitly instead
    mm = {i: "85GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=mm, no_split_module_classes=model._no_split_modules))
    print("[thinker-dir] generator ready", flush=True)

    # INCREMENTAL WRITE. A 208-combo run is multi-hour; buffering everything to the end
    # means a crash at combo 200 loses the lot, and nothing can be inspected for quality
    # until it is over. This project's own rule is to read ROWS, not counts -- which is
    # impossible if the rows only appear at the end. Append as we go, flush every combo.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("a" if args.append else "w", buffering=1)

    rows, rej = [], collections.Counter()
    per_directive = collections.Counter()
    for i, (scene, directive) in enumerate(combos):
        raw = _raw_generate(model, tok, prompt_for(scene, directive, args.per_combo),
                            max_new_tokens=args.max_new_tokens)
        try:
            arr = extract_json_array(raw)
        except Exception as e:
            arr = salvage_pairs(raw)
            print(f"[thinker-dir:{i}] array parse failed ({type(e).__name__}); "
                  f"salvaged {len(arr)} pair(s)", flush=True)
            if not arr:
                continue
        kept = 0
        for o in arr:
            if not isinstance(o, dict):
                continue
            said = ascii_normalize(str(o.get("said", "")).strip())
            think = ascii_normalize(str(o.get("thinking", "")).strip())
            d = ascii_normalize(directive)
            for field, why in (("utterance", reject_utterance(said)),
                               ("reasoning", reject_reasoning(think)),
                               ("directive", reject_directive(d))):
                if why:
                    rej[f"{field}:{why}"] += 1
                    break
            else:
                state = random.choice(STATES)
                rows.append({
                    "prompt": f"You can see: {scene}. The person said: '{said}'. "
                              f"Your state is {state}.",
                    "scene": scene,
                    "said": said,
                    "state": state,
                    "reasoning": think,
                    "directive": d,
                    "category": "thinker_directive",
                    # `said` and `thinking` came out of ONE JSON object and are never
                    # substituted afterwards, so the pair genuinely corresponds. The salvage
                    # path preserves that too (it walks forward from each "said" to the next
                    # "thinking"). Flag both so the training chain can gate on either.
                    "paired": True,
                    "salvaged": bool(o.get("_salvaged", False)),
                })
                per_directive[d] += 1
                kept += 1
                fout.write(json.dumps(rows[-1], ensure_ascii=True) + "\n")
        if i % 5 == 0 or kept == 0:
            print(f"[thinker-dir:{i}/{len(combos)}] kept {kept} (total {len(rows)}) "
                  f"| {directive[:44]}", flush=True)
        # Print a real sample early and periodically so quality is visible DURING the run,
        # not discovered at the end. Counts passed every bad corpus this project ever made.
        if i in (0, 3, 12, 40, 100) and rows:
            r = rows[-1]
            print(f"[sample:{i}] SAID     : {r['said'][:120]!r}\n"
                  f"[sample:{i}] THINKING : {r['reasoning'][:180]!r}\n"
                  f"[sample:{i}] DIRECTIVE: {r['directive'][:90]!r}", flush=True)

    fout.close()

    print(f"\n[thinker-dir] wrote {len(rows)} rows -> {out}", flush=True)
    print(f"[thinker-dir] rejections: {rej.most_common()}", flush=True)
    print(f"[thinker-dir] distinct directives covered: {len(per_directive)}/{len(ALL_DIRECTIVES)}",
          flush=True)
    for d, c in per_directive.most_common():
        print(f"    {c:4d}  {d}", flush=True)

    # Gates. Count alone is never enough -- the speaker's 395-row attempt passed a count gate
    # while 197 of its rows were actively harmful.
    ok = True
    if len(rows) < args.min_rows:
        print(f"!! FAIL only {len(rows)} rows, need {args.min_rows}", flush=True); ok = False
    if len(per_directive) < len(ALL_DIRECTIVES) * 0.8:
        print(f"!! FAIL directive coverage {len(per_directive)}/{len(ALL_DIRECTIVES)}", flush=True)
        ok = False
    if per_directive and per_directive.most_common(1)[0][1] > 0.25 * len(rows):
        top, c = per_directive.most_common(1)[0]
        print(f"!! FAIL collapse: '{top[:50]}' is {c}/{len(rows)} rows", flush=True); ok = False
    dupes = len(rows) - len({r["said"] for r in rows})
    print(f"[thinker-dir] duplicate utterances: {dupes}", flush=True)
    if dupes > 0.15 * max(len(rows), 1):
        print("!! FAIL too many duplicate utterances", flush=True); ok = False
    print(f"\nTHINKER_DIRECTIVE_GEN {'PASS' if ok else 'FAIL'}", flush=True)
    print("THINKER_DIRECTIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
