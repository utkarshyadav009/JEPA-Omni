"""scripts/generate_speaker_directive_rows.py — teach the speaker to OBEY the thinker.

THE MEASUREMENT THAT MOTIVATES THIS (2026-08-16, scripts/speaker_intent_bakeoff.py, real GGUFs
on the Jetson, 6 fixtures x 2 prompt formats):

    model  LONG  SHORT          LONG  = the 453-char scene+CoT the pipeline sends today
    v1     1/6   2/6            SHORT = a compact directive
    v2     4/6   3/6
    v3     5/6   4/6
    v5     4/6   5/6

Two things fall out, and the second is the important one.

1. **The "v1/v2 were better" memory is false.** v1 is far the worst and its failures are
   free-association that ignores the input entirely -- *"BMW's research team works on
   sustainable power solutions for future cars"*, *"I think about how Finn's robotic arm
   works"*. That is the v7-era corpus (88% bare mood lines trained against a fixed user turn
   "Say something.") doing what it was trained to do. v1 always sounds fluent and in-character,
   so in a live demo where nobody is scoring against an intent it *reads* smarter. It was not
   better; it was more confidently irrelevant.

2. **Prompt LENGTH is not the cause.** I predicted LONG would collapse because the corpus
   median prompt is 38 chars and the pipeline sends 453. It does not: LONG vs SHORT moves by
   +/-1 with no consistent direction. That hypothesis is falsified and this script does not
   rely on it.

WHAT IS ACTUALLY WRONG is visible in the failure text, and it is a capability the corpus never
contained. **Zero rows in any speaker corpus, any version, condition on an instruction.** Only
126/3774 rows (3.3%) mention a scene at all, and every one of those is
`short scene + a REAL USER UTTERANCE` -- the scene is context and the user's line is what BMO
answers. Handed a directive instead, the model has no learned behaviour for it and falls into
three failure modes, all present in the bake-off output:

    RESTATE   v3 "Ask them directly. They usually have the best ideas for you."
    THIRD-PERSON  v3 "Are they working on a screen project? If so, give them a clear task."
    LABEL     v3 "Acknowledgement: BMO, you're winning the round - go!"   v5 "Rest is recommended."

It treats the directive as a topic to discuss or a form to fill in, rather than an instruction
to carry out while speaking to the person in front of it. That is a corpus-format gap, not a
v1->v5 regression, and rolling back would have achieved nothing.

THIS SLICE therefore trains exactly one new mapping:

    prompt : "You can see: <scene>. Your private thinking: <directive>"
    text   : the single line BMO says out loud, in BMO's voice, to the PERSON

with the three failure modes above as explicit negative instructions AND as closed-set
rejection rules. Per the lesson already paid for in expand_name_stranger.py, the verifier only
checks things a rule can genuinely decide -- placeholder syntax, label prefixes, directive echo,
second-person address -- and never tries to judge "is this a good line", which is an open-set
question a regex loses.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_bmo_corpus_v10_identity import CHARACTER_NAMES, HARD_RULES, PERSONA

# Scenes phrased exactly as models/query_predictor.py's categories emit them, so the training
# prompt is byte-shaped like what jetson_real_demo.py actually builds at runtime.
SCENES = [
    "wearing: a person wearing headphones; doing: someone is watching a screen; who: a person "
    "wearing glasses; where: a home office; lighting: dim lighting",
    "wearing: a person wearing a red jumper; doing: someone is reading; who: one person; "
    "where: a living room; lighting: natural daylight",
    "wearing: a person in pyjamas; doing: someone is drinking; who: a person sitting; "
    "where: a kitchen; lighting: bright lighting",
    "wearing: a person wearing a black hoodie; doing: someone is typing; who: a person facing "
    "the camera; where: a bedroom; lighting: a lit screen in a dark room",
    "wearing: a person wrapped in a blanket; doing: nothing is happening; who: a person lying "
    "down; where: a living room; lighting: dim lighting; hearing: quiet background noise",
    "wearing: a person wearing a cap; doing: someone is entering the room; who: two people; "
    "where: a hallway; lighting: artificial indoor light",
    "wearing: a person holding a mug; doing: someone is talking; who: a person standing; "
    "where: an office; lighting: bright lighting; hearing: several people talking",
    "wearing: a person with long hair; doing: someone is playing an instrument; who: one "
    "person; where: a bedroom; lighting: dim lighting; hearing: music playing",
]

# The directive vocabulary the THINKER will be trained to emit. Deliberately imperative and
# short -- the thinker's job becomes producing one of these shapes, not 260 chars of prose.
DIRECTIVE_KINDS = [
    "ask what their name is, because you have never met them",
    # REMOVED "greet them by name, because you recognise them": unsatisfiable. No SCENE
    # supplies a name and HARD_RULES forbids inventing one, so the generator wrote
    # reasoning that REFUSES the directive ("I don't have a name for you...") paired with
    # a line that does something else -- 23 rows teaching "instruction X -> do Y".
    # Refusing to invent a name is correct behaviour; it just cannot be trained from a
    # directive that presupposes a name the scene never provides.
    "greet them warmly, because you have met them before",
    "stay quiet and brief, because they are concentrating and must not be interrupted",
    "remark warmly on something they are wearing",
    "gently suggest they take a break, because they have been at this a long time",
    "ask what they are working on, because you genuinely cannot tell",
    "check whether they are okay, because something looks off",
    "say nothing about what you can see, and simply offer to help",
    "acknowledge that you heard a sound in the room",
    "tell them your battery is low and ask to be plugged in",
    "say you are switching to a quieter mode to save power",
    "react to them coming back after being away",
    "offer to play something, because they look bored",
    "apologise for interrupting, and step back",
]

# WIDENING 14 -> 26 (2026-08-23). The shipped v12 slice contained only 13 DISTINCT directives
# (~29 rows each), and the measured coordination failures were concentrated exactly in the
# situations it had no directive for: hostility, withdrawal, celebration, admitting ignorance,
# owning a mistake. Injecting a directive on those turns made the line WORSE than no directive
# at all ("You're being annoying." -> "That jittery move makes me nervous.").
#
# THE SPEAKER AND THE THINKER MUST SHARE ONE VOCABULARY. This list is the single source of
# truth; scripts/generate_thinker_directive_rows.py imports ALL_DIRECTIVES from here. If the
# two drift apart the thinker emits directives the speaker was never trained to obey, which is
# the same class of train/deploy mismatch that caused the original problem.
EXTRA_DIRECTIVES = [
    "celebrate with them, because something good just happened",
    "hold your ground gently, because they are being unkind",
    "give them quiet space, because they do not want to talk",
    "acknowledge how long they have been working, because they sound worn out",
    "answer their question directly, because they asked something specific",
    "admit you do not know, because you genuinely cannot tell",
    "reassure them, because they sound worried",
    "match their excitement, because they are happy about something",
    "ask a follow-up question, because they hinted at something they did not finish",
    "suggest something calm, because it is late and they seem tired",
    "thank them, because they did something kind for you",
    "own the mistake plainly, because you got something wrong",
]
ALL_DIRECTIVES = list(dict.fromkeys(DIRECTIVE_KINDS + EXTRA_DIRECTIVES))

# ---- closed-set rejection rules. Only what a rule can genuinely decide. -------------------
PLACEHOLDER = re.compile(r"\{[a-z_]+\}", re.I)
LABEL_PREFIX = re.compile(r"^\s*(acknowledgement|response|answer|note|name|rest|action|output|"
                          r"thinking|thought|directive)\s*:", re.I)
# restating the instruction rather than performing it ("Ask them directly.", "Tell them that...")
RESTATE = re.compile(r"^\s*(ask|tell|give|greet|remind|offer|suggest|check|say)\s+(them|him|her|"
                     r"the (?:user|person))\b", re.I)
# talking ABOUT the person to a third party instead of TO them
THIRD_PERSON = re.compile(r"\b(they are|they're|their|them)\b.*\?$", re.I)
SECOND_PERSON = re.compile(r"\b(you|you're|your|let's|we|shall we)\b", re.I)
# Places/props that only exist inside the show. BMO's personality comes from it; BMO's LIFE
# does not (the standing rule in HARD_RULES, enforced here on the INSTRUCTION side too).
SETTING_LEAK = re.compile(
    r"\b(treehouse|candy kingdom|ooo|rope bridge|dungeon|kingdom|princess|adventurer|quest for"
    r"|magic sword|slime|land of)\b", re.I)


def load_real_reasoning(path: str = "data/bmo_thinker_corpus_v6c.jsonl",
                        limit: int = 400) -> list:
    """Real `reasoning` strings from the thinker corpus, to train on what is ACTUALLY sent.

    Caught before generating, not after: this script's `DIRECTIVE_KINDS` are compact imperatives
    ("ask what their name is, because you have never met them"), but the live pipeline sends the
    thinker's PROSE chain-of-thought --

        "I see the room is dim, and I hear a faint alarm beeping. The person in pyjamas looks
         like they're waiting for something special to happen - maybe a game or a song..."

    Training the speaker on the compact shape and then deploying the prose shape would recreate
    the exact train/deploy mismatch just diagnosed as the cause of the whole problem. So half
    the rows use a real CoT paragraph as the instruction and half use a compact directive, and
    the speaker learns to act on either. The compact form is still worth keeping: if the thinker
    is later taught to emit directives (a genuine architectural option), the speaker is already
    able to consume them."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.open():
        if not line.strip():
            continue
        r = json.loads(line)
        t = (r.get("reasoning") or "").strip()
        # skip the placeholder rows the corpus generator left behind ("...")
        if not (60 <= len(t) <= 420) or t == "..." or "{" in t:
            continue
        # v6c still carries fictional-setting reasoning that task #8 stripped from the SPEAKER
        # corpus but not from the thinker's -- e.g. "a rope bridge, a gigantic companion, and
        # the risk of a crash-reset". Feeding that in as "your private thinking" would train the
        # speaker to answer from inside the show, which is the one thing the user asked to keep
        # out ("retains the personality of the show but isn't bound by it").
        if CHARACTER_NAMES.search(t) or SETTING_LEAK.search(t):
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def prompt_for(scene: str, directive: str, n: int) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        "BMO has a separate reasoning module (the 'thinker') that decides WHAT should be "
        "communicated. BMO's job here is only to SAY it, in BMO's own voice, out loud, to the "
        "person standing in front of it.\n\n"
        f"BMO can currently see:\n  {scene}\n\n"
        f"The thinker's instruction is: {directive}\n\n"
        "Write BMO's spoken line. ABSOLUTE REQUIREMENTS:\n"
        "  * Speak TO the person, using 'you'. Never talk about them in the third person "
        "('they', 'them') as if reporting to someone else.\n"
        "  * CARRY OUT the instruction. Do NOT restate it. 'Ask them their name.' is a "
        "restatement and is WRONG; 'What should I call you?' carries it out and is RIGHT.\n"
        "  * No label prefixes. 'Acknowledgement: ...', 'Response: ...', 'Rest is "
        "recommended.' are all WRONG.\n"
        "  * Never write a placeholder like {name}. If BMO has not been told a name, BMO "
        "uses no name at all.\n"
        "  * One or two short sentences. Warm, playful, curious -- BMO's voice.\n"
        "  * You may reference what BMO can see, but only if it serves the instruction.\n\n"
        f"Write {n} varied examples for this situation. Return ONLY a JSON array of objects, "
        f"each with TWO keys:\n"
        f"  \"thinking\" -- the private reasoning that leads to THIS EXACT line, as 2-3 flowing "
        f"first-person sentences: what BMO notices, what it is unsure about, and why this move "
        f"rather than another. It must lead to the \"bmo\" line beside it. They are a matched "
        f"pair, not two separate ideas.\n"
        f"  \"bmo\" -- the single line BMO says out loud.\n"
    )


# Salvage individual lines when the whole array will not parse.
# MEASURED: the first v12 run lost **27 of 56 generations (48%)** to `json.loads` errors
# ("Expecting value: line 1 column 18", "Expecting ',' delimiter: line 1 column 15") and
# produced only 82 usable rows against a ~224 target. `extract_json_array` is not the problem --
# it already does correct bracket-depth matching (a fix made after an earlier greedy-regex bug).
# The CONTENTS are malformed: BMO's lines are full of apostrophes and inner quotation, and the
# generator intermittently emits unescaped quotes or typographic ones.
#
# One bad line should not cost the other three in the same array. This pulls out each
# `"bmo": "..."` value independently, so a single malformed entry loses only itself.
_BMO_VALUE = re.compile(r'"bmo"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_THINK_VALUE = re.compile(r'"thinking"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_SMART = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def salvage_lines(raw: str) -> list:
    txt = raw
    for a, b in _SMART.items():
        txt = txt.replace(a, b)
    # Pair each "bmo" with the "thinking" that precedes it, so salvaged rows keep the
    # correspondence too. An unpaired line still salvages -- it just becomes a compact-shape
    # row rather than a prose one, which is a graceful degradation, not a mismatch.
    thinks = [(m.start(), m.group(1)) for m in _THINK_VALUE.finditer(txt)]
    out = []
    for m in _BMO_VALUE.finditer(txt):
        v = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if not v:
            continue
        prior = [t for pos, t in thinks if pos < m.start()]
        out.append({"bmo": v,
                    "thinking": prior[-1].replace('\\"', '"').strip() if prior else ""})
    return out


def reject(text: str) -> str | None:
    # TRUNCATION CHECK, required by the salvage path. Recovering a `"bmo": "..."` value from a
    # malformed array can stop at an unescaped inner quote and yield a stump -- measured:
    # `"Nice glasses! You look like a "pro gamer" today."` salvages as
    # `Nice glasses! You look like a`. Losing one line to malformed JSON is fine; training on
    # half a sentence is not. Every real BMO line ends in terminal punctuation.
    if not text.rstrip().endswith((".", "!", "?", '"')):  return "truncated"
    if len(text.split()) < 3:                             return "too-short"
    if PLACEHOLDER.search(text):        return "placeholder"
    if LABEL_PREFIX.search(text):       return "label-prefix"
    if RESTATE.search(text):            return "restates-directive"
    if CHARACTER_NAMES.search(text):    return "cartoon"
    if not SECOND_PERSON.search(text):  return "not-addressed-to-person"
    if THIRD_PERSON.search(text):       return "third-person"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--base", default="data/bmo_companion_corpus_v11.jsonl",
                    help="corpus to append the new slice to (already {name}-clean)")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v12.jsonl")
    ap.add_argument("--per-combo", type=int, default=4)
    ap.add_argument("--combos", type=int, default=56,
                    help="scene x directive pairs to sample (8 scenes x 14 directives = 112)")
    ap.add_argument("--only-new", action="store_true",
                    help="generate ONLY the 12 widening directives; the original set already "
                         "has ~29 rows each in the base corpus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(0)
    kinds = EXTRA_DIRECTIVES if args.only_new else ALL_DIRECTIVES
    combos = [(s, d) for s in SCENES for d in kinds]
    rng.shuffle(combos)
    combos = combos[:args.combos]
    prose = load_real_reasoning()
    print(f"[directive] {len(prose)} real thinker CoT paragraphs loaded for the prose-shaped "
          f"half (the shape the live pipeline actually sends)", flush=True)
    print(f"[directive] {len(combos)} scene x directive combos x {args.per_combo} lines "
          f"= ~{len(combos) * args.per_combo} target rows", flush=True)
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
    model = AutoModelForCausalLM.from_pretrained(       # proven load path; device_map="auto"
        args.model_path, dtype=torch.bfloat16,          # reproducibly hits an illegal memory
        device_map=None, low_cpu_mem_usage=True)        # access on this box
    mm = {i: "85GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=mm, no_split_module_classes=model._no_split_modules))
    print("[directive] generator ready", flush=True)

    new, rej = [], collections.Counter()
    for i, (scene, directive) in enumerate(combos):
        raw = _raw_generate(model, tok, prompt_for(scene, directive, args.per_combo),
                            max_new_tokens=1400)
        try:
            arr = extract_json_array(raw)
        except Exception as e:
            arr = salvage_lines(raw)      # one malformed entry must not cost the whole array
            print(f"[directive:{i}] array parse failed ({type(e).__name__}); "
                  f"salvaged {len(arr)} line(s)", flush=True)
            if not arr:
                continue
        for o in arr:
            if not (isinstance(o, dict) and o.get("bmo")):
                continue
            line = str(o["bmo"]).strip()
            why = reject(line)
            if why:
                rej[why] += 1
                continue
            # Alternate the INSTRUCTION SHAPE row by row: compact directive, then the PAIRED
            # prose reasoning the generator wrote FOR THIS LINE. Same target line either way,
            # so the speaker learns that the instruction's meaning is what matters and its
            # form is not.
            #
            # BUG THIS REPLACES, caught by reading the generated rows rather than the counts:
            # the first version substituted a RANDOM CoT paragraph sampled from the thinker
            # corpus, so the prose instruction and the spoken line described different
            # situations entirely --
            #     INSTR: "I notice both of my friends are feeling upset because each wants to
            #             play a different game..."
            #     SAYS : "Boredom glitch detected! Want me to load a surprise game for you?"
            # That is a training pair which teaches the speaker to IGNORE the instruction --
            # the exact behaviour this whole slice exists to fix, and worse than the original
            # defect. 197 of 395 rows were mismatched that way. The reasoning must be
            # generated alongside its own line so the pair actually corresponds.
            thinking = str(o.get("thinking") or "").strip()
            use_prose = bool(thinking) and (len(new) % 2 == 1)
            instr = thinking if use_prose else f"{directive}."
            new.append({
                "text": ascii_normalize(line),
                # normalize the PROMPT too. ascii_normalize was applied only to `text`, so 100
                # generated prompts carried typographic Unicode (don't/you're/em-dashes) while
                # the rest of the DB had been scrubbed of it -- a tokenisation inconsistency in
                # the same field, for no benefit.
                "prompt": ascii_normalize(f"You can see: {scene}. Your private thinking: {instr}"),
                "category": "speaker_directive",
                "directive": directive,
                "instr_shape": "prose_cot" if use_prose else "compact",
                # provenance flag the training gate checks: True means this row's
                # reasoning was generated WITH its own line, not sampled from elsewhere
                "paired": bool(use_prose),
                "state": {"energy": 0.6, "mood": "curious"},
            })
        print(f"[directive:{i}/{len(combos)}] kept {len(new)}  rejected {sum(rej.values())} "
              f"{dict(rej)}", flush=True)

    base = [json.loads(l) for l in Path(args.base).open() if l.strip()]
    out = base + new
    random.Random(1).shuffle(out)
    with Path(args.out).open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    cc = collections.Counter(r.get("category", "?") for r in out)
    print(f"\n[directive] wrote {args.out}: {len(out)} rows "
          f"({len(base)} base + {len(new)} new directive rows)")
    print(f"[directive] rejections: {dict(rej)}")
    for k, v in cc.most_common(12):
        print(f"    {k:22s} {v}")
    # the check that was missing for five regenerations
    bad = [r for r in out if PLACEHOLDER.search(json.dumps(r))]
    assert not bad, f"{len(bad)} rows contain a template placeholder"
    print("[directive] placeholder assert: clean")


if __name__ == "__main__":
    main()
