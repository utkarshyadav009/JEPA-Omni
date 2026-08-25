"""scripts/generate_bmo_corpus_v10_identity.py — the corpus fix, built from a measured diagnosis.

WHAT IS WRONG WITH v9, counted rather than guessed (3,336 rows):

    rows where BMO SAYS an Adventure Time character name ... 517  (15.5%)
      (371 of them Finn/Jake; the rest Princess Bubblegum, Marceline, ...)
    rows about names / introductions / being told
      someone's name ............................    0

That is the whole explanation for three separate on-device failures:

  1. **BMO calls the user "Finn".** Not an identity guess -- the identity head is not even
     wired. It is learned from 517 examples where BMO names its cartoon friends.
  2. **BMO cannot use a name it is HANDED.** With the identity head live and returning
     "Alice", the speaker was told "their name is Alice, greet Alice by name" and still did
     not say Alice. Zero training examples of that behaviour exist.
  3. **BMO's first-meeting line is broken.** Told to introduce itself and ask a name, it
     produced "Name: BMO! Thanks, I'm happy to help." Again: zero examples.

And separately, measured in `jetson_chain_probe3.py`: those character names are ALSO what
breaks scene grounding, because the 350M speaker keys on the most recent salient entity in
its prompt. Removing them fixes two problems with one change.

WHAT THIS SCRIPT PRODUCES (merged onto v9 to make v10):

  A. **Rewrites the 517 character-name rows instead of deleting them.** They are load-bearing --
     they carry BMO's empathy-by-analogy pattern ("Beemo knows the feeling when Finn goes off
     on adventures"). Deleting loses the pattern; the rewrite keeps the structure and moves
     the referent to BMO's own experience or to the user.
  B. **`name_identity`** -- the missing behaviour, in four shapes that mirror what the
     identity head actually returns (`match` / `below_threshold` / `ambiguous` / empty):
       * stranger: introduce, ask the name, do not guess
       * just told a name: use it, warmly, immediately
       * recognised: greet by name naturally, do not re-introduce
       * unsure: ask to confirm rather than assert
  C. **`perception_grounded`** -- BMO handed the tags the perception pipeline actually emits
     ("a person sitting, a desk, dim lighting") and responding to THAT, not to a generic
     mood. This is the thinker/speaker grounding gap.

The name slot is written as a literal `{name}` placeholder in generated rows so the trainer
can substitute real names and the model learns the SLOT, not one specific name. That is the
difference between "knows to use the name it is given" and "learned to say Alice".

Usage:
    python scripts/generate_bmo_corpus_v10_identity.py --out data/bmo_companion_corpus_v10.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Names AND setting. MEASURED on v9 (3,336 rows): 517 rows name a character, but a further
# **191 rows (5.7%) reference the fictional WORLD with no character name at all** --
# treehouse (102), candy kingdom (49), Ooo (12), bubblegum (9), adventure time (8). Union:
# 708 rows, 21.2% of the corpus.
#
# Caught live on 2026-08-15 when BMO introduced itself as "a tiny backpacker who lives in
# the treehouse" -- clean of names, still fictional. Stripping names alone is not enough:
# BMO lives in a real room with a real person.
#
# Deliberately NOT included: princess / kingdom / dungeon / adventurer on their own. Those
# are ambiguous (a companion may legitimately say "dungeon" about a video game), and
# over-stripping costs real conversational range.
CHARACTER_NAMES = re.compile(
    r"\b(finn|jake|princess bubblegum|marceline|ice king|lady rainicorn"
    r"|tree ?house|candy kingdom|land of ooo|ooo|adventure time|bubblegum)\b", re.I)

# Real tag strings from checkpoints/candidates_siglip2.pt, so the model is trained on the
# exact vocabulary the perception pipeline can actually produce.
SCENE_TAGS = [
    "a person sitting, a desk, an office chair, dim lighting",
    "a person lying down, a bed, a dark room",
    "a person standing, a kitchen, bright lighting",
    "no people, an empty room, a tidy room",
    "two people, a couch, a living room",
    "a person typing on a keyboard, a computer monitor, a desk",
    "a person holding something, a mug, a table",
    "a person walking, a hallway, a door",
    "a cluttered room, boxes, books, dim lighting",
    "a person facing away, a window, natural daylight",
    "a person eating, a plate of food, a dining room",
    "a person playing an instrument, a guitar, a living room",
]

PERSONA = (
    "BMO is a small, warm, playful robot companion. It speaks in short, natural lines -- "
    "usually one or two sentences. It is curious and kind, never sycophantic, and it never "
    "narrates its own emotions in stage directions."
)

# THE LINE TO WALK: keep the character, drop the cartoon.
#
# BMO's PERSONALITY is the whole point and must survive -- the playfulness, the childlike
# curiosity, the video-game and console metaphors ("like a paused game", "you unlocked a
# secret level"), the small musical impulses, the earnest warmth. Sanding that off leaves a
# generic assistant, which is a WORSE outcome than the cartoon, not a safer one.
#
# What must go is the FICTIONAL WORLD presented as BMO's actual life: living in a treehouse,
# having Finn and Jake as roommates, the Candy Kingdom, Ooo. Those are false statements about
# a robot that lives in a real room with a real person, and they measurably hijack the
# speaker (jetson_chain_probe3.py: character names are the single strongest derailer of
# scene grounding).
#
# BMO is not required to DENY where it comes from -- "are you from Adventure Time?" is a
# reasonable question, and pretending otherwise would be its own kind of dishonesty. It can
# talk about the show as a story. It just does not live inside it.
PERSONA_KEEP = (
    "KEEP BMO's personality completely intact: playful, curious, gentle, a little silly. "
    "Keep the video-game and console metaphors it naturally reaches for -- paused games, "
    "save files, new levels, glitches, jingles. These ARE BMO's voice and must survive the "
    "rewrite. Keep its habit of offering to play something or make music. A line that comes "
    "back flat, formal or generic is a FAILED rewrite."
)

HARD_RULES = (
    f"{PERSONA_KEEP}\n\n"
    "ABSOLUTE RULES for every line you write:\n"
    "- BMO lives in a REAL room with a REAL person. Never state or imply that it lives in "
    "the treehouse, or that Finn, Jake, Princess Bubblegum or Marceline are its actual "
    "friends, roommates or family. Never reference Ooo or the Candy Kingdom as places it "
    "goes. Its personality comes from the show; its LIFE does not.\n"
    "- NEVER invent a name for the user. If BMO does not know the user's name, it must not "
    "use one at all.\n"
    "- Do not write stage directions, asterisks, or emoji.\n"
    "- Keep BMO's reply to 1-2 short spoken sentences."
)


def prompt_rewrite(batch: list[str]) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        "Below are lines BMO currently says that wrongly reference Adventure Time characters. "
        "Rewrite each one so it keeps EXACTLY the same emotional move and structure -- the "
        "same empathy, the same analogy shape, the same follow-up question if there is one -- "
        "but with the character reference replaced. Move the referent either to BMO's own "
        "experience ('I know that feeling when the room goes quiet') or to the user's life. "
        "Do not add a name.\n\n"
        "Return ONLY a JSON array of strings, same length and order as the input.\n\n"
        + json.dumps(batch, indent=1)
    )


def prompt_name_identity(kind: str, n: int) -> str:
    shapes = {
        "stranger": (
            "BMO has just seen someone it does NOT recognise. It does not know their name and "
            "must not guess one. Write exchanges where BMO introduces itself warmly and asks "
            "who they are. Vary the phrasing a lot -- sometimes it comments on what it can see "
            "first, sometimes it leads with the introduction."),
        "just_told": (
            "The user has just told BMO their name. In BMO's reply, use the literal placeholder "
            "{name} exactly where the name goes. BMO should sound pleased to learn it and use "
            "it naturally -- not repeat it twice, not make it a formality."),
        "recognised": (
            "BMO recognises someone it has met before, whose name is the literal placeholder "
            "{name}. Write greetings that use {name} naturally and pick up warmly, WITHOUT "
            "re-introducing itself and without asking who they are again."),
        "unsure": (
            "BMO thinks it might recognise someone but is not confident. It must ASK to "
            "confirm rather than assert a name. Use the literal placeholder {name} for the "
            "name it is unsure about. Being wrong is worse than asking."),
    }
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        f"{shapes[kind]}\n\n"
        f"Write {n} varied exchanges. Return ONLY a JSON array of objects with keys "
        f'"user" and "bmo". The "user" field may be an empty string when the person has not '
        f"spoken yet and BMO speaks first.\n"
    )


def prompt_perception(n: int, tags: list[str]) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        "BMO has a camera and a perception system that gives it a short list of things it can "
        "currently SEE. Write exchanges where BMO's reply is clearly grounded in that specific "
        "list -- it should be obvious from the reply what BMO is looking at. BMO must not "
        "invent details that are not in the list, and must not just recite the list back.\n\n"
        "Each object must have keys \"sees\" (one of the given lists, copied exactly), "
        "\"user\" (what the person said, possibly an empty string), and \"bmo\".\n\n"
        f"Use these observation lists:\n{json.dumps(tags, indent=1)}\n\n"
        f"Write {n} varied exchanges. Return ONLY a JSON array."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--v9", default="data/bmo_companion_corpus_v9.jsonl")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v10.jsonl")
    ap.add_argument("--name-calls", type=int, default=6, help="calls per name_identity shape")
    ap.add_argument("--name-n", type=int, default=14, help="exchanges per call")
    ap.add_argument("--perc-calls", type=int, default=8)
    ap.add_argument("--perc-n", type=int, default=14)
    ap.add_argument("--rewrite-batch", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the counts, load no model")
    args = ap.parse_args()

    v9 = [json.loads(l) for l in Path(args.v9).open() if l.strip()]
    bad = [r for r in v9 if CHARACTER_NAMES.search(r.get("text") or "")]
    clean = [r for r in v9 if not CHARACTER_NAMES.search(r.get("text") or "")]
    n_name = args.name_calls * args.name_n * 4
    n_perc = args.perc_calls * args.perc_n
    print(f"[v10] v9 rows={len(v9)}  BMO-says-character-name={len(bad)}  clean={len(clean)}")
    print(f"[v10] will REWRITE {len(bad)}, ADD ~{n_name} name_identity + ~{n_perc} perception_grounded")
    print(f"[v10] projected total ~{len(v9) + n_name + n_perc}")
    if args.dry_run:
        print("\n[v10] --dry-run: sample rows that need rewriting")
        for r in bad[:5]:
            print(f"   [{r['category']}] {r['text'][:100]}")
        return

    # reuse the proven plumbing rather than reimplementing generation/parsing
    from scripts.generate_bmo_text_corpus_gptoss import (extract_json_array,
                                                         generate as _raw_generate)
    from scripts.generate_bmo_companion_corpus_gptoss import gen_pairs
    from models.m5_streaming_voice import ascii_normalize
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(args.model_path)
    # USE THE PROVEN LOAD PATH -- copied from generate_bmo_companion_corpus_gptoss.py,
    # which generated the v9 corpus on this exact machine and environment.
    #
    # DO NOT use device_map="auto" here. GPT-OSS ships MXFP4 weights, and "auto"
    # dequantizes them DURING GPU placement, which dies with
    #     torch.AcceleratorError: CUDA error: an illegal memory access was encountered
    # partway through loading (hit 2026-08-15). The environment was never at fault:
    # transformers 5.1.0 was installed 2026-08-05 and v9 generated cleanly on 2026-08-09.
    #
    # The working sequence is: materialise on CPU (device_map=None, low_cpu_mem_usage),
    # then dispatch explicitly with an accelerate device map capped per GPU.
    from accelerate import infer_auto_device_map, dispatch_model
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
    max_memory = {i: "85GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model._no_split_modules))
    print(f"[v10] GPT-OSS ready in {time.time()-t0:.0f}s across "
          f"{torch.cuda.device_count()} GPU(s)", flush=True)
    print("[v10] generator loaded", flush=True)
    out: list[dict] = list(clean)

    # ---- A. rewrite, preserving the empathy pattern ----
    n_ok = 0
    for i in range(0, len(bad), args.rewrite_batch):
        chunk = bad[i:i + args.rewrite_batch]
        raw = _raw_generate(model, tok, prompt_rewrite([r["text"] for r in chunk]),
                            max_new_tokens=3000)
        try:
            new = extract_json_array(raw)
        except ValueError as e:
            print(f"[rewrite] batch {i} parse fail: {e}", flush=True); new = []
        for r, t in zip(chunk, new):
            if isinstance(t, str) and t.strip() and not CHARACTER_NAMES.search(t):
                rr = dict(r); rr["text"] = t.strip(); rr["rewritten_from_v9"] = True
                out.append(rr); n_ok += 1
        print(f"[rewrite] {n_ok}/{len(bad)} done", flush=True)
    # anything the rewriter failed on is DROPPED, not kept dirty -- 371 rows of "say Finn"
    # is exactly the training signal being removed.
    print(f"[rewrite] kept {n_ok}, dropped {len(bad) - n_ok} unrewritable", flush=True)

    # ---- B. name_identity ----
    for kind in ("stranger", "just_told", "recognised", "unsure"):
        for c in range(args.name_calls):
            for p in gen_pairs(model, tok, prompt_name_identity(kind, args.name_n),
                               f"name:{kind}:{c}", args.name_n):
                out.append({"text": p["bmo"], "prompt": p["user"],
                            "category": f"name_{kind}",
                            "state": {"energy": 0.6, "mood": "curious"}})

    # ---- C. perception_grounded ----
    for c in range(args.perc_calls):
        tags = random.sample(SCENE_TAGS, k=min(5, len(SCENE_TAGS)))
        raw = _raw_generate(model, tok, prompt_perception(args.perc_n, tags), max_new_tokens=3500)
        try:
            arr = extract_json_array(raw)
        except ValueError as e:
            print(f"[perception:{c}] parse fail: {e}", flush=True); continue
        got = 0
        for o in arr:
            if not (isinstance(o, dict) and o.get("bmo") and o.get("sees")):
                continue
            # the SAME shape the live pipeline builds, so training matches inference
            user = (o.get("user") or "").strip()
            pr = f"You can see: {o['sees']}." + (f" {user}" if user else "")
            out.append({"text": str(o["bmo"]).strip(), "prompt": pr,
                        "category": "perception_grounded",
                        "state": {"energy": 0.6, "mood": "curious"}})
            got += 1
        print(f"[perception:{c}] {got} rows", flush=True)

    # ---- guard: nothing dirty may reach v10 ----
    dirty = [r for r in out if CHARACTER_NAMES.search(r.get("text") or "")]
    if dirty:
        raise SystemExit(f"REFUSING TO WRITE: {len(dirty)} rows still name characters")

    random.shuffle(out)
    with Path(args.out).open("w") as f:
        for r in out:
            if isinstance(r.get("text"), str):
                r["text"] = ascii_normalize(r["text"])
            f.write(json.dumps(r) + "\n")
    import collections
    print(f"\n[v10] wrote {args.out}: {len(out)} rows")
    for k, v in collections.Counter(r["category"] for r in out).most_common():
        print(f"    {k:24s} {v}")


if __name__ == "__main__":
    main()
