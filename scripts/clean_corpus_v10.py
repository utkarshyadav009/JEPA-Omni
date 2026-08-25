"""scripts/clean_corpus_v10.py — finish v10: strip the fictional SETTING, and give the
perception rows BMO's voice back.

TWO DEFECTS, both measured on the generated v10 (3,630 rows):

1. **197 rows still reference the Adventure Time WORLD.** The v10 run predated the extended
   regex, so it stripped character NAMES (0 remain) but not settings: treehouse, Candy
   Kingdom, Ooo, bubblegum. Caught live when BMO introduced itself as "a tiny backpacker who
   lives in the treehouse" -- clean of names, still a cartoon. Concentrated in playful (47),
   companion_memory (40), warmth (28), emotional_support (26), tool_use (24).

2. **The 112 `perception_grounded` rows are FLAT.** They are correctly grounded and
   completely voiceless:
       "You can see: a person eating, a plate of food, a dining room."
         -> "The dining room looks nice."
       "...a person typing on a keyboard, a computer monitor, a desk"
         -> "Your keyboard clicks nicely beside the monitor on the desk."
   Accurate, and nothing like BMO. These were generated before the PERSONA_KEEP rules
   existed, so they get REGENERATED rather than patched -- a flat line cannot be edited into
   a characterful one.

THE LINE THIS WALKS, again: keep the character, drop the cartoon. Removing the fictional
setting must not sand off the playfulness, the console metaphors, or the musical impulses --
a corpus that produces a polite generic assistant is a worse outcome than one that produces a
cartoon, because the cartoon at least had a personality.

Usage:
    python scripts/clean_corpus_v10.py --inp data/bmo_companion_corpus_v10.jsonl \
                                       --out data/bmo_companion_corpus_v10c.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_bmo_corpus_v10_identity import (  # single source of truth
    CHARACTER_NAMES, HARD_RULES, PERSONA, SCENE_TAGS, prompt_perception)

SETTING = re.compile(r"\b(tree ?house|candy kingdom|land of ooo|\booo\b|adventure time"
                     r"|bubblegum)\b", re.I)


def prompt_setting_rewrite(batch: list) -> str:
    return (
        f"{PERSONA}\n\n{HARD_RULES}\n\n"
        "Each line below is something BMO says that still places it inside the Adventure "
        "Time world -- living in a treehouse, visiting the Candy Kingdom, being in Ooo. "
        "Rewrite each so BMO is unmistakably in a REAL room with a REAL person, while "
        "keeping the SAME emotional move, the same warmth, and above all the same "
        "playfulness and game/console imagery. Replace a cartoon place with something real "
        "and specific (a quiet room, the window, the corner where the light is bad) or with "
        "BMO's own inner experience.\n\n"
        "A rewrite that is correct but flat has FAILED. Keep BMO sounding like BMO.\n\n"
        "Return ONLY a JSON array of strings, same length and order as the input.\n\n"
        + json.dumps(batch, indent=1)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--inp", default="data/bmo_companion_corpus_v10.jsonl")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v10c.jsonl")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--perc-calls", type=int, default=9)
    ap.add_argument("--perc-n", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).open() if l.strip()]
    dirty = [r for r in rows if SETTING.search(r.get("text") or "")]
    flat = [r for r in rows if r["category"] == "perception_grounded"]
    keep = [r for r in rows
            if not SETTING.search(r.get("text") or "") and r["category"] != "perception_grounded"]
    print(f"[clean] {len(rows)} rows -> rewrite {len(dirty)} setting rows, "
          f"REGENERATE {len(flat)} flat perception rows, keep {len(keep)}", flush=True)
    if args.dry_run:
        for r in dirty[:4]:
            print(f"   dirty [{r['category']}] {r['text'][:95]}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model
    from scripts.generate_bmo_text_corpus_gptoss import (extract_json_array,
                                                         generate as _raw_generate)
    from models.m5_streaming_voice import ascii_normalize

    tok = AutoTokenizer.from_pretrained(args.model_path)
    # proven load path -- device_map="auto" dequantizes MXFP4 during placement and dies
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
    mm = {i: "85GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=mm, no_split_module_classes=model._no_split_modules))
    print("[clean] generator ready", flush=True)

    out = list(keep)

    # ---- 1. de-cartoon the setting rows ----
    ok = 0
    for i in range(0, len(dirty), args.batch):
        chunk = dirty[i:i + args.batch]
        raw = _raw_generate(model, tok, prompt_setting_rewrite([r["text"] for r in chunk]),
                            max_new_tokens=3000)
        try:
            new = extract_json_array(raw)
        except ValueError as e:
            print(f"[setting] batch {i} parse fail: {e}", flush=True); new = []
        for r, t in zip(chunk, new):
            if isinstance(t, str) and t.strip() and not SETTING.search(t) \
               and not CHARACTER_NAMES.search(t):
                rr = dict(r); rr["text"] = t.strip(); rr["setting_cleaned"] = True
                out.append(rr); ok += 1
        print(f"[setting] {ok}/{len(dirty)}", flush=True)
    print(f"[setting] kept {ok}, dropped {len(dirty) - ok} unrewritable", flush=True)

    # ---- 2. regenerate the perception rows WITH personality ----
    got = 0
    for c in range(args.perc_calls):
        tags = random.sample(SCENE_TAGS, k=min(5, len(SCENE_TAGS)))
        raw = _raw_generate(model, tok, prompt_perception(args.perc_n, tags),
                            max_new_tokens=3500)
        try:
            arr = extract_json_array(raw)
        except ValueError as e:
            print(f"[perception:{c}] parse fail: {e}", flush=True); continue
        for o in arr:
            if not (isinstance(o, dict) and o.get("bmo") and o.get("sees")):
                continue
            if CHARACTER_NAMES.search(str(o["bmo"])) or SETTING.search(str(o["bmo"])):
                continue
            user = (o.get("user") or "").strip()
            pr = f"You can see: {o['sees']}." + (f" {user}" if user else "")
            out.append({"text": str(o["bmo"]).strip(), "prompt": pr,
                        "category": "perception_grounded",
                        "state": {"energy": 0.6, "mood": "curious"}})
            got += 1
        print(f"[perception:{c}] {got} rows so far", flush=True)

    # ---- guard: nothing cartoon may reach the trainer ----
    bad = [r for r in out
           if CHARACTER_NAMES.search(r.get("text") or "") or SETTING.search(r.get("text") or "")]
    if bad:
        raise SystemExit(f"REFUSING TO WRITE: {len(bad)} rows still reference the cartoon")

    random.shuffle(out)
    with Path(args.out).open("w") as f:
        for r in out:
            if isinstance(r.get("text"), str):
                r["text"] = ascii_normalize(r["text"])
            f.write(json.dumps(r) + "\n")
    import collections
    print(f"\n[clean] wrote {args.out}: {len(out)} rows")
    for k, v in collections.Counter(r["category"] for r in out).most_common():
        print(f"    {k:24s} {v}")


if __name__ == "__main__":
    main()
