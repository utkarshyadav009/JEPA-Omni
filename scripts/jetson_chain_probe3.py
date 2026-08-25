"""scripts/jetson_chain_probe3.py — the speaker latches onto the TAIL of the prompt.

PROBES 1-2 RULED OUT: prompt shape (all-in-user is the only format this model honours),
truncation (full vs [:200] byte-identical), and scene richness (richer scenes made the
THINKER *less* grounded, not more).

WHAT IS LEFT, from a direct diff of two runs that differed only in the thought's TAIL:
    thought ends "...keep the mood bright for Finn"          -> "Finn, you look a bit tired;
                                                                 maybe I can play a soft
                                                                 lullaby while you rest."  GROUNDED
    thought ends "...keep the mood bright for Finn and Jake" -> "Finn, Jake - just how you
                                                                 like it!"                 NOT grounded
Same scene, same format, same model, greedy decoding. The only delta is two trailing words.

HYPOTHESIS: the 350M speaker keys on the most recent salient entity. The thinker's habit of
signing off with Adventure Time character names (Finn/Jake -- personality artifacts from the
v9 companion corpus, not scene content) hijacks the line. If so the fix needs no retraining:
SANITISE the thought before handing it to the speaker -- keep the plan, drop the sign-off.

Tests four sanitisers against the real deployed GGUF.
"""
from __future__ import annotations
import argparse, json, os, re, sys
PROD = os.path.expanduser("~/bmo_production"); sys.path.insert(0, f"{PROD}/pipeline")

BASE = ("I see a person lying down, and I'm curious about them. Maybe they're resting or just "
        "want to be quiet - I should ask if they need anything while I can keep the mood "
        "bright for Finn and Jake.")
NAMES = re.compile(r"\b(finn|jake|bmo|princess bubblegum|marceline)\b", re.I)

def strip_names(t):        # drop character-name chatter entirely
    out = NAMES.sub("them", t)
    return re.sub(r"\s+", " ", out).strip()

def plan_only(t):          # keep just the decided action
    m = re.search(r"I should (.+?)(?: while | so | because |$)", t, re.I)
    return ("I should " + m.group(1).strip().rstrip(".")) if m else t

def first_two_sentences(t):
    return " ".join(re.split(r"(?<=[.!?]) ", t)[:2]).strip()

GROUND = {"sit","sitting","lying","rest","resting","quiet","need","anything","okay",
          "comfortable","tired","blanket","sleep","nap"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf")
    ap.add_argument("--tok", default=f"{PROD}/tokenizers/lfm25_350m_tok")
    ap.add_argument("--out", default=os.path.expanduser("~/chain_probe3.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier
    fast = GGUFFastTier(a.gguf, AutoTokenizer.from_pretrained(a.tok), max_new_tokens=48,
                        n_gpu_layers=-1)
    print("[probe3] speaker loaded\n", flush=True)
    R = []
    for name, th in (("0_baseline_with_names", BASE),
                     ("1_names_stripped", strip_names(BASE)),
                     ("2_plan_only", plan_only(BASE)),
                     ("3_plan_only_names_stripped", strip_names(plan_only(BASE))),
                     ("4_first_two_sentences", first_two_sentences(BASE))):
        p = (f"You see: a person lying down Your thought: {th} Say one short friendly line "
             f"to the person.")
        t = getattr(fast.generate(p, {"energy": 0.6, "mood": "curious"}), "text", "")
        g = any(k in t.lower() for k in GROUND)
        print(f"── {name:28s} grounded={g}\n   thought: {th[:110]}\n   SAYS:    {t}\n", flush=True)
        R.append({"variant": name, "thought": th, "said": t, "grounded": g})
    json.dump(R, open(a.out, "w"), indent=2); print(f"[probe3] wrote {a.out}")

if __name__ == "__main__":
    main()
