"""scripts/build_candidate_vocab.py — build the two candidate sets the deployed perception
stack can retrieve against, both pre-encoded in SigLIP2's frozen text space.

THE DISTINCTION THAT MATTERS (do not collapse these two):

  * **CAPTIONS** are what the query predictor was TRAINED to hit. Its output is optimised to
    land on a caption embedding for the asked field, so caption retrieval is the in-geometry
    use of the trained model -- and it is the only path that uses AUDIO at all (via the m2 /
    ambient streams). The AV congruence eval measured this mattering: a vision-only arm
    answers sound questions from the picture 93% of the time.
  * **TAGS** are short concept phrases ("a desk", "two people", "dim lighting"). They work
    natively for ZERO-SHOT SigLIP2 scoring, because a tag embedding and an image embedding
    share the pretrained joint space with no trained head in between -- which is exactly why
    they generalise to rooms the corpus never contained. But the query predictor was never
    optimised to land on tag points, so tag retrieval THROUGH the predictor is an
    extrapolation, not a designed capability.

So this script emits both, and `scripts/eval_candidate_sets.py` measures rather than assumes.
Whether tags survive the predictor path is an empirical question; whether they beat captions
zero-shot is a different empirical question.

WHY TAGS ARE WORTH THE TRY. The 121k caption bank is 355 MiB of VGGSound/Action100M
narration averaging 132 characters. Three problems: it is corpus-shaped (so it has no good
sentence for a bedroom), it is large, and SigLIP2's text tower truncates at 64 tokens so long
narration sits at the edge of what that space represents well. SigLIP2 is also trained with a
SIGMOID loss -- independent yes/no per (image, text) pair rather than a softmax over a batch
-- which is precisely the objective a multi-label tag vocabulary wants.

The home/room categories below are hand-written on purpose: BMO is deployed in a room, and
neither corpus contains that vocabulary. The corpus-mined half covers the action/sound space
the hand-written half does not.

Usage:
    python scripts/build_candidate_vocab.py --out checkpoints/candidates_siglip2.pt
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── hand-written: BMO's actual deployment domain, absent from both corpora ──
PLACES = ["a bedroom", "a living room", "a home office", "a kitchen", "a bathroom",
          "a hallway", "a dining room", "a garage", "a basement", "a closet",
          "an office", "a classroom", "a laboratory", "a workshop", "a garden",
          "a balcony", "a staircase", "an empty room", "a cluttered room",
          "a tidy room", "outdoors", "indoors", "a street", "a parking lot",
          "a shop", "a restaurant", "a corridor", "a lobby"]

OBJECTS = ["a desk", "an office chair", "a couch", "a bed", "a table", "a bookshelf",
           "a wardrobe", "a lamp", "a window", "a door", "a mirror", "a rug",
           "a television", "a computer monitor", "a laptop", "a keyboard", "a mouse",
           "a phone", "a tablet", "headphones", "a speaker", "a camera", "a microphone",
           "a robot", "a plant", "a poster on the wall", "a painting on the wall",
           "photographs on the wall", "a clock", "a fan", "a heater", "a refrigerator",
           "a microwave", "an oven", "a sink", "a kettle", "a mug", "a glass of water",
           "a bottle", "a plate of food", "cutlery", "a backpack", "a box", "boxes",
           "clothes", "shoes", "a blanket", "a pillow", "books", "papers", "a notebook",
           "a pen", "cables", "a power strip", "a guitar", "a keyboard instrument",
           "a bicycle", "a car", "a whiteboard", "a curtain", "a shelf", "a trash bin",
           "a staircase railing", "a cardboard box", "a toy", "a stuffed animal"]

PEOPLE = ["no people", "one person", "two people", "three people", "several people",
          "a crowd of people", "a person sitting", "a person standing", "a person walking",
          "a person lying down", "a person facing the camera", "a person facing away",
          "a person's face close to the camera", "a child", "an adult", "a person waving",
          "a person holding something", "a person wearing glasses", "a person smiling",
          "a person talking"]

LIGHT_TIME = ["bright lighting", "dim lighting", "dark", "natural daylight",
              "artificial indoor light", "a lit screen in a dark room", "daytime",
              "nighttime", "a sunlit room", "a shadowy room"]

CAMERA = ["a close-up view", "a wide view of a room", "a blurry image", "a sharp image",
          "an empty scene with no movement", "a cluttered scene"]

SOUNDS = ["speech", "a person talking", "several people talking", "laughter", "silence",
          "quiet background noise", "music playing", "a television playing",
          "typing on a keyboard", "footsteps", "a door opening", "a door closing",
          "a knock on the door", "running water", "dishes clattering", "a fan humming",
          "traffic noise", "birds chirping", "a dog barking", "a cat meowing",
          "a phone ringing", "an alarm beeping", "a baby crying", "coughing",
          "clapping", "wind", "rain", "an engine running", "a vacuum cleaner",
          "machinery", "a bell", "whistling", "singing", "a crash", "glass breaking"]

ACTIONS = ["someone is cooking", "someone is eating", "someone is drinking",
           "someone is reading", "someone is writing", "someone is typing",
           "someone is cleaning", "someone is exercising", "someone is dancing",
           "someone is playing an instrument", "someone is using a phone",
           "someone is watching a screen", "someone is opening something",
           "someone is closing something", "someone is picking something up",
           "someone is putting something down", "someone is entering the room",
           "someone is leaving the room", "someone is gesturing", "nothing is happening",
           "someone is repairing something", "someone is carrying something"]

# APPEARANCE — added 2026-08-15 for a specific companion behaviour: noticing what someone
# is WEARING and remarking on it ("that's a great red jumper").
#
# The gap this closes was measured, not assumed. The mined vocabulary already contained the
# bare words "blue", "red", "shirt", "jacket" -- but bare colours match ANYTHING blue in the
# room, not what a person has on. SigLIP2 scores whole phrases, so the tag has to name the
# garment AND the colour together for the match to mean what we want.
#
# Composed rather than hand-listed: 11 colours x 9 garments = 99 tags at ~768 floats each,
# about 0.15 MiB on a 2 MiB candidate set. Appearance is also the most socially useful thing
# a companion can notice cheaply -- it changes day to day, unlike the room.
_COLOURS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple",
            "pink", "grey", "brown"]
_GARMENTS = ["jumper", "sweater", "shirt", "t-shirt", "jacket", "hoodie", "dress", "coat", "hat"]
APPEARANCE = [f"a person wearing a {c} {g}" for c in _COLOURS for g in _GARMENTS] + [
    "a person wearing glasses", "a person wearing a cap", "a person wearing headphones",
    "a person with long hair", "a person with short hair", "a person with a beard",
    "a person wearing a scarf", "a person in pyjamas", "a person in a suit",
    "a person wearing a backpack", "a person holding a mug", "a person wrapped in a blanket",
]

CURATED = {"place": PLACES, "object": OBJECTS, "people": PEOPLE, "light": LIGHT_TIME,
           "camera": CAMERA, "sound": SOUNDS, "action": ACTIONS,
           "appearance": APPEARANCE}

STOP = set("""a an the of in on at to for with and or is are was were be been being this that
these those it its as by from into over under near very there here they he she his her their
someone something people person while during after before then than which who whom whose
video shows depicts featuring showing appears seen visible camera scene clip footage""".split())


def mine_corpus_terms(captions_path: str, top_k: int, min_len: int = 4) -> list:
    """Frequent content words from the corpus, as '<word>' phrases. Covers the action/sound
    space the hand-written lists do not. Deliberately simple: SigLIP2 scores short phrases
    well, so a curated-ish frequency list beats a parser here."""
    import train_m3
    train_m3.CAPTIONS_PATH = captions_path
    from train_m3 import build_splits
    from models.query_predictor import VGGSOUND_FIELDS

    cnt = collections.Counter()
    tr, te = build_splits(VGGSOUND_FIELDS)
    for pairs in (tr, te):
        for _cid, _f, txt in pairs:
            for w in re.findall(r"[a-z]+", (txt or "").lower()):
                if len(w) >= min_len and w not in STOP:
                    cnt[w] += 1
    terms = [w for w, _ in cnt.most_common(top_k)]
    print(f"[vocab] mined {len(terms)} corpus terms (top by frequency)", flush=True)
    return terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-path",
                    default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl"))
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--mine-top-k", type=int, default=1200)
    ap.add_argument("--out", default="checkpoints/candidates_siglip2.pt")
    ap.add_argument("--out-queries", default="checkpoints/query_vectors_siglip2.pt",
                    help="pre-encoded QUERY_BANK phrasings, so the device needs no text "
                         "encoder to encode the thinker's question either "
                         "(models/text_target.py::PreEncodedTextSpace)")
    args = ap.parse_args()

    tags, cats = [], []
    for cat, items in CURATED.items():
        for t in items:
            tags.append(t); cats.append(cat)
    n_curated = len(tags)
    for w in mine_corpus_terms(args.captions_path, args.mine_top_k):
        tags.append(w); cats.append("mined")

    # de-duplicate, preserving the curated entry when a mined word collides
    seen, keep_t, keep_c = set(), [], []
    for t, c in zip(tags, cats):
        if t.lower() in seen:
            continue
        seen.add(t.lower()); keep_t.append(t); keep_c.append(c)
    tags, cats = keep_t, keep_c
    print(f"[vocab] {len(tags)} tags total ({n_curated} curated + mined, deduped)", flush=True)

    from models.text_target import SigLIP2TextTarget
    tt = SigLIP2TextTarget(repo=args.siglip, device="cuda")
    emb = tt.encode_text(tags).cpu().to(torch.float16)

    torch.save({"emb": emb, "text": tags, "category": cats, "siglip": args.siglip,
                "n_curated": n_curated}, args.out)
    print(f"[vocab] wrote {args.out}  {tuple(emb.shape)}  "
          f"{emb.numel()*2/2**20:.2f} MiB", flush=True)
    for c in CURATED:
        print(f"    {c:8s} {sum(1 for x in cats if x == c):4d}")
    print(f"    {'mined':8s} {sum(1 for x in cats if x == 'mined'):4d}")

    # ── every query phrasing the predictor was trained/evaluated on ──
    from models.query_predictor import QUERY_BANK
    qtexts, qfields = [], []
    for fld, phrasings in QUERY_BANK.items():
        for q in phrasings:
            qtexts.append(q); qfields.append(fld)
    qemb = tt.encode_text(qtexts).cpu().to(torch.float16)
    torch.save({"emb": qemb, "text": qtexts, "field": qfields, "siglip": args.siglip},
               args.out_queries)
    print(f"[vocab] wrote {args.out_queries}  {tuple(qemb.shape)}  "
          f"{qemb.numel()*2/2**20:.3f} MiB  ({len(set(qfields))} fields)", flush=True)


if __name__ == "__main__":
    main()
