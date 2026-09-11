"""Rebuild the perception candidate bank: replace the junk `mined` tags and add categories.

AUDIT THAT MOTIVATES THIS (2026-08-26, candidates_siglip2_v2.pt):
  mined       1186 (80%)  -- 100% SINGLE WORDS: "sound", "playing", "contains", "through".
                             Produced by build_candidate_vocab.py's `--mine-top-k 1200`
                             caption-vocabulary mine, unfiltered. A function word can never be
                             a useful perception answer.
  appearance   110  object 66  sound 34  place 28  action 22  people 20  light 10  camera 6
Only 296/1482 (20%) of the bank could ever be returned usefully.

WHY THIS IS THE RIGHT LEVER. The query predictor knows only 6 field intents and cannot be
extended without retraining (models/query_predictor.py VGGSOUND_FIELDS). But the ANSWER comes
from category-restricted retrieval over this bank -- which is why `wearing` works today despite
there being no `wearing` query field. So detail is added by improving the BANK, offline, with
no training and no resident text encoder. Cost measured at 1.50 KB/tag: 5,000 tags = 7.7 MB.

TEMPLATES, NOT LLM GENERATION. Most perception categories are combinatorial (colour x garment,
posture, count) and a template expansion is deterministic, reviewable and free of hallucinated
phrases. Phrasing matches the existing curated tags ("a person wearing a red jumper") so the
new entries live in the same region of SigLIP2 space as the ones already known to retrieve well.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

COLOURS = ["red", "blue", "green", "black", "white", "grey", "yellow", "orange", "purple",
           "pink", "brown", "navy", "beige", "striped", "dark", "light-coloured"]
# Near-synonyms REMOVED after measuring within-category cosine: sweater/fleece ~ jumper,
# coat ~ jacket (0.988), blouse ~ shirt. Keeping both halves of such a pair guarantees a
# tiny top1-top2 margin on a CORRECT answer, which the confidence gate then discards.
TOPS = ["jumper", "hoodie", "t-shirt", "shirt", "jacket", "cardigan", "vest", "dress"]
HEAD = ["a cap", "a beanie", "a hat", "headphones", "earphones", "glasses", "sunglasses",
        "a headband", "a hood up", "nothing on their head"]
POSTURE = ["sitting upright", "slouching", "leaning forward", "leaning back", "lying down",
           "standing", "crouching", "kneeling", "sitting cross-legged", "curled up",
           "stretching", "turning away", "facing the camera", "facing away", "head in hands",
           "arms crossed", "resting their chin on their hand", "head tilted"]
HELD = ["a mug", "a glass", "a bottle", "a phone", "a book", "a laptop", "a controller",
        "a pen", "a remote", "a plate", "a fork", "headphones", "a bag", "keys", "a notebook",
        "a cable", "a tablet", "nothing"]
ACTIVITY = ["typing", "writing", "reading", "eating", "drinking", "talking", "laughing",
            "yawning", "stretching", "scrolling on a phone", "watching a screen",
            "playing an instrument", "cooking", "tidying up", "pacing", "sitting still",
            "rubbing their eyes", "waving", "pointing at something", "nodding",
            "shaking their head", "clapping", "getting up", "sitting down", "walking in",
            "walking out", "reaching for something", "putting something down"]
EXPRESSION = ["smiling", "frowning", "looking tired", "looking focused", "looking confused",
              "looking surprised", "looking relaxed", "looking worried", "laughing",
              "with their eyes closed", "with a neutral expression", "looking away"]
COUNT = ["one person", "two people", "three people", "several people", "a small group",
         "nobody", "a person and a pet"]
PLACE = ["a home office", "a bedroom", "a living room", "a kitchen", "a hallway", "a study",
         "a dining room", "a bathroom doorway", "a garage", "a balcony", "a garden",
         "a car interior", "a classroom", "an office", "a cluttered room", "a tidy room",
         "a small room", "a large room", "a room with a window", "a windowless room"]
FURNITURE = ["a desk", "a chair", "an office chair", "a sofa", "a bed", "a table",
             "a bookshelf", "a lamp", "a rug", "a cupboard", "a mirror", "a plant",
             "curtains", "a whiteboard", "a monitor", "two monitors", "a television",
             "a keyboard", "a mouse", "a printer", "a fan", "a radiator", "a kettle",
             "a fridge", "a sink", "a power strip", "a laundry basket", "a cardboard box"]
LIGHT = ["dim lighting", "bright lighting", "natural daylight", "artificial indoor light",
         "a lit screen in a dark room", "warm lamplight", "cool white light", "backlit",
         "sunlight through a window", "almost dark", "flickering light", "evening light"]
SOUND = ["a fan humming", "a kettle boiling", "footsteps", "a door closing", "typing",
         "music playing", "a television", "several people talking", "one person talking",
         "quiet background noise", "near silence", "traffic outside", "rain", "a phone ringing",
         "a notification chime", "a chair creaking", "cutlery", "running water",
         "a dog barking", "laughter", "a clock ticking", "an alarm", "wind", "birdsong"]
TIME = ["early morning", "the middle of the day", "late afternoon", "evening", "late at night"]
ANIMAL = ["a cat", "a dog", "a cat on the sofa", "a dog by the door", "no animals"]

def build() -> list[tuple[str, str]]:
    """-> [(text, category)]. Phrasing deliberately mirrors the existing curated tags."""
    out: list[tuple[str, str]] = []
    add = lambda t, c: out.append((t, c))
    for col, top in itertools.product(COLOURS, TOPS):
        add(f"a person wearing a {col} {top}", "appearance")
    for h in HEAD:
        add(f"a person wearing {h}" if not h.startswith("nothing") else "a person with "+h, "appearance")
    for p in POSTURE:   add(f"a person {p}", "posture")
    for h in HELD:      add(f"a person holding {h}", "held_object")
    for a in ACTIVITY:  add(f"someone is {a}", "action")
    for e in EXPRESSION:add(f"a person {e}", "expression")
    for c in COUNT:     add(c, "people")
    for p in PLACE:     add(p, "place")
    for f in FURNITURE: add(f, "object")
    for l in LIGHT:     add(l, "light")
    for s in SOUND:     add(s, "sound")
    # TIME deliberately not emitted: measured mean within-category cosine 0.937
    # ("early morning" vs "late afternoon" = 0.962). SigLIP2 cannot read clock time
    # off a frame, so these tags could only ever contribute confident noise.
    for a in ANIMAL:    add(a, "animal")
    seen, uniq = set(), []
    for t, c in out:
        if t.lower() not in seen:
            seen.add(t.lower()); uniq.append((t, c))
    return uniq

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/candidates_siglip2_v2.pt",
                    help="existing bank; its CURATED categories are kept, `mined` is dropped")
    ap.add_argument("--out", default="checkpoints/candidates_siglip2_v3.pt")
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--dedup-cos", type=float, default=0.95,
                    help="drop a tag this close to one already kept in its category")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new = build()
    import collections
    print(f"[vocab] generated {len(new)} phrases: "
          f"{collections.Counter(c for _, c in new).most_common()}", flush=True)
    if args.dry_run:
        for t, c in new[:12]: print(f"   {c:<12} {t}")
        return

    import torch, torch.nn.functional as F
    import collections

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    keep = [(t, c) for t, c in zip(base["text"], base.get("category", []))
            if c != "mined"]                      # drop the 1,186 single-word tags
    print(f"[vocab] kept {len(keep)} curated tags from the old bank (dropped "
          f"{len(base['text'])-len(keep)} `mined`)", flush=True)

    # RELABEL MISLABELLED v2 TAGS. `a person holding a mug` shipped in v2 under
    # `appearance`, so the `wearing` question could answer with a held object -- observed
    # live as "wearing: a person holding a mug". Exact-string dedup below keeps the FIRST
    # occurrence, which is the v2 entry, so its wrong category would survive even though
    # the template generates the same phrase correctly under `held_object`. Fix the
    # category by rule rather than by hoping dedup order saves us.
    def _fix(t: str, c: str) -> str:
        if " holding " in t and c != "held_object":
            return "held_object"
        return c
    keep = [(t, _fix(t, c)) for t, c in keep]

    merged, seen = [], set()
    for t, c in keep + new:
        if t.lower() not in seen:
            seen.add(t.lower()); merged.append((t, c))
    texts = [t for t, _ in merged]; cats = [c for _, c in merged]
    print(f"[vocab] final bank {len(texts)} tags: "
          f"{collections.Counter(cats).most_common()}", flush=True)

    # CANONICAL ENCODER. Must be the same call encode_captions_siglip2.py used, or the new
    # tags land in a different region than the 296 kept ones and retrieval silently degrades.
    # transformers' get_text_features is NOT interchangeable -- it returned a wrapper object
    # here, and even when unwrapped it applies the head differently.
    from models.text_target import SigLIP2TextTarget
    tt = SigLIP2TextTarget(repo=args.siglip, device="cuda")
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            embs.append(tt.encode_text_frozen_raw(texts[i:i+256]).cpu())
    emb = torch.cat(embs).float()

    # FALSIFIER: re-encoding a tag that already exists in v2 must reproduce v2's vector.
    # If this fails the two halves of the bank are in different spaces and every retrieval
    # comparing a new tag against an old one is meaningless.
    v2t = {t: i for i, t in enumerate(base["text"])}
    probe = [(i, v2t[t]) for i, t in enumerate(texts) if t in v2t][:200]
    cos = torch.stack([F.cosine_similarity(emb[i], base["emb"][j].float(), dim=0)
                       for i, j in probe])
    print(f"[vocab] space check on {len(probe)} kept tags: cos min {cos.min():.4f} "
          f"mean {cos.mean():.4f}", flush=True)
    if cos.min() < 0.99:
        sys.exit(f"FAIL: re-encode does not reproduce v2 (min cos {cos.min():.4f}). "
                 "New tags would be in a different space than the kept ones.")

    # SEMANTIC DEDUP. Exact-string dedup is not enough: "a computer monitor"/"a monitor"
    # measured 0.987 and "orange jacket"/"orange coat" 0.988. A pair that close makes the
    # top1-top2 margin meaningless -- the gate would drop the field precisely when the
    # answer is right. Greedy keep-first within each category.
    emb = F.normalize(emb, dim=-1)
    keep_mask = torch.ones(len(texts), dtype=torch.bool)
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        kept: list[int] = []
        for i in idx:
            if kept and float((emb[i] @ emb[kept].T).max()) > args.dedup_cos:
                keep_mask[i] = False
            else:
                kept.append(i)
    dropped = int((~keep_mask).sum())
    texts = [t for t, k in zip(texts, keep_mask.tolist()) if k]
    cats  = [c for c, k in zip(cats,  keep_mask.tolist()) if k]
    emb   = emb[keep_mask]
    print(f"[vocab] dedup >{args.dedup_cos}: dropped {dropped}, {len(texts)} remain: "
          f"{collections.Counter(cats).most_common()}", flush=True)

    emb = emb.to(torch.float16)   # v2 is fp16, L2-normed
    torch.save({"text": texts, "category": cats, "emb": emb,
                "siglip": args.siglip, "n_curated": len(keep)}, args.out)
    print(f"[vocab] wrote {args.out}  emb {tuple(emb.shape)}  "
          f"{emb.numel()*2/2**20:.2f} MiB", flush=True)
    print("PERCEPTION_VOCAB_DONE", flush=True)

if __name__ == "__main__":
    main()
