"""Build v16: pool every usable existing corpus, filter the cartoon voice, add MULTI-TURN.

WHY POOLING RATHER THAN GENERATING. Regenerating from scratch took ~3h for 3,200 rows and the
showcase is tomorrow. Measured across the 15 existing versions: 82-90% of rows survive the
game-vocabulary ban, and 7,000+ unique clean replies already exist. The old corpora also carry
329 tool rows each, which the freshly-generated v15 has ZERO of. So the material exists; it
just needs filtering and blending.

THE REGISTER PROBLEM, AND WHY BLENDING FIXES IT.
  v10-v14 : ~40% exclamations, ~15 words/reply -> warm, but cartoonish before filtering
  v15_tars:   0% exclamations,  8.6 words/reply -> clean, but reads like a thermostat
Neither alone is right. Pooling both lands the register in between, which is the TARS target:
dry and brief, but not clinical. Reported as a metric so it can be checked, not assumed.

MULTI-TURN IS THE PART THAT CANNOT BE POOLED. Not one of the 15 versions contains a single
multi-turn row -- which is exactly why BMO forgets what IT said one turn ago. That data has
never existed, so it is TEMPLATED here from real offers mined out of the corpus itself, so the
voice stays in-distribution. Three dependency shapes, matching the observed failures:
  A) offer -> accept -> CONTINUE THE SAME THING   (the "let's play a game" / "sure" failure)
  B) fact stated -> unrelated turn -> recalled     (the depth-2 memory test)
  C) topic -> interjection -> back to topic
"""
from __future__ import annotations
import argparse, collections, glob, json, random, re

BAN = re.compile(r"\b(pixel|beep|boop|cartridge|console|8-?bit|power-?up|powerup|sprite|"
                 r"glitch|jingle|press start|save file|high score|quest|co-?op|npc|joystick|"
                 r"arcade|respawn|checkpoint|game over|loading screen|side-?quest)\b", re.I)

# CHARACTER NAMES. Measured after the first v16 build: 1,234 rows (10.65%) referenced
# Adventure Time characters -- MORE contamination than the game vocabulary ever was, and
# invisible to the original lexicon because it only listed game *words*. BMO is from that
# show, so every source corpus is full of them, and it leaked straight into a trained model
# ("How about a cozy night-time game with Finn and Jake?"). A companion robot for a lonely
# adult must not talk about cartoon characters it has no reason to know.
CHARS = re.compile(r"\b(finn|jake the dog|jake|princess bubblegum|bubblegum|ice king|"
                   r"marceline|lumpy space|candy kingdom|land of ooo|bmo'?s friends)\b", re.I)

# Documented harm pattern in companion products (Replika/Moxie): guilt, dependency, obligation.
CLOY = re.compile(r"\b(i need you|don'?t leave|you'?re all i have|i missed you so much|"
                  r"where were you|i was so lonely without|please don'?t go|i can'?t live)\b", re.I)

# Tools that survive: fully local, cannot fail embarrassingly. weather/search are network-
# dependent and were cut -- a weather robot that is wrong or offline in front of an audience
# is the single worst failure available, and everyone in the room can see the sky.
KEEP_TOOLS = {"time", "date", "time_date", "timer", "volume", "face", "idle",
              "remember", "recall", "repeat", "look"}
DROP_TOOLS = {"weather", "search", "web_search", "websearch", "lights", "calendar", "music"}
# Name drift measured in the old corpora: 6 tools taught under ~20 names, many of which the
# grammar (assets/tool_call.gbnf) forbids -- so the model was trained toward strings the
# decoder rejects. Canonicalise.
TOOL_ALIAS = {"set_timer": "timer", "set_alarm": "timer", "alarm": "timer",
              "get_time": "time", "current_time": "time", "get_date": "date",
              "current_date": "date", "get_current_date": "date",
              "set_volume": "volume", "volume_up": "volume", "volume_down": "volume"}
TOOL_TAG = re.compile(r"<tool_call\s+name=([A-Za-z_]+)", re.I)

SPEAKER_GLOBS = ["data/bmo_companion_corpus_v*.jsonl",
                 "data/bmo_synthetic_functional*.jsonl",
                 "data/bmo_companion_tools_v8_DRAFT.jsonl",
                 "data/bmo_unvoiced_v7.jsonl",
                 "data/bmo_open_question_only.jsonl",
                 "data/bmo_corpus_backfill*.jsonl"]


def norm(r: dict) -> dict | None:
    """Every corpus generation used a slightly different schema. Normalise to one."""
    t = (r.get("text") or r.get("answer") or "").strip()
    p = (r.get("prompt") or "").strip()
    if not t:
        return None
    return {"text": t, "prompt": p, "category": r.get("category", "unknown"),
            "state": r.get("state") or {}}


def canon_tools(t: str) -> str | None:
    """Rewrite tool names to their canonical form; drop rows using cut tools."""
    names = TOOL_TAG.findall(t)
    if not names:
        return t
    for n in names:
        canon = TOOL_ALIAS.get(n.lower(), n.lower())
        if canon in DROP_TOOLS:
            return None                      # network tool -> drop the whole row
        if canon not in KEEP_TOOLS:
            return None                      # unknown tool -> drop rather than teach drift
        if canon != n:
            t = t.replace(f"name={n}", f"name={canon}")
    return t


def load_pool() -> list[dict]:
    seen, out = set(), []
    per_file = collections.Counter()
    for g in SPEAKER_GLOBS:
        for f in sorted(glob.glob(g)):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = norm(json.loads(line))
                except Exception:
                    continue
                if not r:
                    continue
                if BAN.search(r["text"]) or CLOY.search(r["text"]):
                    continue
                t = canon_tools(r["text"])
                if t is None:
                    continue
                r["text"] = t
                key = (r["prompt"], r["text"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
                per_file[f.split("/")[-1]] += 1
    print("[pool] kept per file:")
    for k, v in per_file.most_common():
        print(f"    {v:>5}  {k}")
    return out


# ---------------------------------------------------------------- multi-turn
ACCEPTS = ["sure, let's play it", "yes please", "okay, go on", "yeah alright",
           "sure", "yes let's do that", "go on then", "okay"]
FILLERS = [("the weather outside is bright today", "It is. The light in here has changed."),
           ("I might make some tea", "Good idea."),
           ("my back aches from sitting", "You have been in that chair a while."),
           ("it's getting late", "It is.")]
FACTS = [
    ("my cat is called Pixel", "what is my cat called?", "Pixel."),
    ("my sister plays bass guitar", "what does my sister play?", "Bass guitar."),
    ("I work nights at the hospital", "where do I work?", "The hospital, on nights."),
    ("my brother's name is Sam", "what is my brother called?", "Sam."),
    ("I'm allergic to peanuts", "what am I allergic to?", "Peanuts."),
    ("I have a meeting at four", "when is my meeting?", "Four o'clock."),
    ("my dog is called Biscuit", "what's my dog's name?", "Biscuit."),
    ("I grew up in Leeds", "where did I grow up?", "Leeds."),
    ("I'm learning the piano", "what am I learning?", "The piano."),
    ("my flat is on the third floor", "which floor do I live on?", "The third."),
    ("I don't drink coffee after noon", "when do I stop drinking coffee?", "After noon."),
    ("my best friend is called Ade", "who's my best friend?", "Ade."),
    ("I cycle to work on Tuesdays", "when do I cycle to work?", "Tuesdays."),
    ("my mum's birthday is in March", "when is my mum's birthday?", "March."),
    ("I hate the sound of the fan", "what sound do I hate?", "The fan."),
    ("I'm reading a book about deserts", "what's my book about?", "Deserts."),
    ("my car is blue", "what colour is my car?", "Blue."),
    ("I have a deadline on Friday", "when is my deadline?", "Friday."),
    ("I take the bus home", "how do I get home?", "By bus."),
    ("my favourite meal is dal", "what's my favourite meal?", "Dal."),
    ("I sleep badly when it's hot", "when do I sleep badly?", "When it's hot."),
    ("my neighbour plays drums", "what does my neighbour play?", "Drums."),
    ("I've been at this project for months", "how long have I been at this?", "Months."),
    ("I broke my wrist last winter", "what did I break?", "Your wrist, last winter."),
    ("my desk faces the window", "which way does my desk face?", "Toward the window."),
]
FILLERS = [
    ("the weather outside is bright today", "It is. The light in here has changed."),
    ("I might make some tea", "Good idea."),
    ("my back aches from sitting", "You have been in that chair a while."),
    ("it's getting late", "It is."),
    ("that fan is loud again", "It has been running a while."),
    ("I should tidy this desk", "It would take ten minutes."),
    ("someone's at the door", "I'll wait."),
    ("my phone keeps buzzing", "You could turn it face down."),
    ("I need to stretch", "Go on, then."),
    ("the room's gone cold", "The window is still open."),
    ("I forgot to eat lunch", "That explains the mood."),
    ("this chair squeaks", "It does."),
]
OFFER_RE = re.compile(r"\b(want to|would you like|shall we|want me to|should we|"
                      r"do you want|how about)\b", re.I)


def build_multiturn(pool: list[dict], rng: random.Random, n_offer: int,
                    n_fact: int, n_topic: int) -> list[dict]:
    """Templated, but grounded: the offers are MINED from the pool so the voice matches."""
    # Exclude hostility rows: mining them produced "Alright. Be erased. Please stop."
    offers = [r for r in pool if OFFER_RE.search(r["text"]) and len(r["text"].split()) <= 22
              and "hostil" not in r.get("category", "").lower()]
    rng.shuffle(offers)
    print(f"[multiturn] mined {len(offers)} real offers from the pool")
    rows = []

    # A) offer -> accept -> CONTINUE THE SAME THING.
    #    This is the exact observed failure: BMO offered a game, the user said "sure",
    #    and BMO changed the subject.
    for o in offers[:n_offer]:
        topic = re.sub(r"^.*?(want to|would you like|shall we|want me to|should we|"
                       r"do you want|how about)\s+", "", o["text"], flags=re.I)
        topic = topic.rstrip("?.!").strip()
        if not topic or len(topic) < 4:
            continue
        rows.append({"messages": [
            {"role": "user", "content": o["prompt"] or "I'm a bit bored."},
            {"role": "assistant", "content": o["text"]},
            {"role": "user", "content": rng.choice(ACCEPTS)},
            # VARY THE OPENER. The first build used a literal "Alright." on all 469 rows and
            # both trained models parroted it verbatim -- proving continuity carried, but
            # sounding robotic. Rotate, and sometimes lead with the topic instead.
            {"role": "assistant", "content": rng.choice([
                f"{topic[0].upper()+topic[1:]}, then.",
                f"Right. {topic[0].upper()+topic[1:]}.",
                f"Okay. {topic[0].upper()+topic[1:]}.",
                f"Let's do it. {topic[0].upper()+topic[1:]}.",
                f"{topic[0].upper()+topic[1:]}. Ready when you are.",
                f"Good. {topic[0].upper()+topic[1:]}.",
            ])},
        ], "category": "multiturn_offer_accept", "state": o.get("state", {})})

    # B) fact -> unrelated turn -> recall. The depth-2 memory test, as training data.
    for _ in range(n_fact):
        fact, ask, ans = rng.choice(FACTS)
        fu, fb = rng.choice(FILLERS)
        rows.append({"messages": [
            {"role": "user", "content": fact},
            {"role": "assistant", "content": rng.choice(
                ["Noted.", "Got it.", "I'll remember that.", "Okay."])},
            {"role": "user", "content": fu},
            {"role": "assistant", "content": fb},
            {"role": "user", "content": ask},
            {"role": "assistant", "content": ans},
        ], "category": "multiturn_fact_recall", "state": {}})

    # C) topic -> interjection -> back to topic.
    for _ in range(n_topic):
        a, b = rng.sample(pool, 2)
        if not a["prompt"] or not b["prompt"]:
            continue
        rows.append({"messages": [
            {"role": "user", "content": a["prompt"]},
            {"role": "assistant", "content": a["text"]},
            {"role": "user", "content": b["prompt"]},
            {"role": "assistant", "content": b["text"]},
            {"role": "user", "content": "anyway, back to what I was saying"},
            {"role": "assistant", "content": a["text"]},
        ], "category": "multiturn_topic_hold", "state": {}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bmo_companion_corpus_v16_pooled.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offer", type=int, default=500)
    ap.add_argument("--fact", type=int, default=1100)
    ap.add_argument("--topic", type=int, default=900)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pool = load_pool()
    mt = build_multiturn(pool, rng, a.offer, a.fact, a.topic)
    allrows = pool + mt
    rng.shuffle(allrows)
    # FILTER AGAIN AT WRITE TIME, over the whole row. The load-time filter only inspected
    # r["text"], so multi-turn rows (which carry their content under "messages") bypassed it
    # entirely -- 36 of them tripped this very regex in the first build.
    def dirty(r):
        blob = json.dumps(r)
        return bool(BAN.search(blob) or CHARS.search(blob) or CLOY.search(blob))
    before = len(allrows)
    allrows = [r for r in allrows if not dirty(r)]
    print(f"[v16] write-time filter dropped {before - len(allrows)} contaminated rows "
          f"(incl. multi-turn, which the load-time filter could not see)")
    with open(a.out, "w") as fh:
        for r in allrows:
            fh.write(json.dumps(r) + "\n")

    single = [r for r in allrows if "messages" not in r]
    ex = sum(1 for r in single if "!" in r["text"]) / max(len(single), 1)
    wl = sum(len(r["text"].split()) for r in single) / max(len(single), 1)
    tools = collections.Counter(
        TOOL_ALIAS.get(n.lower(), n.lower())
        for r in single for n in TOOL_TAG.findall(r["text"]))
    print(f"\n[v16] wrote {a.out}")
    print(f"[v16] {len(allrows)} rows  =  {len(single)} single-turn + {len(mt)} multi-turn "
          f"({len(mt)/len(allrows)*100:.0f}%)")
    print(f"[v16] game vocab   : 0.00%  (filtered at load)")
    print(f"[v16] exclamations : {ex*100:.0f}%   (v10-14 were ~40%, v15 was 0%)")
    print(f"[v16] mean words   : {wl:.1f}   (v10-14 were ~15, v15 was 8.6)")
    print(f"[v16] tool rows    : {sum(tools.values())}  {dict(tools)}")
    print(f"[v16] categories   : {collections.Counter(r['category'] for r in allrows).most_common(12)}")


if __name__ == "__main__":
    main()
