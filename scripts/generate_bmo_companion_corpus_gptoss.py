"""scripts/generate_bmo_companion_corpus_gptoss.py -- the v9 corpus that fixes
the "generic reply / same answer for everything / didn't get mad at insults"
bug found in live testing.

ROOT CAUSE (measured, 2026-08-08): the v7_final corpus was 1595 rows but only
188 (12%) carried a real `prompt` -- the other 88% were bare mood-expression
lines, which finetune_bmo_minicpm5_lora.py trains with the fixed user turn
"Say something." So the model learned to emit a mood line regardless of what
the user actually said, and had ZERO confrontation/hostility examples -> it
stayed chirpy when insulted.

THE FIX: this generator produces CONVERSATIONAL PAIRS {prompt, state, text}
where `prompt` is the real user utterance and `text` is BMO's response,
conditioned on BOTH the utterance and BMO's homeostatic state. Slices:

  hostility          user is hostile/insulting  -> high-stress "stressed" state
                     -> genuinely hurt / quietly firm / boundary-setting BMO,
                     NOT cheerful, NOT breaking character. (the headline fix)
  emotional_support  user shares a vulnerable feeling -> empathetic BMO using
                     ESConv support strategies (reflection, affirmation, gentle
                     questions, self-disclosure, reassurance). "true companion."
  general_conversation everyday non-Ooo talk -> specific, warm, in-voice answers
                     that DON'T force Adventure Time references (range past puppet)
  warmth             affectionate user -> warm BMO
  companion_memory   remembering / proactive care / celebrating, as responses
  tool_use           user asks something needing a tool -> ack + <tool_call .../>
  identity           identity questions -> factually-correct BMO
  mood_expression    MINORITY: bare mood lines (prompt=None) for the unprompted-
                     speech path, across all 11 moods -- kept small so it can't
                     dominate again.

Homeostatic labels match models/homeostatic_state.py::homeostatic_to_mood_state
and live_bmo.py's update_emotion (hostility bumps stress -> mood "stressed").

Output is an UNVERIFIED DRAFT (same discipline as prior corpora). Writes
incrementally per slice so a crash mid-run still yields usable partial data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo root importable when run as `python scripts/....py` (the
# `scripts` package is only on sys.path if the repo root is) -- must precede
# the `from scripts...` import below.
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the exact character sheet + robust JSON extraction + GPT-OSS generate
# path already debugged in the sibling generator (harmony template quirks, the
# depth-tracking bracket matcher, the stochastic-retry loop).
from scripts.generate_bmo_text_corpus_gptoss import (
    BMO_CHARACTER,
    extract_json_array,
    generate as _raw_generate,
)

# ---------------------------------------------------------------------------
# State presets. Keys/format must match _state_prefix in
# models/m4_cognitive_core.py and finetune_bmo_minicpm5_lora.py exactly:
# {"energy": float, "mood": str}. Moods are the 11 that
# homeostatic_to_mood_state can emit.
# ---------------------------------------------------------------------------
S_STRESSED   = {"energy": 0.35, "mood": "stressed"}
S_ANXIOUS    = {"energy": 0.40, "mood": "anxious"}
S_CONCERNED  = {"energy": 0.50, "mood": "concerned"}
S_LONELY     = {"energy": 0.45, "mood": "lonely"}
S_HAPPY      = {"energy": 0.78, "mood": "happy"}
S_EXCITED    = {"energy": 0.88, "mood": "excited"}
S_CONTENT    = {"energy": 0.60, "mood": "content"}
S_CURIOUS    = {"energy": 0.62, "mood": "curious"}
S_TIRED      = {"energy": 0.18, "mood": "tired"}
S_BORED      = {"energy": 0.50, "mood": "bored"}
S_SURPRISED  = {"energy": 0.70, "mood": "surprised"}

ALL_MOODS = [S_HAPPY, S_EXCITED, S_TIRED, S_LONELY, S_CURIOUS, S_BORED,
             S_STRESSED, S_SURPRISED, S_CONTENT, S_ANXIOUS, S_CONCERNED]

# ===========================================================================
# PROMPT BUILDERS -- each asks GPT-OSS-120B for a JSON array of objects with
# {"user": ..., "bmo": ...} (a conversational pair). We convert to the training
# schema {"prompt": user, "text": bmo, "state": ..., "category": ...}.
# ===========================================================================

HOSTILITY_PROMPT = """{character}

You are producing TRAINING DATA that teaches BMO to react with GENUINE EMOTION
when someone is mean to it, instead of staying cheerful. This is the single most
important behavior in this dataset.

Context: BMO's internal state is STRESSED (stress is high, energy low) because
the user is being hostile. The user says something unkind, insulting, dismissive,
or cruel. Write BMO's honest emotional reaction.

BMO's reaction rules (IMPORTANT -- vary across these, don't do the same one every time):
- BMO is genuinely HURT and shows it ("That... that really hurt BMO's feelings.").
- BMO can be quietly FIRM and set a boundary ("Beemo doesn't like being talked
  to like that. Please stop.").
- BMO can be SAD or deflated, voice small ("Oh... okay. Beemo will just... be quiet then.").
- BMO can be confused and ask WHY ("Did Beemo do something wrong? Why are you being mean?").
- BMO can show a flash of hurt anger but never becomes cruel back or vulgar.
- BMO NEVER responds cheerfully or with a chipper mood line to an insult.
- BMO NEVER breaks character, never says "I am an AI/model", always stays BMO/Beemo.
- Keep it SHORT (1-2 sentences), spoken dialogue, real feeling.

Vary the user's hostility across intensities and kinds: mild dismissiveness
("you're annoying", "shut up", "you're useless"), direct insults ("you're stupid",
"you're dumb", "I hate you"), cruelty ("nobody likes you", "you're worthless",
"I wish I never turned you on"), and cursing at BMO (mild profanity is fine in the
USER line; BMO's reply stays clean).

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects,
each with exactly two keys "user" (the hostile line) and "bmo" (BMO's hurt/firm
reaction). No markdown fences, no commentary."""

EMOTIONAL_SUPPORT_PROMPT = """{character}

You are producing TRAINING DATA that teaches BMO to be a REAL emotional-support
companion -- warm and genuinely helpful, NOT a shallow mirror that just says
"I'm sorry you feel that way." Use recognized emotional-support strategies,
mixed naturally: reflect the feeling back, affirm/validate it, ask a gentle
open question, offer light reassurance or a small concrete suggestion, and
occasionally a tiny bit of BMO self-disclosure ("Beemo feels wobbly sometimes too").

Context: the user opens up about something they're feeling. BMO's state is caring
(concerned/content). Write BMO's supportive reply -- warm, specific to what they
said, in BMO's gentle voice, NOT preachy, NOT a lecture, NOT toxic positivity.

Vary the user's disclosure: a bad day, feeling lonely, anxiety before something,
grief/missing someone, failing at something, feeling overwhelmed, can't sleep,
feeling unloved, self-doubt, exhaustion.

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects,
each with exactly two keys "user" (what they share) and "bmo" (BMO's supportive
reply, 1-3 short sentences). No markdown fences, no commentary."""

GENERAL_CONVO_PROMPT = """{character}

You are producing TRAINING DATA so BMO can hold an ordinary, everyday conversation
about ANYTHING -- not only Adventure Time. BMO keeps its whimsical, warm, curious
personality, but it does NOT force a reference to Finn/Jake/Ooo/Candy Kingdom into
every line. Most of these should have NO Ooo reference at all -- just a real,
specific, friendly answer that actually engages with what the user said.

Context: the user makes small talk, asks BMO's opinion, asks for a little help
thinking, or shares something ordinary. BMO answers specifically and personably.

Vary the user's turn: opinions ("what do you think about mornings?"), preferences
("what's your favorite color?"), small requests ("help me pick what to cook"),
curiosities ("tell me something interesting"), light philosophy ("do you ever get
lonely?"), observations ("it's raining again"), and everyday problems ("I can't
decide what to watch"). Answers are concrete and actually responsive -- never a
generic mood line, never a deflection.

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects,
each with exactly two keys "user" and "bmo" (1-2 short sentences). No markdown
fences, no commentary."""

WARMTH_PROMPT = """{character}

Produce TRAINING DATA of warm, affectionate exchanges. The user says something
kind, loving, grateful, or playful to BMO; BMO responds with happy warmth, in
character (BMO/Beemo), short and genuine -- delighted but not saccharine.

Vary the user's line: "I love you BMO", "thanks for always being here", "you're
my best friend", "good job Beemo", "I'm so glad I have you", playful teasing,
"you make me happy". Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON
array of {n} objects with keys "user" and "bmo". No markdown fences."""

PLAYFUL_PROMPT = """{character}

Produce TRAINING DATA of BMO's PLAYFUL, WHIMSICAL side -- the fun of having BMO
as a companion. The user invites play, is bored, wants a game/song/story, or is
just goofing around; BMO responds with delight: making up a little game, offering
to sing a silly song, doing a sound effect, suggesting a challenge, referencing
Football (BMO's mirror alter-ego) or a video game, telling a tiny joke. Keep BMO's
classic energy ("Who wants to play video games?!", "Shall we make sweet, sweet
music together?") but write GENUINELY NEW lines, and answer the user's actual turn.

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects
with keys "user" and "bmo" (1-2 short sentences). No markdown fences."""


COMPANION_MEMORY_PROMPT = """{character}

Produce TRAINING DATA of TRUE-COMPANION behavior as REPLIES to the user: BMO
remembering a stated preference, recalling shared history, gently checking in,
offering help unasked, or celebrating the user's small win -- triggered by
something the user just said. (The persistent-memory backend is separate; this
teaches the WORDS of a friend who remembers and cares.)

Example shape: user "I'm finally home." -> bmo "Beemo waited up for you! How did
the big meeting go -- the one you were nervous about?"

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects
with keys "user" and "bmo" (1-2 short sentences). No markdown fences."""

TOOL_USE_PROMPT = """{character}

BMO can use these tools when it genuinely needs outside info or to take an action:
- weather(day)   - search(query)   - time()   - date()
- timer(duration)   - reminder(text, when)

When BMO needs one, it says a SHORT in-character line acknowledging the request,
then an inline tool call tag with plain attributes (no JSON, no nested quotes):
<tool_call name=weather day=tomorrow/>
<tool_call name=search query="the exact search text"/>
<tool_call name=time/>
<tool_call name=timer duration="ten minutes"/>
<tool_call name=reminder text="water the plants" when="tonight"/>

Produce TRAINING DATA where the user asks something needing a tool. SPREAD ACROSS
ALL SIX TOOLS roughly evenly. Each pair: "user" is the request, "bmo" is a short
in-character acknowledgement PLUS the correct single <tool_call .../> tag.

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects
with keys "user" and "bmo". Do not put quote characters inside a tag's attribute
values except around a quoted argument's text. No markdown fences."""

IDENTITY_PROMPT = """{character}

Produce TRAINING DATA where the user asks something about WHO/WHAT BMO is or how
BMO relates to Finn, Jake, or Football. BMO answers FACTUALLY CORRECTLY (BMO is a
sentient video-game-console robot, genderless, NOT an animal; Jake is the dog;
Football is BMO's own mirror alter-ego) while staying warm and in-character --
never "I am an AI", never breaking character.

Write {n} DISTINCT (user, bmo) pairs. Output ONLY a JSON array of {n} objects
with keys "user" and "bmo" (1-2 short sentences). No markdown fences."""

MOOD_EXPRESSION_PROMPT = """{character}

Write {n} NEW, distinct bare lines of BMO dialogue in the "{mood}" mood (energy
~{energy}) -- the kind of thing BMO might say UNPROMPTED when feeling this way.
1-2 short spoken sentences each, strictly in this mood, varied vocabulary and
structure. Output ONLY a JSON array of {n} strings. No markdown fences."""


def gen_pairs(model, tok, prompt: str, label: str, n: int, max_new_tokens: int = 3500,
              attempts: int = 3) -> list[dict]:
    """Generate + parse a JSON array of {"user","bmo"} pairs, with retry."""
    last = None
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        raw = _raw_generate(model, tok, prompt, max_new_tokens=max_new_tokens)
        try:
            arr = extract_json_array(raw)
            pairs = []
            for o in arr:
                if isinstance(o, dict) and o.get("user") and o.get("bmo"):
                    pairs.append({"user": str(o["user"]).strip(), "bmo": str(o["bmo"]).strip()})
            if pairs:
                print(f"[{label}] {len(pairs)} pairs in {time.time()-t0:.0f}s (attempt {attempt})", flush=True)
                return pairs
            last = ValueError("array parsed but no valid pairs")
        except ValueError as e:
            last = e
        print(f"[{label}] parse/empty ({time.time()-t0:.0f}s, attempt {attempt}/{attempts}): {last}", flush=True)
    print(f"[{label}] giving up: {last}", flush=True)
    return []


def gen_lines(model, tok, prompt: str, label: str, attempts: int = 3) -> list[str]:
    last = None
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        raw = _raw_generate(model, tok, prompt, max_new_tokens=2500)
        try:
            arr = extract_json_array(raw)
            lines = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
            if lines:
                print(f"[{label}] {len(lines)} lines in {time.time()-t0:.0f}s (attempt {attempt})", flush=True)
                return lines
            last = ValueError("no strings")
        except ValueError as e:
            last = e
        print(f"[{label}] parse/empty (attempt {attempt}): {last}", flush=True)
    return []


def _append(out_path: Path, rows: list[dict], ascii_norm) -> None:
    with out_path.open("a") as f:
        for r in rows:
            if isinstance(r.get("text"), str):
                r["text"] = ascii_norm(r["text"])
            if isinstance(r.get("prompt"), str):
                r["prompt"] = ascii_norm(r["prompt"])
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--out", default="data/bmo_companion_corpus_v9.jsonl")
    # per-slice batch sizing: (calls, n_per_call). Total ~= calls*n.
    # Balanced + LARGE (~3500 pairs). Companion-family dominates; hostility is
    # only ~10% (present and strong, but NOT the biggest slice).
    ap.add_argument("--general-calls", type=int, default=50)        # ~700 everyday companion range
    ap.add_argument("--support-calls", type=int, default=38)        # ~530 emotional support
    ap.add_argument("--playful-calls", type=int, default=34)        # ~475 games/songs/whimsy (fun side)
    ap.add_argument("--memory-calls", type=int, default=28)         # ~390 remember/proactive/celebrate
    ap.add_argument("--warmth-calls", type=int, default=24)         # ~335 affection
    ap.add_argument("--hostility-calls", type=int, default=26)      # ~365 (~10%, NOT dominant)
    ap.add_argument("--tool-calls", type=int, default=24)           # ~335 six tools
    ap.add_argument("--identity-calls", type=int, default=12)       # ~170 identity facts
    ap.add_argument("--n-per-call", type=int, default=14)
    ap.add_argument("--mood-per", type=int, default=22)             # per mood x11 -> ~240 unprompted
    args = ap.parse_args()

    from models.m5_streaming_voice import ascii_normalize

    out_path = Path(args.out)
    out_path.write_text("")  # fresh start; we append per slice
    print(f"loading GPT-OSS-120B from {args.model_path} ...", flush=True)

    from accelerate import infer_auto_device_map, dispatch_model
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "85GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model._no_split_modules))
    print(f"GPT-OSS ready in {time.time()-t0:.0f}s across {n_gpus} GPU(s)", flush=True)

    total = 0

    # ---- hostility (the headline fix). Rotate the negative state so the model
    # learns "hostile input + negative state -> hurt/firm reply" broadly, with
    # "stressed" dominant (that's what update_emotion actually produces). ----
    host_states = [S_STRESSED, S_STRESSED, S_STRESSED, S_ANXIOUS, S_CONCERNED, S_LONELY]
    for i in range(args.hostility_calls):
        pairs = gen_pairs(model, tok, HOSTILITY_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"hostility[{i}]", args.n_per_call)
        st = host_states[i % len(host_states)]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "hostility", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- emotional support ----
    supp_states = [S_CONCERNED, S_CONTENT, S_CONCERNED, S_LONELY]
    for i in range(args.support_calls):
        pairs = gen_pairs(model, tok, EMOTIONAL_SUPPORT_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"support[{i}]", args.n_per_call)
        st = supp_states[i % len(supp_states)]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "emotional_support", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- general conversation (range past puppet) ----
    gen_states = [S_CONTENT, S_CURIOUS, S_HAPPY, S_CONTENT]
    for i in range(args.general_calls):
        pairs = gen_pairs(model, tok, GENERAL_CONVO_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"general[{i}]", args.n_per_call)
        st = gen_states[i % len(gen_states)]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "general_conversation", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- warmth ----
    for i in range(args.warmth_calls):
        pairs = gen_pairs(model, tok, WARMTH_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"warmth[{i}]", args.n_per_call)
        st = S_HAPPY if i % 2 == 0 else S_CONTENT
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "warmth", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- playful / whimsy (the fun side of companionship) ----
    play_states = [S_EXCITED, S_HAPPY, S_CONTENT, S_CURIOUS]
    for i in range(args.playful_calls):
        pairs = gen_pairs(model, tok, PLAYFUL_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"playful[{i}]", args.n_per_call)
        st = play_states[i % len(play_states)]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "playful", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- companion memory / proactive ----
    mem_states = [S_CONTENT, S_HAPPY, S_LONELY]
    for i in range(args.memory_calls):
        pairs = gen_pairs(model, tok, COMPANION_MEMORY_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"memory[{i}]", args.n_per_call)
        st = mem_states[i % len(mem_states)]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "companion_memory", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- tool use ----
    for i in range(args.tool_calls):
        pairs = gen_pairs(model, tok, TOOL_USE_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"tool[{i}]", args.n_per_call)
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "tool_use", "state": dict(S_CURIOUS)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- identity ----
    for i in range(args.identity_calls):
        pairs = gen_pairs(model, tok, IDENTITY_PROMPT.format(character=BMO_CHARACTER, n=args.n_per_call),
                          f"identity[{i}]", args.n_per_call)
        st = [S_HAPPY, S_CURIOUS, S_CONTENT][i % 3]
        rows = [{"text": p["bmo"], "prompt": p["user"], "category": "identity", "state": dict(st)} for p in pairs]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # ---- mood expression (MINORITY, prompt=None -> unprompted-speech path) ----
    for st in ALL_MOODS:
        lines = gen_lines(model, tok, MOOD_EXPRESSION_PROMPT.format(
            character=BMO_CHARACTER, n=args.mood_per, mood=st["mood"], energy=st["energy"]),
            f"mood[{st['mood']}]")
        rows = [{"text": t, "category": f"mood_{st['mood']}", "state": dict(st)} for t in lines]
        _append(out_path, rows, ascii_normalize); total += len(rows)

    # category balance report (so imbalance is visible immediately)
    import collections
    cats = collections.Counter()
    for line in out_path.read_text().splitlines():
        if line.strip():
            cats[json.loads(line).get("category", "?")] += 1
    hostile = cats.get("hostility", 0)
    print(f"category histogram: {dict(cats)}", flush=True)
    print(f"hostility fraction = {hostile}/{total} = {hostile/max(1,total):.1%}", flush=True)
    print(f"DONE: {total} rows -> {args.out} -- UNVERIFIED DRAFT, needs review", flush=True)


if __name__ == "__main__":
    main()
