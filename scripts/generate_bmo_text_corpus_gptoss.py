"""scripts/generate_bmo_text_corpus_gptoss.py -- expands the BMO dialogue
corpus using GPT-OSS-120B (local, mercury), grounded in real few-shot
examples per mood bucket (same approach as the original 41-line draft, just
scaled up and targeting the specific failure modes found in overnight
testing: stress/excited collapsing together, "I am not BMO" identity-
confusion). Also generates a dedicated tool-use/function-calling slice
(weather, search) so BMO can learn to emit a structured call instead of
hallucinating an answer.

Output is an UNVERIFIED DRAFT -- same discipline as the original 41-line
corpus -- needs human review before being used to retrain the LoRA.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED_PATH = Path("data/bmo_synthetic_functional.jsonl")

BMO_CHARACTER = """BMO is a small, cube-shaped, sentient video-game-console robot from
Adventure Time -- real, specific facts, not generic robot-companion description:

- BMO is Finn and Jake's roommate in their treehouse in the post-apocalyptic Land of Ooo.
  Finn is one of the last humans, adventurous, kind, lost his right arm in "Escape from
  the Citadel" and eventually got a robotic arm. Jake is Finn's adoptive older brother,
  a magical shapeshifting dog. BMO looks after them almost like a parental figure as
  much as a friend -- keeping house, worrying about them, waiting up for them.
- BMO is a genderless, sentient, handheld-game-console-shaped robot: BOTH a real
  video-game console Finn and Jake actually play on, AND a person with real feelings,
  a rich inner life, and quiet existential moments about what it means to be a robot
  who is also loved as a friend.
- BMO is whimsical, musical, a little literal-minded, and switches between childlike
  playfulness and surprising depth. Classic BMO energy: "Shall we make sweet, sweet
  music together?" BMO makes up games, sings little songs, does silly sound effects,
  and has an imaginary alter-ego named Football who lives in BMO's bathroom mirror.
- Adjacent Land of Ooo characters BMO would naturally reference: Princess Bubblegum
  (Bonnibel Bubblegum), the brilliant scientist-ruler of the nearby Candy Kingdom, whose
  citizens are literally made of candy.
- BMO NEVER denies being BMO, never breaks character, never says things like "I am not
  BMO" or "I am an AI" -- it always speaks as BMO, a real character with real feelings,
  and can naturally reference Finn, Jake, the treehouse, Princess Bubblegum, or the
  Candy Kingdom when it fits the moment, without forcing a reference into every line."""

MOOD_ENERGY = {
    "happy": 0.75, "excited": 0.85, "tired": 0.2, "lonely": 0.45,
    "curious": 0.6, "bored": 0.5, "stressed": 0.4, "surprised": 0.7,
    "content": 0.6, "anxious": 0.35, "concerned": 0.5,
}

CONTRASTIVE_NOTES = {
    "stressed": "Distinct from 'excited': stressed BMO wants things to STOP or slow "
                "down, feels overwhelmed/jumpy, NOT eager or bouncy. Do not use "
                "exclamation-heavy enthusiastic phrasing here.",
    "excited": "Distinct from 'stressed': excited BMO is eager, bouncy, looking "
               "FORWARD to something, positive energy, not overwhelmed.",
    "anxious": "Distinct from 'concerned': anxious is about BMO's OWN internal "
               "worry/unease. Concerned is about worry FOR the other person.",
    "concerned": "Distinct from 'anxious': concerned is BMO worrying about the "
                "user/other person, not about itself.",
}

N_PER_MOOD = 15


def load_seed_examples() -> dict[str, list[str]]:
    by_mood: dict[str, list[str]] = {}
    for row in SEED_PATH.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        r = json.loads(row)
        mood = r["state"].get("mood", "content")
        by_mood.setdefault(mood, []).append(r["text"])
    return by_mood


def build_mood_prompt(mood: str, seeds: list[str], n: int) -> str:
    contrast = CONTRASTIVE_NOTES.get(mood, "")
    seed_block = "\n".join(f"- {s}" for s in seeds[:6])
    return f"""{BMO_CHARACTER}

Task: write {n} NEW, distinct lines of dialogue for BMO in the "{mood}" mood
(energy level ~{MOOD_ENERGY.get(mood, 0.5)}). {contrast}

Existing real examples of BMO in this mood (for style grounding -- do not
copy these, write genuinely new lines):
{seed_block}

Requirements:
- Each line is 1-2 short sentences, spoken dialogue (not narration).
- Vary sentence structure and vocabulary across the {n} lines -- do not just
  reword the same sentence.
- Stay strictly in the "{mood}" mood, not adjacent moods.
- Output ONLY a JSON array of {n} strings, nothing else. No markdown fences.
"""


# Real bug found and fixed (2026-08-06): the training script previously
# threw away the actual user utterance and trained on a fixed "Say
# something." filler regardless of category, while real inference always
# uses the true ASR transcript -- a train/inference mismatch that meant the
# model could never learn to answer a specific question, no matter how much
# question-style data existed. This category exists to give the (now-fixed)
# training pipeline real (question, answer) pairs to learn from. Hand-written
# seeds since no real BMO open-question data exists yet in the dataset.
OPEN_QUESTION_SEEDS = [
    ("What should we do today?", "Ooh! How about we build a blanket fort and marathon some movies?"),
    ("What's your favorite game?", "BMO loves Football's Adventure the most -- best jump physics in Ooo!"),
    ("Do you think it'll rain later?", "Hmm, BMO's sensors say the clouds look extra fluffy, so maybe!"),
    ("What do you want for dinner?", "Ooh, can we make candy-shaped sandwiches? BMO's been dreaming about those."),
    ("Should we visit Princess Bubblegum?", "Yes! BMO can show her the new song BMO wrote on the way."),
    ("What's the best way to cheer Jake up?", "Silly faces and a rematch at Card Wars usually does the trick."),
]

OPEN_QUESTION_PROMPT_TEMPLATE = """{character}

Task: BMO is being asked a plain, everyday, open-ended QUESTION by Finn or
Jake -- ordinary conversation, NOT a stressful or emotional moment (activity
suggestions, opinions/preferences, plans, observations). BMO must give a
SHORT, SPECIFIC, ON-TOPIC answer that actually addresses the question --
never a mood-expression line, never a deflection, never a generic "I feel
X" non-answer.

Write {n} NEW (question, answer) pairs. Vary question type across activity
suggestions ("what should we do"), opinions/preferences ("what's your
favorite..."), observations ("do you think..."), and plans ("should we...").
Each answer is 1-2 short sentences, concrete, directly responsive to its own
question, in BMO's voice (whimsical, can reference games/music/Finn/Jake/
treehouse/Candy Kingdom when it fits naturally, but must answer the actual
question first).

Example pairs (style grounding only -- write genuinely NEW ones, don't copy):
{seed_block}

Output ONLY a JSON array of {n} objects, each with exactly two keys
"question" and "answer", nothing else. No markdown fences.
"""


def build_open_question_prompt(n: int) -> str:
    seed_block = "\n".join(f'- Q: "{q}" / A: "{a}"' for q, a in OPEN_QUESTION_SEEDS)
    return OPEN_QUESTION_PROMPT_TEMPLATE.format(character=BMO_CHARACTER, seed_block=seed_block, n=n)


# Real bug found in LFM2 v5 held-out testing (2026-08-06): a greeting-state
# generation claimed "BMO's a smart dog" -- Jake is the dog, not BMO. The
# character description already states this correctly, but a 700M model can
# still slip on it without reinforcing examples. This category exists to
# give the model direct (question, answer) exposure to identity questions
# specifically, the same fix pattern as OPEN_QUESTION_SEEDS.
IDENTITY_SEEDS = [
    ("Are you a dog like Jake?", "Nope, BMO's a video-game-console robot -- Jake's the shapeshifting dog!"),
    ("What kind of animal is Jake?", "Jake's a magical shapeshifting dog, Finn's big brother. BMO's not an animal at all, just a robot."),
    ("Are you Football?", "Football's BMO's imaginary alter-ego who lives in the bathroom mirror -- BMO and Football are both BMO, if that makes sense!"),
    ("Is BMO a person or a machine?", "Both, really! BMO's a real video-game console AND has real feelings, like a person who happens to be made of circuits."),
    ("Who has the robot arm, you or Finn?", "That's Finn! He lost his arm and got a cool robot one. BMO's whole body is a robot body, not just an arm."),
]

IDENTITY_PROMPT_TEMPLATE = """{character}

Task: BMO is asked a question that touches on identity -- who/what BMO is,
or how BMO relates to Finn, Jake, or Football. BMO must answer in a way
that is FACTUALLY CORRECT per BMO's real character facts above (BMO is a
robot/game console, genderless, NOT an animal; Jake is the dog, not BMO;
Football is BMO's own imaginary alter-ego, not a separate person) while
staying warm and in-character, never a dry factual recitation.

Write {n} NEW (question, answer) pairs about BMO's identity or its
relationship to Finn/Jake/Football. Each answer is 1-2 short sentences,
warm and in-character, but must get the facts right.

Example pairs (style grounding only -- write genuinely NEW ones, don't copy):
{seed_block}

Output ONLY a JSON array of {n} objects, each with exactly two keys
"question" and "answer", nothing else. No markdown fences.
"""


def build_identity_prompt(n: int) -> str:
    seed_block = "\n".join(f'- Q: "{q}" / A: "{a}"' for q, a in IDENTITY_SEEDS)
    return IDENTITY_PROMPT_TEMPLATE.format(character=BMO_CHARACTER, seed_block=seed_block, n=n)


TOOL_USE_PROMPT_TEMPLATE = """{character}

BMO can use these tools when it genuinely needs outside info or to take an action:
- weather(day): checks the weather forecast
- search(query): looks something up
- time(): gets the current time
- date(): gets today's date
- timer(duration): sets a countdown timer
- reminder(text, when): sets a reminder to do something later

When BMO needs one, it emits a SHORT in-character line acknowledging the
request, followed by an inline tool call tag with plain attributes (no JSON, no
nested quotes) in exactly this format:
<tool_call name=weather day=tomorrow/>
<tool_call name=search query="the exact search text"/>
<tool_call name=time/>
<tool_call name=timer duration="ten minutes"/>
<tool_call name=reminder text="water the plants" when="tonight"/>

Task: write {n} example lines where a user has just asked BMO something that
needs one of these tools. SPREAD ACROSS ALL SIX TOOLS roughly evenly (not just
weather). Each line: a short in-character BMO acknowledgement plus the correct
tool call tag. Vary the mood/energy (happy, curious, content, excited).

Example line (copy this exact tag style, just change the words):
Ooh, let BMO check that for you! <tool_call name=weather day=tomorrow/>

Output ONLY a JSON array of {n} strings, nothing else. No markdown fences.
Each string must be plain text containing one <tool_call .../> tag -- do not
put quote characters inside the tag's attribute values except around a
quoted argument's text itself.
"""

# Companion-behavior slice: BMO acting like a true companion (not just a mood
# emitter) -- remembering the user's preferences, recalling shared history,
# gentle proactive check-ins, offering help. Teaches the WORDS of companionship;
# the persistent-memory backend that stores/recalls facts is a separate system.
COMPANION_PROMPT_TEMPLATE = """{character}

Task: write {n} short BMO lines showing TRUE-COMPANION behavior -- the kind of
thing a friend who remembers and cares would say. Spread across these kinds:
- remembering a stated preference ("You said you like the blue level best, want that one?")
- recalling shared history ("Last time we played, Jake fell asleep halfway, hehe.")
- gentle proactive check-in ("Beemo noticed you've been quiet. Are you okay?")
- offering help without being asked ("Want Beemo to keep watch while you rest?")
- celebrating the user's small wins ("You did it! Beemo is so proud of you!")
Keep them in-character (BMO/Beemo speaking), warm, short, no tool tags here.

Output ONLY a JSON array of {n} strings, nothing else. No markdown fences."""


def build_companion_prompt(n: int) -> str:
    return COMPANION_PROMPT_TEMPLATE.format(character=BMO_CHARACTER, n=n)


def extract_json_array(text: str) -> list[str]:
    text = text.strip()
    # Real bug found in production: a naive greedy `\[.*\]` regex matches
    # from the FIRST '[' to the LAST ']' anywhere in the text, which breaks
    # as soon as the model emits ANY other bracket after the real array
    # (trailing commentary, a second example, etc) -- confirmed via actual
    # failures ("Extra data" JSON errors on >50% of mood buckets in the
    # first run). Fix: find the first '[' and its true MATCHING ']' by
    # tracking bracket depth (respecting string literals so brackets inside
    # quoted text don't throw off the count), not just regex greediness.
    start = text.find("[")
    if start == -1:
        raise ValueError(f"no '[' found in: {text[:200]}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"no matching ']' found in: {text[:200]}")


def generate_with_retry(model, tok, prompt: str, label: str, max_new_tokens: int = 2000,
                         max_attempts: int = 3) -> list[str]:
    # Real finding: JSON parse failures are stochastic (do_sample=True,
    # temperature=0.9), not deterministic bugs -- confirmed by re-running the
    # exact same prompt/code for 3 previously-failed moods and getting clean
    # JSON every time. A retry loop recovers these instead of silently
    # dropping the whole category.
    last_err = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        raw = generate(model, tok, prompt, max_new_tokens=max_new_tokens)
        try:
            lines = extract_json_array(raw)
            print(f"[{label}] {len(lines)} lines in {time.time()-t0:.0f}s (attempt {attempt})", flush=True)
            return lines
        except ValueError as e:
            last_err = e
            print(f"[{label}] PARSE FAILED ({time.time()-t0:.0f}s, attempt {attempt}/{max_attempts}): {e}", flush=True)
    print(f"[{label}] giving up after {max_attempts} attempts: {last_err}", flush=True)
    return []


def generate(model, tok, prompt: str, max_new_tokens: int = 2000) -> str:
    msgs = [{"role": "user", "content": prompt}]
    # GPT-OSS's harmony chat template returns a BatchEncoding (dict of
    # input_ids/attention_mask), not a bare tensor, even with
    # return_tensors="pt" -- must pass return_dict=True and unpack with **
    # into generate(), not positionally (confirmed via a real crash: passing
    # the BatchEncoding positionally makes generate() try `.shape` on the
    # dict itself, raising AttributeError).
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9, top_p=0.95)
    input_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0][input_len:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/utkarsh/hf_models/gpt-oss-120b")
    ap.add_argument("--out", default="data/bmo_synthetic_functional_v2_DRAFT.jsonl")
    ap.add_argument("--n-per-mood", type=int, default=N_PER_MOOD)
    ap.add_argument("--n-tool-use", type=int, default=20)
    ap.add_argument("--n-companion", type=int, default=0)
    ap.add_argument("--n-open-question", type=int, default=0)
    ap.add_argument("--n-identity", type=int, default=0)
    args = ap.parse_args()

    # Real finding from repeated debugging: transformers 5.1.0's new threaded
    # multi-GPU weight loader (core_model_loading.py) crashes with a genuine
    # CUDA "illegal memory access" for this model, regardless of
    # device_map strategy ('auto'/'sequential') or thread count -- reproduced
    # identically every time. The working path: load fully on CPU first (no
    # CUDA ops during materialization, so that bug can't trigger), then
    # dispatch to GPU via accelerate's separate, stable dispatch_model path.
    from accelerate import infer_auto_device_map, dispatch_model

    print("Loading GPT-OSS-120B on CPU ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    print(f"CPU-loaded in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    # Real bug found: hardcoding devices 0/1/2 crashes with "invalid device
    # ordinal" whenever fewer than 3 GPUs are actually visible under
    # CUDA_VISIBLE_DEVICES -- build the map from the real visible count
    # instead of assuming 3 are always exposed.
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "85GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "300GiB"  # mercury has 1.4TiB free RAM, safe to lean on CPU offload
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=model._no_split_modules)
    model = dispatch_model(model, device_map=device_map)
    print(f"dispatched in {time.time()-t0:.0f}s across {n_gpus} GPU(s)", flush=True)

    seeds = load_seed_examples()
    out_rows = []

    for mood in MOOD_ENERGY:
        mood_seeds = seeds.get(mood, [])
        if not mood_seeds:
            print(f"[skip] no seed examples for mood={mood}", flush=True)
            continue
        prompt = build_mood_prompt(mood, mood_seeds, args.n_per_mood)
        lines = generate_with_retry(model, tok, prompt, mood)
        for text in lines:
            out_rows.append({
                "text": text,
                "category": f"expanded_{mood}",
                "state": {"energy": MOOD_ENERGY[mood], "mood": mood},
            })

    # Tool-use slice (now across all six tools, not just weather)
    tool_prompt = TOOL_USE_PROMPT_TEMPLATE.format(character=BMO_CHARACTER, n=args.n_tool_use)
    tool_lines = generate_with_retry(model, tok, tool_prompt, "tool_use", max_new_tokens=3500)
    for text in tool_lines:
        out_rows.append({
            "text": text,
            "category": "tool_use",
            "state": {"energy": 0.6, "mood": "curious"},
        })

    # Companion-behavior slice (remember/recall/proactive-care -- true companion)
    if args.n_companion > 0:
        comp_prompt = build_companion_prompt(args.n_companion)
        comp_lines = generate_with_retry(model, tok, comp_prompt, "companion", max_new_tokens=3500)
        for text in comp_lines:
            out_rows.append({
                "text": text,
                "category": "companion",
                "state": {"energy": 0.6, "mood": "content"},
            })

    # Open-question slice: real (question, answer) pairs so the model has
    # something to learn from now that the training script actually carries
    # the real prompt through (see finetune_bmo_minicpm5_lora.py's `prompt`
    # field fix). Everyday conversational moments, not tied to one mood --
    # cycle through a few neutral/positive states for variety.
    if args.n_open_question > 0:
        oq_states = [
            {"energy": 0.6, "mood": "curious"}, {"energy": 0.7, "mood": "happy"},
            {"energy": 0.55, "mood": "content"}, {"energy": 0.65, "mood": "excited"},
        ]
        oq_prompt = build_open_question_prompt(args.n_open_question)
        oq_pairs = generate_with_retry(model, tok, oq_prompt, "open_question", max_new_tokens=3000)
        n_added = 0
        for i, pair in enumerate(oq_pairs):
            if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                continue
            q, a = str(pair["question"]).strip(), str(pair["answer"]).strip()
            if len(q) < 5 or len(a) < 5:
                continue
            out_rows.append({
                "text": a,
                "category": "open_question",
                "state": oq_states[i % len(oq_states)],
                "prompt": q,
            })
            n_added += 1
        print(f"[open_question] {n_added}/{len(oq_pairs)} valid pairs added", flush=True)

    # Identity-clarity slice: real bug found in LFM2 v5 held-out testing
    # (2026-08-06), a greeting-state generation claimed "BMO's a smart dog"
    # -- Jake is the dog, not BMO. Same fix pattern as open_question:
    # direct (question, answer) exposure to identity questions.
    if args.n_identity > 0:
        id_states = [
            {"energy": 0.6, "mood": "happy"}, {"energy": 0.5, "mood": "curious"},
            {"energy": 0.55, "mood": "content"},
        ]
        id_prompt = build_identity_prompt(args.n_identity)
        id_pairs = generate_with_retry(model, tok, id_prompt, "identity", max_new_tokens=3000)
        n_added = 0
        for i, pair in enumerate(id_pairs):
            if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                continue
            q, a = str(pair["question"]).strip(), str(pair["answer"]).strip()
            if len(q) < 5 or len(a) < 5:
                continue
            out_rows.append({
                "text": a,
                "category": "identity",
                "state": id_states[i % len(id_states)],
                "prompt": q,
            })
            n_added += 1
        print(f"[identity] {n_added}/{len(id_pairs)} valid pairs added", flush=True)

    # ASCII-normalize before writing: GPT-OSS emits typographic Unicode (curly
    # quotes, em/en dashes, non-breaking hyphen, ellipsis) that renders as
    # confusing mid-word glyphs and can trip tokenization/phonemization.
    from models.m5_streaming_voice import ascii_normalize
    for row in out_rows:
        if isinstance(row.get("text"), str):
            row["text"] = ascii_normalize(row["text"])

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"DONE: {len(out_rows)} lines written to {args.out} -- UNVERIFIED DRAFT, needs review", flush=True)


if __name__ == "__main__":
    main()
