"""scripts/generate_bmo_voice_corpus_fishapi.py -- generates a BMO voice
corpus using the Fish Audio API against the real, public "BMO from
Adventure Time" community voice model (reference_id
323847d4c5394c678e5909c2206725f6, creator @Studio Coille, 15K uses),
s2.1-pro-free model. Complements scripts/generate_bmo_voice_corpus_s2pro.py
(which clones OUR OWN reference clip locally) -- this uses an already-cloned
BMO voice on Fish's platform instead, no local reference clip needed.

Reads the same real dialogue corpus + mood->tag mapping as the S2 Pro
script so both voice corpora cover identical text/emotion coverage.

Requires FISH_API_KEY in the environment (not hardcoded, not written to any
committed file).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests

REFERENCE_ID = "323847d4c5394c678e5909c2206725f6"  # public "BMO from Adventure Time" model
API_URL = "https://api.fish.audio/v1/tts"

# Same real preset-tag mapping used for S2 Pro (Fish's S2 family shares the
# same inline [tag] emotion-control syntax).
MOOD_TAG = {
    "expanded_happy": "[delight]", "expanded_excited": "[excited]", "expanded_tired": "[sigh]",
    "expanded_lonely": "[low voice]", "expanded_curious": "", "expanded_bored": "[low volume]",
    "expanded_stressed": "[panting]", "expanded_surprised": "[surprised]", "expanded_content": "",
    "expanded_anxious": "[panting]", "expanded_concerned": "",
}

NONVERBAL_CLIPS = [
    ("whisper_01", "[whisper] Hey... are you awake?", "whisper"),
    ("whisper_02", "[whisper] Something is moving over there.", "whisper"),
    ("scream_01", "[screaming] Whoa!!", "scream"),
    ("scream_02", "[screaming] No no no no!", "scream"),
    ("laugh_01", "[laughing] Haha, that is so silly!", "laugh"),
    ("laugh_02", "[chuckle] Heh, good one.", "laugh"),
    ("cry_01", "[sad] [moaning] I do not want to be alone right now.", "cry"),
    ("gasp_01", "[inhale] Oh! I did not expect that.", "gasp"),
    ("sigh_01", "[sigh] Okay. Just a long day, I guess.", "sigh"),
]


def call_fish_api(text: str, api_key: str) -> bytes:
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "model": "s2.1-pro-free",
        },
        json={"text": text, "reference_id": REFERENCE_ID, "format": "wav"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bmo_synthetic_functional_v3_final.jsonl")
    ap.add_argument("--out-dir", default="data/bmo_fishapi_synth")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("FISH_API_KEY")
    if not api_key:
        raise SystemExit("FISH_API_KEY not set in environment")

    out_dir = Path(args.out_dir).resolve()
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lines = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    dialogue_lines = [l for l in lines if l["category"] != "tool_use"]
    if args.limit:
        dialogue_lines = dialogue_lines[: args.limit]

    for i, item in enumerate(dialogue_lines):
        mood = item["state"].get("mood", "content")
        cat = item["category"]
        tag = MOOD_TAG.get(cat, "")
        text = f"{tag} {item['text']}".strip() if tag else item["text"]
        fname = f"dialogue_{i:03d}_{mood}.wav"
        try:
            t0 = time.time()
            audio = call_fish_api(text, api_key)
            (wav_dir / fname).write_bytes(audio)
            print(f"[{i}] {time.time()-t0:.1f}s mood={mood} -> {fname}", flush=True)
            rows.append((fname, item["text"], mood))
        except Exception as e:
            print(f"[{i}] FAILED mood={mood}: {type(e).__name__}: {e}", flush=True)

    for name, text, tone in NONVERBAL_CLIPS:
        fname = f"nonverbal_{name}.wav"
        try:
            t0 = time.time()
            audio = call_fish_api(text, api_key)
            (wav_dir / fname).write_bytes(audio)
            print(f"[nonverbal] {time.time()-t0:.1f}s tone={tone} -> {fname}", flush=True)
            rows.append((fname, text, tone))
        except Exception as e:
            print(f"[nonverbal] FAILED tone={tone}: {type(e).__name__}: {e}", flush=True)

    with open(out_dir / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for row in rows:
            w.writerow(row)

    print(f"DONE: {len(rows)} clips written to {wav_dir}", flush=True)


if __name__ == "__main__":
    main()
