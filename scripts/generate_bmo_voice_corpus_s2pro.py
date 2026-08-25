"""scripts/generate_bmo_voice_corpus_s2pro.py -- generates a synthetic BMO
voice corpus using Fish Audio S2 Pro's real voice cloning (locked to the real
BMO reference clip) + inline [tag] emotion/prosody control, covering every
homeostatic mood bucket plus dedicated non-verbal-only clips (whisper,
scream, laugh, cry, gasp, sigh).

Must be run inside the `fish-speech` conda env (needs fish_speech installed).
Calls the two real fish-speech CLI scripts directly via subprocess, same
commands verified manually in the pilot test:
  1. fish_speech/models/text2semantic/inference.py --text ... --prompt-audio
     ... --prompt-text ... --checkpoint-path <s2-pro dir> --output-dir <tmp>
  2. fish_speech/models/dac/inference.py -i codes_0.npy -o out.wav
     --checkpoint-path <s2-pro dir>/codec.pth

Output: data/bmo_s2pro_synth/wavs/*.wav +
data/bmo_s2pro_synth/metadata.csv (filename|text|tone), same format as the
real BMO_SpeechDataset/metadata.csv, so this drops straight into Piper /
NeuTTS-Air fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

FISH_SPEECH_DIR = Path.home() / "repos" / "fish-speech"
S2_PRO_DIR = Path.home() / "hf_models" / "s2-pro"
REF_WAV = Path(
    "/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad/bmo_ref.wav"
)
REF_TEXT = "Hello, my name is Football, and you are Carlos, my best friend since college."

# Real S2 Pro preset tags (from README.md's "Rich Emotion Library"), mapped to
# our existing homeostatic mood buckets (models/homeostatic_state.py's
# homeostatic_to_mood_state()). Left as "" (no tag) where no preset is a good
# fit -- forcing a mismatched tag is worse than plain delivery.
MOOD_TAG = {
    "happy": "[delight]",
    "excited": "[excited]",
    "tired": "[sigh]",
    "lonely": "[low voice]",
    "curious": "",
    "bored": "[low volume]",
    "stressed": "[panting]",
    "surprised": "[surprised]",
    "content": "",
    "anxious": "[panting]",
    "concerned": "",
}

# Dedicated non-verbal-only clips: minimal lexical content, tag carries the
# whole signal. Not tied to any specific dialogue line.
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


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(FISH_SPEECH_DIR))


def generate_one(text: str, out_wav: Path, tmp_dir: Path) -> float:
    t0 = time.time()
    run(
        [
            sys.executable,
            "fish_speech/models/text2semantic/inference.py",
            "--text", text,
            "--prompt-text", REF_TEXT,
            "--prompt-audio", str(REF_WAV),
            "--checkpoint-path", str(S2_PRO_DIR),
            "--output-dir", str(tmp_dir),
            "--num-samples", "1",
        ]
    )
    run(
        [
            sys.executable,
            "fish_speech/models/dac/inference.py",
            "-i", str(tmp_dir / "codes_0.npy"),
            "-o", str(out_wav),
            "--checkpoint-path", str(S2_PRO_DIR / "codec.pth"),
        ]
    )
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bmo_synthetic_functional.jsonl")
    ap.add_argument("--out-dir", default="data/bmo_s2pro_synth")
    ap.add_argument("--limit", type=int, default=None, help="cap number of dialogue lines, for a quick smoke run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    lines = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    # tool_use lines contain literal <tool_call .../> tags meant to be read
    # as text by the LLM, not spoken aloud -- exclude from voice generation.
    lines = [l for l in lines if l.get("category") != "tool_use"]
    if args.limit:
        lines = lines[: args.limit]

    for i, item in enumerate(lines):
        mood = item["state"].get("mood", "content")
        tag = MOOD_TAG.get(mood, "")
        text = f"{tag} {item['text']}".strip() if tag else item["text"]
        fname = f"dialogue_{i:03d}_{mood}.wav"
        out_wav = wav_dir / fname
        try:
            elapsed = generate_one(text, out_wav, tmp_dir)
            print(f"[{i}] {elapsed:.1f}s mood={mood} tag={tag!r} -> {fname}", flush=True)
            rows.append((fname, item["text"], mood))
        except subprocess.CalledProcessError as e:
            print(f"[{i}] FAILED mood={mood}: {e}", flush=True)

    for name, text, tone in NONVERBAL_CLIPS:
        fname = f"nonverbal_{name}.wav"
        out_wav = wav_dir / fname
        try:
            elapsed = generate_one(text, out_wav, tmp_dir)
            print(f"[nonverbal] {elapsed:.1f}s tone={tone} -> {fname}", flush=True)
            rows.append((fname, text, tone))
        except subprocess.CalledProcessError as e:
            print(f"[nonverbal] FAILED tone={tone}: {e}", flush=True)

    with open(out_dir / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for row in rows:
            w.writerow(row)

    print(f"DONE: {len(rows)} clips written to {wav_dir}, metadata at {out_dir / 'metadata.csv'}", flush=True)


if __name__ == "__main__":
    main()
