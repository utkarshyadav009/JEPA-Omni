"""Regenerate the ENTIRE Fish emotion voice corpus from CLEANED text. The weird
typographic Unicode (em-dash->pause, non-breaking hyphen->glued word, curly
quotes) in the original corpus was audible as odd sounds in the trained voice.
metadata.csv is already ASCII-cleaned; we also apply normalize_bmo_text so Fish
says "Beemo" (not B-M-O) and the text is fully normalized. Overwrites every wav
in place; the (fname, mood, tag) mapping is preserved. Free s2.1-pro-free access,
so we regenerate all ~1421 clips for a fully clean dataset."""
import sys, csv, pathlib, time, requests
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
from models.m5_streaming_voice import normalize_bmo_text

KEY = pathlib.Path("/home/utkarsh/.config/fish_audio/api_key").read_text().strip()
REF = "323847d4c5394c678e5909c2206725f6"
IN = pathlib.Path("/home/utkarsh/JEPA-Omni/data/bmo_emotion_fish")
WAVS = IN / "wavs"

MOOD_TAG = {"excited":"[excited]","happy":"[delight]","content":"[relaxed]",
            "surprised":"[surprised]","stressed":"[panting]","anxious":"[nervous]",
            "concerned":"[worried]","lonely":"[sad]","tired":"[sigh]",
            "bored":"[indifferent]","curious":"[curious]"}


def call(text):
    r = requests.post("https://api.fish.audio/v1/tts",
                      headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                               "model": "s2.1-pro-free"},
                      json={"text": text, "reference_id": REF, "format": "wav"}, timeout=60)
    r.raise_for_status(); return r.content


def main():
    rows = [r for r in csv.reader(open(IN / "metadata.csv"), delimiter="|") if len(r) >= 3]
    print(f"regenerating {len(rows)} clips from cleaned text", flush=True)
    ok = fail = 0
    for i, r in enumerate(rows):
        fname, text, mood = r[0].strip(), r[1].strip(), r[2].strip()
        tag = MOOD_TAG.get(mood, "")
        clean = normalize_bmo_text(text)  # Beemo + ASCII, idempotent
        try:
            audio = call(f"{tag} {clean}".strip())
            (WAVS / fname).write_bytes(audio)
            ok += 1
            if ok % 100 == 0:
                print(f"  ... {ok}/{len(rows)} regenerated (last {mood})", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {fname}: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)
    print(f"DONE clean-regen ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
