"""Regenerate ONLY the emotion clips whose text contains "BMO", using the
"Beemo" spelling so Fish voices the character's name correctly (it spells out
"B-M-O" as letters otherwise -- the source of the micro-pause mispronunciation
the model learned). Overwrites those wavs in place; metadata is unchanged (the
dataset prep normalizes text to "Beemo" itself, so audio + text then align).

Only the ~338 BMO-containing lines are touched -- the rest are already fine."""
import csv, pathlib, re, time, requests

KEY = pathlib.Path("/home/utkarsh/.config/fish_audio/api_key").read_text().strip()
REF = "323847d4c5394c678e5909c2206725f6"
IN = pathlib.Path("/home/utkarsh/JEPA-Omni/data/bmo_emotion_fish")
WAVS = IN / "wavs"
BMO_RE = re.compile(r"\bBMO\b", re.IGNORECASE)

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
    todo = [(r[0].strip(), r[1].strip(), r[2].strip()) for r in rows if BMO_RE.search(r[1])]
    print(f"{len(todo)} BMO-containing clips to regenerate with 'Beemo'", flush=True)
    ok = fail = 0
    for i, (fname, text, mood) in enumerate(todo):
        beemo = BMO_RE.sub("Beemo", text)
        tag = MOOD_TAG.get(mood, "")
        try:
            audio = call(f"{tag} {beemo}".strip())
            (WAVS / fname).write_bytes(audio)
            ok += 1
            if ok % 40 == 0:
                print(f"  ... {ok}/{len(todo)} regenerated (last {mood}: {beemo[:40]})", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {fname}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE regen ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
