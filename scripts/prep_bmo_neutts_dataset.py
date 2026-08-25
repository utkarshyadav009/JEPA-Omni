"""scripts/prep_bmo_neutts_dataset.py -- builds a local HF `datasets.Dataset`
from the real BMO_SpeechDataset (971 clips, filename|text|tone metadata),
matching the field schema NeuTTS-Air's own finetune.py expects
(`text`, `codes`, `__key__`), so we can point examples/finetune.py at our
data instead of `neuphonic/emilia-yodas-english-neucodec`.

Filtering criteria copied from finetune.py's own preprocessing (so our data
gets the same treatment the reference dataset gets): skip empty text, text
containing digits, all-caps acronyms, text not ending in . , ? !, and text
containing currency symbols. Also skips pure non-verbal tags like [cry]
(not real spoken text, no phoneme content to learn from for this dataset).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, Features, Value, Sequence
from neucodec import NeuCodec


def to_mono_tmp(wav_path: Path, tmp_path: Path) -> Path:
    """NeuCodec's _prepare_audio does not downmix stereo -> mono; our real
    BMO clips are 44.1kHz stereo, so do it ourselves before encoding."""
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(str(tmp_path), data.astype(np.float32), sr)
    return tmp_path

ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
DIGIT_RE = re.compile(r"\d")
CURRENCY_RE = re.compile(r"[£$€]")


def passes_filter(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        return False  # non-verbal tag, not spoken text
    if DIGIT_RE.search(text):
        return False
    if ACRONYM_RE.search(text):
        return False
    if CURRENCY_RE.search(text):
        return False
    if text[-1] not in ".,?!":
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/bmo_speech_dataset/metadata.csv")
    ap.add_argument("--wavs-dir", default="data/bmo_speech_dataset/wavs")
    ap.add_argument("--out", default="data/bmo_neutts_dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("Loading NeuCodec ...", flush=True)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(args.device).eval()

    wavs_dir = Path(args.wavs_dir)
    rows = {"text": [], "codes": [], "__key__": []}
    skipped = 0
    tmp_mono = Path(args.out).parent / "_tmp_mono.wav"
    tmp_mono.parent.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        lines = f.read().splitlines()
    header = lines[0]
    for i, row in enumerate(lines[1:]):
        parts = row.split("|")
        if len(parts) < 2:
            skipped += 1
            continue
        fname, text = parts[0].strip(), parts[1].strip()
        if not passes_filter(text):
            skipped += 1
            continue
        wav_path = wavs_dir / fname
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            mono_path = to_mono_tmp(wav_path, tmp_mono)
            with torch.no_grad():
                codes = codec.encode_code(str(mono_path))
        except Exception as e:
            # One bad file (degenerate tensor shape in NeuCodec's internal
            # resample layer, seen for real on a specific clip) must not
            # kill the whole run -- isolate per-file and keep going.
            print(f"[{i}] ENCODE FAILED {fname}: {type(e).__name__}: {e}", flush=True)
            skipped += 1
            continue
        rows["text"].append(text)
        rows["codes"].append(codes.flatten().cpu().tolist())
        rows["__key__"].append(fname)
        if i % 50 == 0:
            print(f"[{i}] encoded {fname}", flush=True)
        if len(rows["text"]) % 200 == 0:
            # Incremental checkpoint -- a late failure must not lose all
            # prior encoding work (real incident: a crash at row ~850/969
            # with no checkpointing lost 100% of the run's work).
            ckpt_features = Features({
                "text": Value("string"), "codes": Sequence(Value("int64")), "__key__": Value("string"),
            })
            Dataset.from_dict(rows, features=ckpt_features).save_to_disk(args.out + "_checkpoint")
            print(f"[checkpoint] saved {len(rows['text'])} rows", flush=True)

    print(f"Kept {len(rows['text'])}, skipped {skipped}", flush=True)

    features = Features({
        "text": Value("string"),
        "codes": Sequence(Value("int64")),
        "__key__": Value("string"),
    })
    ds = Dataset.from_dict(rows, features=features)
    ds.save_to_disk(args.out)
    print(f"DONE: saved dataset with {len(ds)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
