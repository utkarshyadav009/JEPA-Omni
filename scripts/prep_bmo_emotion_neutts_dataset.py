"""scripts/prep_bmo_emotion_neutts_dataset.py -- build an HF `datasets.Dataset`
from the Fish-rendered emotion corpus (data/bmo_emotion_fish, produced by the
Fish s2.1-pro-free community-BMO voice, one clip per (mood, line), verified
tags per EMOTION_MAPPING.md). Same field schema as the neutral BMO dataset
(text, codes, __key__) PLUS a `mood` column so the emotion fine-tune can
prepend the matching `<|MOOD|>` control token.

Mirrors prep_bmo_neutts_dataset.py's encoding path exactly (NeuCodec
encode_code on a downmixed-mono temp wav, same text filter), so the emotion
codes are directly compatible with the neutral codes in the same training mix.

Notes learned this session:
  - Fish returns 44.1kHz WAV with a STREAMING placeholder in the header's
    data-size field (Python's `wave` misreads nframes as ~2^31); soundfile
    reads the real content fine, and NeuCodec.encode_code resamples internally.
  - Per-file try/except: one bad clip must not kill a 1400+ clip run.
  - Incremental checkpoint every 200 rows (a late crash once lost a whole run).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import sys
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, Features, Value, Sequence
from neucodec import NeuCodec

sys.path.insert(0, ".")
from models.m5_streaming_voice import normalize_bmo_text  # noqa: E402

ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
DIGIT_RE = re.compile(r"\d")
CURRENCY_RE = re.compile(r"[£$€]")


def passes_filter(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        return False
    if DIGIT_RE.search(text):
        return False
    if ACRONYM_RE.search(text):
        return False
    if CURRENCY_RE.search(text):
        return False
    if text[-1] not in ".,?!":
        return False
    return True


def to_mono_tmp(wav_path: Path, tmp_path: Path) -> Path:
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # Trim leading near-silence / very-quiet breath padding (the user heard a
    # breath at the START of some clips). Conservative: only cut the leading
    # region below 8% of peak, so a real speech onset (much louder) is never
    # clipped; teaches the model a clean start. 20ms pre-roll kept for a natural
    # attack + to avoid a hard cut click.
    if len(data) > sr // 10:
        peak = float(np.max(np.abs(data))) or 1.0
        thr = 0.08 * peak
        above = np.flatnonzero(np.abs(data) > thr)
        if above.size:
            start = max(0, int(above[0]) - int(0.02 * sr))
            data = data[start:]
    sf.write(str(tmp_path), data, sr)
    return tmp_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/bmo_emotion_fish")
    ap.add_argument("--out", default="data/bmo_emotion_neutts_dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0=all; >0 for a smoke test")
    ap.add_argument("--max-codes", type=int, default=320, help="drop clips longer than this "
                    "(~6.4s) so the emotion tokens don't learn long-clip distributions and "
                    "fail to emit EOS on short lines -- the v2 '7s garble' root cause (Task 185)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    wavs_dir = in_dir / "wavs"
    meta_path = in_dir / "metadata.csv"

    print("Loading NeuCodec ...", flush=True)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(args.device).eval()

    rows = {"text": [], "codes": [], "__key__": [], "mood": []}
    skipped = 0
    tmp_mono = Path(args.out).parent / "_tmp_mono_emotion.wav"
    tmp_mono.parent.mkdir(parents=True, exist_ok=True)

    # metadata.csv is delimiter="|": fname | text | mood | tag
    with open(meta_path) as f:
        records = [r for r in csv.reader(f, delimiter="|") if r and len(r) >= 3]
    if args.limit:
        records = records[: args.limit]

    for i, rec in enumerate(records):
        fname, mood = rec[0].strip(), rec[2].strip()
        # normalize BMO->Beemo BEFORE filtering: "BMO" is an all-caps acronym the
        # filter rejects, which had deleted most self-referential lines (lonely
        # 113/132, happy 70/111). Normalizing first recovers them AND fixes the
        # spoken pronunciation (the saved text is what gets phonemized in training).
        text = normalize_bmo_text(rec[1].strip())
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
            print(f"[{i}] ENCODE FAILED {fname}: {type(e).__name__}: {e}", flush=True)
            skipped += 1
            continue
        code_list = codes.flatten().cpu().tolist()
        if args.max_codes and len(code_list) > args.max_codes:
            skipped += 1  # too-long clip: EOS-calibration hazard (Task 185)
            continue
        rows["text"].append(text)
        rows["codes"].append(code_list)
        rows["__key__"].append(fname)
        rows["mood"].append(mood)
        if i % 100 == 0:
            print(f"[{i}/{len(records)}] encoded {fname} ({mood})", flush=True)
        if len(rows["text"]) % 200 == 0:
            ck = Features({"text": Value("string"), "codes": Sequence(Value("int64")),
                          "__key__": Value("string"), "mood": Value("string")})
            Dataset.from_dict(rows, features=ck).save_to_disk(args.out + "_checkpoint")
            print(f"[checkpoint] saved {len(rows['text'])} rows", flush=True)

    print(f"Kept {len(rows['text'])}, skipped {skipped}", flush=True)
    features = Features({"text": Value("string"), "codes": Sequence(Value("int64")),
                         "__key__": Value("string"), "mood": Value("string")})
    ds = Dataset.from_dict(rows, features=features)
    ds.save_to_disk(args.out)
    print(f"DONE: saved {len(ds)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
