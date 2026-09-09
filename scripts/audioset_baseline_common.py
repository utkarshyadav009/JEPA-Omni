"""scripts/audioset_baseline_common.py — shared parquet/decode helper for the
AudioSet audio-tower feature extraction scripts (audioset_extract_<model>.py).

NOT a modification of scripts/audioset_extract_features.py (the reference
implementation for M2's own ambient/world_state extraction) — this is a new,
separate helper used only by the baseline-model extraction scripts. Decode
convention matches the task spec exactly: soundfile decode from bytes, mean
over channels for mono mixdown, native sample rate returned to the caller so
each model can resample to its own required rate.
"""
from __future__ import annotations
import glob
import io
import os
import time
from typing import Iterator, Dict, Any, List

# Cap CPU thread pools BEFORE numpy/torch/torchaudio touch them. This machine is
# shared and heavily loaded; each of our extraction processes previously
# oversubscribed all 256 cores (torch/MKL default to nproc threads per
# process), which measured a ~40x user/real CPU-time blowup and tanked
# throughput to <1 clip/s. Also set by the launching shell (belt & suspenders --
# env vars set here can be too late if a C extension already read them).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import torch as _torch
_torch.set_num_threads(8)

DATA_ROOT = "/mnt/Raid-Storage-2/utkarsh-data/audioset_hf/data"


def decode_flac_mono(b: bytes) -> "tuple[np.ndarray, int]":
    """FLAC bytes -> (mono float32 waveform, sample_rate). Mono by mean over channels."""
    w, sr = sf.read(io.BytesIO(b), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w, sr


def iter_audioset_rows(split: str, limit: int = 0) -> Iterator[Dict[str, Any]]:
    """Yields dicts {video_id, labels, audio_bytes} in file order across all
    parquet shards for the given split ('bal_train' or 'eval')."""
    files = sorted(glob.glob(os.path.join(DATA_ROOT, split, "*.parquet")))
    assert files, f"no parquet files found for split={split!r} under {DATA_ROOT}"
    n = 0
    for fp in files:
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=64):
            for d in batch.to_pylist():
                yield {
                    "video_id": d["video_id"],
                    "labels": list(d["labels"]),
                    "audio_bytes": d["audio"]["bytes"],
                }
                n += 1
                if limit and n >= limit:
                    return


class RateTracker:
    def __init__(self, split: str):
        self.split = split
        self.t0 = time.time()

    def maybe_print(self, n: int, bad: int, every: int = 1024):
        if n % every < 1:
            return
        # print roughly every `every` clips
    def log(self, n: int, bad: int):
        el = time.time() - self.t0
        print(f"[extract] {self.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
