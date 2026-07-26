"""scripts/phase_ego4d_heldout_gallery_build.py — item 2a: replace the
EasyCom retrieval gate (retired: wrong domain, see falsifier_tracking.md
2026-07-26 diagnostic entry -- EasyCom is ~84% speech, nothing for the
ambient/WavJEPA path to distinguish windows by) with a FROZEN, SOURCE-FILE-
DISJOINT held-out split of the 23,303 filtered Ego4D windows
(checkpoints/vjepa21_shelved/ego4d_kept_v5_vadexcl.json).

Split rule (deterministic, seed=42):
  1. Group the 23,303 kept windows by source_id (the underlying video file).
  2. Sort source_ids for a reproducible base order, then shuffle with a
     fixed seed.
  3. Walk the shuffled file list, accumulating ALL of a file's windows into
     the held-out gallery until the running total reaches >=1500 windows
     (matching the 1545-clip VGGSound gallery's scale). The file whose
     addition crosses the threshold is the last one included.
  4. Every remaining file (and ALL of its windows) goes to train. No file
     ever appears in both splits -- this is a file-level partition, not a
     window-level one, so there is no leakage of near-duplicate windows
     from the same source video across the split.

Freezes two manifests (train/heldout) to disk. A downstream scoring script
must hard-assert its loaded gallery size against EXPECTED_HELDOUT_N below
before reporting any retrieval number.

Usage:
    python scripts/phase_ego4d_heldout_gallery_build.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

KEPT_PATH = "checkpoints/vjepa21_shelved/ego4d_kept_v5_vadexcl.json"
TARGET_HELDOUT_N = 1500
SEED = 42
TRAIN_OUT = "checkpoints/vjepa21_shelved/EGO4D_TRAIN_SPLIT_FILEDISJOINT.json"
HELDOUT_OUT = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json"
SUMMARY_OUT = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_SPLIT_SUMMARY.json"


def main() -> None:
    with open(KEPT_PATH) as f:
        kept = json.load(f)
    print(f"[ego4d-split] loaded {len(kept)} kept windows", flush=True)

    by_file = defaultdict(list)
    for c in kept:
        by_file[c["source_id"]].append(c)
    files = sorted(by_file.keys())
    print(f"[ego4d-split] {len(files)} unique source files", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(files)

    heldout_files, heldout_windows = [], []
    for fid in files:
        if len(heldout_windows) >= TARGET_HELDOUT_N:
            break
        heldout_files.append(fid)
        heldout_windows.extend(by_file[fid])

    heldout_file_set = set(heldout_files)
    train_files = [fid for fid in files if fid not in heldout_file_set]
    train_windows = [w for fid in train_files for w in by_file[fid]]

    # Hard partition check: no file in both, every kept window accounted for.
    assert heldout_file_set.isdisjoint(set(train_files)), "file leaked across split"
    assert len(heldout_windows) + len(train_windows) == len(kept), "window count mismatch after split"

    with open(HELDOUT_OUT, "w") as f:
        json.dump(heldout_windows, f, indent=2)
    with open(TRAIN_OUT, "w") as f:
        json.dump(train_windows, f, indent=2)

    summary = {
        "seed": SEED,
        "target_heldout_n": TARGET_HELDOUT_N,
        "n_kept_total": len(kept),
        "n_files_total": len(files),
        "n_heldout_files": len(heldout_files),
        "n_train_files": len(train_files),
        "n_heldout_windows": len(heldout_windows),
        "n_train_windows": len(train_windows),
        "file_disjoint_verified": True,
        "heldout_manifest": HELDOUT_OUT,
        "train_manifest": TRAIN_OUT,
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ego4d-split] FROZEN: {HELDOUT_OUT} ({len(heldout_windows)} windows, "
          f"{len(heldout_files)} files) + {TRAIN_OUT} ({len(train_windows)} windows, "
          f"{len(train_files)} files)", flush=True)


if __name__ == "__main__":
    main()
