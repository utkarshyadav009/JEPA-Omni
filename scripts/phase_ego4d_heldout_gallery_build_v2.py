"""scripts/phase_ego4d_heldout_gallery_build_v2.py — item 1d: rebuild the
Ego4D held-out gate. The v1 gallery (EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json,
n=1542 from only 81 files, mean 19 windows/file, up to 48) was shown
(EGO4D_GATE_DIAGNOSTIC.json) to be ambiguity-bound: file-level R@k is
consistently above chance (1.7-3.8x) while instance-level R@1 sits far below
the pre-registered 10% threshold, largely because each query competes
against up to 47 near-duplicate siblings from the same continuous footage.

Fix: cap AT MOST 2 windows per held-out file, drawn from ~350 files
(~600-800 total windows) instead of 81 -- same file-disjoint-from-train
guarantee, far fewer near-duplicate distractors per query.

Split rule (deterministic, seed=43 -- distinct from v1's seed=42 to avoid
any confusion between the two manifests):
  1. Group the 23,303 kept windows (ego4d_kept_v5_vadexcl.json) by source_id.
  2. Sort source_ids for a reproducible base order, shuffle with seed=43.
  3. Walk the shuffled file list, selecting files for held-out until 350
     files are selected (TARGET_HELDOUT_FILES). From each selected file,
     take the FIRST 2 windows in the kept-list's existing order (already
     deterministic from the original scoring pass) -- or all of them if
     the file has fewer than 2.
  4. Every remaining file (and ALL of its windows) goes to train -- a
     file-level partition, same guarantee as v1.

Usage:
    python scripts/phase_ego4d_heldout_gallery_build_v2.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

KEPT_PATH = "checkpoints/vjepa21_shelved/ego4d_kept_v5_vadexcl.json"
TARGET_HELDOUT_FILES = 350
CAP_PER_FILE = 2
SEED = 43
TRAIN_OUT = "checkpoints/vjepa21_shelved/EGO4D_TRAIN_SPLIT_FILEDISJOINT_V2.json"
HELDOUT_OUT = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json"
SUMMARY_OUT = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_SPLIT_SUMMARY_V2.json"


def main() -> None:
    with open(KEPT_PATH) as f:
        kept = json.load(f)
    print(f"[ego4d-split-v2] loaded {len(kept)} kept windows", flush=True)

    by_file = defaultdict(list)
    for c in kept:
        by_file[c["source_id"]].append(c)
    files = sorted(by_file.keys())
    print(f"[ego4d-split-v2] {len(files)} unique source files", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(files)

    heldout_files = files[:TARGET_HELDOUT_FILES]
    train_files = files[TARGET_HELDOUT_FILES:]

    heldout_windows = []
    for fid in heldout_files:
        heldout_windows.extend(by_file[fid][:CAP_PER_FILE])
    train_windows = [w for fid in train_files for w in by_file[fid]]

    assert set(heldout_files).isdisjoint(set(train_files)), "file leaked across split"
    assert len(heldout_windows) + sum(len(by_file[fid][CAP_PER_FILE:]) for fid in heldout_files) \
        + len(train_windows) == len(kept), "window accounting mismatch"

    with open(HELDOUT_OUT, "w") as f:
        json.dump(heldout_windows, f, indent=2)
    with open(TRAIN_OUT, "w") as f:
        json.dump(train_windows, f, indent=2)

    from collections import Counter
    wpf = Counter(w["source_id"] for w in heldout_windows)
    summary = {
        "seed": SEED,
        "cap_per_file": CAP_PER_FILE,
        "target_heldout_files": TARGET_HELDOUT_FILES,
        "n_kept_total": len(kept),
        "n_files_total": len(files),
        "n_heldout_files": len(heldout_files),
        "n_train_files": len(train_files),
        "n_heldout_windows": len(heldout_windows),
        "n_train_windows": len(train_windows),
        "heldout_windows_per_file_min_max": [min(wpf.values()), max(wpf.values())],
        "file_disjoint_verified": True,
        "heldout_manifest": HELDOUT_OUT,
        "train_manifest": TRAIN_OUT,
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ego4d-split-v2] FROZEN: {HELDOUT_OUT} ({len(heldout_windows)} windows, "
          f"{len(heldout_files)} files) + {TRAIN_OUT} ({len(train_windows)} windows, "
          f"{len(train_files)} files)", flush=True)


if __name__ == "__main__":
    main()
