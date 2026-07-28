"""scripts/subsample_vggsound_60k.py — item 2 (corpus mix): subsample the
persistent VGGSound feature cache (~199,007 clips,
/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k) to ~60,000 clips so
Ego4D's 21,761-clip (well, 17,140 in the v2 rebuild's train split) train
set is a meaningful ~20-27% of the combined corpus instead of a 9.9%
minority too small to move the domain gap.

Rule (deterministic, seed=60000):
  1. Discover every cached clip_id by scanning the cache's shard dirs.
  2. Exclude the 1545 eval-gallery ids (data/vggsound_eval_1545.txt) and the
     1700 fresh-holdout ids (data/vggsound_fresh_holdout_1700.txt) -- both
     must stay held out of training, matching the project's existing
     convention (train_m2.py already excludes the eval set via
     exclude_ids; the fresh-holdout set exists specifically to be a
     never-trained-on check).
  3. Sort the remaining ids for a reproducible base order, shuffle with
     seed=60000, take the first 60,000.

Usage:
    python scripts/subsample_vggsound_60k.py
"""
from __future__ import annotations

import json
import os
import random

CACHE_DIR = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k"
EVAL_1545_PATH = "data/vggsound_eval_1545.txt"
FRESH_HOLDOUT_1700_PATH = "data/vggsound_fresh_holdout_1700.txt"
TARGET_N = 60000
SEED = 60000
OUT_PATH = "data/vggsound_train_60k.txt"
SUMMARY_OUT = "checkpoints/vjepa21_shelved/VGGSOUND_60K_SUBSAMPLE_SUMMARY.json"


def discover_clip_ids(cache_dir: str):
    ids = []
    for shard in sorted(os.listdir(cache_dir)):
        shard_dir = os.path.join(cache_dir, shard)
        if not os.path.isdir(shard_dir) or shard.startswith("."):
            continue
        for fname in sorted(os.listdir(shard_dir)):
            if fname.endswith(".pt"):
                ids.append(fname[:-3])
    return ids


def main() -> None:
    all_ids = discover_clip_ids(CACHE_DIR)
    print(f"[subsample-60k] {len(all_ids)} clips discovered in cache", flush=True)

    with open(EVAL_1545_PATH) as f:
        eval_ids = set(l.strip() for l in f if l.strip())
    with open(FRESH_HOLDOUT_1700_PATH) as f:
        holdout_ids = set(l.strip() for l in f if l.strip())
    print(f"[subsample-60k] excluding {len(eval_ids)} eval + {len(holdout_ids)} fresh-holdout ids", flush=True)

    excluded = eval_ids | holdout_ids
    candidates = sorted(set(all_ids) - excluded)
    print(f"[subsample-60k] {len(candidates)} candidates after exclusion", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    subsample = sorted(candidates[:TARGET_N])

    with open(OUT_PATH, "w") as f:
        for cid in subsample:
            f.write(cid + "\n")

    summary = {
        "seed": SEED,
        "target_n": TARGET_N,
        "n_total_cached": len(all_ids),
        "n_excluded_eval": len(eval_ids),
        "n_excluded_fresh_holdout": len(holdout_ids),
        "n_candidates_after_exclusion": len(candidates),
        "n_subsampled": len(subsample),
        "out_path": OUT_PATH,
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[subsample-60k] wrote {OUT_PATH} + {SUMMARY_OUT}", flush=True)


if __name__ == "__main__":
    main()
