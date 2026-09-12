"""scripts/temporal_probe/stream_extract.py — P2.6/P2.7 stream-processing pipeline.

download one source file -> extract features at a fixed stride -> write -> DELETE the video.

Peak storage becomes a WORKING SET (one video + one file's features), not the corpus. This
is what makes Epic-Kitchens-100 (741 GiB of video) and an Ego4D subset tractable on a box
with ~1.03 TB free.

TWO OUTPUT MODES, because the two consumers have very different needs:

  --mode world_state   ~5.5 KB/window. W(t) + mean-pooled vision + mean-pooled ambient.
                       Enough for the persistence/prediction probes (Phases 2-3) and for
                       fitting g: W(t) -> W(t+delta), which is RUN-5 Arm B's SPEC as written
                       (ridge / 2-layer MLP over FROZEN world-states).

  --mode features      ~3.94 MB/window (measured, not estimated: 518 GB / 134,491 windows in
                       feature_cache_ego4d_train_v1). Full token features in the SAME layout
                       as the existing caches -- vision (32,16,1024), ambient_base/nat
                       (T,768), per-token timestamps, clip_duration_s -- so it is a drop-in
                       corpus for train_m2.py. REQUIRED if Arm B retrains the bridge rather
                       than fitting a head on frozen W: you cannot backprop into the bridge
                       through a stored world-state. The 734x reduction does NOT apply here.

*** ONE-SHOT CONSTRAINT ***
The video is DELETED after extraction. The stride cannot be revisited without
re-downloading the corpus. ERR FINER: 1 s costs 2x the disk of 2 s but 2 s can never be
made finer after the fact. Re-deriving a coarse stride from a fine one is a decimation;
the reverse is a re-download.

NOT RUN YET. Epic-Kitchens is CC BY-NC 4.0 and awaits a human licence decision; Ego4D
awaits fresh credentials (the previous AWS key is deleted server-side -- STS reports
InvalidClientTokenId and S3 InvalidAccessKeyId).
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MB_PER_WINDOW_FEATURES = 3.94        # MEASURED
KB_PER_WINDOW_WORLDSTATE = 5.5       # (1024 + 1024 + 768) floats at fp16


def plan(hours: float, stride_s: float, mode: str) -> Dict:
    n = hours * 3600.0 / stride_s
    gb = (n * MB_PER_WINDOW_FEATURES / 1024) if mode == "features" else (n * KB_PER_WINDOW_WORLDSTATE / 1024 / 1024)
    return {"hours": hours, "stride_s": stride_s, "mode": mode,
            "windows": int(n), "output_GB": round(gb, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["epic_kitchens", "ego4d"], required=True)
    ap.add_argument("--mode", choices=["world_state", "features"], required=True)
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--file-list", default=None, help="newline-delimited source ids/URLs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default="/mnt/Raid-Storage-2/utkarsh-data/_stream_work")
    ap.add_argument("--keep-video", action="store_true",
                    help="do NOT delete after extraction (disables the storage saving)")
    ap.add_argument("--plan-only", action="store_true", help="print the cost plan and exit")
    ap.add_argument("--hours", type=float, default=100.0, help="for --plan-only")
    a = ap.parse_args()

    if a.plan_only:
        print(json.dumps(plan(a.hours, a.stride_s, a.mode), indent=2))
        return

    raise SystemExit(
        "REFUSING TO RUN. Acquisition is gated on a human decision that has not been given:\n"
        "  epic_kitchens -- CC BY-NC 4.0 (non-commercial). Licence acceptance is the user's\n"
        "                   to make, not this script's. See docs/CORPUS_OPTIONS.md.\n"
        "  ego4d         -- requires fresh AWS credentials; the existing key is deleted\n"
        "                   server-side (InvalidAccessKeyId), so nothing can be fetched.\n"
        "Re-run with --plan-only for costing. The extraction body is intentionally not\n"
        "implemented until a corpus is actually authorised, so that it is written against\n"
        "the real directory layout rather than a guessed one.")


if __name__ == "__main__":
    main()
