"""scripts/extract_features_ego4d_expand.py — extract feature cache for the
NEW Ego4D windows recovered by raising --per-file-cap (50->150) on the 946
already-safe train files (checkpoints/vjepa21_shelved/ego4d_expand_kept_cap150.json).
Writes into the SAME cache dir as scripts/extract_features_ego4d_train.py
(additive -- the original 17,140 windows' files are untouched, this only
adds new ones), using the identical clip_id convention and tensor schema
so data/av_cached_dataset.py reads the combined set transparently.

Multi-GPU: run one process per GPU with --shard-idx/--num-shards.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_ego4d_expand.py --shard-idx 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import VISION_DIM, _spatial_pool, _vision_ts, _audio_ts
from scripts.extract_features_ego4d_train import (
    decode_video, decode_audio, window_idx_from_start, _feat_path, CACHE_DIR,
)
from models.audio_encoder import WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO

KEPT_PATH = "checkpoints/vjepa21_shelved/ego4d_expand_kept_cap150.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--kept-path", default=KEPT_PATH)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ego4d-expand-extract] shard {args.shard_idx}/{args.num_shards} device={device}", flush=True)

    with open(args.kept_path) as f:
        kept = json.load(f)
    my_windows = [w for i, w in enumerate(kept) if i % args.num_shards == args.shard_idx]
    if args.limit is not None:
        my_windows = my_windows[:args.limit]
    print(f"[ego4d-expand-extract] shard {args.shard_idx}: {len(my_windows)}/{len(kept)} windows", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    t_start = time.time()
    n_done, n_failed, n_skipped = 0, 0, 0

    for i, m in enumerate(my_windows):
        vid = f"ego4d_{m['source_id']}_w{window_idx_from_start(m['start_sec']):04d}"
        out_path = _feat_path(args.cache_dir, vid)
        if os.path.isfile(out_path):
            n_skipped += 1
            continue
        try:
            frames, t0, t1 = decode_video(m["path"], m["start_sec"], device)
            audio, true_dur = decode_audio(m["path"], t0, t1)

            with torch.no_grad():
                raw = vision_enc.encode(frames.unsqueeze(0).to(device))
            raw = raw[0]
            vis_full = raw.view(32, 256, VISION_DIM)
            vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16)
            vis_ts = _vision_ts()

            wav1 = audio.unsqueeze(0).unsqueeze(0)
            wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0)
            with torch.no_grad():
                base_feat = base_enc.encode(wav1.to(device))[0].to(torch.bfloat16)
                nat_feat = nat_enc.encode(wav2.to(device))[0].to(torch.bfloat16)
            base_ts = _audio_ts(base_feat.shape[0], true_dur)
            nat_ts = _audio_ts(nat_feat.shape[0], true_dur)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({
                "vision": vis_pooled.cpu(),
                "ambient_base": base_feat.cpu(),
                "ambient_nat": nat_feat.cpu(),
                "vision_ts": vis_ts,
                "ambient_base_ts": base_ts,
                "ambient_nat_ts": nat_ts,
                "clip_duration_s": true_dur,
            }, out_path + ".tmp")
            os.rename(out_path + ".tmp", out_path)
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[ego4d-expand-extract] shard {args.shard_idx}: window {i} "
                  f"({m['path']}@{m['start_sec']}) FAILED: {e!r}", flush=True)

        if (i + 1) % 200 == 0 or i == len(my_windows) - 1:
            elapsed = time.time() - t_start
            print(f"[ego4d-expand-extract] shard {args.shard_idx}: {i+1}/{len(my_windows)} "
                  f"(done={n_done}, failed={n_failed}, skipped={n_skipped}), "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    print(f"[ego4d-expand-extract] shard {args.shard_idx} DONE: done={n_done} failed={n_failed} "
          f"skipped={n_skipped}", flush=True)


if __name__ == "__main__":
    main()
