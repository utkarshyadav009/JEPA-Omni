"""scripts/extract_features_action100m.py — Phase 2 of the VL-JEPA-style
embedding predictor plan (/home/utkarsh/.claude/plans/serene-soaring-abelson.md).

Extracts AV features + captions from Action100M-preview's REAL per-segment
timestamps (the Tree-of-Captions hierarchy: `nodes[i]["start"]`/`["end"]`,
multiple granularity `level`s per video, `gpt.action.brief`/`gpt.action.detailed`
captions) -- NOT a fixed 10s crop per video. A single video contributes many
segments at different time points and granularities as SEPARATE training
examples, directly answering "use all the timestamps."

Each segment is still one bounded forward pass through the frozen V-JEPA2
ViT-L / WavJEPA encoders (they are fixed-window encoders, not causal/
streaming) -- a segment of whatever real duration it has gets uniformly
sampled into NUM_FRAMES, same convention scripts/extract_features_av.py
already uses for VGGSound regardless of source clip length. This is NOT
on-the-fly decoding during training -- extraction happens ONCE here, cached
to disk in the exact tensor format data/av_cached_dataset.py already reads,
so Phase 3's training loop is the same fast batched InfoNCE loop Phase 1
already used, no extra GPUs needed for training itself (this project's own
M1 history already learned that lesson the hard way -- see
m1_experiment_results.md's "Feature Caching" section).

Runs INCREMENTALLY against whatever's on disk right now (the download is
still in progress, ~36% done as of 2026-08-01) -- re-run periodically as
more videos land; already-extracted segments are skipped (checkpointed by
output file existence, same pattern as extract_features_av.py's atomic-save
convention).

Duration filter: segments outside [MIN_SEG_S, MAX_SEG_S] are skipped --
sub-second leaf segments don't give V-JEPA2's 64-frame sampling anything
meaningful to work with (near-duplicate frames), and very long segments
(whole-video root nodes can be 400s+) would need such heavy uniform
downsampling that most temporal detail is lost. The range is chosen to
roughly match VGGSound's ~10s scale, not arbitrary.

Usage:
    python scripts/extract_features_action100m.py --limit 2000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO, WAVJEPA_AUDIO_SAMPLE_RATE
from scripts.extract_features_av import _spatial_pool, _audio_ts, VISION_SPAT, VISION_DIM, VISION_TEMP

PREVIEW_DIR = "/mnt/Raid-Storage-2/utkarsh-data/action100m_preview/data"
VIDEO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/action100m_videos"
CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_action100m"
CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "action100m_captions.jsonl")

NUM_FRAMES = 64
RESOLUTION = 256
AUDIO_SR = WAVJEPA_AUDIO_SAMPLE_RATE
MIN_SEG_S = 3.0
MAX_SEG_S = 30.0


def _shard(clip_id: str) -> str:
    return clip_id[:2] if len(clip_id) >= 2 else "_"


def _feat_path(cache_dir: str, clip_id: str) -> str:
    return os.path.join(cache_dir, _shard(clip_id), f"{clip_id}.pt")


def _write_manifest_if_missing(cache_dir: str) -> None:
    """data/av_cached_dataset.py's AVCachedDataset REQUIRES a manifest.json
    with these exact keys (REQUIRED_MANIFEST_KEYS) -- found missing
    (2026-08-01) when Phase 3 training first tried to load this cache.
    video_fps/video_n_frames describe the FIXED encoder input shape (64
    uniformly-sampled frames per segment, matching VGGSound's convention),
    NOT a literal source-video frame rate -- Action100M segment durations
    vary 3-30s (see MIN_SEG_S/MAX_SEG_S), unlike VGGSound's fixed 10s."""
    path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(path):
        return
    m = {
        "vision_repo": "facebook/vjepa2-vitl-fpc64-256",
        "wavjepa_base_repo": WAVJEPA_BASE_REPO,
        "wavjepa_nat_repo": WAVJEPA_NAT_REPO,
        "vision_spatial_pool": VISION_SPAT,
        "video_fps": NUM_FRAMES / MIN_SEG_S,   # placeholder rate, see note above -- not literal
        "video_n_frames": NUM_FRAMES,
        "video_resolution": RESOLUTION,
        "audio_sample_rate": AUDIO_SR,
        "wavjepa_base_token_rate_hz": 99.6,
        "wavjepa_nat_token_rate_hz": 99.6,
        "source": "Action100M-preview, real per-segment start/end timestamps (not fixed 10s crops)",
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.rename(tmp, path)
    print(f"[action100m-extract] wrote {path}", flush=True)


def _uniform_frame_indices(lo: int, hi: int, num_frames: int) -> List[int]:
    """Uniformly sample num_frames indices from the CLOSED range [lo, hi]
    (inclusive), clamped to at least 1 real frame of range -- same
    linspace-based convention as data/video_text_dataset.py's
    _uniform_frame_indices, just parameterized on an arbitrary sub-range
    instead of [0, num_total)."""
    hi = max(hi, lo + 1)
    if hi - lo <= 1:
        return [lo] * num_frames
    idx = torch.linspace(lo, hi - 1, steps=num_frames)
    return idx.round().long().clamp_(lo, hi - 1).tolist()


def build_video_index() -> Dict[str, str]:
    """{video_uid: path} for every COMPLETE (not .part) mp4 on disk. yt-dlp
    names files either {uid}.mp4 or {uid}.f134.mp4 (format-tagged) --
    YouTube IDs never contain '.', so basename.split('.')[0] safely
    recovers the uid either way."""
    index: Dict[str, str] = {}
    for path in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        if path.endswith(".part"):
            continue
        uid = os.path.basename(path).split(".")[0]
        index[uid] = path
    return index


def _video_shard(uid: str, num_shards: int) -> int:
    """Deterministic video->shard assignment (md5, not the first-2-chars
    _shard() used for cache directory sharding above -- that one's for
    filesystem fanout, this one's for splitting work across parallel
    extraction processes). Same video always lands in the same shard
    across separate process launches, so a shard's own progress is
    resumable independently of the others."""
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % num_shards


def iter_action100m_segments(max_per_video: int = 8, mode: str = "deoverlap",
                              shard_idx: int = 0, num_shards: int = 1):
    """Yields (video_uid, video_path, node) for qualifying segments.

    mode="deoverlap" (the first pass, run 2026-08-01): capped at
    `max_per_video` PER VIDEO, greedily de-overlapped -- without this, a
    single video's hundreds of hierarchy nodes (many of them parent/child,
    i.e. heavily overlapping in time -- e.g. level=3 [4.4,23.3] containing
    level=4 [10.2,23.3] containing level=5 [10.2,15.6]) would dominate the
    extracted set and starve diversity across the other ~34k videos on
    disk (found: a first, uncapped run pulled 8000 segments from only 54
    distinct videos). Per video: sort qualifying nodes by start time,
    greedily keep non-overlapping ones, up to max_per_video.

    mode="all" (the second, "extra data" pass, per direct instruction --
    the overlapping/nested segments have real value too, not just noise:
    multiple caption granularities for the same/overlapping content is
    richer supervision, not necessarily harmful to InfoNCE): yields EVERY
    qualifying node per video, uncapped, including ones the deoverlap pass
    would have dropped for overlapping a kept segment. Relies on the
    existing "already extracted, skip" check in main() (by clip_id =
    f"{uid}__{node_id[:8]}", one row per distinct node_id) to avoid
    redundant work on segments the first pass already extracted -- this
    mode is additive, not a replacement, run it AFTER the deoverlap pass.

    Streams parquet files one at a time -- doesn't load the whole preview
    dataset into memory at once (it's 37GB of annotations across ~120k videos)."""
    import pandas as pd
    video_index = build_video_index()
    print(f"[action100m-extract] {len(video_index)} complete videos on disk, mode={mode}, "
          f"shard {shard_idx}/{num_shards}", flush=True)

    parquet_files = sorted(glob.glob(os.path.join(PREVIEW_DIR, "*.parquet")))
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            uid = row["video_uid"]
            if uid not in video_index:
                continue
            if num_shards > 1 and _video_shard(uid, num_shards) != shard_idx:
                continue
            path = video_index[uid]
            candidates = []
            for node in row["nodes"]:
                if node["gpt"] is None:
                    continue
                dur = node["end"] - node["start"]
                if dur < MIN_SEG_S or dur > MAX_SEG_S:
                    continue
                candidates.append(node)

            if mode == "all":
                for node in candidates:
                    yield uid, path, node
                continue

            candidates.sort(key=lambda n: n["start"])
            kept = []
            last_end = -1.0
            for node in candidates:
                if node["start"] >= last_end:
                    kept.append(node)
                    last_end = node["end"]
                if len(kept) >= max_per_video:
                    break
            for node in kept:
                yield uid, path, node


def decode_video_segment(path: str, start_s: float, end_s: float,
                          num_frames: int, resolution: int) -> Optional[Tensor]:
    from torchcodec.decoders import VideoDecoder
    try:
        decoder = VideoDecoder(path, device="cpu")
        fps = decoder.metadata.average_fps
        num_total = decoder.metadata.num_frames
        lo = max(0, int(start_s * fps))
        hi = min(num_total, int(end_s * fps))
        if hi <= lo:
            return None
        indices = _uniform_frame_indices(lo, hi, num_frames)
        unique = sorted(set(indices))
        remap = {o: i for i, o in enumerate(unique)}
        batch = decoder.get_frames_at(indices=unique)
        decoded = batch.data
        gi = torch.tensor([remap[i] for i in indices], dtype=torch.long)
        frames = decoded.index_select(0, gi.to(decoded.device))
        if frames.shape[-2] != resolution or frames.shape[-1] != resolution:
            x = frames.float()
            x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                              align_corners=False, antialias=True)
            frames = x.round_().clamp_(0, 255).to(torch.uint8)
        return frames
    except Exception as e:
        print(f"[action100m-extract]   video decode failed for {path} [{start_s:.1f},{end_s:.1f}]: {e!r}", flush=True)
        return None


def decode_audio_segment(path: str, start_s: float, end_s: float, target_sr: int) -> Optional[Tensor]:
    from torchcodec.decoders import AudioDecoder
    try:
        dec = AudioDecoder(path, sample_rate=target_sr)
        frames = dec.get_samples_played_in_range(start_seconds=start_s, stop_seconds=end_s)
        wav = frames.data
        if wav.shape[0] > 1:
            wav = wav.mean(0)
        else:
            wav = wav[0]
        return wav
    except Exception as e:
        print(f"[action100m-extract]   audio decode failed for {path} [{start_s:.1f},{end_s:.1f}]: {e!r}", flush=True)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=2000, help="max segments to extract this run")
    p.add_argument("--max-per-video", type=int, default=8,
                   help="mode=deoverlap only: cap segments taken from any single video, "
                        "de-overlapped by start time -- prevents a handful of videos with many "
                        "hierarchy nodes from dominating the extracted set (found 2026-08-01: "
                        "uncapped, 8000 segments came from only 54 distinct videos)")
    p.add_argument("--mode", default="deoverlap", choices=["deoverlap", "all"],
                   help="'deoverlap' (default, run first): capped, non-overlapping segments per "
                        "video, for broad video diversity. 'all' (run second, additive): every "
                        "qualifying node per video including overlapping/nested ones, for richer "
                        "multi-granularity supervision on the same content -- per direct "
                        "instruction that overlap isn't necessarily harmful. Already-extracted "
                        "clip_ids are skipped either way, so running 'all' after 'deoverlap' adds "
                        "new segments without redoing work.")
    p.add_argument("--fields", nargs="+", default=["brief", "detailed"],
                   help="which gpt.action.{field} entries to save as captions")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="run only videos where md5(video_uid) %% num-shards == shard-idx -- "
                        "for running N parallel processes (one per GPU) over disjoint video "
                        "sets, so they don't race on the same videos/segments. Set CUDA_VISIBLE_"
                        "DEVICES per-process to actually land each shard on a different GPU.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cpu-threads", type=int, default=0,
                   help="torch.set_num_threads() cap for this process (0 = torch default, which "
                        "is os.cpu_count() -- FOUND 2026-08-02: with the default, each concurrent "
                        "extraction process independently grabs the full core count for its own "
                        "threadpool, so N processes oversubscribe by ~Nx and the CPU-bound decode/ "
                        "resize step thrashes instead of parallelizing -- load average hit 232 on a "
                        "256-core box with just 4 processes, while GPUs sat idle at ~30%% waiting on "
                        "CPU. Set this to cpu_count // (planned concurrent process count) when "
                        "running multiple shards at once.")
    args = p.parse_args()
    assert 0 <= args.shard_idx < args.num_shards or args.num_shards == 1
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[action100m-extract] device={device} shard={args.shard_idx}/{args.num_shards}", flush=True)

    print("[action100m-extract] loading frozen encoders...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    os.makedirs(CACHE_DIR, exist_ok=True)
    _write_manifest_if_missing(CACHE_DIR)
    # Separate per-shard caption file -- concurrent processes appending to the
    # SAME file risks interleaved/corrupted lines under multi-process writes;
    # merge shard files into CAPTIONS_PATH once all shards finish.
    captions_path = CAPTIONS_PATH if args.num_shards == 1 else f"{CAPTIONS_PATH}.shard{args.shard_idx}"
    captions_out = open(captions_path, "a")

    n_done, n_skipped_exists, n_failed = 0, 0, 0
    t0 = time.time()
    for uid, path, node in iter_action100m_segments(max_per_video=args.max_per_video, mode=args.mode,
                                                      shard_idx=args.shard_idx, num_shards=args.num_shards):
        if n_done >= args.limit:
            break
        clip_id = f"{uid}__{node['node_id'][:8]}"
        out_path = _feat_path(CACHE_DIR, clip_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            n_skipped_exists += 1
            continue

        start_s, end_s = node["start"], node["end"]
        true_dur = end_s - start_s

        frames = decode_video_segment(path, start_s, end_s, NUM_FRAMES, RESOLUTION)
        audio = decode_audio_segment(path, start_s, end_s, AUDIO_SR)
        if frames is None or audio is None or audio.numel() < AUDIO_SR // 2:
            n_failed += 1
            continue

        with torch.no_grad():
            raw = vision_enc.encode(frames.unsqueeze(0).to(device))  # (1, 8192, 1024)
            vis_full = raw[0].view(VISION_TEMP, 256, VISION_DIM)
            vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16)   # (32, 16, 1024)

            wav1 = audio.unsqueeze(0).unsqueeze(0).to(device)
            wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0).to(device)
            base_feat = base_enc.encode(wav1)[0].to(torch.bfloat16)
            nat_feat = nat_enc.encode(wav2)[0].to(torch.bfloat16)

        vis_ts = torch.zeros(VISION_TEMP, 2)
        fpg = NUM_FRAMES / VISION_TEMP
        fd = true_dur / NUM_FRAMES
        for t in range(VISION_TEMP):
            vis_ts[t, 0] = t * fpg * fd
            vis_ts[t, 1] = (t + 1) * fpg * fd

        torch.save({
            "vision": vis_pooled.cpu(),
            "ambient_base": base_feat.cpu(),
            "ambient_nat": nat_feat.cpu(),
            "vision_ts": vis_ts,
            "ambient_base_ts": _audio_ts(base_feat.shape[0], true_dur),
            "ambient_nat_ts": _audio_ts(nat_feat.shape[0], true_dur),
            "clip_duration_s": true_dur,
        }, out_path + ".tmp")
        os.rename(out_path + ".tmp", out_path)

        cap_row = {"clip_id": clip_id, "video_uid": uid, "start_s": start_s, "end_s": end_s,
                   "level": node["level"]}
        for field in args.fields:
            cap_row[f"gpt_action_{field}"] = node["gpt"]["action"].get(field)
        captions_out.write(json.dumps(cap_row) + "\n")
        captions_out.flush()

        n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            print(f"[action100m-extract] {n_done}/{args.limit} extracted "
                  f"({n_skipped_exists} already existed, {n_failed} failed) "
                  f"elapsed={elapsed:.0f}s ({elapsed/max(1,n_done):.2f}s/clip)", flush=True)

    captions_out.close()
    print(f"[action100m-extract] DONE. {n_done} new segments extracted, "
          f"{n_skipped_exists} already existed, {n_failed} failed. "
          f"Cache: {CACHE_DIR}  Captions: {captions_path}", flush=True)


if __name__ == "__main__":
    main()
