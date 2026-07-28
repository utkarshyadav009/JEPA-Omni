"""scripts/download_audioset_mp4.py — High-Performance AudioSet MP4 Downloader.

Downloads official AudioSet 10-second MP4 video segments (containing BOTH Video and Audio)
from YouTube using official Google AudioSet CSV metadata and multi-threaded yt-dlp + ffmpeg.

Dataset URL References:
  - Official Downloads: https://research.google.com/audioset/download.html
  - Strong Labels: https://research.google.com/audioset/download_strong.html

Usage:
  # Download balanced train set (22k clips)
  python scripts/download_audioset_mp4.py --subset balanced --workers 16

  # Download strong labels set (103k clips)
  python scripts/download_audioset_mp4.py --subset strong --workers 16

  # Download first 50,000 clips from unbalanced set
  python scripts/download_audioset_mp4.py --subset unbalanced --max-clips 50000 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = "/home/utkarsh/raid2-data/audioset_mp4"

# Official Google AudioSet CSV URLs
CSV_URLS = {
    "eval": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
    "balanced": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "unbalanced": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
    "strong": "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_train_strong.tsv",
}

YTDLP_BIN = sys.executable.replace("python", "yt-dlp")
if not os.path.exists(YTDLP_BIN):
    YTDLP_BIN = "yt-dlp"


def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


def download_csv(url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        logging.info(f"Downloading CSV metadata from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        logging.info(f"Saved CSV to {dest_path}")
    return dest_path


def parse_audioset_csv(csv_path: str, is_tsv: bool = False) -> List[Tuple[str, float, float, List[str]]]:
    """Parses AudioSet CSV/TSV file returning list of (ytid, start_sec, end_sec, labels)."""
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        if is_tsv:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or row[0].startswith("#") or len(row) < 4:
                    continue
                if row[1] == "start_time_seconds":
                    continue
                clip_id, event_start, event_end, mid = row[0], float(row[1]), float(row[2]), row[3]
                parts = clip_id.rsplit("_", 1)
                ytid = parts[0]
                start_ms = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                clip_start = start_ms / 1000.0
                clip_end = clip_start + 10.0
                items.append((ytid, clip_start, clip_end, mid, clip_id))

            # Deduplicate by unique clip_id (ytid_startms)
            unique_map: Dict[str, Tuple[str, float, float, List[str]]] = {}
            for ytid, c_start, c_end, mid, cid in items:
                if cid not in unique_map:
                    unique_map[cid] = (ytid, c_start, c_end, [])
                unique_map[cid][3].append(mid)

            deduped_items = []
            for cid, (ytid, c_start, c_end, mids) in unique_map.items():
                deduped_items.append((ytid, c_start, c_end, sorted(list(set(mids)))))
            return deduped_items
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) >= 4:
                    ytid = parts[0]
                    start_s = float(parts[1])
                    end_s = float(parts[2])
                    labels = [l.strip() for l in parts[3:]]
                    items.append((ytid, start_s, end_s, labels))
    return items


def download_single_clip(
    ytid: str,
    start_sec: float,
    end_sec: float,
    out_dir: str,
) -> Tuple[str, bool, str]:
    """Downloads a single 10-second MP4 video clip containing BOTH video and audio streams."""
    safe_ytid = ytid.replace("/", "_").replace("\\", "_")
    out_filename = f"{safe_ytid}_{int(start_sec)}_{int(end_sec)}.mp4"
    out_path = os.path.join(out_dir, out_filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return ytid, True, "exists"

    section_arg = f"*{start_sec:.3f}-{end_sec:.3f}"
    url = f"https://www.youtube.com/watch?v={ytid}"

    cmd = [
        YTDLP_BIN,
        url,
        "--remote-components",
        "ejs:github",
        "-N",
        "4",
        "--download-sections",
        section_arg,
        "-f",
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    cookie_path = "/home/utkarsh/cookies.txt"
    if os.path.exists(cookie_path):
        cmd.extend(["--cookies", cookie_path])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return ytid, True, "downloaded"
        else:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return ytid, False, f"error: {res.stderr.strip()[:100]}"
    except Exception as e:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return ytid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="AudioSet MP4 Video Downloader")
    parser.add_argument(
        "--subset",
        default="balanced",
        choices=["balanced", "eval", "strong", "unbalanced"],
        help="AudioSet subset to download",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory on RAID storage",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum number of clips to download (None = all)",
    )
    args = parser.parse_args()

    subset_dir = os.path.join(args.output_dir, args.subset)
    os.makedirs(subset_dir, exist_ok=True)
    log_file = os.path.join(PROJECT_ROOT, "logs", f"download_audioset_mp4_{args.subset}.log")
    setup_logging(log_file)

    logging.info("=" * 65)
    logging.info(f"AudioSet MP4 Video Downloader — Subset: {args.subset.upper()}")
    logging.info(f"Output Directory: {subset_dir}")
    logging.info(f"Max Workers: {args.workers}")
    logging.info("=" * 65)

    # 1. Download & Parse CSV Metadata
    csv_url = CSV_URLS[args.subset]
    csv_filename = f"{args.subset}_segments.csv" if args.subset != "strong" else "audioset_train_strong.tsv"
    local_csv = os.path.join(PROJECT_ROOT, "data", csv_filename)
    download_csv(csv_url, local_csv)

    is_tsv = args.subset == "strong"
    all_items = parse_audioset_csv(local_csv, is_tsv=is_tsv)
    logging.info(f"Parsed {len(all_items):,} total clips from {local_csv}")

    if args.max_clips:
        all_items = all_items[: args.max_clips]
        logging.info(f"Limited download to first {len(all_items):,} clips")

    # Save manifest mapping for downstream dataloaders
    manifest_path = os.path.join(subset_dir, "manifest.json")
    manifest_dict = {}
    for ytid, start, end, labels in all_items:
        key = f"{ytid}_{int(start)}_{int(end)}"
        manifest_dict[key] = {
            "ytid": ytid,
            "start": start,
            "end": end,
            "labels": labels,
            "mp4": f"{key}.mp4",
        }
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    logging.info(f"Saved dataset manifest mapping to {manifest_path}")

    # 2. Multi-threaded Video Download
    logging.info(f"Starting download of {len(all_items):,} clips with {args.workers} threads...")
    n_success = 0
    n_skipped = 0
    n_failed = 0
    missing_ytids = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_single_clip, ytid, start, end, subset_dir): (ytid, start, end)
            for ytid, start, end, _ in all_items
        }

        total = len(futures)
        count = 0
        for future in as_completed(futures):
            count += 1
            ytid, success, status = future.result()
            if success:
                if status == "exists":
                    n_skipped += 1
                else:
                    n_success += 1
            else:
                n_failed += 1
                missing_ytids.append(ytid)

            if count % 100 == 0 or count == total:
                pct = (count / total) * 100
                logging.info(
                    f"Progress: {count}/{total} ({pct:.1f}%) | Downloaded: {n_success:,} | Skipped: {n_skipped:,} | Failed: {n_failed:,}"
                )

    logging.info("=" * 65)
    logging.info("Download Run Complete!")
    logging.info(f"  Successfully Downloaded : {n_success:,}")
    logging.info(f"  Already Existed (Skipped): {n_skipped:,}")
    logging.info(f"  Unavailable / Failed    : {n_failed:,}")
    logging.info(f"  Output MP4 Directory    : {subset_dir}")
    logging.info("=" * 65)

    # Save missing/unavailable video list for tracking
    missing_file = os.path.join(subset_dir, "missing_ytids.json")
    with open(missing_file, "w") as f:
        json.dump(missing_ytids, f, indent=2)


if __name__ == "__main__":
    main()
