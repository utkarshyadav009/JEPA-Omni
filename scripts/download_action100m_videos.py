"""scripts/download_action100m_videos.py — Meta Action100M Video Downloader.

Extracts all 120,000 unique video_uid entries from Meta Action100M Preview Parquet files
and downloads the corresponding MP4 videos from YouTube using multi-threaded yt-dlp + Deno JS solver.

Usage:
    # Download first 10,000 videos
    python scripts/download_action100m_videos.py --max-videos 10000 --workers 16

    # Download all 120,000 videos with 24 threads
    python scripts/download_action100m_videos.py --workers 24
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = "/home/utkarsh/raid2-data/action100m_preview/data"
DEFAULT_OUTPUT_DIR = "/home/utkarsh/raid2-data/action100m_videos"

YTDLP_BIN = sys.executable.replace("python", "yt-dlp")
if not os.path.exists(YTDLP_BIN):
    YTDLP_BIN = "yt-dlp"

COOKIE_DIR = "/home/utkarsh"


def get_cookie_files() -> List[str]:
    """Dynamically scans for all available cookie files in COOKIE_DIR."""
    cookies = sorted(glob.glob(os.path.join(COOKIE_DIR, "cookies[0-9]*.txt")))
    if not cookies:
        cookies = sorted(glob.glob(os.path.join(COOKIE_DIR, "cookies_*.txt")))
    if not cookies:
        cookies = [os.path.join(COOKIE_DIR, "cookies.txt")]
    return cookies


def get_cookie_for_thread() -> str:
    """Assigns a cookie file to the current thread via round-robin from currently available cookies."""
    cookie_files = get_cookie_files()
    tid = threading.get_ident()
    idx = abs(hash(tid)) % len(cookie_files)
    return cookie_files[idx]


# Tor SOCKS5 proxy (if running)
TOR_PROXY = "socks5://127.0.0.1:9050"
TOR_CONTROL_PORT = 9051
TOR_DATA_DIR = "/tmp/tor_data_active"

_newnym_lock = threading.Lock()
_last_newnym = 0.0
NEWNYM_MIN_INTERVAL_S = 12  # Tor itself rate-limits NEWNYM effect to roughly every 10s;
                            # calling more often than that just wastes a control-port round
                            # trip without forcing an actually-new circuit.


def renew_tor_circuit() -> bool:
    """Signals Tor to abandon its current circuits so the NEXT stream gets a fresh exit
    node -- does NOT affect already-open connections. FOUND (2026-08-02): with no
    ControlPort configured, this script's 30 concurrent workers were all reusing
    whatever small set of circuits Tor's default stream-isolation happened to build for
    repeated connections to the same destination (youtube.com), so the "6 cookie" account
    rotation was the only real request-level diversity -- the underlying Tor exit-IP
    diversity was much lower than the worker count implied, causing both severe
    bandwidth contention (workers sharing a handful of circuits) and YouTube bot-detection
    (many requests fingerprinted to a concentrated set of exit IPs). This uses Tor's raw
    control-port protocol directly (cookie auth) rather than adding a `stem` dependency --
    the AUTHENTICATE + SIGNAL NEWNYM exchange is two lines once the cookie is read.
    Thread-safe and rate-limited to avoid re-triggering faster than Tor honors anyway."""
    global _last_newnym
    with _newnym_lock:
        now = time.time()
        if now - _last_newnym < NEWNYM_MIN_INTERVAL_S:
            return False
        _last_newnym = now
    try:
        cookie_path = os.path.join(TOR_DATA_DIR, "control_auth_cookie")
        with open(cookie_path, "rb") as f:
            cookie_hex = f.read().hex()
        with socket.create_connection(("127.0.0.1", TOR_CONTROL_PORT), timeout=5) as s:
            s.sendall(f"AUTHENTICATE {cookie_hex}\r\n".encode())
            auth_resp = s.recv(1024)
            if not auth_resp.startswith(b"250"):
                logging.warning(f"Tor control AUTHENTICATE failed: {auth_resp!r}")
                return False
            s.sendall(b"SIGNAL NEWNYM\r\n")
            signal_resp = s.recv(1024)
            return signal_resp.startswith(b"250")
    except Exception as e:
        logging.warning(f"Tor circuit renewal failed: {e!r}")
        return False


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


def extract_action100m_uids(parquet_dir: str) -> List[str]:
    """Extracts all unique video_uid values from Action100M Parquet files."""
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")

    all_uids: List[str] = []
    seen: Set[str] = set()

    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["video_uid"])
        uids = df["video_uid"].dropna().unique()
        for u in uids:
            u_str = str(u).strip()
            if u_str and u_str not in seen:
                seen.add(u_str)
                all_uids.append(u_str)

    return all_uids


def download_single_video(vuid: str, out_dir: str) -> Tuple[str, bool, str]:
    """Downloads a full MP4 video for a given Action100M video_uid."""
    safe_uid = vuid.replace("/", "_").replace("\\", "_")
    out_filename = f"{safe_uid}.mp4"
    out_path = os.path.join(out_dir, out_filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return vuid, True, "exists"

    url = f"https://www.youtube.com/watch?v={vuid}"

    cmd = [
        YTDLP_BIN,
        url,
        "--cookies",
        get_cookie_for_thread(),
        "--proxy",
        TOR_PROXY,
        "--remote-components",
        "ejs:github",
        "--sleep-requests",
        "1",
        "--min-sleep-interval",
        "1",
        "--max-sleep-interval",
        "2",
        "--limit-rate",
        "10M",
        "--extractor-retries",
        "1",
        "-N",
        "2",
        "-f",
        "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return vuid, True, "downloaded"
        else:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return vuid, False, f"error: {res.stderr.strip()[:100]}"
    except Exception as e:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return vuid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Meta Action100M Video Downloader")
    parser.add_argument(
        "--parquet-dir",
        default=PARQUET_DIR,
        help="Directory containing Action100M Parquet files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory on RAID storage for MP4 videos",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to download (None = all 120,000)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(PROJECT_ROOT, "logs", "download_action100m_videos.log")
    setup_logging(log_file)

    logging.info("=" * 65)
    logging.info("Meta Action100M MP4 Video Downloader")
    logging.info(f"Parquet Source: {args.parquet_dir}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Max Workers: {args.workers}")
    logging.info("=" * 65)

    # 1. Extract Unique Video UIDs
    all_uids = extract_action100m_uids(args.parquet_dir)
    logging.info(f"Extracted {len(all_uids):,} unique video_uids from Parquet files")

    if args.max_videos:
        all_uids = all_uids[: args.max_videos]
        logging.info(f"Limited download to first {len(all_uids):,} videos")

    # 2. Multi-threaded Video Download
    logging.info(f"Starting download of {len(all_uids):,} videos with {args.workers} threads...")
    n_success = 0
    n_skipped = 0
    n_failed = 0

    consecutive_fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_single_video, vuid, args.output_dir): vuid
            for vuid in all_uids
        }

        # FOUND (2026-08-02): the `reason` string below was always computed but never
        # logged anywhere -- there was no visibility into how much of the failure rate
        # was genuinely-unavailable videos (unfixable) vs proxy/bot-detection (fixable
        # via circuit renewal below). Categorize into buckets and report a breakdown
        # alongside the existing progress line.
        reason_buckets: Counter = Counter()
        n_since_renew = 0

        for idx, future in enumerate(as_completed(futures), 1):
            vuid = futures[future]
            try:
                vuid_res, ok, reason = future.result()
                if ok:
                    if reason == "exists":
                        n_skipped += 1
                    else:
                        n_success += 1
                    consecutive_fails = 0
                else:
                    n_failed += 1
                    reason_lower = str(reason).lower()
                    if "bot" in reason_lower or "sign in" in reason_lower or "429" in reason_lower:
                        consecutive_fails += 1
                        reason_buckets["bot_or_ratelimit"] += 1
                        # Strongest fixable-failure signal -- renew the Tor circuit so the
                        # NEXT request (this one already failed and isn't retried within
                        # this run) has a fresh exit IP instead of hitting the same
                        # already-flagged one again.
                        renew_tor_circuit()
                    elif "private" in reason_lower or "unavailable" in reason_lower or "removed" in reason_lower or "not available" in reason_lower:
                        reason_buckets["genuinely_unavailable"] += 1
                        consecutive_fails = 0
                    elif "timeout" in reason_lower or "timed out" in reason_lower:
                        reason_buckets["timeout"] += 1
                        consecutive_fails = 0
                    else:
                        reason_buckets["other"] += 1
                        consecutive_fails = 0
            except Exception as exc:
                n_failed += 1
                reason_buckets["exception"] += 1
                consecutive_fails = 0

            # Defensive periodic rotation regardless of failures -- keeps exit-IP
            # diversity high even when failures aren't yet spiking (renew_tor_circuit's
            # own rate-limit makes this cheap to call this often).
            n_since_renew += 1
            if n_since_renew >= 15:
                n_since_renew = 0
                renew_tor_circuit()

            if consecutive_fails >= 30:
                logging.warning("⚠️ Detected 30 consecutive bot-lock / rate-limit failures. Initiating 15-minute automatic cooldown...")
                import time
                time.sleep(900)
                logging.info("✅ 15-minute cooldown complete. Resuming download pipeline...")
                consecutive_fails = 0

            if idx % 100 == 0 or idx == len(all_uids):
                pct = (idx / len(all_uids)) * 100
                logging.info(
                    f"Progress: {idx}/{len(all_uids)} ({pct:.1f}%) | "
                    f"Downloaded: {n_success:,} | Skipped: {n_skipped:,} | Failed: {n_failed:,} | "
                    f"FailReasons: {dict(reason_buckets)}"
                )

    logging.info("=" * 65)
    logging.info("Meta Action100M Download Run Complete!")
    logging.info(f"  Successfully Downloaded : {n_success:,}")
    logging.info(f"  Already Existed (Skipped): {n_skipped:,}")
    logging.info(f"  Unavailable / Failed    : {n_failed:,}")
    logging.info(f"  Output Video Directory  : {args.output_dir}")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
