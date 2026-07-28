"""scripts/download_action100m_preview.py — Meta Action100M Preview Downloader.

Downloads the facebook/action100m-preview Parquet dataset (36.5 GB) containing
14.7 Million V-JEPA 2 temporally-segmented action captions and LLM tree-of-captions.

Repo: https://huggingface.co/datasets/facebook/action100m-preview
Destination: /home/utkarsh/raid2-data/action100m_preview
"""

import os
import sys
import logging
from huggingface_hub import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "download_action100m_preview.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a")
    ]
)

LOCAL_DIR = "/home/utkarsh/raid2-data/action100m_preview"

def main():
    logging.info("=" * 65)
    logging.info("Starting Meta Action100M Preview Dataset Download")
    logging.info(f"Target Directory: {LOCAL_DIR}")
    logging.info("=" * 65)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id="facebook/action100m-preview",
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            max_workers=8,
        )
        logging.info(f"Download complete! Saved to {path}")

        # Verification of files
        parquet_files = [
            os.path.join(dp, f) for dp, dn, filenames in os.walk(LOCAL_DIR)
            for f in filenames if f.endswith(".parquet")
        ]
        total_size = sum(os.path.getsize(f) for f in parquet_files)
        logging.info(f"Total Parquet files downloaded: {len(parquet_files)}")
        logging.info(f"Total dataset size on disk: {total_size / (1024**3):.2f} GB")

    except Exception as e:
        logging.exception(f"Error during Action100M preview download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
