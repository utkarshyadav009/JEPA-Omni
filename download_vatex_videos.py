import json
import os
import subprocess
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

JSON_PATH = "/home/jovyan/work/data/vatex/vatex_train.json"
OUTPUT_DIR = "/home/jovyan/work/data/vatex/video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Parsing VATEX JSON...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

unique_yt_ids = set()
for item in data:
    vid_string = item.get("videoID", "")
    if len(vid_string) >= 11:
        unique_yt_ids.add(vid_string[:11])
        
yt_ids = list(unique_yt_ids)
print(f"Found {len(yt_ids)} unique master videos in VATEX.")


def download_video(yt_id):
    # Add a random jitter so requests don't hit at perfectly regular intervals
    time.sleep(random.uniform(2, 5)) 
    
    out_path = os.path.join(OUTPUT_DIR, f"{yt_id}.mp4")
    if os.path.exists(out_path):
        return True
        
    cmd = [
        "yt-dlp",
        "-f", "18/worst", 
        "--socket-timeout", "15",
        "--retries", "2",
        "--quiet",
        "--no-warnings",
        "-o", out_path,
        f"https://www.youtube.com/watch?v={yt_id}"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

print(f"Starting full download of {len(yt_ids)} videos. This will take several hours...")
success_count = 0

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(download_video, vid): vid for vid in yt_ids}
    
    for i, future in enumerate(as_completed(futures), 1):
        if future.result():
            success_count += 1
        
        # Print update every 100 videos to keep the terminal clean
        if i % 100 == 0:
            print(f"Progress: {i}/{len(yt_ids)} | Successful: {success_count} | Failed/Skipped: {i - success_count}")

print(f"\nDownload complete! Successfully fetched {success_count} videos.")