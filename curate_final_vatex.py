import json
import os
import glob
from datasets import load_from_disk

BASE_DIR = "/home/jovyan/work/data/vatex"
VIDEO_DIR = os.path.join(BASE_DIR, "video")
OUTPUT_JSON = os.path.join(BASE_DIR, "vatex_final_curated.json")

# 1. Collect all available video IDs from disk
print(f"Scanning {VIDEO_DIR} for videos...")
available_videos = {}
for ext in ["*.mp4", "*.mkv"]:
    for path in glob.glob(os.path.join(VIDEO_DIR, ext)):
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        # Store both the full filename and the YT ID (first 11 chars)
        yt_id = stem[:11]
        available_videos[filename] = path
        if yt_id not in available_videos:
            available_videos[yt_id] = path

print(f"Found {len(available_videos)} video entries (full names or YT IDs) on disk.")

# 2. Extract all captions from Arrow files
print("Extracting captions from all splits (Train, Val, Public Test)...")
all_data = {}

# We only need to read from one part as they are identical
part_path = os.path.join(BASE_DIR, "part1", "vatex", "json")
for split in ["train", "validation", "public_test"]:
    ds_path = os.path.join(part_path, split)
    if os.path.exists(ds_path):
        print(f"  Processing {split}...")
        ds = load_from_disk(ds_path)
        for row in ds:
            vid = row["videoID"]
            captions = row["enCap"]
            if vid and captions:
                all_data[vid] = {
                    "videoID": vid,
                    "enCap": captions,
                    "split": split
                }

print(f"Extracted {len(all_data)} unique video IDs with captions from annotations.")

# 3. Match annotations with videos on disk
final_data = []
matched_ids = set()

for vid, entry in all_data.items():
    yt_id = vid[:11]
    
    # Try various match candidates
    possible_names = [
        f"{vid}.mp4",
        f"{yt_id}.mp4",
        f"{vid}.mkv",
        f"{yt_id}.mkv"
    ]
    
    found = False
    for name in possible_names:
        if name in available_videos:
            final_data.append(entry)
            matched_ids.add(vid)
            found = True
            break

print(f"\n--- Curation Summary ---")
print(f"Total annotations available: {len(all_data)}")
print(f"Total videos on disk: 34372")
print(f"Successfully matched: {len(final_data)}")
print(f"Unmatched annotations: {len(all_data) - len(final_data)}")

# 4. Check for videos without annotations
unmatched_videos = 34372 - len(final_data)
if unmatched_videos > 0:
    print(f"Videos on disk without matching annotations: {unmatched_videos}")

# 5. Save the final JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(final_data, f)

print(f"\nSaved final curated dataset to: {OUTPUT_JSON}")
