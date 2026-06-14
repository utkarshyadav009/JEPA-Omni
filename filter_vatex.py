import json
import os

JSON_PATH = "/home/jovyan/work/data/vatex/vatex_train.json"
VIDEO_DIR = "/home/jovyan/work/data/vatex/video"
OUTPUT_JSON = "/home/jovyan/work/data/vatex/vatex_train_filtered.json"

print("Loading original VATEX JSON...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

filtered_data = []
missing_count = 0

print(f"Checking {len(data)} entries against local disk...")
for item in data:
    vid_string = item.get("videoID", "")
    
    # Kaggle datasets usually save the video using just the 11-character YouTube ID
    yt_id = vid_string[:11]
    
    # Check if either the 11-char ID or the full ID exists on disk
    path_11_char = os.path.join(VIDEO_DIR, f"{yt_id}.mp4")
    path_full = os.path.join(VIDEO_DIR, f"{vid_string}.mp4")
    
    if os.path.exists(path_11_char) or os.path.exists(path_full):
        filtered_data.append(item)
    else:
        missing_count += 1

print("\n--- Filtering Complete ---")
print(f"Original entries: {len(data)}")
print(f"Valid entries kept: {len(filtered_data)}")
print(f"Missing videos dropped: {missing_count}")

with open(OUTPUT_JSON, 'w') as f:
    json.dump(filtered_data, f)
print(f"Saved clean JSON to: {OUTPUT_JSON}")