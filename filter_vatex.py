import json
import os
import glob

JSON_PATH = "/home/jovyan/work/data/vatex/vatex_train.json"
BASE_DIR = "/home/jovyan/work/data/vatex"
OUTPUT_JSON = "/home/jovyan/work/data/vatex/vatex_train_filtered.json"

print("Loading original VATEX JSON...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Build a set of available video filenames by searching recursively
print(f"Scanning {BASE_DIR} for mp4/mkv files...")
available_videos = set()
for ext in ["*.mp4", "*.mkv"]:
    # Using recursive glob to find all videos in subdirectories (video, part2, part3, etc.)
    for path in glob.glob(os.path.join(BASE_DIR, "**", ext), recursive=True):
        available_videos.add(os.path.basename(path))

print(f"Found {len(available_videos)} unique video files on disk.")

filtered_data = []
missing_count = 0

print(f"Checking {len(data)} entries against available videos...")
for item in data:
    vid_string = item.get("videoID", "")
    yt_id = vid_string[:11]
    
    # Check for various filename formats
    possible_names = [
        f"{yt_id}.mp4",
        f"{vid_string}.mp4",
        f"{yt_id}.mkv",
        f"{vid_string}.mkv"
    ]
    
    found = False
    for name in possible_names:
        if name in available_videos:
            found = True
            break
            
    if found:
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
