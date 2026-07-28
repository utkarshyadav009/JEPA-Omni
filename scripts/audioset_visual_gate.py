"""scripts/audioset_visual_gate.py — P4 mitigation: visual admissibility gate
for AudioSet-Strong's audio-scored kept set.

Motivation (found via manual spot-check with decoded frames, 16/16 samples
reviewed): scripts/audioset_av_relevance_filter.py scores AUDIO ONLY (AST
event confidence * (1-vad_speech_frac) * energy_cov). Confident, dynamic
audio can come from a screenshot with a voiceover, a static stock photo
with background music, a cartoon/claymation soundtrack, or a title card --
none of which have a genuine visible sound-CAUSE. Manual review of the
capped kept set found ~9-10/16 samples were non-genuine (CAD screenshot,
stock guinea-pig photo, horror digital art, stick-figure/claymation
animation, title card, cartoon, a person's face mislabeled "Animal").

This does NOT re-run VAD/AST (expensive, audio scores are still valid) --
it adds a cheap VISUAL admissibility check on top of the already-scored
pool: (1) a staticness check (frame-to-frame pixel diff -- catches static
photos, screenshots, slideshows, title cards) and (2) a CLIP zero-shot
real-photo-vs-cartoon/screenshot/graphic check (catches animation/claymation
and anything staticness misses).

Usage:
    python scripts/audioset_visual_gate.py --in checkpoints/vjepa21_shelved/audioset_av_filter_scores_kept_CAPPED.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

STATIC_DIFF_THRESHOLD = 0.015   # mean abs pixel diff (0-1 scale) between two frames; below = static
CLIP_REAL_PROB_THRESHOLD = 0.35  # argmax must be "real photo" AND prob above this (chance = 1/5 = 0.20)
FRAME_SIZE = 224

PROMPTS = [
    "a real photograph or video frame of a real-world scene",
    "a cartoon, animation, or claymation illustration",
    "a screenshot of a computer application, software, or document",
    "a slideshow, title card, or text image",
    "a stock photo on a plain background",
]


def extract_two_frames(path: str, tmp_dir: str, idx: int):
    f1 = os.path.join(tmp_dir, f"vg_{idx}_a.jpg")
    f2 = os.path.join(tmp_dir, f"vg_{idx}_b.jpg")
    ok1 = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "0.4", "-i", path, "-frames:v", "1",
         "-vf", f"scale={FRAME_SIZE}:{FRAME_SIZE}", "-q:v", "4", f1, "-y"],
        timeout=15,
    ).returncode == 0 and os.path.isfile(f1)
    ok2 = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "3.0", "-i", path, "-frames:v", "1",
         "-vf", f"scale={FRAME_SIZE}:{FRAME_SIZE}", "-q:v", "4", f2, "-y"],
        timeout=15,
    ).returncode == 0 and os.path.isfile(f2)
    return (f1 if ok1 else None), (f2 if ok2 else None)


def worker(args):
    idx, path, tmp_dir = args
    try:
        f1, f2 = extract_two_frames(path, tmp_dir, idx)
        if f1 is None:
            return idx, None, None, None
        img1 = np.asarray(Image.open(f1).convert("RGB"), dtype=np.float32) / 255.0
        static = True
        if f2 is not None:
            img2 = np.asarray(Image.open(f2).convert("RGB"), dtype=np.float32) / 255.0
            diff = float(np.abs(img1 - img2).mean())
            static = diff < STATIC_DIFF_THRESHOLD
        else:
            diff = None
        return idx, f1, static, diff
    except Exception:
        return idx, None, None, None
    finally:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="checkpoints/vjepa21_shelved/audioset_av_filter_scores_kept_CAPPED.json")
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/audioset_visual_gate_result.json")
    p.add_argument("--tmp-dir", default="/dev/shm/audioset_visual_gate_frames")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    kept = json.load(open(args.inp))
    print(f"[visual-gate] {len(kept)} candidates from {args.inp}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import CLIPModel, CLIPProcessor
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_name)
    text_inputs = clip_proc(text=PROMPTS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_pooled = clip_model.text_model(**text_inputs).pooler_output
        text_feat = clip_model.text_projection(text_pooled)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    t_start = time.time()
    frame_paths = {}
    static_flags = {}
    diffs = {}
    tasks = [(i, c["path"], args.tmp_dir) for i, c in enumerate(kept)]
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            idx, f1, static, diff = fut.result()
            if f1 is not None:
                frame_paths[idx] = f1
                static_flags[idx] = static
                diffs[idx] = diff
            n_done += 1
            if n_done % 1000 == 0:
                elapsed = time.time() - t_start
                print(f"[visual-gate] frames: {n_done}/{len(tasks)} elapsed={elapsed/60:.1f}min", flush=True)
    print(f"[visual-gate] frame extraction done: {len(frame_paths)}/{len(kept)} usable, "
          f"elapsed={(time.time()-t_start)/60:.1f}min", flush=True)

    idxs = sorted(frame_paths.keys())
    clip_real_prob = {}
    BATCH = 256
    t_clip = time.time()
    for b0 in range(0, len(idxs), BATCH):
        batch_idxs = idxs[b0:b0 + BATCH]
        imgs = [Image.open(frame_paths[i]).convert("RGB") for i in batch_idxs]
        inputs = clip_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_pooled = clip_model.vision_model(**inputs).pooler_output
            img_feat = clip_model.visual_projection(img_pooled)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_feat @ text_feat.T
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for j, i in enumerate(batch_idxs):
            clip_real_prob[i] = float(probs[j, 0])
        if (b0 // BATCH) % 10 == 0:
            print(f"[visual-gate] clip scored {b0+len(batch_idxs)}/{len(idxs)} "
                  f"elapsed={(time.time()-t_clip)/60:.1f}min", flush=True)

    results = []
    for i, c in enumerate(kept):
        if i not in frame_paths:
            results.append({**c, "visual_admissible": False, "reason": "frame_extract_failed"})
            continue
        static = static_flags[i]
        real_prob = clip_real_prob.get(i, 0.0)
        admissible = (not static) and (real_prob >= CLIP_REAL_PROB_THRESHOLD)
        reason = None if admissible else ("static" if static else "not_real_photo")
        results.append({**c, "visual_admissible": admissible, "static_flag": static,
                         "clip_real_prob": real_prob, "reason": reason})

    n_admissible = sum(1 for r in results if r["visual_admissible"])
    n_static = sum(1 for r in results if r.get("reason") == "static")
    n_not_real = sum(1 for r in results if r.get("reason") == "not_real_photo")
    n_extract_fail = sum(1 for r in results if r.get("reason") == "frame_extract_failed")

    summary = {
        "n_input": len(kept),
        "n_visual_admissible": n_admissible,
        "n_dropped_static": n_static,
        "n_dropped_not_real_photo": n_not_real,
        "n_dropped_frame_extract_failed": n_extract_fail,
        "visual_retention_rate": n_admissible / max(1, len(kept)),
        "static_diff_threshold": STATIC_DIFF_THRESHOLD,
        "clip_real_prob_threshold": CLIP_REAL_PROB_THRESHOLD,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out.replace(".json", "_kept.json"), "w") as f:
        json.dump([r for r in results if r["visual_admissible"]], f, indent=2)
    with open(args.out.replace(".json", "_dropped.json"), "w") as f:
        json.dump([r for r in results if not r["visual_admissible"]], f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[visual-gate] wrote {args.out} + _kept.json + _dropped.json", flush=True)


if __name__ == "__main__":
    main()
