"""scripts/extract_siglip2_scene.py — extract the SCENE stream (SigLIP2) for clips that
already have V-JEPA2/WavJEPA/M2 features cached.

WHAT THIS ADDS AND WHAT IT DOES NOT REPLACE
The JEPA trunk stays exactly as it is. This adds a FOURTH stream next to it:

    vision   V-JEPA2 ViT-L    motion / actions over a 10 s window   (unchanged)
    ambient  WavJEPA base+nat  sound events                          (unchanged)
    m2       AVJepaPredictor   audio-visual congruence, the fused view (unchanged)
    scene    SigLIP2-base      WHAT THINGS ARE: rooms, objects, people  <-- NEW

Measured justification for adding rather than swapping: the query-predictor ablation showed
combining streams beats any single one (`m2+vision` 0.478 > `vision` 0.447 > `m2` 0.385), and
SigLIP2 covers precisely the blind spot — on a real room frame it ranked room sentences
1-2-3-4 while V-JEPA2/M2 retrieved "opening a microwave oven".

FORMAT: K frames sampled across the segment, each encoded independently and pooled by
SigLIP2's own image tower -> (K, D) per clip. This is the "meanP" frame-wise recipe that
`baseline_siglip2.py` already validated in this repo (zero-shot R@1 32.5 on MSR-VTT video
retrieval, vs the trained V-JEPA2 spine's 22.5). Keeping the K frames SEPARATE rather than
mean-pooling them preserves coarse temporal structure for the query predictor to attend over.

CORPUS: **Action100M only.** The raw VGGSound videos no longer exist on this machine (only
`feature_cache_vgg51k` survives), so SigLIP2 features cannot be extracted for them. Action100M
raw video IS present (134,989 files) and its captions carry `video_uid`/`start_s`/`end_s`,
so segments are directly recoverable. This is a real constraint on the retrain, not a choice.

Usage (4-way shard, one per GPU):
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/extract_siglip2_scene.py \
        --shard-idx $i --num-shards 4 --limit 60000 &
    done; wait
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

VIDEO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/action100m_videos"
CAPTIONS = os.path.join(PROJECT_ROOT, "scripts", "action100m_captions.jsonl")
CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_action100m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out-dir", default="/dev/shm/siglip2_scene")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=60000,
                    help="cap on segments considered; 0 = no cap (the whole corpus)")
    ap.add_argument("--cpu-threads", type=int, default=16)
    ap.add_argument("--skip-existing", default="",
                    help="dir of already-extracted scene shards. Clips found there are "
                         "skipped, so this script is RESUMABLE and can top up a partial "
                         "corpus without re-decoding what is already done.")
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    # clips that ALREADY have cached AV features -- the scene stream must align with them
    have = set()
    for root, _, files in os.walk(CACHE_DIR):
        for f in files:
            if f.endswith(".pt"):
                have.add(f[:-3])
    print(f"[sig] {len(have)} clips have cached AV features", flush=True)

    done = set()
    if args.skip_existing:
        import glob as _g
        for sp in sorted(_g.glob(os.path.join(args.skip_existing, "*.pt"))):
            done |= set(torch.load(sp, map_location="cpu", weights_only=False).keys())
        print(f"[sig] {len(done)} clips already have scene features -- skipping", flush=True)

    rows = []
    with open(CAPTIONS) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["clip_id"]
            if cid not in have or cid in done:
                continue
            rows.append((cid, r["video_uid"], float(r["start_s"]), float(r["end_s"])))
    rows.sort()
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard_idx::args.num_shards]
    print(f"[sig:{args.shard_idx}] {len(rows)} segments to extract", flush=True)

    from transformers import AutoModel, AutoProcessor
    from torchcodec.decoders import VideoDecoder
    model = AutoModel.from_pretrained(args.siglip, dtype=torch.bfloat16).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.siglip)
    print(f"[sig:{args.shard_idx}] {args.siglip} loaded", flush=True)

    # group by video so each file is opened once (decode dominates cost)
    by_vid: Dict[str, List] = {}
    for cid, uid, s, e in rows:
        by_vid.setdefault(uid, []).append((cid, s, e))

    out_feats: Dict[str, torch.Tensor] = {}
    n_ok = n_bad = 0
    t0 = time.time()
    for vi, (uid, segs) in enumerate(by_vid.items()):
        path = os.path.join(VIDEO_DIR, f"{uid}.mp4")
        if not os.path.exists(path):
            n_bad += len(segs); continue
        try:
            dec = VideoDecoder(path, device="cpu", num_ffmpeg_threads=args.cpu_threads)
            fps = float(dec.metadata.average_fps or 25.0)
            nf = dec.metadata.num_frames
        except Exception:
            n_bad += len(segs); continue
        for cid, s, e in segs:
            try:
                i0, i1 = int(s * fps), min(nf - 1, int(e * fps))
                if i1 <= i0:
                    n_bad += 1; continue
                idx = torch.linspace(i0, i1, args.frames).long().clamp(0, nf - 1).tolist()
                frames = dec.get_frames_at(indices=idx).data          # (K,3,H,W) uint8
                imgs = [f.permute(1, 2, 0).numpy() for f in frames]
                with torch.no_grad():
                    px = proc(images=imgs, return_tensors="pt").to(device)
                    px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                          for k, v in px.items()}
                    o = model.get_image_features(**px)
                    o = o.pooler_output if hasattr(o, "pooler_output") else o
                    z = F.normalize(o.float(), dim=-1)               # (K, D)
                out_feats[cid] = z.cpu().to(torch.float16)
                n_ok += 1
            except Exception:
                n_bad += 1
        if (vi + 1) % 200 == 0:
            print(f"[sig:{args.shard_idx}] video {vi+1}/{len(by_vid)} ok={n_ok} bad={n_bad} "
                  f"{time.time()-t0:.0f}s", flush=True)

    path = os.path.join(args.out_dir, f"shard{args.shard_idx}.pt")
    torch.save(out_feats, path)
    print(f"[sig:{args.shard_idx}] DONE ok={n_ok} bad={n_bad} -> {path} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
