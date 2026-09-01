"""verify_scene.py -- Fix 1c gate.

Rebuilds the SigLIP2 scene stream with the recipe read out of
scripts/extract_siglip2_scene_vgg.py and compares against the CACHED
vgg_shard*.pt features for the same clips. Below ~0.99 means my live
construction does not match training and must not ship.
"""
import glob, os, sys
import torch, torch.nn.functional as F
import numpy as np

PROJECT_ROOT = "/home/utkarsh/JEPA-Omni"
sys.path.insert(0, PROJECT_ROOT)
VIDEO_DIR = "/home/utkarsh/raid2-data/vggsound_raw/extracted/video"
SIGLIP = "google/siglip2-base-patch16-224"
K = 8

# ---- pull cached scene features for a few clips that also have raw video ----
want, cached = [], {}
for sp in sorted(glob.glob("/dev/shm/scene_all/vgg_shard*.pt")):
    d = torch.load(sp, map_location="cpu", weights_only=False)
    for cid, v in d.items():
        if os.path.exists(os.path.join(VIDEO_DIR, cid + ".mp4")):
            cached[cid] = v
            want.append(cid)
            if len(want) >= 3:
                break
    del d
    if len(want) >= 3:
        break
print("[gate] clips:", want, flush=True)
for c in want:
    print("   cached %s shape=%s dtype=%s" % (c, tuple(cached[c].shape), cached[c].dtype), flush=True)

# ---- rebuild live, replicating extract_siglip2_scene_vgg.py exactly ----
from transformers import AutoModel, AutoProcessor
from torchcodec.decoders import VideoDecoder
device = torch.device("cuda")
model = AutoModel.from_pretrained(SIGLIP, dtype=torch.bfloat16).to(device).eval()
proc = AutoProcessor.from_pretrained(SIGLIP)
print("[gate] %s loaded" % SIGLIP, flush=True)

def build_scene(path, n_frames=K):
    dec = VideoDecoder(path, device="cpu", num_ffmpeg_threads=16)
    nf = dec.metadata.num_frames
    i0, i1 = 0, nf - 1
    idx = torch.linspace(i0, i1, n_frames).long().clamp(0, nf - 1).tolist()
    frames = dec.get_frames_at(indices=idx).data            # (K,3,H,W) uint8
    imgs = [f.permute(1, 2, 0).numpy() for f in frames]
    with torch.no_grad():
        px = proc(images=imgs, return_tensors="pt").to(device)
        px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in px.items()}
        o = model.get_image_features(**px)
        o = o.pooler_output if hasattr(o, "pooler_output") else o
        z = F.normalize(o.float(), dim=-1)
    return z.cpu().to(torch.float16)

print("\n=== 1c GATE: live vs cached ===", flush=True)
worst = 1.0
for cid in want:
    live = build_scene(os.path.join(VIDEO_DIR, cid + ".mp4"))
    ref = cached[cid]
    if live.shape != ref.shape:
        print("  %s SHAPE MISMATCH live=%s cached=%s" % (cid, tuple(live.shape), tuple(ref.shape)), flush=True)
        worst = 0.0
        continue
    per = F.cosine_similarity(live.float(), ref.float(), dim=-1)   # per frame
    flat = F.cosine_similarity(live.float().flatten().unsqueeze(0),
                               ref.float().flatten().unsqueeze(0)).item()
    worst = min(worst, float(per.min()), flat)
    print("  %-28s per-frame min=%.6f mean=%.6f   flattened=%.6f"
          % (cid, float(per.min()), float(per.mean()), flat), flush=True)

print("\n[GATE] worst cosine = %.6f -> %s" % (worst, "PASS" if worst >= 0.99 else "FAIL"), flush=True)
sys.exit(0 if worst >= 0.99 else 1)
