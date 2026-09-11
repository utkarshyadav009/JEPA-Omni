"""scripts/temporal_probe/extract_scene_subset.py — P0.4 enabler.

sig_runB/C/D use a 4th 'scene' stream whose SigLIP2 features lived in
/dev/shm/scene_all. /dev/shm is empty (the box rebooted), so those arms cannot be
scored. VGGSound raw video IS still on RAID (197,970 files), so the features are
re-extractable.

This re-extracts them for an EXPLICIT clip-id list only (the query-predictor eval
split), using the SAME recipe as scripts/extract_siglip2_scene_vgg.py, verified
line by line against it: K=8 frames via torch.linspace over the whole file,
torchcodec get_frames_at, HWC numpy -> AutoProcessor(images=...), get_image_features
-> pooler_output, F.normalize(dim=-1), stored (8,768) float16, frames kept SEPARATE.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER THIS ENABLES: the original run selected its
eval clips with `restrict_to_scene=True` against the /dev/shm set, which covered
13,579 of 13,679 eval clips (99.3%). That exact membership is unrecoverable, so a
re-extracted subset will not reproduce the original 624-clip gallery exactly. The
paired batch-48-vs-batch-1 DELTA is unaffected (both sides use the identical subset);
the ABSOLUTE value is a near-match, not a reproduction.

Usage:
    python scripts/temporal_probe/extract_scene_subset.py \
        --n-clips 900 --out /mnt/Raid-Storage-2/utkarsh-data/scene_qp_eval_subset.pt
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Dict

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

VIDEO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video"
CAPTIONS = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--n-clips", type=int, default=900)
    ap.add_argument("--cpu-threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda")

    import train_m3
    train_m3.CAPTIONS_PATH = CAPTIONS
    from train_query_predictor import build_splits, group_by_clip
    FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief",
              "gpt_summary_detailed", "gpt_sound_acoustic",
              "gpt_sound_acoustic_v1_original"]
    _, te = build_splits(FIELDS)
    vte = group_by_clip(te, FIELDS)
    # QueryClipDataset iterates sorted(clips) order; take the head of that order so the
    # subset is the one the eval loader would reach first with shuffle=False.
    ids = sorted(vte.keys())[: args.n_clips]
    print(f"[scene] {len(ids)} target clips", flush=True)

    from transformers import AutoModel, AutoProcessor
    from torchcodec.decoders import VideoDecoder
    model = AutoModel.from_pretrained(args.siglip, dtype=torch.bfloat16).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.siglip)

    out: Dict[str, torch.Tensor] = {}
    n_ok = n_bad = 0
    t0 = time.time()
    for i, cid in enumerate(ids):
        path = os.path.join(VIDEO_DIR, f"{cid}.mp4")
        if not os.path.exists(path):
            n_bad += 1; continue
        try:
            dec = VideoDecoder(path, device="cpu", num_ffmpeg_threads=args.cpu_threads)
            nf = dec.metadata.num_frames
            idx = torch.linspace(0, nf - 1, args.frames).long().clamp(0, nf - 1).tolist()
            frames = dec.get_frames_at(indices=idx).data
            imgs = [f.permute(1, 2, 0).numpy() for f in frames]
            with torch.no_grad():
                px = proc(images=imgs, return_tensors="pt").to(device)
                px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                      for k, v in px.items()}
                o = model.get_image_features(**px)
                o = o.pooler_output if hasattr(o, "pooler_output") else o
                z = F.normalize(o.float(), dim=-1)
            out[cid] = z.cpu().to(torch.float16)
            n_ok += 1
        except Exception as e:
            n_bad += 1
        if (i + 1) % 200 == 0:
            print(f"[scene] {i+1}/{len(ids)} ok={n_ok} bad={n_bad} {time.time()-t0:.0f}s", flush=True)

    torch.save(out, args.out)
    print(f"[scene] DONE ok={n_ok} bad={n_bad} -> {args.out} ({time.time()-t0:.0f}s)", flush=True)
    shapes = {tuple(v.shape) for v in out.values()}
    print(f"[scene] distinct shapes: {shapes}  (a single (8,768) means NO scene padding "
          f"can occur in collate, so the unmasked scene stream is leak-free in practice)",
          flush=True)


if __name__ == "__main__":
    main()
