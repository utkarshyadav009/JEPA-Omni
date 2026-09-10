"""scripts/imagebind_retrieval.py — ImageBind audio<->visual retrieval baseline.

Runs Meta's public ImageBind-Huge checkpoint on the fixed 1545-clip VGGSound
eval gallery (data/vggsound_eval_1545.txt) and computes:
    audio->visual  R@1, R@5, R@10
    visual->audio  R@1, R@5, R@10

Preprocessing is ImageBind's OWN default preprocessing, taken verbatim from
imagebind/data.py in the official repo (not reimplemented):

  Audio (load_and_transform_audio_data defaults):
    - torchaudio.load() the clip's audio track directly from the .mp4,
      resampled to 16 kHz if needed (torchaudio.functional.resample)
    - ConstantClipsPerVideoSampler(clip_duration=2s, clips_per_video=3)
      -> 3 non-overlapping-ish 2s sub-clips per video (uniformly spaced)
    - each 2s clip -> log-mel spectrogram via torchaudio.compliance.kaldi.fbank
      (num_mel_bins=128, frame_shift=10ms, htk_compat=True, hanning window),
      padded/cropped to target_length=204 frames
    - normalised with ImageBind's fixed constants mean=-4.268, std=9.138
    - the 3 clip embeddings are mean-pooled by the model's own forward()
      (ndim>=5 multi-clip reduction in ImageBindModel.forward)

  Video (load_and_transform_video_data defaults):
    - decord-backed decode (decode_audio=False)
    - ConstantClipsPerVideoSampler(clip_duration=2s, clips_per_video=5)
      -> 5 temporal sub-clips per video
    - UniformTemporalSubsample(num_samples=2) -> 2 frames per sub-clip
    - ShortSideScale(224) + NormalizeVideo(CLIP mean/std
      (0.48145466,0.4578275,0.40821073)/(0.26862954,0.26130258,0.27577711))
    - SpatialCrop(224, num_crops=3) -> 3 spatial crops per temporal clip
      => 5*3 = 15 (2-frame, 224x224) clips per video, mean-pooled by the
      model's own forward() exactly as for audio

No values in this file were adapted from our own CAV-MAE preprocessing;
every constant above is ImageBind's own default kwarg value, read directly
from the official repo (commit at clone time, see PROVENANCE in the output
JSON) and left untouched.

Usage:
    conda run -n jepa-omni python scripts/imagebind_retrieval.py
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
from torch import Tensor

REPO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/imagebind/repo"
sys.path.insert(0, REPO_DIR)

from imagebind.models import imagebind_model                      # noqa: E402
from imagebind.models.imagebind_model import ModalityType         # noqa: E402
from imagebind import data as ib_data                              # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recall_at_k(sim: Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    """sim: (N, N) cosine sim. Row i retrieved by col i (matched pairs)."""
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/imagebind/imagebind_huge.pth")
    parser.add_argument("--video-dir", default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    parser.add_argument("--eval-list", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "imagebind_retrieval_results.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}", flush=True)
    n_requested = len(clip_ids)

    present, missing = [], []
    for cid in clip_ids:
        p = os.path.join(args.video_dir, cid + ".mp4")
        (present if os.path.exists(p) else missing).append(cid)
    if missing:
        print(f"WARNING: {len(missing)}/{n_requested} clips have no video file on disk "
              f"(pre-existing gap in the raw-video extraction, not model-specific). "
              f"Evaluating on the intersection: {len(present)} clips.", flush=True)
        print("  missing (first 5):", missing[:5], flush=True)
    clip_ids = present
    N = len(clip_ids)
    print(f"Evaluating {N} clips on {args.device}", flush=True)

    model = imagebind_model.imagebind_huge(pretrained=False)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    result = model.load_state_dict(sd, strict=True)
    print(f"ImageBind: loaded state_dict strictly, missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}", flush=True)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(v.numel() for v in sd.values())
    print(f"ImageBind-Huge total params: {n_params} ({n_params/1e6:.1f}M)", flush=True)

    audio_embeds: List[Tensor] = []
    vision_embeds: List[Tensor] = []
    failed: List[str] = []

    t0 = time.time()
    B = args.batch_size
    for start in range(0, N, B):
        batch_ids = clip_ids[start:start + B]
        batch_paths = [os.path.join(args.video_dir, cid + ".mp4") for cid in batch_ids]

        ok_paths, ok_ids = [], []
        for p, cid in zip(batch_paths, batch_ids):
            try:
                ok_paths.append(p)
                ok_ids.append(cid)
            except Exception as e:
                failed.append(cid)
                print(f"  FAILED (pre-check) {cid}: {e}", flush=True)

        try:
            with torch.no_grad():
                inputs = {
                    ModalityType.VISION: ib_data.load_and_transform_video_data(ok_paths, args.device),
                    ModalityType.AUDIO: ib_data.load_and_transform_audio_data(ok_paths, args.device),
                }
                out = model(inputs)
            audio_embeds.append(out[ModalityType.AUDIO].float().cpu())
            vision_embeds.append(out[ModalityType.VISION].float().cpu())
        except Exception as e:
            # fall back to per-clip so one bad file doesn't drop the whole batch
            print(f"  batch failed ({e}); retrying per-clip", flush=True)
            for p, cid in zip(ok_paths, ok_ids):
                try:
                    with torch.no_grad():
                        inputs = {
                            ModalityType.VISION: ib_data.load_and_transform_video_data([p], args.device),
                            ModalityType.AUDIO: ib_data.load_and_transform_audio_data([p], args.device),
                        }
                        out = model(inputs)
                    audio_embeds.append(out[ModalityType.AUDIO].float().cpu())
                    vision_embeds.append(out[ModalityType.VISION].float().cpu())
                except Exception as e2:
                    failed.append(cid)
                    print(f"  FAILED {cid}: {e2}", flush=True)

        if (start // B) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{start+len(ok_ids)}/{N}] elapsed={elapsed:.1f}s", flush=True)

    if failed:
        print(f"WARNING: {len(failed)} clips failed during embedding extraction (decode/other errors).", flush=True)
        print("  failed (first 10):", failed[:10], flush=True)

    A = torch.cat(audio_embeds, dim=0)   # (N_eval, D), already L2-normalised by ImageBind's postprocessor
    V = torch.cat(vision_embeds, dim=0)  # (N_eval, D)
    N_eval = A.shape[0]
    print(f"clips_seen={N_eval} (requested {n_requested}, video-present {N}, "
          f"failed-during-extraction {len(failed)})", flush=True)
    assert N_eval == N - len(failed), "embedding count mismatch"
    assert V.shape[0] == N_eval, "audio/visual count mismatch"

    A = torch.nn.functional.normalize(A, dim=-1)
    V = torch.nn.functional.normalize(V, dim=-1)

    sim_a2v = A @ V.T   # audio->visual: row=audio query, col=visual gallery
    sim_v2a = V @ A.T   # visual->audio

    a2v = recall_at_k(sim_a2v)
    v2a = recall_at_k(sim_v2a)
    print(f"audio->visual: R@1={a2v['R@1']}% R@5={a2v['R@5']}% R@10={a2v['R@10']}%", flush=True)
    print(f"visual->audio: R@1={v2a['R@1']}% R@5={v2a['R@5']}% R@10={v2a['R@10']}%", flush=True)

    ckpt_sha256 = hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest()

    out = {
        "n_clips": N_eval,
        "clips_requested": n_requested,
        "clips_missing_video": len(missing),
        "clips_failed_extraction": len(failed),
        "eval_list": os.path.relpath(args.eval_list, PROJECT_ROOT),
        "audio_to_visual": a2v,
        "visual_to_audio": v2a,
        "ckpt": args.ckpt,
        "ckpt_sha256": ckpt_sha256,
        "repo": "https://github.com/facebookresearch/ImageBind",
        "params_millions": round(n_params / 1e6, 1),
        "pretrain_corpus": "Audio tower: AudioSet. Image/vision tower: web-scale image-text pairs "
                            "(CLIP init) + naturally-paired video-audio web data. NOT trained on VGGSound "
                            "per the ImageBind paper/authors.",
        "contamination_flag": "HELD-OUT",
        "preprocessing": (
            "ImageBind's own defaults from imagebind/data.py, unmodified. Audio: torchaudio.load direct "
            "from mp4 -> resample 16kHz -> ConstantClipsPerVideoSampler(2s x 3 clips) -> kaldi fbank "
            "(128 mel bins, 10ms frame shift, target_length=204) -> normalise(mean=-4.268,std=9.138) -> "
            "mean-pooled over 3 clips by the model. Video: decord decode -> ConstantClipsPerVideoSampler "
            "(2s x 5 clips) -> UniformTemporalSubsample(2 frames/clip) -> ShortSideScale(224) -> "
            "NormalizeVideo(CLIP mean/std) -> SpatialCrop(224, 3 crops) -> 15 (2-frame,224x224) views, "
            "mean-pooled over the 15 views by the model's forward()."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda run -n jepa-omni python scripts/imagebind_retrieval.py",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
