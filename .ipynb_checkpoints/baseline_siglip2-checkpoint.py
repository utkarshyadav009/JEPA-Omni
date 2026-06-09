"""
baseline_siglip2.py — zero-shot SigLIP2 retrieval baseline for the M1 gate.

WHY THIS EXISTS
---------------
The M1 gate is "spine R@1 within `within_margin` of a SigLIP2 baseline". For that
comparison to mean anything, SigLIP2 and the spine must be scored on the SAME
test population with the SAME metric, while each encoder uses its OWN native input
preprocessing (forcing SigLIP2 through V-JEPA's 256px/64-frame pipeline would run it
off-distribution and produce a meaningless, deflated baseline).

So this script:
  * builds the EXACT same eval set as eval_m1.py (`build_dataset(cfg, "eval", ...)`,
    `eval_captions_per_video: 1` -> one caption per video, diagonal = positive), and
  * imports `retrieval_metrics` from eval_m1 VERBATIM (no reimplementation), but
  * feeds SigLIP2 its native input: K uniformly-sampled frames per clip, the model's
    OWN AutoProcessor at its native resolution, mean-pooled into one video embedding
    (the standard "meanP" CLIP-style zero-shot video-retrieval recipe).

This is zero-shot: SigLIP2 is frozen and never sees MSR-VTT in training. Note the
spine, by contrast, IS trained on MSR-VTT — so matching/slightly beating this number
is the bar, and LOSING to a zero-shot image model is a clear "switch encoder" signal.

USAGE
-----
    CUDA_VISIBLE_DEVICES=1 python baseline_siglip2.py --config configs/m1.yaml
    # then paste the printed R@1 into configs/m1.yaml -> eval.baseline_r1
"""

from __future__ import annotations

import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoModel, AutoProcessor

from utils import load_config, cfg_get
from data.video_text_dataset import build_dataset, _uniform_frame_indices
from eval_m1 import retrieval_metrics, _print_metrics


def _decode_frames(path: str, num_frames: int) -> Tensor:
    """Decode `num_frames` uniformly-sampled frames -> uint8 [K, C, H, W] (CPU).

    Uses the same uniform-index helper as the training dataset, so the temporal
    sampling philosophy matches; only the count (K, small) differs.
    """
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(path, device="cpu")
    num_total = getattr(dec.metadata, "num_frames", None) or len(dec)
    idx = _uniform_frame_indices(int(num_total), num_frames)
    # Per-frame indexing is the version-stable torchcodec API; dec[i] -> uint8 [C,H,W].
    return torch.stack([dec[i] for i in idx], dim=0)


@torch.no_grad()
def encode_videos(
    model, processor, samples, num_frames: int, device, dtype
) -> tuple[Tensor, List[int]]:
    """Mean-pooled SigLIP2 image embedding per video. Returns (embeds, kept_indices)."""
    embeds: List[Tensor] = []
    kept: List[int] = []
    for i, s in enumerate(samples):
        try:
            frames = _decode_frames(s.video_path, num_frames)         # [K,C,H,W] uint8
        except Exception as e:  # skip unreadable clips, keep alignment via `kept`
            print(f"[baseline] skip {s.video_id}: {e}")
            continue
        # HF image processors accept a list of HWC uint8 arrays.
        imgs = [f.permute(1, 2, 0).cpu().numpy() for f in frames]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(dtype)
        feats = model.get_image_features(**inputs)                    # [K, D]
        vid = F.normalize(feats.pooler_output.float(), dim=-1).mean(dim=0)         # mean-pool frames
        embeds.append(F.normalize(vid, dim=-1).cpu())                 # re-norm -> [D]
        kept.append(i)
    return torch.stack(embeds, dim=0), kept


@torch.no_grad()
def encode_texts(model, processor, captions: List[str], device, dtype, batch: int = 256) -> Tensor:
    out: List[Tensor] = []
    for start in range(0, len(captions), batch):
        chunk = captions[start : start + batch]
        # SigLIP text tower expects max_length padding (its training convention).
        inputs = processor(
            text=chunk, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        ).to(device)
        feats = model.get_text_features(**inputs)
        out.append(F.normalize(feats.pooler_output.float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="google/siglip2-so400m-patch14-384",
                    help="SigLIP2 checkpoint (~400M, scale-matched to V-JEPA ViT-L).")
    ap.add_argument("--frames", type=int, default=8, help="Frames mean-pooled per clip.")
    ap.add_argument("--limit", type=int, default=None, help="Cap eval size (debug).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = load_config(args.config)
    # Same eval population as eval_m1.py: same split, same 1:1 caption collapse, same order.
    dataset = build_dataset(cfg, "eval", limit=args.limit, decode_device="cpu")
    samples = dataset.samples
    print(f"[baseline] eval samples: {len(samples)}  | ckpt={args.ckpt}  frames={args.frames}")

    model = AutoModel.from_pretrained(args.ckpt, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)

    video_embeds, kept = encode_videos(model, processor, samples, args.frames, device, dtype)
    captions = [samples[i].caption for i in kept]   # align captions to successfully-decoded videos
    text_embeds = encode_texts(model, processor, captions, device, dtype)

    if len(kept) < len(samples):
        print(f"[baseline] WARNING: {len(samples) - len(kept)} clips skipped; "
              f"scoring on {len(kept)} aligned pairs.")

    sim = video_embeds @ text_embeds.t()            # [N, N]; diagonal = positive
    v2t = retrieval_metrics(sim)
    t2v = retrieval_metrics(sim.t())
    _print_metrics("SigLIP2 v2t", v2t)
    _print_metrics("SigLIP2 t2v", t2v)

    print("\n" + "=" * 60)
    print(f" Paste into configs/m1.yaml ->  eval.baseline_r1: {v2t['R@1']:.2f}")
    print(" (video->text R@1; the M1 gate compares the spine against this.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
