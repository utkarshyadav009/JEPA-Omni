"""scripts/step1_diagnostic_rank.py — pre-fix diagnostic (STEP 1 re-gate).

Cheap check BEFORE writing any normalisation code: is the RAW WavJEPA ambient
target even non-trivially high-rank?  If the target's effective rank is very
low (e.g. <30 of 768), per-token normalisation cannot manufacture signal that
isn't there and the task itself may need to change.

Computes, over a large pool of cached ambient tokens:
  - effective_rank(tokens)  — participation-ratio rank of the (N, 768)
    token-covariance, using the SAME effective_rank() used for World-State
    (imported from models.av_jepa_predictor, not reimplemented).
  - per-token std: std within each token vector (over the 768 feature dim),
    reported as mean/min/max across tokens — this is the quantity that
    per-token layer-norm would act on.
  - per-channel std (for reference / comparison to the earlier std=0.1529
    figure, which was per-channel-across-corpus, not per-token).

Usage:
    conda run -n jepa-omni python scripts/step1_diagnostic_rank.py \
        --config configs/m2.yaml --n-clips 200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import effective_rank
from train_m2 import build_dataloader, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--n-clips", type=int, default=200,
                        help="Number of clips to pool tokens from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader, _ = build_dataloader(cfg, limit=args.n_clips, batch_size_override=32)

    vision_tokens = []
    ambient_tokens = []
    n_clips = 0
    for batch in loader:
        feats = batch["feats"]
        pad = batch["padding_mask"]
        # Only keep real (non-padding) tokens
        v = feats["vision"][~pad["vision"]].float()     # (n_v, 1024)
        a = feats["ambient"][~pad["ambient"]].float()    # (n_a, 768)
        vision_tokens.append(v)
        ambient_tokens.append(a)
        n_clips += feats["vision"].shape[0]
        if n_clips >= args.n_clips:
            break

    vision_tokens = torch.cat(vision_tokens, 0)
    ambient_tokens = torch.cat(ambient_tokens, 0)

    print(f"[diag] pooled from {n_clips} clips", flush=True)
    print(f"[diag] vision tokens:  {tuple(vision_tokens.shape)}", flush=True)
    print(f"[diag] ambient tokens: {tuple(ambient_tokens.shape)}", flush=True)

    for name, tokens in [("vision", vision_tokens), ("ambient", ambient_tokens)]:
        D = tokens.shape[1]
        rank = effective_rank(tokens)

        per_token_std = tokens.std(dim=1)          # (N,) — std within each token, over feat dim
        per_channel_std = tokens.std(dim=0)        # (D,) — std within each channel, over tokens

        print()
        print("=" * 60)
        print(f"RAW TARGET DIAGNOSTIC — {name}  (dim={D})")
        print("-" * 60)
        print(f"  effective_rank            = {rank:.2f}  (of {D})")
        print(f"  per-token std   mean/min/max = "
              f"{per_token_std.mean():.4f} / {per_token_std.min():.4f} / {per_token_std.max():.4f}")
        print(f"  per-channel std mean/min/max = "
              f"{per_channel_std.mean():.4f} / {per_channel_std.min():.4f} / {per_channel_std.max():.4f}")
        print(f"  corpus mean magnitude       = {tokens.mean():.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
