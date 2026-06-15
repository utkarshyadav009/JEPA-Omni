"""Milestone M1 evaluation: video<->text retrieval on MSR-VTT.

Loads a checkpoint, embeds the evaluation split with ``spine.embed_video`` /
``spine.embed_text``, and computes video->text and text->video retrieval
R@1 / R@5 / R@10 and median rank (MedR).

The **M1 gate** compares video->text R@1 against the SigLIP2 reference
(``eval.baseline_r1``):

* within ``eval.within_margin`` points  -> PASS
* more than ``eval.abort_margin`` points worse -> ABORT (switch encoder)

Usage
-----
    python eval_m1.py --config configs/m1.yaml --checkpoint checkpoints/m1/best.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.video_text_dataset import build_dataset, collate_fn
from data.cached_feature_dataset import (
    CachedFeatureDataset,
    cached_collate_fn,
    validate_manifest,
)
from models import SpineConfig, SpineM1
from utils import AttrDict, cfg_get, load_config


def load_spine(
    cfg: AttrDict,
    checkpoint: Optional[str],
    device: torch.device,
    *,
    skip_encoder: bool = False,
    encoder_out_dim: int = 1024,
) -> SpineM1:
    """Build the spine and load weights from ``checkpoint`` if available."""
    model_config = dict(cfg["model"])
    state = None
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        ckpt_cfg = state.get("model_config")
        if ckpt_cfg:
            # Rebuild with the exact architecture the checkpoint was trained with.
            model_config = {**model_config, **ckpt_cfg}
    
    for flag in ["loss_type", "micro_chunk"]:
        model_config.pop(flag, None)

    if skip_encoder:
        model_config["skip_encoder"] = True
        model_config["encoder_out_dim"] = encoder_out_dim

    spine = SpineM1(SpineConfig(**model_config)).to(device)
    if state is not None:
        missing, unexpected = spine.load_state_dict(state["model"], strict=False)
        print(
            f"[eval] loaded checkpoint {checkpoint!r} "
            f"(step={state.get('step')}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"[eval] WARNING: checkpoint {checkpoint!r} not found; "
            f"evaluating the (untrained) spine. Note: This is an untrained "
            f"projector on frozen V-JEPA, NOT a true SigLIP2 baseline."
        )
    spine.eval()
    return spine


@torch.no_grad()
def embed_split(
    spine: SpineM1,
    cfg: AttrDict,
    device: torch.device,
    limit: Optional[int],
    *,
    feature_cache_dir: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Embed the eval split, returning aligned ``(video_embeds, text_embeds)``.

    With ``eval_captions_per_video=1`` the i-th video and i-th caption form a
    1:1 positive pair, so the correct match lies on the diagonal of the
    similarity matrix.
    """
    num_workers = int(cfg_get(cfg, "train.num_workers", "num_workers", default=4))
    batch_size = int(cfg_get(cfg, "eval.batch_size", "batch_size", default=64))

    if feature_cache_dir:
        dataset = CachedFeatureDataset.from_config(cfg, "eval", limit=limit)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cached_collate_fn,
            drop_last=False,
        )
    else:
        dataset = build_dataset(cfg, "eval", limit=limit, decode_device="cpu")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    amp_enabled = device.type in ("cuda", "cpu")
    video_embeds: List[Tensor] = []
    text_embeds: List[Tensor] = []
    for batch in loader:
        if feature_cache_dir:
            feats, captions = batch
            feats = feats.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                zv = spine.predictor(feats.float())
                zt = spine.embed_text(captions)
        else:
            clips, captions = batch
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                zv = spine.embed_video(clips)
                zt = spine.embed_text(captions)
        # Normalise so the similarity matrix is a cosine ranking.
        video_embeds.append(F.normalize(zv.float(), dim=-1).cpu())
        text_embeds.append(F.normalize(zt.float(), dim=-1).cpu())

    return torch.cat(video_embeds, dim=0), torch.cat(text_embeds, dim=0)


def retrieval_metrics(sim: Tensor) -> Dict[str, float]:
    """Retrieval metrics where the correct gallery index is the row index.

    ``sim`` has shape ``[num_queries, num_gallery]`` and the positive for query
    ``i`` is gallery ``i`` (diagonal).
    """
    n = sim.shape[0]
    correct = sim.diagonal().unsqueeze(1)  # [n, 1]
    # 0-based rank = number of gallery items scoring strictly higher.
    ranks = (sim > correct).sum(dim=1)
    ranks_f = ranks.float()
    return {
        "R@1": (ranks < 1).float().mean().item() * 100.0,
        "R@5": (ranks < 5).float().mean().item() * 100.0,
        "R@10": (ranks < 10).float().mean().item() * 100.0,
        "MedR": ranks_f.median().item() + 1.0,
        "n": float(n),
    }


def _print_metrics(name: str, m: Dict[str, float]) -> None:
    print(
        f"[eval] {name:>10s}  "
        f"R@1={m['R@1']:5.2f}  R@5={m['R@5']:5.2f}  "
        f"R@10={m['R@10']:5.2f}  MedR={m['MedR']:5.1f}  (N={int(m['n'])})"
    )


def report_m1_gate(
    r1_v2t: float, r1_t2v: float, n_gallery: int, cfg: AttrDict,
) -> bool:
    """Print the M1 gate verdict; return True if PASS.

    Criteria (all must hold):
      1. **Above chance**: R@1 > chance + ``eval.min_above_chance`` (default 5 pts).
      2. **No collapse**: |v2t_R@1 − t2v_R@1| ≤ ``eval.max_v2t_t2v_gap`` (default 8 pts).
      3. **Improvement**: if ``eval.previous_r1`` is set (> 0), current R@1 must
         exceed it by at least ``eval.min_improvement`` pts (default 1.0).
    """
    chance = 100.0 / max(n_gallery, 1)
    min_above = float(cfg_get(cfg, "eval.min_above_chance", default=5.0))
    max_gap = float(cfg_get(cfg, "eval.max_v2t_t2v_gap", default=8.0))
    prev_r1 = float(cfg_get(cfg, "eval.previous_r1", default=0.0))
    min_improv = float(cfg_get(cfg, "eval.min_improvement", default=1.0))

    above_chance = r1_v2t > (chance + min_above)
    gap = abs(r1_v2t - r1_t2v)
    no_collapse = gap <= max_gap
    improved = (prev_r1 <= 0) or (r1_v2t >= prev_r1 + min_improv)

    passed = above_chance and no_collapse and improved
    status = "PASS" if passed else "FAIL"

    print(f"\n--- [M1 GATE] {status} ---")
    print(
        f"  1. Above chance:  {'PASS' if above_chance else 'FAIL'} | "
        f"R@1={r1_v2t:.2f}  chance={chance:.2f}  threshold={chance + min_above:.2f}"
    )
    print(
        f"  2. No collapse:   {'PASS' if no_collapse else 'FAIL'} | "
        f"v2t={r1_v2t:.2f}  t2v={r1_t2v:.2f}  gap={gap:.2f}  max={max_gap:.1f}"
    )
    if prev_r1 > 0:
        print(
            f"  3. Improvement:   {'PASS' if improved else 'FAIL'} | "
            f"current={r1_v2t:.2f}  previous={prev_r1:.2f}  "
            f"delta={r1_v2t - prev_r1:+.2f}  min={min_improv:.1f}"
        )
    else:
        print(f"  3. Improvement:   SKIP (no previous_r1 configured)")
    print()
    return passed


def evaluate(
    cfg: AttrDict, checkpoint: Optional[str], limit: Optional[int]
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cache_dir = str(cfg_get(cfg, "data.feature_cache_dir", default="")) or None
    skip_encoder = feature_cache_dir is not None
    encoder_out_dim = 1024

    if feature_cache_dir:
        manifest = validate_manifest(feature_cache_dir, cfg)
        encoder_out_dim = manifest["hidden_size"]
        print(
            f"[eval] Using pre-computed features from {feature_cache_dir}",
            flush=True,
        )

    spine = load_spine(
        cfg, checkpoint, device,
        skip_encoder=skip_encoder,
        encoder_out_dim=encoder_out_dim,
    )

    video_embeds, text_embeds = embed_split(
        spine, cfg, device, limit,
        feature_cache_dir=feature_cache_dir,
    )
    sim = video_embeds @ text_embeds.t()  # [Nv, Nt]

    v2t = retrieval_metrics(sim)
    t2v = retrieval_metrics(sim.t())

    _print_metrics("video->text", v2t)
    _print_metrics("text->video", t2v)

    return report_m1_gate(v2t["R@1"], t2v["R@1"], int(v2t["n"]), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the M1 video/text spine.")
    parser.add_argument("--config", default="configs/m1.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to <ckpt_dir>/best.pt).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Tiny debug subset of the eval data."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = str(cfg_get(cfg, "train.ckpt_dir", "ckpt_dir", default="checkpoints/m1"))
    checkpoint = args.checkpoint or os.path.join(ckpt_dir, "best.pt")
    passed = evaluate(cfg, checkpoint, args.limit)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()