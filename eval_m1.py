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
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data.video_text_dataset import build_dataset, collate_fn
from models import SpineConfig, SpineM1
from utils import AttrDict, load_config


def load_spine(cfg: AttrDict, checkpoint: Optional[str], device: torch.device) -> SpineM1:
    """Build the spine and load weights from ``checkpoint`` if available."""
    spine_cfg = SpineConfig(**cfg["model"])
    state = None
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        ckpt_cfg = state.get("spine_config")
        if ckpt_cfg:
            # Rebuild with the exact architecture the checkpoint was trained with.
            spine_cfg = SpineConfig(**{**asdict(spine_cfg), **ckpt_cfg})

    spine = SpineM1(spine_cfg).to(device)
    if state is not None:
        missing, unexpected = spine.load_state_dict(state["model"], strict=False)
        print(
            f"[eval] loaded checkpoint {checkpoint!r} "
            f"(step={state.get('step')}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"[eval] WARNING: checkpoint {checkpoint!r} not found; "
            f"evaluating the (untrained) spine, which approximates the "
            f"SigLIP2 zero-shot baseline."
        )
    spine.eval()
    return spine


@torch.no_grad()
def embed_split(
    spine: SpineM1,
    cfg: AttrDict,
    device: torch.device,
    limit: Optional[int],
) -> Tuple[Tensor, Tensor]:
    """Embed the eval split, returning aligned ``(video_embeds, text_embeds)``.

    With ``eval_captions_per_video=1`` the i-th video and i-th caption form a
    1:1 positive pair, so the correct match lies on the diagonal of the
    similarity matrix.
    """
    dataset = build_dataset(cfg, "eval", limit=limit, decode_device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["eval"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        collate_fn=collate_fn,
        drop_last=False,
    )

    amp_enabled = device.type in ("cuda", "cpu")
    video_embeds: List[Tensor] = []
    text_embeds: List[Tensor] = []
    for clips, captions in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            zv = spine.embed_video(clips)
            zt = spine.embed_text(captions)
        video_embeds.append(zv.float().cpu())
        text_embeds.append(zt.float().cpu())

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


def report_m1_gate(r1_v2t: float, cfg: AttrDict) -> bool:
    """Print the M1 gate verdict; return True if PASS."""
    eval_cfg = cfg["eval"]
    baseline = float(eval_cfg["baseline_r1"])
    within = float(eval_cfg.get("within_margin", 5.0))
    abort = float(eval_cfg.get("abort_margin", 10.0))
    deficit = baseline - r1_v2t  # positive => worse than baseline

    print(
        f"[M1 GATE] video->text R@1={r1_v2t:.2f} vs SigLIP2 baseline={baseline:.2f} "
        f"(delta={r1_v2t - baseline:+.2f})"
    )
    if deficit <= within:
        print(
            f"[M1 GATE] PASS: within {within:.1f} pts of the SigLIP2 baseline."
        )
        return True
    if deficit > abort:
        print(
            f"[M1 GATE] ABORT: {deficit:.2f} pts worse than baseline "
            f"(> {abort:.1f}); switch the encoder."
        )
        return False
    print(
        f"[M1 GATE] FAIL: {deficit:.2f} pts worse than baseline "
        f"(> {within:.1f} but <= {abort:.1f}); keep iterating on the spine."
    )
    return False


def evaluate(cfg: AttrDict, checkpoint: Optional[str], limit: Optional[int]) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spine = load_spine(cfg, checkpoint, device)

    video_embeds, text_embeds = embed_split(spine, cfg, device, limit)
    sim = video_embeds @ text_embeds.t()  # [Nv, Nt]

    v2t = retrieval_metrics(sim)
    t2v = retrieval_metrics(sim.t())

    _print_metrics("video->text", v2t)
    _print_metrics("text->video", t2v)

    return report_m1_gate(v2t["R@1"], cfg)


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
    checkpoint = args.checkpoint or os.path.join(str(cfg["train"]["ckpt_dir"]), "best.pt")
    passed = evaluate(cfg, checkpoint, args.limit)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
