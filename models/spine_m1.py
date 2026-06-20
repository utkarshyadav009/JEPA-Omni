"""
models/spine_m1.py

Assembles the M1 perception spine:

    frozen V-JEPA 2 ViT-L  --(B,N,1024)-->  Predictor (trainable)  --(B,1536)-->|
                                                                                 |--InfoNCE
    frozen TextTarget (EmbeddingGemma/MiniLM) --------------------- (B,1536) ---->|

A MoCo-style negative QUEUE (cfg.queue_size > 0) lets each step contrast against
B-1 in-batch + K queued negatives, so the contrastive difficulty no longer collapses
to the GPU batch size (~8 on one H100). Queue entries are detached and stored in
NON-persistent buffers, so checkpoints and eval are unaffected.

forward() returns (loss, metrics). embed_video()/embed_text() are used by eval_m1.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .vision_encoder import VisionEncoder
from .text_target import TextTarget
from .predictor import Predictor
from .losses import info_nce, info_nce_with_queue, sigreg_jepa_loss


@dataclass
class SpineConfig:
    vision_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    text_backbone: str = "minilm"          # "embeddinggemma" for the faithful (gated) run
    predictor_mode: str = "mlp"            # "transformer" | "llama_last8"
    shared_dim: int = 1536
    temperature: float = 0.07
    stack_factor: int = 1
    predictor_layers: int = 4
    text_max_length: int = 512
    unfreeze_text: bool = False
    queue_size: int = 0                    # 0 = disabled; e.g. 2048 to flex the negatives
    loss_type: str = "info_nce"            # "info_nce" | "sigreg"
    sigreg_lambda: float = 10.0            # weight for SigReg term
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    skip_encoder: bool = False             # True when using pre-computed feature cache
    encoder_out_dim: int = 1024            # only used when skip_encoder=True


class SpineM1(nn.Module):
    def __init__(self, cfg: SpineConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # When using pre-computed feature cache, skip the ~26GB VisionEncoder
        # entirely so workers don't inherit its memory footprint.
        if cfg.skip_encoder:
            self.encoder = None
            enc_out_dim = cfg.encoder_out_dim
        else:
            self.encoder = VisionEncoder(cfg.vision_repo, dtype=cfg.dtype, device=cfg.device)
            enc_out_dim = self.encoder.out_dim

        self.text = TextTarget(
            backbone=cfg.text_backbone, shared_dim=cfg.shared_dim,
            max_length=cfg.text_max_length, unfreeze_base=cfg.unfreeze_text,
            device=cfg.device, dtype=cfg.dtype,
        )
        self.predictor = Predictor(
            in_dim=enc_out_dim, shared_dim=cfg.shared_dim,
            mode=cfg.predictor_mode, n_layers=cfg.predictor_layers,
            stack_factor=cfg.stack_factor,
        ).to(cfg.device)

        # ---- MoCo-style negative queue (non-persistent: not saved in checkpoints) ----
        self.queue_size = int(cfg.queue_size)
        if self.queue_size > 0:
            D = cfg.shared_dim
            dev = cfg.device
            self.register_buffer("queue_v", torch.zeros(self.queue_size, D, device=dev), persistent=False)
            self.register_buffer("queue_t", torch.zeros(self.queue_size, D, device=dev), persistent=False)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long, device=dev), persistent=False)
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long, device=dev), persistent=False)

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        params = list(self.predictor.parameters()) + list(self.text.proj.parameters())
        if self.cfg.unfreeze_text:
            params += [p for p in self.text.base.parameters() if p.requires_grad]
        return params

    def text_base_parameters(self):
        """Only the text BACKBONE params (for the 0.05 LR group); excludes the projection."""
        return [p for p in self.text.base.parameters() if p.requires_grad]

    def embed_video(self, videos) -> Tensor:
        # If videos is already a feature tensor [B, N, D], just pass it to predictor
        if isinstance(videos, torch.Tensor) and videos.dim() == 3:
            feats = videos
        elif self.encoder is not None:
            feats = self.encoder.encode(videos)      # frozen, no grad
        else:
            feats = videos                           # already features (cached path)
        return self.predictor(feats)                 # (B, shared_dim), normalized

    def embed_text(self, texts: List[str]) -> Tensor:
        return self.text.encode_text(texts)          # (B, shared_dim), normalized

    # ------------------------------------------------------------------ #
    def _valid_queue(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        f = int(self.queue_filled.item())
        if f == 0:
            return None, None
        return self.queue_v[:f], self.queue_t[:f]

    @torch.no_grad()
    def _enqueue(self, z_v: Tensor, z_t: Tensor) -> None:
        z_v = z_v.detach().to(self.queue_v.dtype)
        z_t = z_t.detach().to(self.queue_t.dtype)
        B, K = z_v.shape[0], self.queue_size
        if B >= K:                                   # batch bigger than queue: keep last K
            self.queue_v.copy_(z_v[-K:]); self.queue_t.copy_(z_t[-K:])
            self.queue_ptr[0] = 0; self.queue_filled[0] = K
            return
        ptr = int(self.queue_ptr.item())
        end = ptr + B
        if end <= K:
            self.queue_v[ptr:end] = z_v
            self.queue_t[ptr:end] = z_t
        else:                                        # wrap-around
            first = K - ptr
            self.queue_v[ptr:] = z_v[:first]; self.queue_t[ptr:] = z_t[:first]
            self.queue_v[: end - K] = z_v[first:]; self.queue_t[: end - K] = z_t[first:]
        self.queue_ptr[0] = end % K
        self.queue_filled[0] = min(int(self.queue_filled.item()) + B, K)

    # ------------------------------------------------------------------ #
    def forward(self, videos, texts: List[str], global_step: int = 0) -> Tuple[Tensor, Dict[str, float]]:
        z_v = self.embed_video(videos)
        z_t = self.embed_text(texts)

        if self.cfg.loss_type == "sigreg":
            return sigreg_jepa_loss(z_v, z_t, global_step=global_step, lamb=self.cfg.sigreg_lambda)

        # Use the queue only on training steps (grad on). Never enqueue under no_grad
        # (e.g. diagnostic embeds), and never let a queued/stale vector be the positive.
        if self.queue_size > 0 and z_v.requires_grad:
            neg_v, neg_t = self._valid_queue()
            loss, metrics = info_nce_with_queue(
                z_v, z_t, neg_t=neg_t, neg_v=neg_v, temperature=self.cfg.temperature)
            self._enqueue(z_v, z_t)                  # AFTER the loss: positives never queued
            return loss, metrics

        return info_nce(z_v, z_t, self.cfg.temperature)


if __name__ == "__main__":
    cfg = SpineConfig(queue_size=2048)
    spine = SpineM1(cfg)
    n = sum(p.numel() for p in spine.trainable_parameters())
    print(f"[spine_m1] trainable params = {n/1e6:.1f}M  (encoder + text base frozen)")
    B, T, C, H, W = 2, 64, 3, 256, 256
    dummy = (torch.rand(B, T, C, H, W) * 255).to(torch.uint8)
    loss, m = spine(dummy, ["a dog catching a frisbee", "glass shattering"])
    print(f"[spine_m1] loss={m['loss']:.3f} acc_v2t={m['acc_v2t']:.2f} queue_negs={m['queue_negatives']}")
