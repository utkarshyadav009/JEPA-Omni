"""
models/spine_m1.py

Assembles the M1 perception spine:

    frozen V-JEPA 2 ViT-L  --(B,N,1024)-->  Predictor (trainable)  --(B,1536)-->|
                                                                                 |--InfoNCE
    frozen TextTarget (EmbeddingGemma/MiniLM) --------------------- (B,1536) ---->|

No audio, no streaming, no LLM generation. forward() returns (loss, metrics).
embed_video()/embed_text() are used by eval_m1.py for retrieval/classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .text_target import TextTarget
from .predictor import Predictor
from .losses import info_nce


@dataclass
class SpineConfig:
    vision_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    text_backbone: str = "minilm"          # "embeddinggemma" for the faithful (gated) run
    predictor_mode: str = "mlp"            # or "transformer"
    shared_dim: int = 1536
    temperature: float = 0.07
    stack_factor: int = 1
    predictor_layers: int = 4
    text_max_length: int = 512
    unfreeze_text: bool = False
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class SpineM1(nn.Module):
    def __init__(self, cfg: SpineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionEncoder(cfg.vision_repo, dtype=cfg.dtype, device=cfg.device)
        self.text = TextTarget(
            backbone=cfg.text_backbone, shared_dim=cfg.shared_dim,
            max_length=cfg.text_max_length, unfreeze_base=cfg.unfreeze_text,
            device=cfg.device, dtype=cfg.dtype,
        )
        self.predictor = Predictor(
            in_dim=self.encoder.out_dim, shared_dim=cfg.shared_dim,
            mode=cfg.predictor_mode, n_layers=cfg.predictor_layers,
            stack_factor=cfg.stack_factor,
        ).to(cfg.device)

    def trainable_parameters(self):
        params = list(self.predictor.parameters()) + list(self.text.proj.parameters())
        if self.cfg.unfreeze_text:
            params += [p for p in self.text.base.parameters() if p.requires_grad]
        return params

    def embed_video(self, videos) -> torch.Tensor:
        feats = self.encoder.encode(videos)          # frozen, no grad
        return self.predictor(feats)                 # (B, shared_dim)

    def embed_text(self, texts: List[str]) -> torch.Tensor:
        return self.text.encode_text(texts)          # (B, shared_dim)

    def forward(self, videos, texts: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        z_v = self.embed_video(videos)
        z_t = self.embed_text(texts)
        return info_nce(z_v, z_t, self.cfg.temperature)


if __name__ == "__main__":
    cfg = SpineConfig()
    spine = SpineM1(cfg)
    n = sum(p.numel() for p in spine.trainable_parameters())
    print(f"[spine_m1] trainable params = {n/1e6:.1f}M  (encoder + text base frozen)")
    B, T, C, H, W = 2, 64, 3, 256, 256
    dummy = (torch.rand(B, T, C, H, W) * 255).to(torch.uint8)
    loss, m = spine(dummy, ["a dog catching a frisbee", "glass shattering"])
    print(f"[spine_m1] loss={m['loss']:.3f} acc_v2t={m['acc_v2t']:.2f}")
