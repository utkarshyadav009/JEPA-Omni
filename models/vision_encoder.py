"""
models/vision_encoder.py

Frozen V-JEPA 2 ViT-L vision encoder (the X-Encoder of the VL-JEPA recipe).

Grounded facts (sources):
- Checkpoint `facebook/vjepa2-vitl-fpc64-256`, HF class VJEPA2Model, hidden_size=1024,
  default 256px / 64 frames, license MIT.  [HF model card + transformers config]
- Encoder features via `model.get_vision_features(**inputs)`.  [HF model card]
- Tubelet temporal-2 + 16x16 spatial patches => default config yields ~8192 tokens
  (64/2 temporal * (256/16)^2 spatial = 32 * 256), each 1024-dim.  [derived from
  tubelet/patch sizes verified in the project Research_Report]

This module is FROZEN: eval(), requires_grad_(False), bf16, no_grad on the forward.
It is the only place the V-JEPA HF API is touched — everything downstream sees a
clean (B, N, 1024) tensor.
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoVideoProcessor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Needs a recent `transformers` with V-JEPA 2 support. Install:\n"
        "  pip install -U git+https://github.com/huggingface/transformers torchcodec einops timm"
    ) from e


VideoInput = Union[torch.Tensor, List[torch.Tensor]]  # [B,T,C,H,W] or list of [T,C,H,W]


class VisionEncoder(nn.Module):
    def __init__(
        self,
        hf_repo: str = "facebook/vjepa2-vitl-fpc64-256",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.hf_repo = hf_repo
        self.dtype = dtype
        self.device_str = device

        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(
            hf_repo,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)

        # Freeze hard.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 1024 for ViT-L; read it instead of hardcoding so a ViT-H/g swap "just works".
        self.hidden_size: int = int(self.model.config.hidden_size)

    @property
    def out_dim(self) -> int:
        return self.hidden_size

    def _to_list(self, videos: VideoInput) -> List[torch.Tensor]:
        if isinstance(videos, torch.Tensor):
            if videos.dim() == 4:  # single [T,C,H,W]
                return [videos]
            if videos.dim() == 5:  # [B,T,C,H,W]
                return [videos[i] for i in range(videos.shape[0])]
            raise ValueError(f"Expected 4D or 5D video tensor, got {videos.dim()}D")
        return list(videos)

    @torch.no_grad()
    def encode(self, videos: VideoInput) -> torch.Tensor:
        """videos: [B,T,C,H,W] (or list of [T,C,H,W]) uint8 0-255 OR float.
        Returns frozen features: (B, N, 1024)."""
        vid_list = self._to_list(videos)
        inputs = self.processor(vid_list, return_tensors="pt")
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        # pixel_values must match model dtype.
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        feats = self.model.get_vision_features(**inputs)  # (B, N, hidden)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    def forward(self, videos: VideoInput) -> torch.Tensor:
        return self.encode(videos)


if __name__ == "__main__":
    # Smoke test: random "video", confirm shape is (B, N, 1024).
    enc = VisionEncoder()
    B, T, C, H, W = 2, 64, 3, 256, 256
    dummy = (torch.rand(B, T, C, H, W) * 255).to(torch.uint8)
    out = enc.encode(dummy)
    print(f"[vision_encoder] features: {tuple(out.shape)}  dtype={out.dtype}  hidden={enc.out_dim}")
    assert out.shape[0] == B and out.shape[-1] == enc.out_dim
    print(f"[vision_encoder] tokens per clip N = {out.shape[1]} (expect ~8192 at 256px/64f)")
