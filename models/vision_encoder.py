"""SigLIP2 vision tower wrapper used by the M1 spine.

:class:`VisionEncoder` turns a *list* of video clips into a single embedding
per clip by encoding every sampled frame with the SigLIP image tower and
mean-pooling over time.

It is a lightweight wrapper around an already-instantiated vision tower
(``model.vision_model``) plus its image processor -- it deliberately does not
own / re-register the tower as a submodule, so the owning :class:`SpineM1`
registers the backbone parameters exactly once.

Whatever keys the matching image processor produces (standard
``pixel_values`` for the fixed-resolution SigLIP2 checkpoints, or
``pixel_values`` + ``pixel_attention_mask`` + ``spatial_shapes`` for the
NaFlex variants) are forwarded straight to the tower, so the same code path
works for both.

Interface contract (matched by ``data``/``train_m1``/``eval_m1``):

    embeds = vision_encoder.encode(clips)   # clips: List[uint8 [T, C, H, W]]
                                            # -> Tensor [B, D]
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn


class VisionEncoder:
    """SigLIP image tower + processor with temporal mean-pooling.

    Parameters
    ----------
    tower:
        An instantiated SigLIP vision tower (e.g. ``siglip_model.vision_model``).
    processor:
        The matching HuggingFace image processor.
    frame_chunk_size:
        Maximum number of frames pushed through the tower at once.
    temporal_pool:
        Temporal reduction across frames; only ``"mean"`` is supported.
    """

    def __init__(
        self,
        tower: nn.Module,
        processor,
        *,
        frame_chunk_size: int = 256,
        temporal_pool: str = "mean",
    ) -> None:
        if temporal_pool != "mean":
            raise ValueError(f"Unsupported temporal_pool={temporal_pool!r}; only 'mean'.")
        self.tower = tower
        self.processor = processor
        self.frame_chunk_size = int(frame_chunk_size)
        self.temporal_pool = temporal_pool
        self.embed_dim: int = int(tower.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.tower.parameters()).device

    def _is_trainable(self) -> bool:
        for p in self.tower.parameters():
            return bool(p.requires_grad)
        return False

    def _process_frames(self, frames_u8: Tensor) -> Dict[str, Tensor]:
        """Run the image processor on uint8 [N, C, H, W] frames -> model inputs."""
        images = [f.permute(1, 2, 0).cpu().numpy() for f in frames_u8]
        proc = self.processor(images=images, return_tensors="pt")
        device = self.device
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in proc.items()
        }

    def _run_tower(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Pooled per-frame embeddings, chunked along the frame axis."""
        n = inputs["pixel_values"].shape[0]
        step = max(1, self.frame_chunk_size)
        outputs: List[Tensor] = []
        for start in range(0, n, step):
            chunk = {
                k: (v[start : start + step] if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            outputs.append(self.tower(**chunk).pooler_output)
        return torch.cat(outputs, dim=0)

    def encode(self, clips: Sequence[Tensor]) -> Tensor:
        """Encode a *list* of clips into one embedding per clip.

        Parameters
        ----------
        clips:
            Sequence of uint8 tensors shaped ``[T, C, H, W]`` (``T`` may vary
            per clip), already resized to the processor's resolution.

        Returns
        -------
        Tensor of shape ``[len(clips), embed_dim]`` (un-normalised pooled
        video embeddings).
        """
        if not isinstance(clips, (list, tuple)):
            raise TypeError(
                "VisionEncoder.encode expects a list/tuple of clips, "
                f"got {type(clips)!r}."
            )
        if len(clips) == 0:
            return torch.zeros((0, self.embed_dim), device=self.device)

        lengths = [int(c.shape[0]) for c in clips]
        flat = torch.cat([c for c in clips], dim=0)  # [sumT, C, H, W]
        inputs = self._process_frames(flat)

        ctx = nullcontext() if self._is_trainable() else torch.no_grad()
        with ctx:
            frame_embeds = self._run_tower(inputs)  # [sumT, D]

        pooled: List[Tensor] = []
        idx = 0
        for length in lengths:
            pooled.append(frame_embeds[idx : idx + length].mean(dim=0))
            idx += length
        return torch.stack(pooled, dim=0)

    def __call__(self, clips: Sequence[Tensor]) -> Tensor:  # pragma: no cover - alias
        return self.encode(clips)
