"""SigLIP2 vision tower used by the M1 spine.

The :class:`VisionEncoder` wraps a SigLIP2 vision model and turns a *list*
of video clips into a single embedding per clip by encoding every sampled
frame with the image tower and mean-pooling over time.

Interface contract (matched by ``data``/``train_m1``/``eval_m1``):

    encoder = VisionEncoder(model_name, ...)
    embeds  = encoder.encode(clips)   # clips: List[uint8 Tensor [T, C, H, W]]
                                      # -> Tensor [B, D]

``encode`` accepts a Python list (clips may have a different number of
frames each) and never expects a pre-stacked batch tensor.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence

import torch
from torch import Tensor, nn

try:  # transformers is an optional import at module load time for fast unit tests
    from transformers import AutoImageProcessor, Siglip2VisionModel
except Exception:  # pragma: no cover - exercised only without transformers installed
    AutoImageProcessor = None  # type: ignore[assignment]
    Siglip2VisionModel = None  # type: ignore[assignment]


class VisionEncoder(nn.Module):
    """SigLIP2 image tower with temporal mean-pooling.

    Parameters
    ----------
    model_name:
        HuggingFace id of the SigLIP2 checkpoint (e.g.
        ``"google/siglip2-base-patch16-256"``).
    trainable:
        If ``False`` (default) the backbone is frozen and its forward pass is
        run under ``torch.no_grad`` to save memory; only downstream modules
        (adapters / logit parameters living in the spine) are optimised.
    frame_chunk_size:
        Maximum number of frames pushed through the backbone at once. Keeps
        peak memory bounded when ``batch_size * num_frames`` is large.
    temporal_pool:
        Temporal reduction across frames; only ``"mean"`` is supported.
    """

    def __init__(
        self,
        model_name: str,
        *,
        trainable: bool = False,
        frame_chunk_size: int = 256,
        temporal_pool: str = "mean",
    ) -> None:
        super().__init__()
        if Siglip2VisionModel is None:
            raise ImportError(
                "transformers is required for VisionEncoder. "
                "Install it with `pip install transformers`."
            )
        if temporal_pool != "mean":
            raise ValueError(f"Unsupported temporal_pool={temporal_pool!r}; only 'mean'.")

        self.model_name = model_name
        self.temporal_pool = temporal_pool
        self.frame_chunk_size = int(frame_chunk_size)

        self.backbone = Siglip2VisionModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)

        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        size = getattr(processor, "size", {}) or {}
        self.image_size: int = int(size.get("height") or size.get("shortest_edge") or 256)

        self.embed_dim: int = int(self.backbone.config.hidden_size)
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the backbone in-place."""
        self.trainable = bool(trainable)
        self.backbone.requires_grad_(self.trainable)

    def _normalize(self, frames_u8: Tensor) -> Tensor:
        """Rescale uint8 [N, C, H, W] frames to the SigLIP2 normalised range."""
        x = frames_u8.to(self.pixel_mean.dtype) / 255.0
        return (x - self.pixel_mean) / self.pixel_std

    def _run_backbone(self, pixel_values: Tensor) -> Tensor:
        """Return pooled per-frame embeddings, chunked to bound memory."""
        outputs: List[Tensor] = []
        n = pixel_values.shape[0]
        step = max(1, self.frame_chunk_size)
        for start in range(0, n, step):
            chunk = pixel_values[start : start + step]
            pooled = self.backbone(pixel_values=chunk).pooler_output
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def encode(self, clips: Sequence[Tensor]) -> Tensor:
        """Encode a *list* of clips into one embedding per clip.

        Parameters
        ----------
        clips:
            Sequence of uint8 tensors shaped ``[T, C, H, W]`` (``T`` may vary
            per clip). Frames are expected to already be at the processor's
            resolution (``image_size`` x ``image_size``).

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
            return torch.zeros((0, self.embed_dim), device=self.pixel_mean.device)

        device = self.pixel_mean.device
        lengths = [int(c.shape[0]) for c in clips]
        flat = torch.cat([c.to(device) for c in clips], dim=0)
        pixel_values = self._normalize(flat)

        ctx = nullcontext() if self.trainable else torch.no_grad()
        with ctx:
            frame_embeds = self._run_backbone(pixel_values)

        # Split back into per-clip groups and pool over time.
        pooled: List[Tensor] = []
        idx = 0
        for length in lengths:
            seg = frame_embeds[idx : idx + length]
            pooled.append(seg.mean(dim=0))
            idx += length
        return torch.stack(pooled, dim=0)

    def forward(self, clips: Sequence[Tensor]) -> Tensor:  # pragma: no cover - thin alias
        return self.encode(clips)
