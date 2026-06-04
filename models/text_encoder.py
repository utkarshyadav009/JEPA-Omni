"""SigLIP2 text tower used by the M1 spine.

The :class:`TextEncoder` wraps a SigLIP2 text model + tokenizer and turns a
list of caption strings into one embedding per caption.

Interface contract:

    encoder = TextEncoder(model_name, ...)
    embeds  = encoder.encode(captions)   # captions: List[str] -> Tensor [B, D]
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence

import torch
from torch import Tensor, nn

try:
    from transformers import AutoTokenizer, Siglip2TextModel
except Exception:  # pragma: no cover - exercised only without transformers installed
    AutoTokenizer = None  # type: ignore[assignment]
    Siglip2TextModel = None  # type: ignore[assignment]


class TextEncoder(nn.Module):
    """SigLIP2 text tower.

    Parameters
    ----------
    model_name:
        HuggingFace id of the SigLIP2 checkpoint.
    trainable:
        If ``False`` (default) the backbone is frozen; its forward pass runs
        under ``torch.no_grad`` to save memory.
    max_length:
        Tokenizer sequence length. SigLIP/SigLIP2 are trained with
        ``padding="max_length"`` and ``max_length=64``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        trainable: bool = False,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        if Siglip2TextModel is None:
            raise ImportError(
                "transformers is required for TextEncoder. "
                "Install it with `pip install transformers`."
            )
        self.model_name = model_name
        self.max_length = int(max_length)

        self.backbone = Siglip2TextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.embed_dim: int = int(self.backbone.config.hidden_size)
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the backbone in-place."""
        self.trainable = bool(trainable)
        self.backbone.requires_grad_(self.trainable)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def encode(self, captions: Sequence[str]) -> Tensor:
        """Encode a list of captions into one embedding per caption.

        Returns a Tensor of shape ``[len(captions), embed_dim]`` (un-normalised
        pooled text embeddings).
        """
        if not isinstance(captions, (list, tuple)):
            raise TypeError(
                "TextEncoder.encode expects a list/tuple of strings, "
                f"got {type(captions)!r}."
            )
        if len(captions) == 0:
            return torch.zeros((0, self.embed_dim), device=self.device)

        tokens = self.tokenizer(
            list(captions),
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        ctx = nullcontext() if self.trainable else torch.no_grad()
        with ctx:
            pooled = self.backbone(**tokens).pooler_output
        return pooled

    def forward(self, captions: Sequence[str]) -> Tensor:  # pragma: no cover - thin alias
        return self.encode(captions)
