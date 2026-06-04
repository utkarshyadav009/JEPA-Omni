"""SigLIP2 text tower wrapper used by the M1 spine.

:class:`TextEncoder` turns a list of caption strings into one embedding per
caption. Like :class:`VisionEncoder` it is a lightweight wrapper around an
already-instantiated text tower (``model.text_model``) plus its tokenizer; it
does not own / re-register the tower as a submodule.

Interface contract:

    embeds = text_encoder.encode(captions)   # captions: List[str] -> Tensor [B, D]
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from torch import Tensor, nn

# Keys consumed by the SigLIP text tower forward.
_TEXT_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids")


class TextEncoder:
    """SigLIP text tower + tokenizer.

    Parameters
    ----------
    tower:
        An instantiated SigLIP text tower (e.g. ``siglip_model.text_model``).
    tokenizer:
        The matching HuggingFace tokenizer.
    max_length:
        Tokenizer sequence length. SigLIP/SigLIP2 are trained with
        ``padding="max_length"`` and ``max_length=64``.
    """

    def __init__(
        self,
        tower: nn.Module,
        tokenizer,
        *,
        max_length: int = 64,
    ) -> None:
        self.tower = tower
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.embed_dim: int = int(tower.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.tower.parameters()).device

    def _is_trainable(self) -> bool:
        for p in self.tower.parameters():
            return bool(p.requires_grad)
        return False

    def encode(self, captions: Sequence[str]) -> Tensor:
        """Encode a list of captions into one embedding per caption ``[B, D]``."""
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
        inputs = {
            k: v.to(self.device)
            for k, v in tokens.items()
            if k in _TEXT_INPUT_KEYS and isinstance(v, torch.Tensor)
        }

        ctx = nullcontext() if self._is_trainable() else torch.no_grad()
        with ctx:
            pooled = self.tower(**inputs).pooler_output
        return pooled

    def __call__(self, captions: Sequence[str]) -> Tensor:  # pragma: no cover - alias
        return self.encode(captions)
