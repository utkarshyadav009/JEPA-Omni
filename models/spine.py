"""The M1 video/text alignment "spine".

``SpineM1`` connects a SigLIP2 vision tower and a SigLIP2 text tower with a
small set of trainable parameters (residual adapters + learnable SigLIP
logit scale/bias). By default both backbones are frozen, so the only
optimised parameters are the connective "spine". The adapters are
zero-initialised on their output projection, so at initialisation the spine
reproduces the underlying SigLIP2 embeddings (and therefore the SigLIP2
zero-shot retrieval baseline) before any training happens.

Public interface (consumed by ``train_m1.py`` / ``eval_m1.py``):

    spine = SpineM1(SpineConfig(**cfg["model"]))
    spine.trainable_parameters()          # -> Iterator[nn.Parameter]
    spine.text_base_parameters()          # -> Iterator[nn.Parameter]
    zv = spine.embed_video(list_of_clips) # -> [B, D] L2-normalised
    zt = spine.embed_text(list_of_caps)   # -> [B, D] L2-normalised
    out = spine(list_of_clips, captions)  # -> dict(loss, acc_v2t, alignment, uniformity, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder


@dataclass
class SpineConfig:
    """Configuration for :class:`SpineM1`.

    Instantiated as ``SpineConfig(**cfg["model"])`` from ``configs/m1.yaml``.
    """

    vision_model_name: str = "google/siglip2-base-patch16-256"
    text_model_name: str = "google/siglip2-base-patch16-256"
    proj_dim: Optional[int] = None
    adapter_hidden: int = 2048
    adapter_dropout: float = 0.0
    use_adapters: bool = True
    temporal_pool: str = "mean"
    frame_chunk_size: int = 256
    max_text_length: int = 64
    unfreeze_text: bool = False
    unfreeze_vision: bool = False
    init_logit_scale: float = math.log(10.0)
    init_logit_bias: float = -10.0
    # Accept and ignore unknown keys so the YAML can carry extra annotations.
    extra: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temporal_pool != "mean":
            raise ValueError(f"Unsupported temporal_pool={self.temporal_pool!r}.")


class ResidualAdapter(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear residual block.

    The second linear layer is zero-initialised so the module is the identity
    at initialisation (``forward(x) == x``) when ``in_dim == out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Identity at init (only valid as residual when in_dim == out_dim).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.project: Optional[nn.Linear] = None
        if out_dim != in_dim:
            self.project = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.project is None else self.project(x)
        h = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))
        return residual + h


class _Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - trivial
        return x


class SpineM1(nn.Module):
    """Video/text alignment model for milestone M1."""

    def __init__(self, config: SpineConfig) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(
            config.vision_model_name,
            trainable=config.unfreeze_vision,
            frame_chunk_size=config.frame_chunk_size,
            temporal_pool=config.temporal_pool,
        )
        self.text_encoder = TextEncoder(
            config.text_model_name,
            trainable=config.unfreeze_text,
            max_length=config.max_text_length,
        )

        v_dim = self.vision_encoder.embed_dim
        t_dim = self.text_encoder.embed_dim
        proj_dim = config.proj_dim or v_dim
        self.embed_dim = proj_dim

        if config.use_adapters:
            self.vision_adapter: nn.Module = ResidualAdapter(
                v_dim, config.adapter_hidden, proj_dim, config.adapter_dropout
            )
            self.text_adapter: nn.Module = ResidualAdapter(
                t_dim, config.adapter_hidden, proj_dim, config.adapter_dropout
            )
        else:
            if v_dim != proj_dim or t_dim != proj_dim:
                raise ValueError(
                    "use_adapters=False requires proj_dim to match backbone dims."
                )
            self.vision_adapter = _Identity()
            self.text_adapter = _Identity()

        # Learnable SigLIP logit scale/bias -- the connective spine parameters.
        self.logit_scale = nn.Parameter(torch.tensor(float(config.init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(config.init_logit_bias)))

    # ------------------------------------------------------------------ #
    # Parameter groups
    # ------------------------------------------------------------------ #
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield every parameter that currently requires gradients."""
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def text_base_parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters of the *text backbone* only.

        Used by the optimiser to place the (optionally unfrozen) text base in
        a separate, lower-learning-rate parameter group.
        """
        for p in self.text_encoder.backbone.parameters():
            if p.requires_grad:
                yield p

    def vision_base_parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters of the vision backbone only."""
        for p in self.vision_encoder.backbone.parameters():
            if p.requires_grad:
                yield p

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    def embed_video(self, clips: Sequence[Tensor]) -> Tensor:
        """Embed a *list* of clips into L2-normalised vectors ``[B, D]``."""
        feats = self.vision_encoder.encode(clips)
        z = self.vision_adapter(feats)
        return F.normalize(z, p=2, dim=-1)

    def embed_text(self, captions: Sequence[str]) -> Tensor:
        """Embed a list of captions into L2-normalised vectors ``[B, D]``."""
        feats = self.text_encoder.encode(captions)
        z = self.text_adapter(feats)
        return F.normalize(z, p=2, dim=-1)

    # ------------------------------------------------------------------ #
    # Loss / metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _uniformity(x: Tensor, t: float = 2.0) -> Tensor:
        """Wang & Isola uniformity: log E_{i!=j} exp(-t * ||x_i - x_j||^2)."""
        if x.shape[0] < 2:
            return x.new_zeros(())
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def forward(
        self,
        clips: Sequence[Tensor],
        captions: Sequence[str],
    ) -> Dict[str, Tensor]:
        """Compute the SigLIP contrastive loss and alignment metrics.

        Returns a dict with ``loss``, ``acc_v2t``, ``alignment``,
        ``uniformity`` (scalar tensors) plus the embeddings and logits.
        """
        zv = self.embed_video(clips)  # [B, D]
        zt = self.embed_text(captions)  # [B, D]
        n = zv.shape[0]

        # SigLIP sigmoid loss (mirrors transformers' SiglipModel.forward).
        logit_scale = self.logit_scale.exp()
        logits_per_text = zt @ zv.t() * logit_scale + self.logit_bias  # [B_t, B_v]
        eye = torch.eye(n, device=zv.device, dtype=zv.dtype)
        m1_diag1 = -torch.ones_like(logits_per_text) + 2.0 * eye
        loglik = F.logsigmoid(m1_diag1 * logits_per_text)
        loss = (-loglik.sum(dim=-1)).mean()

        with torch.no_grad():
            sims_v2t = zv @ zt.t()  # [B_v, B_t]
            targets = torch.arange(n, device=zv.device)
            acc_v2t = (sims_v2t.argmax(dim=1) == targets).float().mean()
            alignment = (zv - zt).pow(2).sum(dim=-1).mean()
            uniformity = 0.5 * (self._uniformity(zv) + self._uniformity(zt))

        return {
            "loss": loss,
            "acc_v2t": acc_v2t,
            "alignment": alignment,
            "uniformity": uniformity,
            "logits_per_text": logits_per_text,
            "zv": zv,
            "zt": zt,
        }
