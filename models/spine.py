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
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder

try:  # transformers imported lazily so logic-only unit tests need no deps
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - exercised only without transformers installed
    AutoImageProcessor = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


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


def uniformity(x: Tensor, t: float = 2.0) -> Tensor:
    """Wang & Isola uniformity: log E_{i!=j} exp(-t * ||x_i - x_j||^2)."""
    if x.shape[0] < 2:
        return x.new_zeros(())
    # pdist has no bf16/fp16 kernel; compute in float32.
    sq_pdist = torch.pdist(x.float(), p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def alignment_loss(
    zv: Tensor,
    zt: Tensor,
    logit_scale: Tensor,
    logit_bias: Tensor,
) -> Dict[str, Tensor]:
    """SigLIP sigmoid loss + alignment/uniformity/accuracy metrics.

    ``zv`` and ``zt`` are L2-normalised ``[B, D]`` paired embeddings (positive
    pair ``i`` lies on the diagonal). Mirrors ``transformers``'
    ``SiglipModel.forward`` loss.
    """
    n = zv.shape[0]
    scale = logit_scale.exp()
    logits_per_text = zt @ zv.t() * scale + logit_bias  # [B_t, B_v]
    eye = torch.eye(n, device=zv.device, dtype=logits_per_text.dtype)
    m1_diag1 = -torch.ones_like(logits_per_text) + 2.0 * eye
    loglik = F.logsigmoid(m1_diag1 * logits_per_text)
    loss = (-loglik.sum(dim=-1)).mean()

    with torch.no_grad():
        zv_f = zv.float()
        zt_f = zt.float()
        sims_v2t = zv_f @ zt_f.t()  # [B_v, B_t]
        targets = torch.arange(n, device=zv.device)
        acc_v2t = (sims_v2t.argmax(dim=1) == targets).float().mean()
        align = (zv_f - zt_f).pow(2).sum(dim=-1).mean()
        unif = 0.5 * (uniformity(zv_f) + uniformity(zt_f))

    return {
        "loss": loss,
        "acc_v2t": acc_v2t,
        "alignment": align,
        "uniformity": unif,
        "logits_per_text": logits_per_text,
        "zv": zv,
        "zt": zt,
    }


def _load_backbones(
    config: SpineConfig,
) -> Tuple[nn.Module, nn.Module, object, object, object]:
    """Load the SigLIP backbone(s) and return towers + processor + tokenizer.

    Returns ``(vision_tower, text_tower, image_processor, tokenizer, modules)``
    where ``modules`` is an ``nn.ModuleDict`` holding the backbone(s) so the
    owning module registers their parameters exactly once.
    """
    if AutoModel is None:
        raise ImportError(
            "transformers (and torchvision for the image processor) are "
            "required for SpineM1. Install them with `pip install "
            "transformers torchvision`."
        )

    vision_name = config.vision_model_name
    text_name = config.text_model_name
    image_processor = AutoImageProcessor.from_pretrained(vision_name)
    tokenizer = AutoTokenizer.from_pretrained(text_name)

    modules = nn.ModuleDict()
    if vision_name == text_name:
        backbone = AutoModel.from_pretrained(vision_name)
        modules["backbone"] = backbone
        vision_tower = backbone.vision_model
        text_tower = backbone.text_model
    else:
        vision_backbone = AutoModel.from_pretrained(vision_name)
        text_backbone = AutoModel.from_pretrained(text_name)
        modules["vision_backbone"] = vision_backbone
        modules["text_backbone"] = text_backbone
        vision_tower = vision_backbone.vision_model
        text_tower = text_backbone.text_model

    return vision_tower, text_tower, image_processor, tokenizer, modules


class SpineM1(nn.Module):
    """Video/text alignment model for milestone M1."""

    def __init__(self, config: SpineConfig) -> None:
        super().__init__()
        self.config = config

        (
            vision_tower,
            text_tower,
            image_processor,
            tokenizer,
            backbones,
        ) = _load_backbones(config)

        # Register backbone parameters exactly once. The towers are referenced
        # (not re-registered) through the lightweight encoder wrappers below.
        self.backbones = backbones

        # Freeze everything, then selectively unfreeze the requested towers.
        self.backbones.requires_grad_(False)
        if config.unfreeze_vision:
            vision_tower.requires_grad_(True)
        if config.unfreeze_text:
            text_tower.requires_grad_(True)

        # Lightweight (non-Module) encode wrappers around the shared towers.
        self.vision_encoder = VisionEncoder(
            vision_tower,
            image_processor,
            frame_chunk_size=config.frame_chunk_size,
            temporal_pool=config.temporal_pool,
        )
        self.text_encoder = TextEncoder(
            text_tower,
            tokenizer,
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
        """Yield trainable parameters of the *text tower* only.

        Used by the optimiser to place the (optionally unfrozen) text base in
        a separate, lower-learning-rate parameter group.
        """
        for p in self.text_encoder.tower.parameters():
            if p.requires_grad:
                yield p

    def vision_base_parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters of the vision tower only."""
        for p in self.vision_encoder.tower.parameters():
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
        return alignment_loss(zv, zt, self.logit_scale, self.logit_bias)
