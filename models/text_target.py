"""
models/text_target.py

Y-Encoder: maps the text target into the shared embedding space.

VL-JEPA recipe (2512.10942): Y-Encoder = EmbeddingGemma-300M (native 768-dim,
Matryoshka), projected (with the predictor) into a shared 1,536-dim space; trained
with bi-directional InfoNCE. The paper's ablation shows a *frozen* text encoder with a
trainable linear projection on top is a valid configuration, so that is the M1 default
(simplest, no gated-model gradients). Set `unfreeze_base=True` to train it jointly with
a small LR multiplier (paper uses ~0.05) later.

Defaults to an UNGATED encoder so the smoke test runs with no HF license:
  - "minilm"          -> sentence-transformers/all-MiniLM-L6-v2 (384-dim, ungated)  [default]
  - "embeddinggemma"  -> google/embeddinggemma-300m (768-dim, GATED: accept license + HF token)

Native dim is read from config; never hardcoded.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


_PRESETS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "embeddinggemma": "google/embeddinggemma-300m",
}


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class TextTarget(nn.Module):
    def __init__(
        self,
        backbone: str = "minilm",
        shared_dim: int = 1536,
        max_length: int = 512,
        unfreeze_base: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        repo = _PRESETS.get(backbone, backbone)
        self.device_str = device
        self.dtype = dtype
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.base = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device)
        self.unfreeze_base = unfreeze_base
        self.base.train(unfreeze_base)
        for p in self.base.parameters():
            p.requires_grad_(unfreeze_base)

        native = int(self.base.config.hidden_size)
        # Trainable projection into the shared space (kept in fp32 for stability).
        self.proj = nn.Linear(native, shared_dim).to(device)
        self.shared_dim = shared_dim
        self.native_dim = native

    def encode_text(self, texts: Union[List[str], dict], return_prenorm: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode a list of texts -> (B, shared_dim) L2-normalised embeddings.

        If return_prenorm=True, returns (z_norm, z_prenorm) where z_prenorm is
        the pre-L2-normalised embedding for the SIGReg head.
        """
        if isinstance(texts, dict):
            tok = {k: v.to(self.device_str) for k, v in texts.items()}
        else:
            tok = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device_str)

        ctx = torch.enable_grad() if self.unfreeze_base else torch.no_grad()
        with ctx:
            out = self.base(**tok).last_hidden_state
            pooled = _mean_pool(out, tok["attention_mask"])

        z = self.proj(pooled.float())
        if return_prenorm:
            return F.normalize(z, dim=-1), z
        return F.normalize(z, dim=-1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)


if __name__ == "__main__":
    yt = TextTarget(backbone="minilm")
    z = yt.encode_text(["a dog catching a frisbee", "glass shattering on the floor"])
    print(f"[text_target] z={tuple(z.shape)} native={yt.native_dim} shared={yt.shared_dim} dtype={z.dtype}")
