"""models/m4_control.py — M4 control-token + LoRA wiring utilities.

Additive, new file (no existing model code touched). Two pieces:

1. Control tokens. Qwen2.5-1.5B-Instruct's tokenizer vocab (151,665 entries)
   already contains several special tokens that exist only because the 2.5
   family shares one tokenizer schema across text/VL/coder variants -- for
   THIS text-only instruct checkpoint (no vision tower, no FIM/repo tooling
   wired up in our pipeline) they are dead weight: `<|quad_start|>` /
   `<|quad_end|>` (VL bounding-quad tokens) are never produced or consumed
   anywhere in this project. We repurpose their existing embedding rows for
   <silence> / <stop_interruption> instead of resizing the embedding matrix
   (which would require touching the tied LM head and re-deriving init
   statistics for brand-new rows). Their embeddings still need real training
   (LoRA won't reach them -- see below), but the row already exists.

2. LoRA wrapper. Thin helper around peft.get_peft_model so train_m4.py has
   one place to change LoRA hyperparameters. Base LLM weights stay frozen;
   only the low-rank adapters (and, separately, the two repurposed control-
   token embedding rows -- unfrozen explicitly, since they start from
   pretrained-but-irrelevant vectors and need to move freely, not just via a
   low-rank update) are trainable.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

CONTROL_TOKENS: Dict[str, str] = {
    "silence": "<|quad_start|>",
    "stop_interruption": "<|quad_end|>",
}

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def get_control_token_ids(tokenizer) -> Dict[str, int]:
    ids = {name: tokenizer.convert_tokens_to_ids(tok) for name, tok in CONTROL_TOKENS.items()}
    for name, tid in ids.items():
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"control token {name!r} ({CONTROL_TOKENS[name]!r}) not found in tokenizer vocab")
    return ids


def wrap_lora(
    llm: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
):
    """Wrap a frozen causal LM with LoRA adapters. Returns the peft-wrapped
    model; base weights frozen, only adapters trainable (peft handles the
    freezing internally via requires_grad_)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(llm, cfg)


class ControlTokenEmbedding(nn.Module):
    """Fix for the original unfreeze_control_token_rows() approach (removed
    2026-07-24): that version set requires_grad=True on the FULL 233M-param
    tied embedding/LM-head matrix and zeroed gradients for all but 2 rows
    via a backward hook. Behaviorally correct (only those 2 rows ever
    actually moved) but wasteful: the optimizer still allocated Adam state
    (exp_avg + exp_avg_sq) for all 233M entries, ~1.8GB for nothing.

    This version keeps the base embedding matrix (and, since Qwen2.5-1.5B
    ties weights, the LM head that reads the SAME tensor) COMPLETELY frozen
    -- zero rows unfrozen -- and adds one tiny genuinely-new parameter,
    shape (2, hidden_dim) = 3,072 floats, as an additive delta on top of the
    base (frozen) lookup for the two control-token ids only. Only the INPUT
    side needs a new representation (the tokens' pretrained meaning as VL
    bounding-box markers is irrelevant conditioning context, so the model
    needs to actually learn something new to read them meaningfully). The
    OUTPUT/scoring side deliberately does NOT get a separate parameter:
    since the matrix is tied, moving the input-side row would silently move
    the LM-head row too and reintroduce exactly the coupling problem this
    fix removes; instead we rely on LoRA reshaping the hidden state space so
    the frozen (but already distinct, already-unused-in-practice) existing
    row for each control token becomes the right target to dot-product
    against -- the same "steer toward a fixed target vector via upstream
    representation changes" mechanism prompt/prefix-tuning already relies on
    for every one of the LLM's other 151,663 frozen vocab rows."""

    def __init__(self, base_embedding: nn.Embedding, control_token_ids: Dict[str, int]):
        super().__init__()
        self.base_embedding = base_embedding
        for p in self.base_embedding.parameters():
            p.requires_grad_(False)
        self.ids = list(control_token_ids.values())
        self._id_to_slot = {tid: i for i, tid in enumerate(self.ids)}
        d = base_embedding.embedding_dim
        ref = base_embedding.weight
        # match device/dtype so this drops in cleanly regardless of where the
        # base LLM was already placed (.to(device)/.to(dtype) happen before
        # wrapping); trained in fp32 regardless, autocast handles the matmul
        self.delta = nn.Parameter(torch.zeros(len(self.ids), d, device=ref.device, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base_embedding(input_ids)
        for slot, tid in enumerate(self.ids):
            match = (input_ids == tid)
            if match.any():
                out = out + match.unsqueeze(-1).to(out.dtype) * self.delta[slot].to(out.dtype)
        return out


def attach_control_token_embedding(peft_model: nn.Module, tokenizer) -> ControlTokenEmbedding:
    """Replaces the (frozen) base LLM's input-embedding module with a
    ControlTokenEmbedding wrapper. Returns the wrapper so callers can read
    `.delta` directly if needed (e.g. for inspection/logging)."""
    ids = get_control_token_ids(tokenizer)
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    base_embed = base.get_input_embeddings()
    wrapped = ControlTokenEmbedding(base_embed, ids)
    base.set_input_embeddings(wrapped)
    return wrapped


if __name__ == "__main__":
    # CPU-friendly smoke test: control-token lookup only (LoRA wrap needs the
    # real LLM, exercised separately in train_m4.py's --smoke-test path).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ids = get_control_token_ids(tok)
    print(f"control token ids: {ids}")
    assert len(set(ids.values())) == 2, "control tokens must map to distinct ids"
    print("OK")
