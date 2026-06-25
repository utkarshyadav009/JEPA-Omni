"""
models/predictor.py

The ONLY trainable visual-side module in M1: maps frozen V-JEPA tokens (B, N, 1024)
into the shared embedding space (default 1536-dim), L2-normalized.

Two runnable, ungated modes:
  - "mlp"          -> attention-pool tokens -> SwiGLU MLP + RMSNorm (Ultravox-style). Cheap, O(N).
  - "transformer"  -> learnable CLS + projected tokens through a small NON-CAUSAL
                      Transformer encoder, take CLS (spiritual match to VL-JEPA's predictor).

FAITHFUL upgrade (delegated to the coding agent, see project notes): "llama_last8" =
init the predictor from the last 8 layers of Llama-3.2-1B (490M trainable), causal mask
DISABLED so vision+query co-attend, projecting to the shared 1536-dim space. That mode
reproduces VL-JEPA (2512.10942) exactly but needs the gated Llama-3.2-1B; validate the
layer-slicing against your transformers version before trusting numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AttnPool(nn.Module):
    """Single learnable query attends over tokens -> (B, dim)."""
    def __init__(self, dim: int, n_heads: int = 8) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(pooled.squeeze(1))


class Predictor(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        shared_dim: int = 1536,
        mode: str = "mlp",
        hidden_mult: int = 4,
        n_layers: int = 4,
        n_heads: int = 8,
        stack_factor: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.shared_dim = shared_dim
        self.stack_factor = stack_factor

        eff_in = in_dim * stack_factor  # Ultravox-style stack pooling (1 = off)

        if mode == "mlp":
            self.in_norm = RMSNorm(eff_in)
            self.pool = AttnPool(eff_in, n_heads)
            self.ffn = SwiGLU(eff_in, eff_in * hidden_mult)
            self.out_norm = RMSNorm(eff_in)
            self.head = nn.Linear(eff_in, shared_dim)

        elif mode == "transformer":
            self.in_proj = nn.Linear(eff_in, shared_dim)
            self.cls = nn.Parameter(torch.randn(1, 1, shared_dim) * shared_dim ** -0.5)
            layer = nn.TransformerEncoderLayer(
                d_model=shared_dim, nhead=n_heads,
                dim_feedforward=shared_dim * hidden_mult,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(  # non-causal by default
                layer, num_layers=n_layers, enable_nested_tensor=False)
            self.out_norm = RMSNorm(shared_dim)
            self.head = nn.Linear(shared_dim, shared_dim)
            
        elif mode == "llama_last8":
            try:
                from transformers import LlamaModel
            except ImportError:
                raise ImportError("mode='llama_last8' requires the transformers library.")
            
            llama = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="sdpa")
            hidden_size = llama.config.hidden_size
            
            self.in_proj = nn.Linear(eff_in, hidden_size)
            self.cls = nn.Parameter(torch.randn(1, 1, hidden_size) * hidden_size ** -0.5)
            
            self.layers = nn.ModuleList(llama.layers[-8:])
            self.norm = llama.norm
            
            # Llama 3.2+ stores rotary_emb in the model, older versions in the first layer's attn
            if hasattr(llama, "rotary_emb"):
                self.rotary_emb = llama.rotary_emb
            else:
                self.rotary_emb = llama.layers[0].self_attn.rotary_emb
            
            for layer in self.layers:
                if hasattr(layer.self_attn, "is_causal"):
                    layer.self_attn.is_causal = False
                    
            del llama
            self.head = nn.Linear(hidden_size, shared_dim)

        else:
            raise ValueError(f"Unknown predictor mode '{mode}'. Use 'mlp', 'transformer', or 'llama_last8'.")

    def _stack(self, x: torch.Tensor) -> torch.Tensor:
        if self.stack_factor == 1:
            return x
        B, N, C = x.shape
        n = (N // self.stack_factor) * self.stack_factor
        x = x[:, :n, :].reshape(B, n // self.stack_factor, C * self.stack_factor)
        return x

    def forward(self, tokens: torch.Tensor, return_prenorm: bool = False):
        """tokens: (B, N, in_dim) from the frozen encoder -> (B, shared_dim) normalized.

        If return_prenorm=True, returns (z_norm, z_prenorm) where z_prenorm is
        the pre-L2-normalised embedding — needed by the SIGReg head so it can
        push per-direction variance toward 1.0 without fighting unit-norm.
        """
        x = self._stack(tokens.float())
        if self.mode == "mlp":
            x = self.in_norm(x)
            pooled = self.pool(x)
            pooled = pooled + self.ffn(self.out_norm(pooled))
            z = self.head(pooled)
        elif self.mode == "transformer":
            x = self.in_proj(x)
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.encoder(x)
            z = self.head(self.out_norm(x[:, 0]))
        elif self.mode == "llama_last8":
            x = self.in_proj(x)
            
            # Match the Llama parameters' dtype (usually bfloat16)
            llama_dtype = next(self.layers.parameters()).dtype
            x = x.to(dtype=llama_dtype)
            
            cls = self.cls.expand(x.shape[0], -1, -1).to(dtype=llama_dtype)
            x = torch.cat([cls, x], dim=1)
            
            position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            
            # The latest transformers dev version has a decorator that might 
            # fail if positional arguments are used while it expects keywords or vice-versa.
            # We call it with explicit keywords to be safe.
            try:
                position_embeddings = self.rotary_emb(x, position_ids)
            except TypeError:
                # Fallback for some dev versions where signature might be (x, position_ids, seq_len=None)
                # but positional count check is strict.
                position_embeddings = self.rotary_emb(x=x, position_ids=position_ids)
            
            # Create a 4D attention mask of zeros (batch, 1, seq, seq) to allow full bidirectional attention
            # and avoid SDPA automatic causal masking or NestedTensor shape collapsing when mask is None.
            attention_mask = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            
            for layer in self.layers:
                layer_outputs = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                if isinstance(layer_outputs, tuple):
                    x = layer_outputs[0]
                else:
                    x = layer_outputs
                
            x = self.norm(x)
            # Mean-pooling across visual tokens (excluding CLS token at index 0)
            x_pooled = x[:, 1:].mean(dim=1)
            # Cast back to head weight dtype (usually float32) before linear projection
            z = self.head(x_pooled.to(dtype=self.head.weight.dtype))

        if return_prenorm:
            return F.normalize(z, dim=-1), z
        return F.normalize(z, dim=-1)



if __name__ == "__main__":
    # Light standalone test (N=256) so it runs on a laptop; real seqs are ~8192 on GPU.
    for mode in ("mlp", "transformer", "llama_last8"):
        # Skip llama_last8 if we don't want to download the 1B model by default during tests
        if mode == "llama_last8":
            continue
        p = Predictor(in_dim=1024, shared_dim=1536, mode=mode)
        z = p(torch.randn(2, 256, 1024))
        n_params = sum(t.numel() for t in p.parameters() if t.requires_grad)
        print(f"[predictor:{mode}] z={tuple(z.shape)} trainable_params={n_params/1e6:.1f}M")
