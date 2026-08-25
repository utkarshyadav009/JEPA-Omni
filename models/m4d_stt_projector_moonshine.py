"""models/m4d_stt_projector_moonshine.py -- Ultravox-style STT fusion on the
MOONSHINE encoder (per user direction 2026-08-08: Moonshine, NOT Whisper).

Why Moonshine: the deployed live loop (live_bmo.py) already loads
`UsefulSensors/moonshine-base` for ASR, so reusing THAT SAME encoder for the
speech->LLM projector means one encoder in memory, one consistent front-end,
and the latency win we're after -- a single encoder forward with NO
autoregressive text-decode, feeding the LLM directly via inputs_embeds.

Architecture (Ultravox's real published design): frozen audio encoder +
trainable frame-stacking projector that maps encoder hidden states straight
into the LLM's token-embedding space. Frame-stacking (concatenating `stack`
consecutive encoder frames before projecting) shortens the audio sequence
toward the LLM's own token rate and gives each projected embedding more
temporal context. Moonshine's encoder already runs at a lower frame rate than
Whisper's 50Hz, so a smaller stack (default 4) is appropriate.

Trained (scripts/train_stt_projector_moonshine.py) with both encoder AND LLM
frozen -- only the projector learns -- so the projected audio embeddings make
the frozen LLM predict the same continuation it would from real text
(CE on the transcript + KL-distillation from a text-only teacher, the
Ultravox objective).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class MoonshineAudioProjector(nn.Module):
    """Frame-stacking + 2-layer MLP: Moonshine encoder hidden states -> LLM
    embedding space. Identical structure to the Whisper-side projector so the
    two are directly comparable; only the input dim and default stack differ."""

    def __init__(self, enc_dim: int, llm_dim: int, stack: int = 4, hidden_mult: int = 4):
        super().__init__()
        self.stack = stack
        in_dim = enc_dim * stack
        hidden_dim = llm_dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, enc_hidden: torch.Tensor) -> torch.Tensor:
        """enc_hidden: (B, T, enc_dim) -> (B, ceil(T/stack), llm_dim)."""
        b, t, d = enc_hidden.shape
        pad = (-t) % self.stack
        if pad:
            enc_hidden = torch.nn.functional.pad(enc_hidden, (0, 0, 0, pad))
            t = t + pad
        stacked = enc_hidden.reshape(b, t // self.stack, d * self.stack)
        return self.net(stacked)


class ConvDownsampler(nn.Module):
    """Deeper 1D-conv stride network: compresses the Moonshine frame sequence by
    `total_stride` (2^n) before projecting to the LLM space, so the LLM's small
    attention window isn't flooded with too many continuous acoustic frames
    (the ramble/loop failure mode). Variable-length output, fixed compression."""

    def __init__(self, enc_dim: int, llm_dim: int, total_stride: int = 8):
        super().__init__()
        n = max(1, int(round(__import__("math").log2(total_stride))))
        chans = [enc_dim] + [llm_dim] * n
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(nn.Conv1d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1))
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,enc_dim)
        x = x.transpose(1, 2)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i < len(self.convs) - 1:
                x = self.act(x)
        x = x.transpose(1, 2)                            # (B, T/stride, llm_dim)
        return self.norm(x)


class PerceiverResampler(nn.Module):
    """Flamingo/Ultravox-style resampler: a fixed set of learned latent queries
    cross-attend to the (arbitrarily long) audio frames, producing a FIXED
    number of `n_latents` soft tokens. Caps the acoustic sequence length the LLM
    sees regardless of utterance length -- directly arrests attention overload."""

    def __init__(self, enc_dim: int, llm_dim: int, n_latents: int = 64,
                 n_heads: int = 8, depth: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, llm_dim)
        self.latents = nn.Parameter(torch.randn(n_latents, llm_dim) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "ln_q": nn.LayerNorm(llm_dim),
            "ln_kv": nn.LayerNorm(llm_dim),
            "attn": nn.MultiheadAttention(llm_dim, n_heads, batch_first=True),
            "ff": nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, llm_dim * 4),
                                nn.GELU(), nn.Linear(llm_dim * 4, llm_dim)),
        }) for _ in range(depth)])
        self.out = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,enc_dim)
        kv = self.in_proj(x)
        q = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous()
        for l in self.layers:
            a, _ = l["attn"](l["ln_q"](q), l["ln_kv"](kv), l["ln_kv"](kv))
            q = q + a
            q = q + l["ff"](q)
        return self.out(q)                               # (B, n_latents, llm_dim)


class MoonshineEncoderProjector(nn.Module):
    """Wraps a frozen Moonshine encoder + the trainable MoonshineAudioProjector.

    Moonshine consumes RAW audio (not Whisper's log-mel): the processor yields
    `input_values` (B, n_samples) + `attention_mask`, and the encoder conv-stack
    downsamples internally. We never run the Moonshine decoder, so there is no
    autoregressive text generation on this path at all."""

    def __init__(self, moonshine_repo: str = "UsefulSensors/moonshine-base",
                 llm_dim: int = 1024, stack: int = 4, arch: str = "stack",
                 n_latents: int = 64, total_stride: int = 8, device: str = "cuda"):
        super().__init__()
        model = AutoModel.from_pretrained(moonshine_repo, dtype=torch.bfloat16)
        # encoder-decoder model: keep only the encoder (freeze it)
        self.encoder = model.encoder.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        enc_dim = getattr(model.config, "hidden_size", None) or model.config.encoder_hidden_size
        self.enc_dim = enc_dim
        self.arch = arch
        if arch == "stack":
            self.projector = MoonshineAudioProjector(enc_dim, llm_dim, stack=stack)
        elif arch == "conv":
            self.projector = ConvDownsampler(enc_dim, llm_dim, total_stride=total_stride)
        elif arch == "perceiver":
            self.projector = PerceiverResampler(enc_dim, llm_dim, n_latents=n_latents)
        else:
            raise ValueError(f"unknown projector arch: {arch}")
        self.projector = self.projector.to(device)
        self.device = device

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """input_values: (B, n_samples) raw 16kHz audio (Moonshine processor
        output). Returns (B, T', llm_dim) audio embeddings ready to splice into
        the LLM's inputs_embeds in place of transcript token embeddings."""
        with torch.no_grad():
            kw = {}
            if attention_mask is not None:
                kw["attention_mask"] = attention_mask
            out = self.encoder(input_values, **kw)
            hidden = out.last_hidden_state
        dt = next(self.projector.parameters()).dtype
        return self.projector(hidden.to(dt))
