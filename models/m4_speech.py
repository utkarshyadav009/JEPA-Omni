"""models/m4_speech.py — M4b speech path: frozen Whisper-medium encoder +
Ultravox-style projector (stack-8, RMSNorm, SwiGLU) into the shared LLM
input space (1536-d, Qwen2.5-1.5B-Instruct).

Additive, new file -- does not touch models/av_jepa_predictor.py, models/
m3_connector.py, or any other existing model code.

Design (approved 2026-07-24):
  - Encoder: whisper-medium (769M, frozen). Chosen over WavLM because our
    only training supervision (EasyCom Close_Microphone_Audio <->
    Transcription pairs) is content/ASR-alignment supervision, which is
    exactly Whisper's training objective (and Ultravox's own real choice
    for the same task shape). Chosen over whisper-large-v3 because our LLM
    is only 1.5B -- an oversized frozen encoder would dwarf the trainable
    stack and is hard to leverage on EasyCom's small (6,692-segment) real
    supervision; scale up once training data grows.
  - Projector: stack 8 consecutive 20ms (50Hz) Whisper frames -> 160ms/
    token, RMSNorm, SwiGLU-gated MLP -> llm_hidden=1536. Deliberately
    VARIABLE-length (unlike M3's fixed-32-latent Perceiver connector):
    speech-content grounding needs sequential/word-order structure
    preserved, a fixed-latent bottleneck is the wrong shape for that job.
  - Whisper's encoder has a FIXED 1500-position (30s) positional embedding,
    so the frozen encoder always runs on the full padded-to-30s window
    (that's how Whisper works, can't be avoided) -- but we slice the
    output down to each utterance's own real frame count before stacking/
    projecting, so short utterances don't pay for hundreds of wasted
    padding tokens downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

WHISPER_FRAME_RATE_HZ = 50.0   # Whisper encoder: 20ms/frame after conv stride


class RMSNorm(nn.Module):
    """Matches the normalization convention used inside Qwen/Llama-family
    transformer blocks -- consistent geometry for what's about to be fed
    into that same family of models."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


@dataclass
class UltravoxProjectorConfig:
    whisper_hidden: int = 1024      # whisper-medium d_model
    stack_factor: int = 8           # 8 * 20ms = 160ms/token
    llm_hidden: int = 1536          # Qwen2.5-1.5B-Instruct hidden_size
    mlp_dim: int = 4096
    dropout: float = 0.0


class UltravoxProjector(nn.Module):
    """stack -> RMSNorm -> SwiGLU MLP -> llm_hidden. ONLY trainable
    component of M4b (Whisper and the LLM are both frozen)."""

    def __init__(self, cfg: UltravoxProjectorConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.whisper_hidden * cfg.stack_factor
        self.norm = RMSNorm(in_dim)
        self.gate_up = nn.Linear(in_dim, 2 * cfg.mlp_dim, bias=False)
        self.down = nn.Linear(cfg.mlp_dim, cfg.llm_hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden: Tensor, valid_frames: Tensor) -> Tuple[Tensor, Tensor]:
        """hidden: (B, T, whisper_hidden) frozen Whisper encoder output,
        already truncated to this batch's max real length (see
        WhisperSpeechEncoder.forward). valid_frames: (B,) real (non-padding)
        frame count per sample. Returns (projected (B, T//stack, llm_hidden),
        key_padding_mask (B, T//stack) bool True=PADDING)."""
        B, T, D = hidden.shape
        sf = self.cfg.stack_factor
        pad = (-T) % sf
        if pad > 0:
            hidden = F.pad(hidden, (0, 0, 0, pad))
        T2 = hidden.shape[1]
        n_tokens = T2 // sf
        stacked = hidden.reshape(B, n_tokens, D * sf)

        x = self.norm(stacked)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dropout(x)
        x = self.down(x)   # (B, n_tokens, llm_hidden)

        valid_tokens = torch.clamp((valid_frames.float() / sf).ceil().long(), max=n_tokens)
        pos = torch.arange(n_tokens, device=hidden.device).unsqueeze(0).expand(B, -1)
        key_padding_mask = pos >= valid_tokens.unsqueeze(1)   # True = padding
        return x, key_padding_mask


class WhisperSpeechEncoder(nn.Module):
    """Frozen whisper-medium encoder. Runs on the padded-to-30s window
    (unavoidable given Whisper's fixed 1500-position embedding), but the
    caller (forward()) returns each sample's REAL valid-frame count so
    downstream code can slice off the padding-derived frames."""

    def __init__(self, model_id: str = "openai/whisper-medium", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        full = WhisperModel.from_pretrained(model_id, dtype=dtype)
        self.encoder = full.encoder
        del full.decoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.hidden_size = self.encoder.config.d_model
        self.native_sr = self.feature_extractor.sampling_rate   # 16000
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, waveforms: List, durations_sec: List[float], device) -> Tuple[Tensor, Tensor]:
        """waveforms: list of 1D float32 numpy arrays at self.native_sr.
        Returns (hidden (B, maxT, whisper_hidden) truncated to this batch's
        max real length, valid_frames (B,) long)."""
        inputs = self.feature_extractor(waveforms, sampling_rate=self.native_sr, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=self.dtype)
        out = self.encoder(input_features)
        full_hidden = out.last_hidden_state   # (B, 1500, whisper_hidden)

        valid_frames = torch.tensor(
            [min(full_hidden.shape[1], max(1, math.ceil(d * WHISPER_FRAME_RATE_HZ))) for d in durations_sec],
            device=device, dtype=torch.long,
        )
        max_valid = int(valid_frames.max().item())
        hidden = full_hidden[:, :max_valid]   # drop the pure-padding tail shared by the whole batch
        return hidden, valid_frames


class MoonshineSpeechEncoder(nn.Module):
    """Task 150: real replacement for the decision-head-facing leg of
    WhisperSpeechEncoder -- NOT a replacement for Whisper everywhere. The
    M4b speech projector / AsyncThinker deep-grounding path still uses
    WhisperSpeechEncoder (untouched, out of scope here -- confirmed by
    reading scripts/bmo_jetson_startup.py that the currently-DEPLOYED
    production stack never wires that path up at all, only the 3-class
    decision head, which is the actual load-bearing consumer this swap
    targets).

    Real, measured win (2026-08-07, Jetson): 20-26ms encode vs Whisper-
    medium's 277ms -- 10-13x faster, and (unlike Whisper's fixed 1500-
    position/30s-padded window) Moonshine's ergodic sliding-window encoder
    naturally outputs a length proportional to the real input, so there is
    no padding-derived tail to slice off -- valid_frames below is just the
    real output length, kept only for interface parity with
    WhisperSpeechEncoder so callers don't need two code paths.

    Real, measured cost: retraining SpeechOnlyThreeClassHead on Moonshine-
    base features (416-dim, vs Whisper-medium's 1024-dim) scored 90.67%
    test accuracy vs the Whisper-based head's previously-reported 95.00%
    -- a real ~4.3pt regression, not hidden. Backchannel recall dropped
    most (0.85). If that gap matters more than the latency win, moonshine-
    medium-streaming (245M params, still far smaller than whisper-medium's
    769M) is the next lever to try -- not attempted here."""

    def __init__(self, model_id: str = "UsefulSensors/moonshine-base", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoProcessor, MoonshineModel

        self.processor = AutoProcessor.from_pretrained(model_id)
        full = MoonshineModel.from_pretrained(model_id, dtype=dtype)
        self.encoder = full.encoder
        del full.decoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.hidden_size = full.config.hidden_size
        self.native_sr = 16000
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, waveforms: List, durations_sec: List[float], device) -> Tuple[Tensor, Tensor]:
        """Same call signature as WhisperSpeechEncoder.forward() for
        drop-in compatibility. Single-sample batches only for now (matches
        every real call site today, models/m4_duplex_loop.py's
        compute_speech_activity -- extend to true batching if a caller
        ever needs it)."""
        assert len(waveforms) == 1, "MoonshineSpeechEncoder currently only supports batch size 1"
        inputs = self.processor(waveforms[0], sampling_rate=self.native_sr, return_tensors="pt")
        input_values = inputs.input_values.to(device, dtype=self.dtype)
        out = self.encoder(input_values)
        hidden = out.last_hidden_state   # (1, T, hidden_size) -- T already matches real input length
        valid_frames = torch.tensor([hidden.shape[1]], device=device, dtype=torch.long)
        return hidden, valid_frames


if __name__ == "__main__":
    # CPU-friendly smoke test: projector mechanics only (Whisper needs a
    # real checkpoint download, exercised separately in train_m4b.py).
    torch.manual_seed(0)
    cfg = UltravoxProjectorConfig()
    proj = UltravoxProjector(cfg)
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"UltravoxProjector params={n_params:,} ({n_params/1e6:.1f}M)")

    B = 3
    lengths = [37, 150, 8]   # ragged real Whisper-frame counts
    maxT = max(lengths)
    hidden = torch.randn(B, maxT, cfg.whisper_hidden)
    valid_frames = torch.tensor(lengths)
    out, mask = proj(hidden, valid_frames)
    print(f"output shape={tuple(out.shape)}  mask shape={tuple(mask.shape)}")
    for i, L in enumerate(lengths):
        expected_tokens = math.ceil(L / cfg.stack_factor)
        actual_valid = int((~mask[i]).sum().item())
        assert actual_valid == expected_tokens, (actual_valid, expected_tokens)
    print("valid-token counts match expected stack-factor ceiling per sample: OK")
    loss = out.sum()
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in proj.parameters() if p.grad is not None)
    print(f"backward OK, total grad norm={gnorm:.3f}")
