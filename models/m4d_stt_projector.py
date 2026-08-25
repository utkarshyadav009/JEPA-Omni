"""models/m4d_stt_projector.py -- Ultravox-style STT fusion: a frozen
Whisper encoder + a trainable projector (with temporal frame-stacking, per
Ultravox's real published architecture -- not a plain linear layer) that
maps Whisper's audio hidden states directly into LFM2-700M's token
embedding space, so speech audio can be consumed by the LLM without an
intermediate text-transcription step.

Real, load-bearing finding from earlier research this project did: there is
NO weight-reuse shortcut here. This projector must be trained from scratch
against paired audio-transcript data -- confirmed via Ultravox's own docs,
not assumed. Training objective (models/../scripts/train_stt_projector.py):
freeze both Whisper and the LLM, train ONLY the projector so that feeding
the projected audio embeddings in place of the transcript's text-token
embeddings lets the (frozen) LLM predict the same transcript continuation
it would predict from real text -- i.e. the projector learns to make audio
"look like" text embeddings to the LLM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import WhisperModel


class AudioProjector(nn.Module):
    """Frame-stacking + 2-layer MLP, mapping Whisper encoder hidden states
    into an LLM's embedding space. Frame-stacking (not a bare per-frame
    linear layer) is the real Ultravox design -- concatenating `stack`
    consecutive 50Hz Whisper frames before projecting reduces the audio
    sequence length to be closer to the LLM's own token rate, and gives the
    projector more context per output embedding."""

    def __init__(self, whisper_dim: int, llm_dim: int, stack: int = 8, hidden_mult: int = 4):
        super().__init__()
        self.stack = stack
        in_dim = whisper_dim * stack
        hidden_dim = llm_dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, whisper_hidden: torch.Tensor) -> torch.Tensor:
        """whisper_hidden: (B, T, whisper_dim) -> (B, T//stack, llm_dim)."""
        b, t, d = whisper_hidden.shape
        pad = (-t) % self.stack
        if pad:
            whisper_hidden = torch.nn.functional.pad(whisper_hidden, (0, 0, 0, pad))
            t = t + pad
        stacked = whisper_hidden.reshape(b, t // self.stack, d * self.stack)
        return self.net(stacked)


class AudioEncoderProjector(nn.Module):
    """Wraps a frozen Whisper encoder + the trainable AudioProjector."""

    def __init__(self, whisper_repo: str, llm_dim: int, stack: int = 8, device: str = "cuda"):
        super().__init__()
        whisper = WhisperModel.from_pretrained(whisper_repo, dtype=torch.bfloat16)
        self.encoder = whisper.encoder.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projector = AudioProjector(self.encoder.config.d_model, llm_dim, stack=stack).to(device)
        self.device = device

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """input_features: (B, n_mels, T) log-mel spectrogram, Whisper's
        standard preprocessing. Returns (B, T', llm_dim) audio embeddings
        ready to be spliced into the LLM's input_embeds."""
        with torch.no_grad():
            hidden = self.encoder(input_features).last_hidden_state
        return self.projector(hidden)
