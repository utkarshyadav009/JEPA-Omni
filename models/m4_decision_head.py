"""models/m4_decision_head.py — M4 speak/silence decision head.

PIVOT (2026-07-24): the LLM+LoRA route for control-token emission hit a
flat-zero silence-collapse across 40 checkpoints, 2 LoRA ranks, 6x class
reweighting, and 4x training duration -- not slow convergence, a genuine
optimization obstacle. SIGReg doesn't apply (it regularizes continuous
embedding distributions against collapse; this is a discrete token-emission
failure with no embedding distribution to shape). Moving the decision OUT
of the LLM entirely: a small dedicated head reads World-State (+ a speech-
activity feature) and outputs a binary speak/silence decision. The LLM is
NOT modified by this at all -- M3/M4b grounding stays exactly as banked.

This is a genuine 2-class classification problem (not discrete next-token
generation over a 150K-token vocabulary competing with a strong fluent-text
prior) -- standard class-imbalance handling (BCE + pos_weight) applies
cleanly, unlike the LoRA route's thin 2-token gradient fighting a ~150K-way
distribution.

Design: small MLP (not a transformer over a World-State sequence -- our
current tick data is i.i.d. single-tick samples, not temporal sequences;
building genuine sequence context would need restructuring the data
pipeline, flagged as a limitation, not attempted here) over the
concatenation of:
  - World-State (1024-d, from M2's encode_world_state -- REAL for VGGSound
    pseudo-timeline ticks, ZERO-VECTOR for EasyCom ticks, since EasyCom has
    no cached M2 features)
  - speech-activity feature (1024-d, mean-pooled frozen Whisper encoder
    hidden state over the tick's audio window -- REAL for EasyCom ticks,
    ZERO-VECTOR for VGGSound ticks, which have no real speech audio)
Missing-modality-as-zero mirrors the same single-stream-at-a-time principle
used throughout Phase 1/1a (no genuinely-paired data yet, per the Phase 2
scoping).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DecisionHeadConfig:
    world_state_dim: int = 1024
    speech_feat_dim: int = 1024
    hidden: int = 512
    dropout: float = 0.1


class SpeakSilenceHead(nn.Module):
    """Binary logit: positive = SPEAK, negative/low = SILENCE."""

    def __init__(self, cfg: DecisionHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.world_state_dim + cfg.speech_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, world_state: torch.Tensor, speech_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([world_state, speech_feat], dim=-1)
        return self.net(x).squeeze(-1)


class ThreeClassHead(nn.Module):
    """3-class extension (2026-07-25): SPEAK / SILENCE / BACKCHANNEL.
    Same input contract as SpeakSilenceHead ([World-State; speech-activity
    feature]), same architecture shape, just a 3-way softmax output instead
    of a binary sigmoid. Class order fixed as (silence=0, speak=1,
    backchannel=2) -- see LABEL_TO_IDX."""

    def __init__(self, cfg: DecisionHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.world_state_dim + cfg.speech_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 3),
        )

    def forward(self, world_state: torch.Tensor, speech_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([world_state, speech_feat], dim=-1)
        return self.net(x)   # (B, 3) logits


LABEL_TO_IDX = {"silence": 0, "speak": 1, "backchannel": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = DecisionHeadConfig()
    head = SpeakSilenceHead(cfg)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"SpeakSilenceHead params={n_params:,} ({n_params/1e6:.2f}M)")
    B = 5
    ws = torch.randn(B, cfg.world_state_dim)
    sf = torch.randn(B, cfg.speech_feat_dim)
    logits = head(ws, sf)
    print(f"output shape={tuple(logits.shape)}")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(B))
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
    print(f"backward OK, grad norm={gnorm:.3f}")
