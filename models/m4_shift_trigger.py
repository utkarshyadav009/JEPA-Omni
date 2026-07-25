"""models/m4_shift_trigger.py — M4 proactive-speech trigger: World-State
shift detection.

DECISION STATUS (2026-07-25): SELECTED ON REAL DATA. The earlier VGGSound
clip-transition proxy (scripts/m4_shift_metric_eval.py) found all three
metrics near-ceiling (ROC-AUC >= 0.9999) -- too coarse to discriminate
(whole different clips are trivially far apart). scripts/
m4_shift_metric_easycom.py re-ran the comparison on REAL graded turn
transitions: 4,168 consecutive EasyCom speech-segment pairs, boundary =
genuine speaker change (Participant_ID differs), within = same speaker
continuing, scored in Whisper speech-activity feature space (World-State
doesn't exist for EasyCom -- no M2/vision features). Result:

    metric        ROC-AUC   Cohen's d
    euclidean     0.614     0.400
    cosine        0.612     0.363
    mahalanobis   0.497     -0.004   <- essentially CHANCE

Mahalanobis -- the theoretically-motivated default this module previously
carried, reasoning from the M2 World-State's measured anisotropy
(effective_rank=37.71/1024, see checkpoints/m2_fusion_20k_best/
PROVENANCE.txt) -- performs at chance on the task that actually matters.
Likely explanation: whitening equalizes the natural high-variance
directions (which happen to carry most of the speaker-identity signal
Euclidean already exploits) against low-variance directions (mostly
linguistic-content noise for a mean-pooled utterance embedding), diluting
the exact signal a turn-boundary detector needs. The theoretical argument
for Mahalanobis was reasonable but wrong for this task -- it made a
testable prediction and the test failed it.

Default is now EUCLIDEAN, selected on this evidence (highest AUC and
Cohen's d of the three, and the simplest/cheapest to compute). Honest
caveat: AUC=0.614 is real but modest -- mean-pooling a whole utterance's
Whisper hidden states discards most speaker-specific acoustic detail
(pitch, timbre); a dedicated speaker-embedding model would likely do
better, but that's out of scope here. All three metrics stay implemented
and selectable via ShiftTriggerConfig.metric for future re-evaluation
(e.g. once Ego4D LAM data or a dedicated speaker embedding is available).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ShiftTriggerConfig:
    metric: str = "euclidean"            # "euclidean" | "cosine" | "mahalanobis" -- see module docstring
    tau: Optional[float] = None          # trigger threshold; None = score-only, caller decides
    pca_basis_path: str = "checkpoints/m4/shift_metric_pca_basis.pt"   # only used for "mahalanobis"


class ShiftTrigger:
    """Stateless scorer + stateful reference-tracking trigger.

    Usage:
        trigger = ShiftTrigger(ShiftTriggerConfig(metric="mahalanobis", tau=5.0))
        for w_t in world_state_stream:
            if trigger.step(w_t):
                ... run the speak/silence decision ...
    """

    def __init__(self, cfg: ShiftTriggerConfig):
        if cfg.metric not in ("euclidean", "cosine", "mahalanobis"):
            raise ValueError(f"unknown shift metric {cfg.metric!r}")
        self.cfg = cfg
        self._basis = None
        if cfg.metric == "mahalanobis":
            self._basis = torch.load(cfg.pca_basis_path, map_location="cpu")
        self.w_ref: Optional[Tensor] = None

    def _whiten(self, w: Tensor) -> Tensor:
        mu = self._basis["mu"]
        components = self._basis["components"]      # (K, d)
        whiten_scale = self._basis["whiten_scale"]   # (K,)
        x = (w.float() - mu)
        proj = components @ x
        return proj * whiten_scale

    def score(self, w_prev: Tensor, w_curr: Tensor) -> float:
        """Shift score between two World-State vectors (d_model,). Higher = more shift."""
        if self.cfg.metric == "euclidean":
            return float(torch.norm(w_curr.float() - w_prev.float()).item())
        if self.cfg.metric == "cosine":
            sim = F.cosine_similarity(w_prev.float().unsqueeze(0), w_curr.float().unsqueeze(0)).item()
            return 1.0 - sim
        # mahalanobis
        m_prev = self._whiten(w_prev)
        m_curr = self._whiten(w_curr)
        return float(torch.norm(m_curr - m_prev).item())

    def step(self, w_t: Tensor) -> bool:
        """Update internal reference and return True if a trigger fires
        (shift(w_t, w_ref) > tau). On the first call (no reference yet) or
        on any trigger, w_ref resets to w_t -- see M4 design doc: reset
        regardless of whether the model decided to speak, so a persistent-
        but-static state doesn't re-trigger every tick."""
        if self.w_ref is None:
            self.w_ref = w_t.clone()
            return False
        s = self.score(self.w_ref, w_t)
        fired = self.cfg.tau is not None and s > self.cfg.tau
        if fired:
            self.w_ref = w_t.clone()
        return fired


if __name__ == "__main__":
    # CPU smoke test: mechanics only, tiny random basis substitute isn't
    # available (mahalanobis needs the real saved basis) -- exercise
    # euclidean/cosine directly, and mahalanobis only if the basis file
    # exists (produced by scripts/m4_shift_metric_eval.py).
    import os

    torch.manual_seed(0)
    w1, w2, w3 = torch.randn(1024), torch.randn(1024), torch.randn(1024)
    for metric in ("euclidean", "cosine"):
        t = ShiftTrigger(ShiftTriggerConfig(metric=metric, tau=0.0))
        s = t.score(w1, w2)
        print(f"{metric}: score(w1,w2)={s:.4f}")
    basis_path = ShiftTriggerConfig().pca_basis_path
    if os.path.isfile(basis_path):
        t = ShiftTrigger(ShiftTriggerConfig(metric="mahalanobis", tau=0.0))
        s = t.score(w1, w2)
        print(f"mahalanobis: score(w1,w2)={s:.4f}")
    else:
        print(f"mahalanobis: skipped, {basis_path} not found yet")
    print("OK")
