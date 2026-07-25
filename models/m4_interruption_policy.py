"""models/m4_interruption_policy.py — post-halt interruption policy: what
happens after the duplex loop halts generation because of a real user
interruption (not a backchannel -- those are handled separately by the
3-class decision head, models/m4_decision_head.py, and never reach here).

Three outcomes, decided ONCE per resolved interruption (i.e. once the
interrupting user utterance itself ends and the tick loop's decision
returns to SILENCE again) -- not every tick:
  - RESUME: continue the SAME halted generation from its cached KV-state
    -- the interruption didn't materially change context (e.g. a short
    clarifying aside), so finishing the original utterance still makes
    sense.
  - RE-PLAN: discard the halted generation and its cache; re-run
    compute_world_state/compute_speech_activity + decide() fresh and, if
    still SPEAK, build a brand-new soft prompt and generate from scratch
    -- the interruption changed context (new question, topic shift), so
    the old partial utterance is no longer the right thing to say.
  - ABANDON: discard the halted generation, return to IDLE, and let the
    NEXT natural tick decide whether to speak -- the user has taken an
    extended turn and the robot shouldn't force its way back in.

── Grounding the thresholds in real EasyCom data (not invented) ───────────
scripts/m4_interruption_mining.py mined 1,796 genuine (non-backchannel)
overlapping-speech events across all 12 EasyCom sessions and classified
what the interrupted speaker did next, using real timing (ground truth) +
word-overlap similarity (a stated PROXY, not a verified intent label) for
RESUME vs RE-PLAN:

    RESUME     4.7%   median interrupter duration = 1.40s
    RE-PLAN   61.1%   median interrupter duration = 1.90s
    ABANDON   34.1%   median interrupter duration = 2.15s

This tells us two things this module leans on directly: (1) RESUME is the
rare exception, RE-PLAN is the default outcome of a real interruption --
the policy below is built so RE-PLAN is what you get when the other two
signals are ambiguous, matching the real prior; (2) interrupter duration
is a real but WEAK monotonic signal for abandonment (1.40 -> 1.90 -> 2.15s
is a small, noisy gap, not a clean separation) -- used as a soft
co-requirement for ABANDON, not a hard sole trigger.

The context-shift signal reuses models/m4_shift_trigger.py's EUCLIDEAN
metric, already selected on real data (scripts/m4_shift_metric_easycom.py,
n=4,168 real segment-pairs, ROC-AUC=0.614, Cohen's d=0.400) for the
adjacent task of detecting genuine speaker/turn transitions in this exact
Whisper speech-feature space. shift_tau defaults to the midpoint between
that study's measured within-speaker mean (17.65) and boundary/speaker-
change mean (19.66) -- i.e. this module's default threshold is read off an
already-verified real-data histogram, not chosen blind.

HONEST LIMITATION: no classifier was TRAINED on the RESUME/RE-PLAN/ABANDON
labels above -- they're a text-similarity proxy over a small, skewed mined
set, not a verified ground-truth intent label EasyCom actually provides.
This module is therefore a grounded HEURISTIC state machine, not a learned
policy; report it as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from models.m4_shift_trigger import ShiftTrigger, ShiftTriggerConfig


class InterruptionOutcome(Enum):
    RESUME = "resume"
    REPLAN = "re-plan"
    ABANDON = "abandon"


@dataclass
class InterruptionPolicyConfig:
    shift_metric: str = "euclidean"
    # midpoint of EasyCom's measured within-speaker (17.65) vs
    # boundary/speaker-change (19.66) euclidean means -- see module docstring
    shift_tau: float = 18.65
    # soft co-requirement for ABANDON, informed by mined medians
    # (RESUME=1.40s / RE-PLAN=1.90s / ABANDON=2.15s) -- weak signal, used
    # only in combination with a high shift score, never alone
    abandon_duration_sec: float = 2.0


class InterruptionPolicy:
    def __init__(self, cfg: InterruptionPolicyConfig):
        self.cfg = cfg
        self._shift = ShiftTrigger(ShiftTriggerConfig(metric=cfg.shift_metric, tau=None))

    def resolve(self, feat_at_halt: torch.Tensor, feat_at_resolve: torch.Tensor,
                interruption_duration_sec: float) -> InterruptionOutcome:
        """feat_at_halt / feat_at_resolve: speech-activity or World-State
        feature vectors (d,) captured at the moment generation halted and
        at the moment the interrupting speech ended, respectively."""
        shift = self._shift.score(feat_at_halt, feat_at_resolve)
        if shift >= self.cfg.shift_tau and interruption_duration_sec >= self.cfg.abandon_duration_sec:
            return InterruptionOutcome.ABANDON
        if shift < self.cfg.shift_tau:
            return InterruptionOutcome.RESUME
        return InterruptionOutcome.REPLAN


if __name__ == "__main__":
    torch.manual_seed(0)
    policy = InterruptionPolicy(InterruptionPolicyConfig())

    w = torch.randn(1024) * 5.0
    # near-identical context, short interruption -> RESUME
    w_same = w + torch.randn(1024) * 0.01
    out = policy.resolve(w, w_same, interruption_duration_sec=1.0)
    print(f"low shift, short duration -> {out}")
    assert out == InterruptionOutcome.RESUME

    # very different context, long interruption -> ABANDON
    w_far = torch.randn(1024) * 5.0
    out = policy.resolve(w, w_far, interruption_duration_sec=3.0)
    print(f"high shift, long duration -> {out}")
    assert out == InterruptionOutcome.ABANDON

    # very different context, but SHORT interruption -> RE-PLAN (context
    # changed, but not long enough to conclude the user took over the floor)
    out = policy.resolve(w, w_far, interruption_duration_sec=0.5)
    print(f"high shift, short duration -> {out}")
    assert out == InterruptionOutcome.REPLAN

    print("OK — all 3 transitions reachable and behave as designed")
