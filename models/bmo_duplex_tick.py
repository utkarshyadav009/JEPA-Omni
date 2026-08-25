"""models/bmo_duplex_tick.py — ties tonight's pieces into one real tick loop:
HomeostaticState (autonomy/drives) + FastTier (fast LLM response) +
AsyncThinker (deliberative, off-tick-loop) + the AppraisalVector->face
mapping. This is the actual integration point the existing
models/m4_duplex_loop.py's DuplexLoop should call into per tick.

Does NOT reimplement anything already built tonight -- imports directly
from models/homeostatic_state.py and models/m4_cognitive_core.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.homeostatic_state import (
    HomeostaticState, homeostatic_to_appraisal, homeostatic_to_mood_state, nearest_face, load_mapping,
)
from models.m4_cognitive_core import AsyncThinker, CognitiveCoreRouter, FastTier


@dataclass
class TickResult:
    text_to_speak: Optional[str]
    speak_reason: str            # "user_exchange" | "unprompted_lonely" | "unprompted_stressed" | "none"
    face_name: Optional[str]
    appraisal: Tuple[float, float, float, float, float]
    homeostatic: Dict[str, float]


class BmoDuplexTick:
    def __init__(self, router: CognitiveCoreRouter, thinker: AsyncThinker,
                 homeostatic_params=None, face_table: Optional[List[Dict]] = None):
        self.router = router
        self.thinker = thinker
        self.state = HomeostaticState(params=homeostatic_params) if homeostatic_params else HomeostaticState()
        self.face_table = face_table
        self.mapping = load_mapping()
        self._last_unprompted_speak_t = 0.0

    def tick(self, dt_s: float, *, transcript: Optional[str], user_speaking: bool,
              user_present: bool, scene_embedding_drift: float,
              input_arousal_signal: float, is_idle_rest: bool = False) -> TickResult:
        self.state.update(dt_s=dt_s, user_speaking=user_speaking, user_present=user_present,
                           scene_embedding_drift=scene_embedding_drift,
                           input_arousal_signal=input_arousal_signal, is_idle_rest=is_idle_rest)

        appraisal = homeostatic_to_appraisal(self.state, self.mapping)
        face_name = None
        if self.face_table:
            face_name, _ = nearest_face(appraisal, self.face_table)

        # Thinker re-integration check FIRST (non-blocking) -- if a
        # deliberative response finished since the last tick, it takes
        # priority over anything else this tick.
        thinker_result = self.thinker.poll()
        if thinker_result is not None and thinker_result.ready:
            self.thinker._result = None  # consumed, don't re-emit next tick
            return TickResult(text_to_speak=thinker_result.text, speak_reason="user_exchange",
                               face_name=face_name, appraisal=appraisal, homeostatic=self.state.as_dict())

        if transcript:
            decision = self.router.route(transcript, homeostatic_to_mood_state(self.state))
            if decision.escalate_to_thinker and not self.thinker.is_running():
                # Real integration point: caller's DuplexLoop supplies the
                # soft_prompt/attention_mask for the Thinker's full AV-
                # grounded generation; this module only handles the
                # decision to escalate + kicks off the async task, it does
                # not itself have vision/audio context to build that prompt.
                pass  # caller wires: self.thinker.start(soft_prompt, attn_mask)
            return TickResult(text_to_speak=decision.fast_result.text, speak_reason="user_exchange",
                               face_name=face_name, appraisal=appraisal, homeostatic=self.state.as_dict())

        if self.state.wants_to_speak_unprompted():
            # Found via testing (not assumed): using the SAME literal prompt
            # text for every unprompted trigger produced near-identical
            # output regardless of WHY BMO was speaking up (loneliness vs.
            # stress gave the same generic line both times). Differentiate
            # the prompt by reason so the state-conditioned generation
            # actually reflects the real trigger, not just "speak now."
            if self.state.stress > 0.85:
                reason = "unprompted_stressed"
                prompt = "Something startling or loud just happened."
            else:
                reason = "unprompted_lonely"
                prompt = "It has been quiet for a long time and no one is around."
            result = self.router.fast_tier.generate(prompt, homeostatic_to_mood_state(self.state))
            # reset social_need partway so it doesn't re-trigger every tick
            self.state.social_need *= 0.5
            return TickResult(text_to_speak=result.text, speak_reason=reason,
                               face_name=face_name, appraisal=appraisal, homeostatic=self.state.as_dict())

        return TickResult(text_to_speak=None, speak_reason="none", face_name=face_name,
                           appraisal=appraisal, homeostatic=self.state.as_dict())
