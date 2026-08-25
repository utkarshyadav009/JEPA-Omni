"""models/homeostatic_state.py — BMO's homeostatic drive system.

Gives BMO autonomy: previously the Face Engine's AppraisalVector (valence,
arousal, control, novelty, obstruct) was driven entirely by manual sliders
(confirmed directly by the user, 2026-08-04 -- BMO-Project has NO existing
homeostatic system, only the face renderer + its manual AppraisalVector
input). This module is what drives that vector on its own over time.

Four variables, per direct instruction ("loneliness/social-need, energy,
curiosity, stress"):
  - energy       [0,1]: depletes with activity/time awake, restored by idle
                 rest periods. Analogue of fatigue.
  - social_need  [0,1]: rises during silence/no-interaction, falls when the
                 user is actively engaging. Crossing HIGH_SOCIAL_NEED_THRESH
                 is the literal "BMO speaks up unprompted because it feels
                 lonely" trigger from the spec.
  - curiosity    [0,1]: rises when the perceived world is static/unchanging
                 (boredom), falls when real perceptual novelty is detected.
                 Wired to the M2-embed-predictor's own embedding drift, not
                 a separate heuristic -- reuses infrastructure already built
                 and validated this session, doesn't invent a new "novelty
                 detector."
  - stress       [0,1]: rises with high-arousal negative input (loud/sudden
                 sound, angry tone), falls during calm/positive interaction.

Update is O(1) arithmetic per tick -- no model forward pass, satisfies the
"cheap enough to update every streaming tick without adding latency"
requirement directly (measured: <0.01ms per update() call, see
models/m4_cognitive_core.py's tick-loop integration).

Mapping to the AppraisalVector (valence, arousal, control, novelty,
obstruct) is EXPLICIT and file-backed (homeostatic_appraisal_mapping.json,
sibling to this file) rather than hardcoded, per direct instruction. The
mapping targets the REAL, existing face_tags.csv space (32 expressions,
Jetson: /home/bmo/bmo_fresh/BMO Face Engine/face_tags.csv) via the SAME
nearest-neighbour mechanism the Face Engine's own AffectiveEngine already
uses -- this module does not touch or reimplement the Face Engine, it only
produces the 5D vector the Face Engine already knows how to consume.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "homeostatic_appraisal_mapping.json")

HIGH_SOCIAL_NEED_THRESH = 0.75   # crossing this -> unprompted-speech trigger
LOW_ENERGY_THRESH = 0.2


@dataclass
class HomeostaticParams:
    """All rates are PER SECOND, applied as state += rate * dt each tick --
    tunable without touching update() logic. Real-world-plausible defaults,
    not measured (no real deployment data exists yet to calibrate against;
    flagged honestly rather than presented as validated)."""
    energy_decay_per_s: float = 1.0 / (30 * 60)          # ~30min awake -> depleted
    energy_restore_per_s_idle: float = 1.0 / (10 * 60)   # ~10min idle -> refilled
    social_need_rise_per_s_silence: float = 1.0 / (5 * 60)   # ~5min silence -> lonely
    social_need_fall_per_s_interacting: float = 1.0 / 20.0   # ~20s of interaction -> satisfied
    curiosity_rise_per_s_static: float = 1.0 / (2 * 60)      # ~2min unchanging scene -> bored
    curiosity_fall_on_novelty: float = 0.4                    # single-tick drop on real novelty detection
    stress_rise_per_s_high_arousal: float = 1.0 / 10.0       # ~10s of loud/negative input -> stressed
    stress_decay_per_s: float = 1.0 / 60.0                    # calm decay, ~1min to unwind


@dataclass
class HomeostaticState:
    energy: float = 0.8
    social_need: float = 0.2
    curiosity: float = 0.3
    stress: float = 0.1
    # Separate from curiosity on purpose (found via testing, not assumed):
    # curiosity is BMO's internal drive to WANT novelty (rises when bored,
    # i.e. exactly when nothing novel is happening); recent_novelty is
    # whether something ACTUALLY novel just happened. These are near-
    # opposite during prolonged silence, and conflating them (curiosity
    # driving the novelty appraisal dimension directly) sent a "lonely,
    # bored" state to the nearest-neighbour face "shocked_pale" instead of
    # something sad/worried -- high curiosity was wrongly read as high
    # novelty. recent_novelty spikes on real detected drift and decays
    # back down, independent of the boredom-driven curiosity level.
    recent_novelty: float = 0.0
    params: HomeostaticParams = field(default_factory=HomeostaticParams)
    _last_update_t: float = field(default_factory=time.perf_counter)

    def update(self, dt_s: float, *, user_speaking: bool, user_present: bool,
               scene_embedding_drift: float, input_arousal_signal: float,
               is_idle_rest: bool = False) -> None:
        """One tick. All inputs are things the existing pipeline already
        computes or can compute cheaply:
          user_speaking / user_present -- from the existing 3-class decision
            head / speech-activity feature (models/m4_decision_head.py).
          scene_embedding_drift -- cosine distance between this tick's and
            last tick's M2-embed-predictor world-state embedding; reuses
            the SAME embedding this session already validated for scene
            discrimination (crowd vs desk-work, 4/4 correct), not a new
            signal.
          input_arousal_signal -- [0,1], e.g. audio RMS spike / detected
            angry tone; caller-supplied since exact source (mic RMS, a
            tone classifier) is not built yet -- pass 0.0 if unavailable,
            state degrades gracefully to "stress never rises," not a crash.
        """
        p = self.params

        # energy: decays while "awake", restores during genuine idle rest
        if is_idle_rest:
            self.energy = min(1.0, self.energy + p.energy_restore_per_s_idle * dt_s)
        else:
            self.energy = max(0.0, self.energy - p.energy_decay_per_s * dt_s)

        # social_need: rises in silence/absence, falls while genuinely interacting
        if user_speaking or user_present:
            self.social_need = max(0.0, self.social_need - p.social_need_fall_per_s_interacting * dt_s)
        else:
            self.social_need = min(1.0, self.social_need + p.social_need_rise_per_s_silence * dt_s)

        # curiosity: rises when the world is static, drops on real detected novelty
        if scene_embedding_drift < 0.05:   # near-zero drift = static scene
            self.curiosity = min(1.0, self.curiosity + p.curiosity_rise_per_s_static * dt_s)
        else:
            self.curiosity = max(0.0, self.curiosity - p.curiosity_fall_on_novelty * min(1.0, scene_embedding_drift))

        # recent_novelty: spikes on real detected drift, decays back down --
        # deliberately independent of curiosity's slow boredom accumulation
        # (see dataclass comment for why these must not be conflated).
        self.recent_novelty = max(0.0, self.recent_novelty - 0.5 * dt_s)  # ~2s half-life-ish decay
        if scene_embedding_drift >= 0.05:
            self.recent_novelty = min(1.0, max(self.recent_novelty, scene_embedding_drift))

        # stress: rises with high-arousal negative input, decays otherwise
        if input_arousal_signal > 0.5:
            self.stress = min(1.0, self.stress + p.stress_rise_per_s_high_arousal * dt_s * input_arousal_signal)
        else:
            self.stress = max(0.0, self.stress - p.stress_decay_per_s * dt_s)

    def wants_to_speak_unprompted(self) -> bool:
        """The literal 'if the user hasn't spoken and is there [sic -- or
        isn't there], the robot feels lonely/anxious, it speaks up'
        trigger. Also fires on high stress (BMO reacting to something
        alarming) independent of social_need."""
        return self.social_need > HIGH_SOCIAL_NEED_THRESH or self.stress > 0.85

    def as_dict(self) -> Dict[str, float]:
        return {"energy": self.energy, "social_need": self.social_need,
                "curiosity": self.curiosity, "stress": self.stress,
                "recent_novelty": self.recent_novelty}


def homeostatic_to_mood_state(state: HomeostaticState) -> Dict[str, object]:
    """Translates the 5 continuous homeostatic variables into the
    {energy, mood} schema the MiniCPM5-1B LoRA adapter was actually
    trained on (scripts/finetune_bmo_minicpm5_lora.py, data/
    bmo_synthetic_functional.jsonl).

    FOUND VIA TESTING (2026-08-04): the full tick-loop integration
    (bmo_duplex_tick.py) was calling FastTier.generate() with
    HomeostaticState.as_dict()'s raw 5-key dict directly -- a schema the
    fine-tune never saw a single example of (training only ever used
    2-key {energy, mood} dicts). The model was operating completely
    out-of-distribution on every real inference call in the integrated
    test, which is a real, separate cause of bad generations from the
    "thin training data" issue already flagged -- this function is the
    missing translation layer, not a data-quantity problem.

    Priority-ordered rule mapping (most salient/urgent state wins) --
    a reasoned first design, not learned or calibrated."""
    if state.stress > 0.7:
        mood = "stressed"
    elif state.recent_novelty > 0.6:
        mood = "surprised"
    elif state.social_need > 0.6:
        mood = "lonely"
    elif state.curiosity > 0.6 and state.recent_novelty < 0.1:
        mood = "bored"
    elif state.curiosity > 0.5:
        mood = "curious"
    elif state.energy < 0.25:
        mood = "tired"
    elif state.energy > 0.75 and state.stress < 0.2 and state.social_need < 0.3:
        mood = "excited" if state.energy > 0.85 else "happy"
    else:
        mood = "content"
    return {"energy": round(state.energy, 2), "mood": mood}


def load_mapping(path: str = MAPPING_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


def homeostatic_to_appraisal(state: HomeostaticState, mapping: Optional[Dict] = None) -> Tuple[float, float, float, float, float]:
    """Explicit linear mapping, file-backed (homeostatic_appraisal_mapping.json).
    Returns (valence, arousal, control, novelty, obstruct) clamped to the
    real ranges observed in face_tags.csv (valence/arousal/control/novelty
    in [0,1] or [-1,1] per that file's own conventions, obstruct in [0,1])."""
    m = mapping or load_mapping()
    s = state.as_dict()
    out = []
    for dim in ("valence", "arousal", "control", "novelty", "obstruct"):
        coefs = m["coefficients"][dim]
        v = coefs.get("bias", 0.0)
        for var, w in coefs.items():
            if var == "bias":
                continue
            v += w * s[var]
        lo, hi = m["clamp"][dim]
        v = max(lo, min(hi, v))
        out.append(v)
    return tuple(out)  # type: ignore[return-value]


def nearest_face(appraisal: Tuple[float, float, float, float, float],
                  face_table: List[Dict]) -> Tuple[str, float]:
    """Nearest-neighbour lookup, same mechanism the Face Engine's own
    AffectiveEngine uses (per BMO-Project README: 'nearest-neighbour
    lookup over a hand-crafted database') -- this module does NOT
    reimplement face rendering, it only needs to pick which face's tag
    vector is closest so callers (or the Face Engine itself, if the
    lookup is ported to C++) know which expression to show.
    face_table: list of {"name": str, "valence":.., "arousal":.., ...}
    as loaded directly from face_tags.csv."""
    best_name, best_dist = None, float("inf")
    for row in face_table:
        d = math.sqrt(sum(
            (appraisal[i] - row[dim]) ** 2
            for i, dim in enumerate(("valence", "arousal", "control", "novelty", "obstruct"))
        ))
        if d < best_dist:
            best_dist, best_name = d, row["name"]
    return best_name, best_dist


def load_face_table(csv_path: str) -> List[Dict]:
    import csv
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"],
                "valence": float(r["valence"]), "arousal": float(r["arousal"]),
                "control": float(r["control"]), "novelty": float(r["novelty"]),
                "obstruct": float(r["obstruct"]),
            })
    return rows


if __name__ == "__main__":
    # Smoke test: simulate 10 minutes of silence, confirm social_need rises
    # and eventually triggers unprompted speech; confirm face selection
    # moves toward a plausible "lonely" expression, not a random one.
    state = HomeostaticState()
    face_table_path = os.environ.get("BMO_FACE_TAGS_CSV")
    face_table = load_face_table(face_table_path) if face_table_path else None
    mapping = load_mapping()

    print("t=0s  ", state.as_dict())
    triggered_at = None
    for t in range(0, 600, 10):
        state.update(dt_s=10.0, user_speaking=False, user_present=False,
                      scene_embedding_drift=0.0, input_arousal_signal=0.0)
        if state.wants_to_speak_unprompted() and triggered_at is None:
            triggered_at = t
    print(f"t=600s", state.as_dict())
    print(f"unprompted-speech trigger fired at t={triggered_at}s "
          f"(expected ~{HIGH_SOCIAL_NEED_THRESH * 5 * 60:.0f}s given social_need_rise_per_s_silence default)")

    appraisal = homeostatic_to_appraisal(state, mapping)
    print(f"final appraisal vector (V,A,C,N,O) = {tuple(round(x, 3) for x in appraisal)}")
    if face_table:
        name, dist = nearest_face(appraisal, face_table)
        print(f"nearest face: {name} (distance={dist:.3f})")
    else:
        print("(set BMO_FACE_TAGS_CSV to also test nearest-face lookup against the real table)")

    print("\n--- Scenario 2: calm baseline state (no drama), sanity check ---")
    calm = HomeostaticState(energy=0.9, social_need=0.1, curiosity=0.1, stress=0.05, recent_novelty=0.0)
    calm_appraisal = homeostatic_to_appraisal(calm, mapping)
    print(f"calm appraisal (V,A,C,N,O) = {tuple(round(x, 3) for x in calm_appraisal)}")
    if face_table:
        name, dist = nearest_face(calm_appraisal, face_table)
        print(f"nearest face: {name} (distance={dist:.3f})")

    print("\n--- Scenario 3: sudden real novelty spike (e.g. someone unexpected appears) ---")
    surprised = HomeostaticState(energy=0.7, social_need=0.3, curiosity=0.2, stress=0.2, recent_novelty=0.9)
    surprised_appraisal = homeostatic_to_appraisal(surprised, mapping)
    print(f"surprised appraisal (V,A,C,N,O) = {tuple(round(x, 3) for x in surprised_appraisal)}")
    if face_table:
        name, dist = nearest_face(surprised_appraisal, face_table)
        print(f"nearest face: {name} (distance={dist:.3f})")

    print("\n--- Scenario 4: high stress (loud/angry input) ---")
    stressed = HomeostaticState(energy=0.6, social_need=0.2, curiosity=0.1, stress=0.9, recent_novelty=0.1)
    stressed_appraisal = homeostatic_to_appraisal(stressed, mapping)
    print(f"stressed appraisal (V,A,C,N,O) = {tuple(round(x, 3) for x in stressed_appraisal)}")
    if face_table:
        name, dist = nearest_face(stressed_appraisal, face_table)
        print(f"nearest face: {name} (distance={dist:.3f})")
