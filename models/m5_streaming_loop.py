"""models/m5_streaming_loop.py — M5 Day 1: the real streaming milestone.

V-JEPA2 and WavJEPA are BIDIRECTIONAL transformers -- unlike the LLM there
is no KV-cache and no way to feed them one new frame/sample at a time and
get an incremental update. Every "refresh" is a full forward pass over a
window of recent frames/samples. The core streaming design question is
choosing that window's length and how often (stride) to re-run it --
re-running too often wastes compute (ViT-L forward alone measured 2.43s on
the Jetson at max clocks, Phase 0), too rarely means the World-State lags
real events. This module makes that choice explicit and configurable
rather than implicit.

WINDOW_VISION_SEC = 10.0, matching CLIP_DURATION_S in
scripts/extract_features_av.py -- the real-world duration a 64-frame
V-JEPA2 input was TRAINED to represent (64 frames uniformly sampled across
a 10s clip). A shorter window puts ViT-L outside its trained input
distribution; not an arbitrary pick.

STRIDE_VISION_SEC = 2.0 (default, tunable -- Day 3 profiles this
tradeoff): re-encode vision 5x within one window's span rather than every
tick. Between refreshes the loop reuses the most recently computed
World-State (stale-but-cheap), matching the duplex loop's existing
principle that perception refresh is decoupled from generation/tick
cadence (models/m4_duplex_loop.py's docstring).

WINDOW_AUDIO_SEC = 2.0, matching EasyCom's SILENCE_WINDOW_SEC
(data/m4_easycom_turntaking.py) -- audio (WavJEPA + Whisper) is far
cheaper per forward pass than ViT-L, so it is refreshed every tick, not
strided.

TICK_INTERVAL_SEC = 0.25 -- the decision head is a ~1.3M-param MLP
(microseconds), so it polls far more often than perception refreshes.

Echo/self-interruption handling reuses the project's own measured
conclusion (models/m4_echo_cancellation.py): MIC GATING is the safe
default (0% false-interruption, measured), NOT AEC alone (measured
insufficient -- its comfort-noise floor resembles real quiet speech to the
current decision head). No real TTS engine exists in this repo; "TTS
output" is simulated the same way scripts/m4_echo_test.py already
established for the echo-cancellation gate -- a real generated-text token
count converted to an estimated speech duration via a standard words-per-
minute rate, during which MicGate.is_playing=True gates the mic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional
from collections import deque

import numpy as np
import torch

from models.m4_duplex_loop import DuplexLoop, GenerationResult
from models.m4_echo_cancellation import MicGate
from models.m4_interruption_policy import InterruptionPolicy, InterruptionOutcome

TTS_WORDS_PER_MINUTE = 150.0   # standard speech-synthesis rate estimate


@dataclass
class StreamingConfig:
    window_vision_sec: float = 10.0
    stride_vision_sec: float = 2.0
    window_audio_sec: float = 2.0
    tick_interval_sec: float = 0.25
    audio_sr: int = 16000
    video_fps: float = 6.4        # 64 frames / 10s -- matches CLIP_DURATION_S


class RollingAudioBuffer:
    def __init__(self, window_sec: float, sr: int):
        self.window_sec = window_sec
        self.sr = sr
        self.max_samples = int(window_sec * sr)
        self._buf: Deque[np.ndarray] = deque()
        self._n = 0

    def push(self, chunk: np.ndarray) -> None:
        self._buf.append(chunk)
        self._n += len(chunk)
        while self._n - len(self._buf[0]) >= self.max_samples:
            popped = self._buf.popleft()
            self._n -= len(popped)

    def get_window(self) -> Optional[np.ndarray]:
        if self._n == 0:
            return None
        full = np.concatenate(list(self._buf))
        return full[-self.max_samples:] if len(full) > self.max_samples else full


class RollingVideoBuffer:
    """Holds raw (C,H,W) frames; get_window() uniformly samples 64 frames
    from whatever's currently buffered (same _uniform_frame_indices
    pattern as scripts/extract_features_av.py, so a partially-filled
    buffer still produces a valid V-JEPA2 input, not an error)."""

    def __init__(self, window_sec: float, fps: float, n_frames_out: int = 64):
        self.window_sec = window_sec
        self.max_frames = int(window_sec * fps)
        self.n_frames_out = n_frames_out
        self._buf: Deque[torch.Tensor] = deque(maxlen=self.max_frames)

    def push(self, frame: torch.Tensor) -> None:
        self._buf.append(frame)

    def get_window(self) -> Optional[torch.Tensor]:
        if len(self._buf) == 0:
            return None
        frames = list(self._buf)
        n = len(frames)
        if n < self.n_frames_out:
            idx = np.linspace(0, n - 1, self.n_frames_out).round().astype(int)
        else:
            idx = np.linspace(0, n - 1, self.n_frames_out).round().astype(int)
        return torch.stack([frames[i] for i in idx], dim=0)


@dataclass
class TickLog:
    t: float
    action: str                    # "silence" | "speak" | "backchannel" | "gated" | "interrupted"
    vision_refreshed: bool
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    decision_label: Optional[str] = None
    decision_probs: Optional[Dict[str, float]] = None
    generation_text: Optional[str] = None


class StreamingLoop:
    """Wraps DuplexLoop with rolling AV buffers, strided perception
    refresh, mic gating during simulated TTS playback, and the
    interruption policy state machine."""

    def __init__(self, duplex_loop: DuplexLoop, cfg: StreamingConfig,
                 interruption_policy: Optional[InterruptionPolicy] = None,
                 vision_encoder=None, max_tdm_bins: int = 512):
        self.loop = duplex_loop
        self.cfg = cfg
        self.mic_gate = MicGate()
        self.interruption_policy = interruption_policy
        self.vision_encoder = vision_encoder   # models.vision_encoder.VisionEncoder -- REAL ViT-L, not dummy features
        self.max_tdm_bins = max_tdm_bins
        self.video_buf = RollingVideoBuffer(cfg.window_vision_sec, cfg.video_fps)
        self.audio_buf = RollingAudioBuffer(cfg.window_audio_sec, cfg.audio_sr)
        self._last_vision_refresh_t: Optional[float] = None
        self._cached_world_state: Optional[torch.Tensor] = None
        self._tts_playing_until: Optional[float] = None
        self._halted_generation: Optional[GenerationResult] = None
        self._halted_soft_prompt = None
        self._halted_attn = None
        self.logs: List[TickLog] = []

    def ingest_video_frame(self, frame: torch.Tensor) -> None:
        self.video_buf.push(frame)

    def ingest_audio_chunk(self, chunk: np.ndarray) -> None:
        self.audio_buf.push(chunk)

    def _maybe_refresh_vision(self, t: float) -> tuple:
        """Returns (world_state, refreshed: bool, latencies: dict). Runs
        the REAL V-JEPA2 ViT-L forward pass (self.vision_encoder.encode)
        over the current rolling window, THEN the M2 fusion predictor's
        encode_world_state on top of those real vision tokens -- both
        stages timed separately so the latency breakdown distinguishes
        "the vision encoder itself" from "the fusion predictor.\""""
        need_refresh = (self._last_vision_refresh_t is None or
                         t - self._last_vision_refresh_t >= self.cfg.stride_vision_sec)
        window = self.video_buf.get_window()
        if not need_refresh or window is None or self.vision_encoder is None:
            return self._cached_world_state, False, {}

        t0 = time.perf_counter()
        with torch.no_grad():
            vision_feats = self.vision_encoder.encode(window.unsqueeze(0))   # REAL ViT-L 64f forward -> (1, N, 1024)
        vitl_lat = (time.perf_counter() - t0) * 1000.0

        n_tok = vision_feats.shape[1]
        bin_idx = torch.linspace(0, self.max_tdm_bins - 1, n_tok, device=vision_feats.device).round().long()
        feats = {"vision": vision_feats.float()}
        tbins = {"vision": bin_idx.unsqueeze(0)}

        t0 = time.perf_counter()
        ws = self.loop.compute_world_state(feats, tbins)
        fusion_lat = (time.perf_counter() - t0) * 1000.0

        self._cached_world_state = ws
        self._last_vision_refresh_t = t
        return ws, True, {"vitl_forward_ms": vitl_lat, "fusion_predictor_ms": fusion_lat}

    def _estimate_tts_duration_sec(self, text: str) -> float:
        n_words = max(1, len(text.split()))
        return n_words / TTS_WORDS_PER_MINUTE * 60.0

    def tick(self, t: float,
             speech_waveform: Optional[np.ndarray] = None, speech_dur_sec: Optional[float] = None,
             generate_fn: Optional[Callable] = None) -> TickLog:
        """One tick. `generate_fn() -> GenerationResult` is caller-supplied
        so the streaming loop doesn't hardcode which connector(s) built the
        soft prompt (M3/M4b/both).

        Vision refresh runs REGARDLESS of mic-gating -- gating exists to
        stop the robot's own TTS output from re-entering the AUDIO
        decision path (self-interruption); the camera doesn't "hear" the
        robot's own voice, so there's no reason to also freeze World-State
        updates during TTS playback. Only audio/decision/generation are
        gated."""
        latencies: Dict[str, float] = {}

        # TTS playback bookkeeping (simulated -- see module docstring)
        if self._tts_playing_until is not None and t >= self._tts_playing_until:
            self.mic_gate.is_playing = False
            self._tts_playing_until = None

        ws, refreshed, vision_lats = self._maybe_refresh_vision(t)
        latencies.update(vision_lats)

        if not self.mic_gate.should_run_decision():
            log = TickLog(t=t, action="gated", vision_refreshed=refreshed, latencies_ms=latencies)
            self.logs.append(log)
            return log

        sf = None
        if speech_waveform is not None:
            t0 = time.perf_counter()
            sf, _ = self.loop.compute_speech_activity(speech_waveform, speech_dur_sec)
            latencies["speech_activity_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        label, probs = self.loop.decide3(ws, sf)
        latencies["decision_ms"] = (time.perf_counter() - t0) * 1000.0

        log = TickLog(t=t, action=label, vision_refreshed=refreshed, latencies_ms=latencies,
                       decision_label=label, decision_probs=probs)

        # currently-generating and a real interruption fires -> halt, run policy
        if self._halted_generation is None and label == "speak" and generate_fn is not None:
            t0 = time.perf_counter()
            result = generate_fn()
            latencies["generation_ms"] = (time.perf_counter() - t0) * 1000.0
            log.generation_text = result.text
            # start simulated TTS playback + mic gate
            dur = self._estimate_tts_duration_sec(result.text)
            self.mic_gate.is_playing = True
            self._tts_playing_until = t + dur
            log.latencies_ms = latencies

        self.logs.append(log)
        return log
