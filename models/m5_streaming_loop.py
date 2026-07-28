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

import threading
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
    # LOCKED (2026-07-26): stride=window=10.0s + priority decision stream is
    # the disclosed operating point (p95=786ms, mean=372ms, duty 37-41%,
    # checkpoints/vjepa21_shelved/JETSON_PHASE4_2_3_RESULTS.json). VAD-on-CPU
    # (54.57ms, JETSON_VAD_CPU_RESULTS.json) already makes interruption
    # independent of GPU contention -- the hard constraint is met. No further
    # latency tuning against this config; opportunistic refresh
    # (start_vision_refresh_thread_opportunistic) remains available but is
    # NOT the default and must be opted into explicitly.
    window_vision_sec: float = 10.0
    stride_vision_sec: float = 10.0
    window_audio_sec: float = 2.0      # Whisper/speech-activity window (decision path) -- unchanged
    window_ambient_sec: float = 10.0   # NEW (2026-07-26): WavJEPA/M2-ambient window, matches
                                        # window_vision_sec/CLIP_DURATION_S -- a SEPARATE buffer
                                        # from window_audio_sec, different consumer (M2 fusion vs
                                        # Whisper/decision head), do not repurpose one for the other
    tick_interval_sec: float = 0.25
    audio_sr: int = 16000
    video_fps: float = 6.4        # 64 frames / 10s -- matches CLIP_DURATION_S
    use_priority_decision_stream: bool = True   # LOCKED default (item 3): p95 786ms vs 1220ms
                                                 # without it (JETSON_PHASE4_2_3_RESULTS.json 4.2 vs 4.3)


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
    overlapped_vision_forward: bool = False   # item 4 root-cause instrumentation: was a vision/
                                               # ambient encoder forward pass in flight (on the
                                               # vision thread) at the moment this tick ran?
    decision_label: Optional[str] = None
    decision_probs: Optional[Dict[str, float]] = None
    generation_text: Optional[str] = None


class StreamingLoop:
    """Wraps DuplexLoop with rolling AV buffers, strided perception
    refresh, mic gating during simulated TTS playback, and the
    interruption policy state machine."""

    def __init__(self, duplex_loop: DuplexLoop, cfg: StreamingConfig,
                 interruption_policy: Optional[InterruptionPolicy] = None,
                 vision_encoder=None, max_tdm_bins: int = 512,
                 ambient_base_encoder=None, ambient_nat_encoder=None):
        self.loop = duplex_loop
        self.cfg = cfg
        self.mic_gate = MicGate()
        self.interruption_policy = interruption_policy
        self.vision_encoder = vision_encoder   # models.vision_encoder.VisionEncoder -- REAL ViT-L, not dummy features
        # NEW (2026-07-26, item 0 fix): WavJEPA-base/nat -- M2's actual audio
        # encoders. Whisper (via compute_speech_activity) is a DIFFERENT
        # pathway feeding the decision head's speech-activity feature, not
        # World-State; these two were previously conflated (ambient was
        # simply absent). None-able for back-compat with any caller not
        # yet passing them (vision-only World-State, degraded but no crash).
        self.ambient_base_encoder = ambient_base_encoder
        self.ambient_nat_encoder = ambient_nat_encoder
        self.max_tdm_bins = max_tdm_bins
        self.video_buf = RollingVideoBuffer(cfg.window_vision_sec, cfg.video_fps)
        self.audio_buf = RollingAudioBuffer(cfg.window_audio_sec, cfg.audio_sr)              # Whisper/decision path
        self.ambient_buf = RollingAudioBuffer(cfg.window_ambient_sec, cfg.audio_sr)           # WavJEPA/World-State path -- separate buffer, separate consumer
        self._last_vision_refresh_t: Optional[float] = None
        self._cached_world_state: Optional[torch.Tensor] = None
        self._tts_playing_until: Optional[float] = None
        self._halted_generation: Optional[GenerationResult] = None
        self._halted_soft_prompt = None
        self._halted_attn = None
        self.logs: List[TickLog] = []
        # LOCKED default (item 3, 2026-07-26): decision path runs on a
        # priority CUDA stream so its kernels interleave at ViT-L's kernel
        # boundaries instead of queueing behind the whole encoder forward --
        # measured p95 786ms vs 1220ms without it, same mean (JETSON_PHASE4_
        # 2_3_RESULTS.json). None on CPU-only or if explicitly disabled via
        # cfg.use_priority_decision_stream=False.
        self._decision_stream = (
            torch.cuda.Stream(priority=-1)
            if (cfg.use_priority_decision_stream and torch.cuda.is_available())
            else None
        )
        # 2026-07-26: vision refresh moved off tick()'s critical path onto
        # its own thread (decide3_speechonly() never receives World-State,
        # so nothing in the decision path needs to wait on ViT-L). Vision
        # is NOT removed from the system -- it still feeds generate_fn's
        # M3-grounded soft prompt via get_cached_world_state().
        self._ws_lock = threading.Lock()
        self._vision_thread: Optional[threading.Thread] = None
        self._vision_thread_stop = threading.Event()
        self.vision_logs: List[Dict] = []   # (t, vitl_forward_ms, fusion_predictor_ms) -- the
                                             # instrumentation tick()'s latencies dict used to carry
        # item 4 (2026-07-26): set for the duration of a vision/ambient
        # forward pass, regardless of which refresh-thread policy is
        # active (strided or opportunistic) -- lets tick() record whether
        # it landed during a forward pass, for the root-cause split
        # (tick latency when overlapping vs not overlapping a refresh).
        self._vision_busy = threading.Event()

    def ingest_video_frame(self, frame: torch.Tensor) -> None:
        self.video_buf.push(frame)

    def ingest_audio_chunk(self, chunk: np.ndarray) -> None:
        self.audio_buf.push(chunk)      # Whisper/decision-path window (unchanged, 2.0s default)
        self.ambient_buf.push(chunk)    # NEW: WavJEPA/World-State window (10.0s default), same
                                         # underlying mic stream, two independent rolling windows

    def start_vision_refresh_thread(self, hz: float = 0.3) -> None:
        """Runs _maybe_refresh_vision on its own cadence, independent of
        tick(). Fix (2026-07-26 review): sleep for (period - elapsed), not
        a full period after the forward returns -- the naive version ran
        at forward_time + period between refreshes (e.g. 2.43s + 3.33s =
        5.76s = 0.17Hz on Jetson, not the requested 0.3Hz)."""
        period = 1.0 / hz

        def _loop():
            while not self._vision_thread_stop.is_set():
                t0 = time.perf_counter()
                self._maybe_refresh_vision(time.time())
                elapsed = time.perf_counter() - t0
                self._vision_thread_stop.wait(max(0.0, period - elapsed))

        self._vision_thread = threading.Thread(target=_loop, daemon=True)
        self._vision_thread.start()

    def start_vision_refresh_thread_opportunistic(self, staleness_deadline_sec: float = 15.0,
                                                   poll_interval_sec: float = 0.25) -> None:
        """Item 4 design: root-cause first (see JETSON_PHASE4_2_3_RESULTS.json
        + the split-by-overlap analysis) showed the p95=786.5ms tail (even
        WITH the priority CUDA stream) coincides with ticks that land during
        a vision/ambient forward pass -- a 2.43-3.4s forward spans ~10-14
        ticks at 0.25s. Fixed-cadence strided refresh (start_vision_refresh_
        thread) is oblivious to what the decision path is doing; this
        alternative policy instead PREFERS to refresh during a window that
        is already gated for other reasons -- MicGate.is_playing (simulated
        TTS playback; Day 1 measured 50/60 ticks gated during playback, i.e.
        the robot is "looking while it talks" and a stale-vision tick costs
        nothing extra since decide3_speechonly() doesn't run during gating
        anyway). A hard staleness deadline forces a refresh even if the
        robot never speaks for a long stretch, so vision cannot go silent
        indefinitely.

        This does NOT replace start_vision_refresh_thread() -- both are
        real, selectable scheduling policies; which one is the deployed
        default is a decision for after on-device p95 numbers exist for
        both (NOT RUN yet, blocked on Jetson SSH access as of this write)."""
        def _loop():
            while not self._vision_thread_stop.is_set():
                t = time.time()
                stale_for = ((t - self._last_vision_refresh_t) if self._last_vision_refresh_t is not None
                             else float("inf"))
                opportunistic = self.mic_gate.is_playing
                forced = stale_for >= staleness_deadline_sec
                if (opportunistic or forced) and not self._vision_busy.is_set():
                    self._maybe_refresh_vision(t, force=True)
                self._vision_thread_stop.wait(poll_interval_sec)

        self._vision_thread = threading.Thread(target=_loop, daemon=True)
        self._vision_thread.start()

    def stop_vision_refresh_thread(self, timeout: float = 5.0) -> None:
        self._vision_thread_stop.set()
        if self._vision_thread is not None:
            self._vision_thread.join(timeout=timeout)

    def get_cached_world_state(self) -> Optional[torch.Tensor]:
        """Thread-safe read for generate_fn (M3 grounding). Returns None
        during the startup window before the first refresh completes (one
        forward-pass duration, ~2.4-2.5s on the Jetson per PHASE0_
        PROVENANCE.txt) -- callers MUST check for None rather than assume a
        vector is always available. Use get_cached_world_state_or_zero()
        for the standard fallback."""
        with self._ws_lock:
            return self._cached_world_state

    def get_cached_world_state_or_zero(self, ws_dim: int, device) -> torch.Tensor:
        """Startup guard (2026-07-26 review): during the one-time window
        before the first vision refresh completes, a "speak" decision must
        not hand None to the M3 connector. Fallback matches this project's
        existing missing-modality-as-zero convention (models/
        m4_duplex_loop.py's decide()/decide3() already default a missing
        WS or SF to a zero tensor) -- a real vision refresh always
        supersedes this within one refresh cycle once the thread is
        running; this only fires during the initial startup gap."""
        ws = self.get_cached_world_state()
        if ws is not None:
            return ws
        return torch.zeros(1, ws_dim, device=device)

    def _maybe_refresh_vision(self, t: float, force: bool = False) -> tuple:
        """Returns (world_state, refreshed: bool, latencies: dict).

        force=True (item 4, opportunistic scheduling): bypass the internal
        stride_vision_sec cadence check -- the caller (start_vision_refresh_
        thread_opportunistic) has already decided this IS the right moment
        to refresh (mic-gated TTS playback or the hard staleness deadline),
        so re-applying the strided check here would just suppress refreshes
        the opportunistic policy exists to trigger.

        REWRITTEN 2026-07-26 to close the item-0 divergence (confirmed by
        direct source reads: no spatial pooling, linspace-ramp tbins
        instead of training's floor-formula staircase, and ambient/WavJEPA
        absent entirely). Now calls models.world_state_builder.
        build_world_state_features() -- the SAME construction verified
        against the cached-feature reference in Phase 1.2 (mean cosine
        0.9985, tbins exact match) and Phase 2.1 (M3 grounding falsifier:
        0.477/0.276/0.144 vs the cached-feature 0.482/0.29/0.15).

        REFRESH BOTH VISION AND AMBIENT TOGETHER, ALWAYS, from the SAME
        timestamp -- congruency was trained on time-aligned pairs; if
        ambient refreshed on its own (cheaper) cadence while vision stayed
        strided, fresh-audio-with-stale-vision would just be a NEW
        divergence replacing the one this rewrite closes. Do not split
        their refresh cadence."""
        need_refresh = force or (self._last_vision_refresh_t is None or
                         t - self._last_vision_refresh_t >= self.cfg.stride_vision_sec)
        window = self.video_buf.get_window()
        ambient_window = self.ambient_buf.get_window()
        if (not need_refresh or window is None or self.vision_encoder is None
                or ambient_window is None or self.ambient_base_encoder is None
                or self.ambient_nat_encoder is None):
            return self._cached_world_state, False, {}

        from models.world_state_builder import build_world_state_features
        device = next(iter(self.vision_encoder.model.parameters())).device if hasattr(self.vision_encoder, "model") else "cuda"
        audio_tensor = torch.from_numpy(ambient_window).float()
        true_dur = audio_tensor.shape[0] / self.cfg.audio_sr

        self._vision_busy.set()
        try:
            t0 = time.perf_counter()
            result = build_world_state_features(
                window, audio_tensor, true_dur,
                self.vision_encoder, self.ambient_base_encoder, self.ambient_nat_encoder,
                self.max_tdm_bins, device,
            )
            # combined ViT-L + WavJEPA-base + WavJEPA-nat cost -- not separable
            # into per-encoder numbers without threading extra timing plumbing
            # through the shared builder (reported as one "encoders_ms" figure
            # instead of the old vitl-only vitl_forward_ms)
            encoders_lat = (time.perf_counter() - t0) * 1000.0

            t0b = time.perf_counter()
            ws = self.loop.compute_world_state(result.feats, result.tbins)
            fusion_lat = (time.perf_counter() - t0b) * 1000.0
        finally:
            self._vision_busy.clear()

        with self._ws_lock:
            self._cached_world_state = ws
            self._last_vision_refresh_t = t
        # Instrumentation fix (2026-07-26 review): this used to reach
        # tick()'s latencies dict via the return value; now that vision
        # refresh runs on its own thread, nobody reads this return value
        # synchronously, so log it here instead -- vitl_forward_ms /
        # fusion_predictor_ms would otherwise silently vanish from every
        # results JSON, including the first real Jetson profiling run.
        self.vision_logs.append({"t": t, "encoders_ms": encoders_lat, "fusion_predictor_ms": fusion_lat})
        return ws, True, {"encoders_ms": encoders_lat, "fusion_predictor_ms": fusion_lat}

    def _estimate_tts_duration_sec(self, text: str) -> float:
        n_words = max(1, len(text.split()))
        return n_words / TTS_WORDS_PER_MINUTE * 60.0

    def tick(self, t: float,
             speech_waveform: Optional[np.ndarray] = None, speech_dur_sec: Optional[float] = None,
             generate_fn: Optional[Callable] = None) -> TickLog:
        """One tick. `generate_fn() -> GenerationResult` is caller-supplied
        so the streaming loop doesn't hardcode which connector(s) built the
        soft prompt (M3/M4b/both). `generate_fn` should call
        get_cached_world_state_or_zero() itself if it needs vision for M3
        grounding -- this method no longer passes World-State through.

        Vision refresh (2026-07-26 review) runs on its OWN THREAD
        (start_vision_refresh_thread()), independent of tick() entirely --
        not "regardless of mic-gating" as before, but structurally absent
        from this method. The decision path (decide3_speechonly) was shown
        not to depend on World-State (A1_PROVENANCE.txt's six-condition
        falsifier); vision still feeds generation via M3, just decoupled
        from tick()'s cadence rather than refreshed inside it. Only audio/
        decision/generation are gated by mic state, same as before."""
        latencies: Dict[str, float] = {}

        # TTS playback bookkeeping (simulated -- see module docstring)
        if self._tts_playing_until is not None and t >= self._tts_playing_until:
            self.mic_gate.is_playing = False
            self._tts_playing_until = None

        refreshed = False   # vision refresh is no longer synchronous with tick(); see vision_logs
        # item 4 root-cause instrumentation: sampled once, at tick entry --
        # a forward pass in flight right now is what would contend with
        # this tick's own GPU work.
        overlapped = self._vision_busy.is_set()

        if not self.mic_gate.should_run_decision():
            log = TickLog(t=t, action="gated", vision_refreshed=refreshed, latencies_ms=latencies,
                          overlapped_vision_forward=overlapped)
            self.logs.append(log)
            return log

        # LOCKED default (item 3): the decision path's GPU work runs on a
        # priority stream so it interleaves at ViT-L's kernel boundaries
        # rather than queueing behind the vision thread's forward pass --
        # baked into tick() itself (not caller-opt-in), matching the
        # measured 4.3 configuration exactly (JETSON_PHASE4_2_3_RESULTS.json).
        if self._decision_stream is not None:
            with torch.cuda.stream(self._decision_stream):
                label, probs, log = self._run_decision_path(
                    t, speech_waveform, speech_dur_sec, generate_fn, latencies, refreshed, overlapped)
            torch.cuda.current_stream().wait_stream(self._decision_stream)
        else:
            label, probs, log = self._run_decision_path(
                t, speech_waveform, speech_dur_sec, generate_fn, latencies, refreshed, overlapped)

        self.logs.append(log)
        return log

    def _run_decision_path(self, t, speech_waveform, speech_dur_sec, generate_fn,
                            latencies, refreshed, overlapped):
        sf = None
        if speech_waveform is not None:
            t0 = time.perf_counter()
            sf, _ = self.loop.compute_speech_activity(speech_waveform, speech_dur_sec)
            latencies["speech_activity_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        label, probs = self.loop.decide3_speechonly(sf)
        latencies["decision_ms"] = (time.perf_counter() - t0) * 1000.0

        log = TickLog(t=t, action=label, vision_refreshed=refreshed, latencies_ms=latencies,
                       decision_label=label, decision_probs=probs, overlapped_vision_forward=overlapped)

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

        return label, probs, log
