"""models/m5_tts.py — Phase B: real TTS, replacing the simulated 150wpm
playback estimate in models/m5_streaming_loop.py.

Engine: Piper (CPU, onnxruntime, no CUDA) -- same choice and voice already
validated in scripts/jetson_tts_latency.py (en_US-lessac-medium), which
measured real time-to-audio on the Jetson: backchannel synthesis 57.8ms
mean, generated-turn synthesis 340.8ms mean
(checkpoints/vjepa21_shelved/JETSON_TTS_LATENCY_RESULTS.json). CPU-only was
a deliberate choice, not an oversight: it keeps TTS off the CUDA allocator
entirely, so it competes with the 854MiB full-stack GPU headroom (A1) only
via shared unified memory for the ~60-70MB ONNX voice model, never for
compute.

Two paths, per the explicit instruction that they must NOT share a
mechanism:
  - TTSEngine.speak(text)      -- LIVE synthesis, for real generated turns
                                   (text is dynamic, can't be pre-computed).
                                   Returns the REAL measured duration, not
                                   an estimate -- this is what should drive
                                   MicGate.is_playing / _tts_playing_until
                                   now, replacing _estimate_tts_duration_sec.
  - BackchannelInventory.play() -- a FIXED set of phrases synthesized ONCE
                                   at construction (startup), cached as
                                   ready-to-play audio arrays. Backchannel
                                   decisions never call the LLM and never
                                   call Piper live -- pure playback, the
                                   fastest and most reliable path available,
                                   by design (this is a deliberate demo-
                                   robustness simplification over the
                                   earlier LLM-generates-8-tokens approach
                                   in scripts/m4_backchannel_production_test.py,
                                   superseded per direct instruction).

Playback: sounddevice (PortAudio backend, works over ALSA on Jetson).
Non-blocking by default (sd.play() returns immediately) so the tick loop
keeps running while audio plays -- MicGate.is_playing is the real gate,
not blocking the caller.

Not yet run on real hardware from this dev machine (no Jetson/Piper/audio
device available here) -- built to the exact API already proven in
scripts/jetson_tts_latency.py so it should run correctly on first try, but
this is a documented, not-yet-verified assumption. Report real numbers
once run.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_VOICE_MODEL = Path.home() / "piper_voices" / "en_US-lessac-medium.onnx"
DEFAULT_VOICE_CONFIG = Path.home() / "piper_voices" / "en_US-lessac-medium.onnx.json"

# Same fixed set already used (and latency-measured) in scripts/jetson_tts_latency.py --
# reusing it rather than inventing a second inventory that's never been timed.
DEFAULT_BACKCHANNEL_PHRASES = ["Mm-hmm.", "I see.", "Right.", "Got it.", "Okay."]

# Task 137: real BMO-voiced bank (Fish Audio API, public "BMO from Adventure
# Time" community reference_id -- see the rendering script referenced in
# PrebuiltVoiceBank's docstring), superseding DEFAULT_BACKCHANNEL_PHRASES'
# generic Piper voice for anything that wants the real BMO voice.
DEFAULT_BACKCHANNEL_DIR = Path.home() / "JEPA-Omni" / "assets" / "bmo_backchannels"


@dataclass
class SynthesisResult:
    audio: np.ndarray          # int16 mono PCM
    sample_rate: int
    duration_sec: float
    synth_latency_sec: float   # wall-clock time.synthesize() took (NOT playback duration)


def _play_pcm(audio: np.ndarray, sr: int, device, blocking: bool) -> None:
    """Shared playback path (factored out of TTSEngine.play so
    PrebuiltVoiceBank -- pure file playback, no TTSEngine involved -- can
    reuse the exact same real-hardware-verified resample/normalize logic
    instead of a second, divergent copy of it."""
    import sounddevice as sd
    # Real-hardware finding (2026-08-01): the ReSpeaker 4 Mic Array only
    # accepts its ALSA-reported default_samplerate (16000Hz) for playback --
    # sd.play() at Piper's native 22050Hz (en_US-lessac-medium.onnx.json's
    # audio.sample_rate) raises PortAudioError('Invalid sample rate'). Query
    # the target device's supported rate and resample if they don't match,
    # rather than hardcoding 16000 -- a different TTS voice or output device
    # could have a different native/accepted rate.
    try:
        target_sr = int(sd.query_devices(device)["default_samplerate"])
    except Exception:
        target_sr = sr
    if target_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        # BUG FIXED (2026-08-01, real hardware): sd.play() interprets a
        # float32 array as normalized to [-1.0, 1.0] -- casting int16 PCM
        # (range +-32767) to float32 WITHOUT rescaling first fed values
        # ~32767x too large, producing hard-clipped/out-of-range output at
        # the driver level (the user reported hearing nothing at all when
        # this was first tried on real hardware -- silent or garbled
        # clipping, not a volume/mixer setting, is the likely cause).
        # Normalize to [-1, 1] before resampling.
        audio = resample_poly(audio.astype(np.float32) / 32768.0, target_sr // g, sr // g)
        sr = target_sr
    elif np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    sd.play(audio, samplerate=sr, device=device)
    if blocking:
        sd.wait()


class TTSEngine:
    """LIVE synthesis path, for real generated-turn text. One-time voice
    load at construction (not counted in per-utterance latency, matching
    jetson_tts_latency.py's convention)."""

    def __init__(self, model_path: Path = DEFAULT_VOICE_MODEL,
                 config_path: Path = DEFAULT_VOICE_CONFIG,
                 device: Optional[object] = 24) -> None:
        # device default (2026-08-01, real hardware confirmed): sounddevice index 24
        # is the ReSpeaker 4 Mic Array's playback path (out=2, confirmed by a real
        # test-tone play() with no error) -- the same physical device as
        # models/m5_live_capture.py's LiveMicCapture default, no separate speaker
        # needed. Verify with sounddevice.query_devices() if hardware changes.
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        t0 = time.perf_counter()
        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._load_s = time.perf_counter() - t0
        self._syn_cfg = SynthesisConfig()
        self._device = device  # sounddevice output device index/name; None = system default
        # warm-up: first real call after load sometimes pays extra one-time JIT/cache cost
        # (jetson_tts_latency.py's own convention) -- do it now so it doesn't land inside
        # the first real turn's measured latency.
        self._synthesize_raw("Warm up.")

    def _synthesize_raw(self, text: str) -> SynthesisResult:
        t0 = time.perf_counter()
        chunks = list(self._voice.synthesize(text, syn_config=self._syn_cfg))
        pcm_bytes = b"".join(c.audio_int16_bytes for c in chunks)
        synth_latency = time.perf_counter() - t0
        sample_rate = chunks[0].sample_rate if chunks else 22050
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        duration_sec = len(audio) / float(sample_rate)
        return SynthesisResult(audio=audio, sample_rate=sample_rate,
                                duration_sec=duration_sec, synth_latency_sec=synth_latency)

    def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize without playing -- used for the pre-synthesized backchannel
        inventory (BackchannelInventory) and for latency measurement."""
        return self._synthesize_raw(text)

    def play(self, result: SynthesisResult, blocking: bool = False) -> None:
        _play_pcm(result.audio, result.sample_rate, self._device, blocking)

    def speak(self, text: str, blocking: bool = False) -> SynthesisResult:
        """Synthesize + start playback. Returns the REAL synthesis result
        (duration_sec is the actual audio length, not a wpm estimate) --
        this is what should set MicGate.is_playing / _tts_playing_until in
        models/m5_streaming_loop.py, replacing _estimate_tts_duration_sec.
        Does NOT include synth_latency_sec in duration_sec -- the caller
        (streaming loop) should gate the mic from the moment playback
        starts, i.e. t_now + result.duration_sec, same convention as the
        estimate it replaces."""
        result = self._synthesize_raw(text)
        self.play(result, blocking=blocking)
        return result

    def stop(self) -> None:
        """Hard-stop any in-progress playback (interruption path)."""
        import sounddevice as sd
        sd.stop()


class BackchannelInventory:
    """Pre-synthesized at construction (startup), never touches Piper or
    the LLM again after __init__. play() is pure playback -- the fastest
    path available by construction, since there is no synthesis step left
    to pay at decision time."""

    def __init__(self, engine: TTSEngine, phrases: List[str] = None) -> None:
        self.phrases = phrases if phrases is not None else list(DEFAULT_BACKCHANNEL_PHRASES)
        t0 = time.perf_counter()
        self._synthesized: List[SynthesisResult] = [engine.synthesize(p) for p in self.phrases]
        self._prep_s = time.perf_counter() - t0
        self._engine = engine
        self._next_idx = 0

    def play(self, index: Optional[int] = None, blocking: bool = False) -> SynthesisResult:
        """index=None cycles round-robin through the inventory (avoids the
        robot saying the exact same backchannel every time)."""
        if index is None:
            index = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self._synthesized)
        result = self._synthesized[index]
        self._engine.play(result, blocking=blocking)
        return result


class PrebuiltVoiceBank:
    """Task 137: a bank of literally prebuilt non-verbal cues -- rendered
    OFFLINE (Fish Audio API against the real BMO voice, not our own
    NeuTTS-Air fine-tune -- that fine-tune proved unstable on very short
    1-3 word phrases, see the rendering script's docstring for the exact
    repetition-loop failure found) into on-disk WAV files, loaded once at
    construction, and never touching any TTS engine or the LLM again after
    that. Same "pure playback, fastest path by construction" property as
    BackchannelInventory, just with the real cloned BMO voice and real
    functional categories instead of one flat generic-voice list.

    Three categories, grounded in real full-duplex-dialogue literature
    (dGSLM/Moshi: backchanneling emerges as a genuinely separate stream
    from turn-taking speech, not a truncated version of it; VAP: short-
    horizon prosody-triggered prediction is what decides WHEN, not WHAT;
    Lex/Sierra/Twilio voice-agent guides: filler audio at TURN START masks
    processing latency, a functionally different moment than a listener
    backchannel):
      - "continuer": fired WHILE the user is still speaking, on the
        existing 3-class decision head's "backchannel" label (models/
        m4_decision_head.py, SpeechOnlyThreeClassHead, ~95% acc on real
        EasyCom data) -- low-key active-listening signal (uh-huh/mm-hmm/
        yeah/okay). This is the direct drop-in replacement for
        BackchannelInventory in models/m5_streaming_loop.py's existing
        wiring (label == "backchannel" -> .play(blocking=False)).
      - "reactive": a more expressive light reaction (oh!/whoa/huh),
        fired from the SAME trigger point as "continuer" for variety --
        the decision head was trained on one "backchannel" label, not
        sub-typed, so this is a weighted random pick inside play(), not a
        second classifier output (would need new labeled data to train
        that split, out of scope here).
      - "thinking_filler": a NEW integration point (not present before
        this task) -- fired at the START of BMO's own SPEAK turn to mask
        LLM/TTS generation latency, most valuable for the reasoning-tier
        escalation path (models/m4_cognitive_core.py's
        GGUFReasoningTier), which is measurably slower than the fast tier.
    """

    def __init__(self, bank_dir: Optional[Path] = None, device=24, reactive_weight: float = 0.3) -> None:
        import soundfile as sf
        self.bank_dir = Path(bank_dir) if bank_dir is not None else DEFAULT_BACKCHANNEL_DIR
        self._device = device
        self.reactive_weight = reactive_weight

        manifest_path = self.bank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no backchannel manifest at {manifest_path} -- render the bank first "
                "(scratchpad's render_bc_fishapi.py pattern, see PrebuiltVoiceBank docstring)")
        manifest = json.loads(manifest_path.read_text())

        self._clips: Dict[str, List[SynthesisResult]] = {"continuer": [], "reactive": [], "thinking_filler": []}
        for entry in manifest:
            audio, sr = sf.read(str(self.bank_dir / entry["file"]), dtype="int16")
            self._clips.setdefault(entry["category"], []).append(SynthesisResult(
                audio=audio, sample_rate=sr, duration_sec=len(audio) / float(sr), synth_latency_sec=0.0))
        for cat, clips in self._clips.items():
            if not clips:
                raise ValueError(f"no clips loaded for category {cat!r} from {self.bank_dir}")
        self._next_idx = {cat: 0 for cat in self._clips}

    def play(self, category: str = "continuer", blocking: bool = False) -> SynthesisResult:
        """Signature-compatible with BackchannelInventory.play() for the
        default case (category="continuer") -- existing call sites like
        `self.backchannel_inventory.play(blocking=False)` in models/
        m5_streaming_loop.py work unchanged if this replaces that
        inventory. Pass category="thinking_filler" from the new SPEAK-turn
        latency-masking hook."""
        if category == "continuer" and self.reactive_weight > 0 and random.random() < self.reactive_weight:
            category = "reactive"
        clips = self._clips[category]
        idx = self._next_idx[category]
        self._next_idx[category] = (idx + 1) % len(clips)
        result = clips[idx]
        _play_pcm(result.audio, result.sample_rate, self._device, blocking)
        return result

    def stop(self) -> None:
        """Hard-stop any in-progress playback (interruption path)."""
        import sounddevice as sd
        sd.stop()
