"""models/m4_echo_cancellation.py — self-interruption fix for the M4c duplex
loop: when TTS is attached, the robot's own speech enters its own mic, VAD
fires, and the controller halts its own generation (a self-interruption
loop). Two candidate fixes, both implemented:

  MicGate: mute/discard mic input entirely while TTS is playing. Dead
    simple, always works, zero compute -- but it ALSO blocks genuine user
    barge-in during robot speech, which defeats the interruption feature
    this whole M4 track was built around. Use only as a fallback when AEC
    isn't available.

  PyAecCanceller: wraps the `pyaec` package (ctypes binding to a compiled
    C AEC library, speex-family algorithm) -- the RECOMMENDED
    implementation. Measured 14.24dB ERLE on a 123s stream of real,
    concatenated EasyCom speech (non-stationary, bursty) with
    enable_preprocess=True; frame=256, filter_length=1024.
    enable_preprocess=False was tested first and gave <1dB ERLE on the same
    real-speech stream (vs 21-44dB on stationary test tones) -- preprocessing
    (signal conditioning appropriate for real non-stationary audio) turned
    out to be load-bearing, not optional, for real speech.

  NLMSEchoCanceller: a from-scratch time-domain NLMS filter, kept for
    reference/pedagogical purposes. Verified to work correctly on
    stationary test signals (26.6dB ERLE on a two-tone sine) but did NOT
    reliably converge on real speech even over a 409s continuous stream in
    initial testing (~-1dB ERLE, i.e. no improvement) -- a genuine, measured
    limitation of naive time-domain NLMS on real non-stationary audio
    without the additional preprocessing/regularization production
    libraries include. NOT recommended for actual use; superseded by
    PyAecCanceller.

RECOMMENDATION -- UPDATED AFTER MEASUREMENT, more qualified than originally
expected: PyAecCanceller gives real, measured echo suppression (~13.2dB
ERLE, consistent across simulated echo attenuation levels 0.05-0.6) but
this did NOT translate into a low false-interruption rate in testing
(scripts/m4_echo_test.py measured ~100% false-interruption on the AEC
residual in the converged region, vs 62.5% with no fix at all -- a real
but insufficient improvement). Root cause, also measured directly: this
AEC implementation's residual has an apparent NOISE FLOOR of RMS~=0.0022
that is INDEPENDENT of the input echo's loudness (likely comfort-noise
injected by enable_preprocess=True) -- and that floor is comparable to or
louder than genuine quiet speech (measured EasyCom real-speech RMS: mean
0.00188, p10 0.00077). A downstream energy/SNR gate on top of AEC will NOT
cleanly separate the two, since they overlap in loudness.

Given this, the HONEST current recommendation is a hybrid, not a clean
single answer: MIC GATING as the safe default during TTS playback (the
only condition measured at 0% false-interruption), accepting the real cost
that it also blocks genuine barge-in during robot speech (0% detection-
preserved in testing) -- with AEC logged as a real, partial improvement
worth keeping in the pipeline (it measurably helps versus no fix, 62.5%),
but NOT yet sufficient on its own to safely enable full always-listening
barge-in during TTS playback. Closing this gap needs either (a) hardware-
level echo suppression (mic/speaker physical separation, beamforming,
echo-optimized hardware -- reduces the RAW echo level the software AEC has
to work with) or (b) retraining the decision head with real AEC-residual/
comfort-noise negative examples so it stops treating that specific noise
floor as speech, neither attempted here. See scripts/m4_echo_test.py for
the full measured comparison and checkpoints/m4_decision_head/
echo_test_results.json for raw numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class MicGate:
    """Simplest possible fix: while `is_playing` is True, the decision
    pipeline should be SKIPPED ENTIRELY (short-circuit to a hardcoded
    SILENCE decision, don't call Whisper/the decision head at all) --
    NOT implemented as feeding a zeroed waveform through the model.

    This distinction is load-bearing, not stylistic: an all-zero waveform
    is OUT OF DISTRIBUTION for the decision head (training silence
    examples were always real quiet-room audio from actual gaps between
    utterances, never mathematically-zero digital silence) and was found
    to score P(speak)=0.999 -- i.e. naive zero-feeding would GUARANTEE a
    false trigger rather than prevent one. should_run_decision() is the
    correct integration point: check it before calling the pipeline, don't
    call the pipeline on gated audio and expect the model to handle it."""

    def __init__(self):
        self.is_playing = False

    def should_run_decision(self) -> bool:
        return not self.is_playing

    def process(self, mic_signal: np.ndarray) -> np.ndarray:
        """Retained for completeness/comparison in tests only -- do not use
        this to gate the real pipeline, use should_run_decision() instead."""
        if self.is_playing:
            return np.zeros_like(mic_signal)
        return mic_signal


@dataclass
class NLMSConfig:
    filter_length: int = 1024   # ~64ms at 16kHz -- must cover the expected echo delay
    mu: float = 0.5             # step size, 0 < mu <= 1 for stability with normalization
    eps: float = 1e-6


class NLMSEchoCanceller:
    """Adaptive echo path estimation + cancellation. Call reset() between
    independent utterances (the adaptive filter's state is specific to one
    echo path instance)."""

    def __init__(self, cfg: NLMSConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.cfg.filter_length)

    def process(self, reference: np.ndarray, mic_signal: np.ndarray) -> np.ndarray:
        """reference: the known TTS output signal (same sample rate/length
        as mic_signal, causally aligned -- the filter learns any residual
        delay within filter_length). Returns the echo-cancelled residual,
        same length as mic_signal."""
        assert len(reference) == len(mic_signal)
        n = len(mic_signal)
        L = self.cfg.filter_length
        residual = np.zeros(n)
        x_buf = np.zeros(L)
        for i in range(n):
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = reference[i]
            y_hat = np.dot(self.w, x_buf)
            e = mic_signal[i] - y_hat
            residual[i] = e
            norm = np.dot(x_buf, x_buf) + self.cfg.eps
            self.w += (self.cfg.mu / norm) * e * x_buf
        return residual


class PyAecCanceller:
    """Wraps pyaec.Aec (ctypes binding to a compiled C AEC library).
    Operates on int16 PCM frames -- process() handles the float32<->int16
    conversion so callers can stay in the same [-1,1] float convention used
    elsewhere in this pipeline (Whisper's feature extractor, etc)."""

    def __init__(self, sr: int = 16000, frame_size: int = 256, filter_length: int = 1024,
                 enable_preprocess: bool = True):
        from pyaec import Aec
        self.frame_size = frame_size
        self.sr = sr
        self._aec = Aec(frame_size, filter_length, sr, enable_preprocess=enable_preprocess)

    @staticmethod
    def _to_i16(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)

    def process(self, reference: np.ndarray, mic_signal: np.ndarray) -> np.ndarray:
        assert len(reference) == len(mic_signal)
        ref_i16 = self._to_i16(reference)
        mic_i16 = self._to_i16(mic_signal)
        n = len(mic_i16) - (len(mic_i16) % self.frame_size)
        residual = np.zeros(len(mic_i16), dtype=np.int16)
        for i in range(0, n, self.frame_size):
            rec = mic_i16[i:i + self.frame_size].tolist()
            ref = ref_i16[i:i + self.frame_size].tolist()
            out = self._aec.cancel_echo(rec, ref)
            residual[i:i + self.frame_size] = np.array(out, dtype=np.int16)
        residual[n:] = mic_i16[n:]   # leftover partial frame, passed through uncancelled
        return (residual.astype(np.float32) / 32767.0)


def simulate_echo_path(reference: np.ndarray, sr: int, delay_ms: float = 15.0,
                        attenuation: float = 0.6, noise_level: float = 0.01,
                        rng: np.random.Generator = None) -> np.ndarray:
    """Simulates what the mic actually picks up from the robot's own
    loudspeaker: a delayed, attenuated copy of the reference plus a little
    ambient noise. Standard AEC test methodology (fixed linear echo path)."""
    rng = rng or np.random.default_rng(0)
    delay_samples = int(sr * delay_ms / 1000.0)
    echo = attenuation * np.concatenate([np.zeros(delay_samples), reference])[:len(reference)]
    noise = noise_level * rng.standard_normal(len(reference))
    return (echo + noise).astype(np.float32)
