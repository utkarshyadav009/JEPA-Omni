"""models/m5_fan_notch.py — make the mic deaf to BMO's own fan.

The user's framing was the right one: *"you know how headphones run the inverted wave through
the speaker to remove that specific tone or frequency, we can try that with the mic so the mic
is deaf to the fan sound."* ANC works on **periodic** noise, and the question was whether the
Jetson fan is periodic. It is.

## The measurement that decided this (scripts/measure_fan_signature.py, 2026-08-16)

`bmo-power` made the fan a controlled variable for the first time, so the fan could be isolated
from the room by differencing two commanded speeds instead of guessed at from a single recording:

    quiet  1915 rpm  29.8%  rms 0.00059
    cool   5403 rpm  84.7%  rms 0.00736

    DELTA spectrum (cool minus quiet) = the fan, and nothing else
        <300 Hz   4.5%      300-1k  48.5%      1k-4k  44.9%      4k-8k  2.1%
        808.6 Hz  18.6%  |  812.5 Hz  20.0%    <- 38.6% of all fan energy in one narrow peak

    blade-pass candidates at 5403 rpm:
         9 blades -> 810.4 Hz   nearest measured peak 808.6 Hz   OFF BY 1.9 Hz

Nine blades, and the dominant tone is the blade-pass frequency. **This corrects an earlier
conclusion in CLAUDE.md that the fan was broadband** -- that reading came from a single
recording of the TOTAL floor, which could not separate fan from room, and it was wrong.

So the tone is not merely tonal, it is *predictable*: BPF = rpm/60 x 9, and the tachometer
reports rpm continuously. There is no need to estimate the frequency from the audio at all --
the machine tells us what it is about to sound like. That is a stronger position than ordinary
ANC, which has to infer the reference.

## Why a notch and not spectral subtraction

Spectral subtraction against a broadband profile was tried and made the percept measurably
WORSE ("an alarm beeping" 0.451 -> 0.478, "glass breaking" appearing) because it leaves
narrowband musical-noise residue across the whole spectrum. A notch does the opposite: it
removes energy only in a few narrow bands whose centres are known exactly, and leaves every
other bin untouched, so it cannot manufacture artefacts where speech lives.

Bandwidth is deliberately tight (Q=30, ~27 Hz at 810 Hz). Speech energy is broadband and loses
almost nothing to three narrow notches; the fan tone loses almost everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Identified from the delta spectrum, not from a datasheet: 9 blades predicts 810.4 Hz at
# 5403 rpm against 808.6 Hz measured, a 1.9 Hz error (0.2%).
FAN_BLADES = 9


@dataclass
class FanNotchConfig:
    sr: int = 16000
    blades: int = FAN_BLADES
    harmonics: int = 3       # BPF, 2xBPF, 3xBPF -- higher ones sit under the 4k-8k tail (2.1%)
    q: float = 30.0          # ~27 Hz wide at 810 Hz
    min_hz: float = 60.0
    max_hz: float = 7000.0


def _iir_notch(f0: float, q: float, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Second-order IIR notch (RBJ cookbook). Written out rather than imported from scipy so
    this runs on the Jetson image regardless of the scipy/numpy version pinning, which has
    already broken a torch install once on this device."""
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1.0, -2.0 * np.cos(w0), 1.0], dtype=np.float64)
    a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha], dtype=np.float64)
    return b / a[0], a / a[0]


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct-form II transposed, applied forward then backward (zero phase).

    Zero-phase matters here: the mic buffer feeds WavJEPA, which is a temporal model, and a
    frequency-dependent group delay from a causal notch would smear onsets relative to the
    video stream the world-state builder aligns against. We filter an already-buffered 10 s
    window offline, so there is no reason to accept phase distortion.

    THIS IS THE FALLBACK PATH, not the default. Measured on a 10 s / 16 kHz buffer with 3
    harmonics: **374 ms** -- a 57% increase on the 650 ms perception leg, which is not a
    trade worth making for noise cleanup. `FanNotch.__call__` uses `scipy.signal.sosfiltfilt`
    (C-implemented, sub-millisecond) whenever scipy imports, and only lands here if it does
    not. Kept because scipy/numpy pinning on this device has broken a torch install before,
    so a pure-numpy path that always works is worth its weight."""
    def _once(sig):
        y = np.zeros_like(sig)
        z1 = z2 = 0.0
        for n in range(len(sig)):
            xn = sig[n]
            yn = b[0] * xn + z1
            z1 = b[1] * xn - a[1] * yn + z2
            z2 = b[2] * xn - a[2] * yn
            y[n] = yn
        return y
    return _once(_once(x.astype(np.float64))[::-1])[::-1]


class FanNotch:
    """Tach-driven notch bank. Coefficients are recomputed only when the RPM actually moves."""

    def __init__(self, cfg: Optional[FanNotchConfig] = None):
        self.cfg = cfg or FanNotchConfig()
        self._rpm: Optional[float] = None
        self._sos: List[Tuple[np.ndarray, np.ndarray]] = []
        self.centres: List[float] = []

    def bpf(self, rpm: float) -> float:
        return rpm / 60.0 * self.cfg.blades

    def update(self, rpm: Optional[float]) -> bool:
        """Rebuild the bank for a new fan speed. Returns True if the bank changed."""
        if not rpm or rpm <= 0:
            return False
        # 1% hysteresis: the tach jitters by a few rpm and rebuilding per read is wasted work
        # for a notch far narrower than the jitter is wide.
        if self._rpm is not None and abs(rpm - self._rpm) / self._rpm < 0.01:
            return False
        self._rpm = float(rpm)
        f0 = self.bpf(rpm)
        self._sos, self.centres = [], []
        for h in range(1, self.cfg.harmonics + 1):
            f = f0 * h
            if not (self.cfg.min_hz <= f <= min(self.cfg.max_hz, 0.45 * self.cfg.sr)):
                continue
            self._sos.append(_iir_notch(f, self.cfg.q, self.cfg.sr))
            self.centres.append(f)
        return True

    def __call__(self, x: np.ndarray, rpm: Optional[float] = None) -> np.ndarray:
        if rpm is not None:
            self.update(rpm)
        if not self._sos:
            return x
        try:
            from scipy.signal import sosfiltfilt          # fast path, sub-ms
            sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]] for b, a in self._sos])
            return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
        except Exception:
            y = x                                         # 374 ms; see _lfilter's docstring
            for b, a in self._sos:
                y = _lfilter(b, a, y)
            return y.astype(np.float32)


def attenuation_db(before: np.ndarray, after: np.ndarray, f0: float,
                   sr: int = 16000, halfwidth: float = 25.0) -> float:
    """Measured attenuation in the notch band -- so the fix reports a number, not a claim."""
    n = 1 << int(np.ceil(np.log2(len(before))))
    fb = np.abs(np.fft.rfft(before, n)) ** 2
    fa = np.abs(np.fft.rfft(after, n)) ** 2
    fr = np.fft.rfftfreq(n, 1 / sr)
    m = (fr >= f0 - halfwidth) & (fr <= f0 + halfwidth)
    if not m.any():
        return 0.0
    return float(10 * np.log10((fb[m].sum() + 1e-20) / (fa[m].sum() + 1e-20)))


if __name__ == "__main__":
    sr, rpm = 16000, 5403.0
    n = FanNotch(FanNotchConfig(sr=sr))
    n.update(rpm)
    print(f"[notch] rpm={rpm} blades={FAN_BLADES} -> BPF={n.bpf(rpm):.1f} Hz")
    print(f"[notch] centres {[f'{c:.1f}' for c in n.centres]}")
    assert abs(n.bpf(rpm) - 810.4) < 1.0, "BPF must match the measured 808.6 Hz peak"

    t = np.arange(sr * 2) / sr
    speech = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.03 * np.sin(2 * np.pi * 1500 * t)
    fan = 0.20 * np.sin(2 * np.pi * n.bpf(rpm) * t)
    x = (speech + fan).astype(np.float32)
    y = n(x)
    print(f"[notch] fan tone   {attenuation_db(x, y, n.bpf(rpm), sr):+.1f} dB")
    print(f"[notch] speech 220 {attenuation_db(x, y, 220.0, sr):+.1f} dB   "
          f"(must stay near 0 -- the point is to be surgical)")
    assert attenuation_db(x, y, n.bpf(rpm), sr) > 20, "notch must kill the fan tone"
    assert abs(attenuation_db(x, y, 220.0, sr)) < 1.0, "notch must not touch speech"
    print("[notch] ok")
