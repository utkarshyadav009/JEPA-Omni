"""scripts/measure_fan_signature.py — what does the fan ACTUALLY sound like?

Every fan-noise fix so far has been guesswork about the spectrum, and the one that got deployed
made things measurably worse: blind spectral subtraction moved the `hearing` percept from
"an alarm beeping" 0.451 to 0.478 and introduced "glass breaking", because a stationary tonal
source leaves narrowband musical-noise residue. Before writing another filter, measure.

WHAT MAKES THIS POSSIBLE NOW. Until 2026-08-16 the fan ran under the closed-loop thermal
governor and could not be commanded, so there was no way to isolate its contribution from the
rest of the room. `bmo-power` (user-built, wired into models/m5_tools.py the same day) can set
the fan to an exact duty cycle and read back real RPM from the tachometer. That turns fan noise
into a CONTROLLED VARIABLE: record the same room at several fan speeds and difference them, and
whatever changes is the fan and nothing else.

THE QUESTION THIS ANSWERS, which decides the whole approach:

  * If the delta spectrum is dominated by NARROWBAND PEAKS at the blade-pass frequency
    (BPF = rpm/60 x n_blades) and its harmonics, then an adaptive NOTCH driven by the tach is
    the right tool -- cheap, causal, and it tracks automatically as the fan speeds up. This is
    the true analogue of what ANC headphones do to a periodic tone.
  * If it is genuinely BROADBAND, no notch will help and the honest options are physical
    distance, spatial nulling across the 4 capsules, or simply commanding the fan quiet before
    listening (also newly possible).

An earlier 4-second sample suggested broadband (lo<300Hz / 300-4k energy ratio 0.12-0.19), but
that measurement could not separate fan from room -- it was the total floor. This one can.

Also reports the observed BPF peaks against the predicted BPF for a range of blade counts, so
the blade count itself is inferred from data rather than assumed.

Usage (on the Jetson):
    python3 scripts/measure_fan_signature.py --out ~/fan_signature.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

SR = 16000
NFFT = 4096


def fan_set(profile: str) -> None:
    """Python module first. The `bmo-power` CLI raises PermissionError when invoked
    non-interactively as the `bmo` user (measured 2026-08-16), while `import bmo_power` sets the
    fan fine -- it writes the same root-owned /sys/class/hwmon/hwmon0/pwm1 through whatever
    privilege path the package sets up. models/m5_tools.py already prefers the module for this
    reason; the CLI is only a fallback."""
    try:
        import bmo_power
        bmo_power.set_fan_speed(profile)
        return
    except Exception:
        pass
    subprocess.run(["bmo-power", "--fan", profile], capture_output=True, timeout=10.0)


def fan_state() -> dict:
    from models.m5_tools import power_status
    st = power_status() or {}
    f = st.get("fan") or {}
    return {"rpm": f.get("rpm"), "pwm_pct": f.get("pwm_percentage"), "mode": f.get("mode")}


def record(sd, idx: int, ch: int, sec: float) -> np.ndarray:
    x = sd.rec(int(SR * sec), samplerate=SR, channels=ch, dtype="float32", device=idx)
    sd.wait()
    return x


def spectrum(mono: np.ndarray):
    w = np.hanning(NFFT).astype(np.float32)
    hop = NFFT // 2
    n = 1 + max(0, len(mono) - NFFT) // hop
    acc = np.zeros(NFFT // 2 + 1, dtype=np.float64)
    for i in range(n):
        seg = mono[i * hop: i * hop + NFFT]
        if len(seg) < NFFT:
            break
        acc += np.abs(np.fft.rfft(seg * w)) ** 2
    return acc / max(n, 1), np.fft.rfftfreq(NFFT, 1 / SR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=6.0)
    ap.add_argument("--profiles", default="quiet,cool,max")
    ap.add_argument("--out", default=os.path.expanduser("~/fan_signature.json"))
    args = ap.parse_args()

    import sounddevice as sd
    idx = next(i for i, d in enumerate(sd.query_devices())
               if "ReSpeaker" in d["name"] and d["max_input_channels"] > 0)
    ch = int(sd.query_devices()[idx]["max_input_channels"])
    print(f"[fan] device {idx} '{sd.query_devices()[idx]['name']}' ch={ch}")
    print("[fan] KEEP THE ROOM QUIET for the whole run -- anything you do is measured too.\n")

    runs = {}
    try:
        for prof in args.profiles.split(","):
            fan_set(prof)
            time.sleep(6.0)             # let the fan actually reach the commanded speed
            st = fan_state()
            x = record(sd, idx, ch, args.sec)
            live = [c for c in range(ch) if np.sqrt((x[:, c] ** 2).mean()) > 1e-6]
            mono = x[:, live].mean(axis=1).astype(np.float32)
            P, f = spectrum(mono)
            runs[prof] = {"state": st, "rms": float(np.sqrt((mono ** 2).mean())), "P": P}
            print(f"[fan] {prof:6s} rpm={st['rpm']} pwm={st['pwm_pct']}% "
                  f"rms={runs[prof]['rms']:.5f}")
    finally:
        # ALWAYS hand the thermal governor back, including on a crash or a timeout kill.
        # This script pins the fan manually and the Jetson idles at ~68C; leaving it latched to
        # `quiet` after an exception is a thermal-throttle waiting to happen. The first version
        # restored `auto` on the happy path only, and a PermissionError mid-run left the fan
        # pinned -- exactly the failure this guards.
        fan_set("auto")
        print(f"[fan] fan restored to auto -> {fan_state()}")

    names = list(runs)
    lo, hi = names[0], names[-1]
    dP = runs[hi]["P"] - runs[lo]["P"]          # what the EXTRA fan speed added, and only that
    dP = np.maximum(dP, 0.0)
    _, f = spectrum(np.zeros(NFFT * 2, dtype=np.float32))

    tot = dP.sum() + 1e-12
    bands = {"<300Hz": (0, 300), "300-1k": (300, 1000), "1k-4k": (1000, 4000),
             "4k-8k": (4000, 8000)}
    print(f"\n[fan] DELTA spectrum ({hi} minus {lo}) — this is the fan, isolated:")
    for nm, (a, b) in bands.items():
        frac = dP[(f >= a) & (f < b)].sum() / tot
        print(f"    {nm:8s} {frac*100:5.1f}%")

    # peak structure: narrowband tone vs broadband hiss
    k = 12
    top = np.argsort(dP)[-k:][::-1]
    peak_frac = dP[top].sum() / tot
    print(f"\n[fan] top {k} bins carry {peak_frac*100:.1f}% of the delta energy")
    print("    (a strongly TONAL source concentrates here; broadband hiss spreads it thin)")
    for i in top[:6]:
        print(f"      {f[i]:7.1f} Hz   {dP[i]/tot*100:5.2f}%")

    rpm = runs[hi]["state"]["rpm"] or 0
    if rpm:
        print(f"\n[fan] blade-pass candidates at {rpm} rpm (BPF = rpm/60 x blades):")
        for nb in (5, 7, 9, 11):
            bpf = rpm / 60.0 * nb
            near = min(top, key=lambda i: abs(f[i] - bpf))
            print(f"      {nb:2d} blades -> {bpf:7.1f} Hz   nearest measured peak "
                  f"{f[near]:7.1f} Hz  (off by {abs(f[near]-bpf):6.1f} Hz)")

    verdict = "TONAL — adaptive notch driven by the tach is the right tool" if peak_frac > 0.25 \
        else "BROADBAND — a notch will not help; use distance, spatial nulling, or fan-quiet-before-listen"
    print(f"\n[fan] VERDICT: {verdict}")

    json.dump({"verdict": verdict, "peak_frac": float(peak_frac),
               "bands": {nm: float(dP[(f >= a) & (f < b)].sum() / tot)
                         for nm, (a, b) in bands.items()},
               "top_hz": [float(f[i]) for i in top],
               "runs": {k2: {"state": v["state"], "rms": v["rms"]} for k2, v in runs.items()}},
              open(args.out, "w"), indent=2)
    print(f"[fan] wrote {args.out}")


if __name__ == "__main__":
    main()
