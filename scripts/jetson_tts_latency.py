"""scripts/jetson_tts_latency.py — item 6a: real TTS on Jetson (Piper,
CPU/onnxruntime, no CUDA dependency). Measures time-to-audio SEPARATELY
from one-time model load: a real streaming loop loads the voice once at
startup and reuses it, so per-utterance latency (not load time) is what
matters for the "time-to-audio" number.

Two conditions, matching the instruction exactly:
  1. A synthesized BACKCHANNEL from a fixed inventory (short, e.g. "Mm-hmm.",
     "I see.", "Right.") -- these are pre-scriptable, so this measures the
     lower bound: how fast can the robot make a sound after deciding to
     backchannel.
  2. A GENERATED turn (longer, ~1-2 sentences, representative of what
     Qwen's soft-prompted generation would actually produce) -- the upper
     bound relevant case: how long from "generation finished" to "audio
     starts playing."

Usage (on the Jetson):
    python3 scripts/jetson_tts_latency.py
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

MODEL_PATH = Path.home() / "piper_voices" / "en_US-lessac-medium.onnx"
CONFIG_PATH = Path.home() / "piper_voices" / "en_US-lessac-medium.onnx.json"
OUT_PATH = Path.home() / "jetson_tts_latency_results.json"

BACKCHANNEL_INVENTORY = ["Mm-hmm.", "I see.", "Right.", "Got it.", "Okay."]
GENERATED_TURN_EXAMPLES = [
    "Sure, I can help with that. Let me take a look and get back to you in just a moment.",
    "That's a good question. Based on what I'm seeing, I'd say the second option looks better.",
    "I understand what you mean. Let's go ahead and try that approach first.",
]


def main() -> None:
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    print("[tts-latency] loading Piper voice (one-time cost)...", flush=True)
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(MODEL_PATH), config_path=str(CONFIG_PATH))
    load_s = time.perf_counter() - t0
    print(f"[tts-latency] voice loaded in {load_s:.2f}s (one-time, NOT counted in time-to-audio)", flush=True)

    syn_config = SynthesisConfig()

    def synth_once(text: str) -> float:
        """Returns wall-clock seconds from call to having a complete audio
        array in hand (time-to-audio for a REUSED, already-loaded voice)."""
        t0 = time.perf_counter()
        chunks = list(voice.synthesize(text, syn_config=syn_config))
        _ = b"".join(c.audio_int16_bytes for c in chunks)  # force full generation, not lazy
        return time.perf_counter() - t0

    # warm-up (first real call after load sometimes pays extra one-time JIT/cache costs)
    _ = synth_once("Warm up.")

    results = {}

    print("[tts-latency] === backchannel inventory (fixed, short) ===", flush=True)
    bc_lat = []
    for text in BACKCHANNEL_INVENTORY:
        for _ in range(5):
            lat = synth_once(text)
            bc_lat.append(lat)
    results["backchannel"] = {
        "n": len(bc_lat), "mean_s": statistics.mean(bc_lat), "median_s": statistics.median(bc_lat),
        "max_s": max(bc_lat), "min_s": min(bc_lat),
        "texts_used": BACKCHANNEL_INVENTORY,
    }
    print(json.dumps(results["backchannel"], indent=2), flush=True)

    print("[tts-latency] === generated turn (longer, representative) ===", flush=True)
    gen_lat = []
    for text in GENERATED_TURN_EXAMPLES:
        for _ in range(5):
            lat = synth_once(text)
            gen_lat.append(lat)
    results["generated_turn"] = {
        "n": len(gen_lat), "mean_s": statistics.mean(gen_lat), "median_s": statistics.median(gen_lat),
        "max_s": max(gen_lat), "min_s": min(gen_lat),
        "texts_used": GENERATED_TURN_EXAMPLES,
    }
    print(json.dumps(results["generated_turn"], indent=2), flush=True)

    results["one_time_voice_load_s"] = load_s
    results["cpu_onnxruntime_no_cuda"] = True

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[tts-latency] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
