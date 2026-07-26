"""scripts/jetson_vad_cpu_latency.py — Phase 3.3: measure Silero VAD
latency on Jetson CPU (not GPU) -- protects the 7ms interruption
guarantee regardless of GPU contention with vision refresh.

Usage:
    python3 jetson_vad_cpu_latency.py --n 30
"""
import argparse
import sys
import time
import types

import numpy as np
import torch

# Jetson-specific workaround: the installed torchaudio's C-extension
# (_torchaudio.abi3.so) doesn't match this torch build's ABI (version-
# mismatched aarch64 wheel), so `import torchaudio` raises OSError.
# silero_vad.utils_vad only uses torchaudio inside read_audio/save_audio
# (file I/O helpers) -- never in the core model/get_speech_timestamps
# path, which we drive with an in-memory tensor. Stub the module so the
# top-level `import torchaudio` in utils_vad.py succeeds without loading
# the broken extension; the real functions are never called.
import importlib.machinery
_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    torch.set_num_threads(4)
    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    get_speech_timestamps = vad_utils[0]

    wave = (np.random.randn(16000 * 2) * 0.01).astype(np.float32)
    wav_t = torch.from_numpy(wave)

    for _ in range(3):
        _ = get_speech_timestamps(wav_t, vad_model, sampling_rate=16000)

    lats = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        _ = get_speech_timestamps(wav_t, vad_model, sampling_rate=16000)
        lats.append((time.perf_counter() - t0) * 1000.0)

    import statistics
    print(f"[vad-cpu] n={len(lats)} mean={statistics.mean(lats):.2f}ms "
          f"median={statistics.median(lats):.2f}ms max={max(lats):.2f}ms", flush=True)
    import json
    with open("/home/bmo/jetson_vad_cpu_results.json", "w") as f:
        json.dump({"n": len(lats), "mean_ms": statistics.mean(lats), "median_ms": statistics.median(lats),
                   "max_ms": max(lats), "device": "Jetson Orin Nano Super CPU (torch.set_num_threads(4))"}, f, indent=2)


if __name__ == "__main__":
    main()
