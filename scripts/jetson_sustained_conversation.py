"""scripts/jetson_sustained_conversation.py — M5 Day 2: does memory creep
across a SUSTAINED multi-turn conversation on the Jetson, or stay bounded?

Loads the full int8 stack (same recipe as jetson_phase0_memory.py: CPU-
quantize with torchao Int8WeightOnlyConfig(version=2) + malloc_trim before
GPU move -- both were needed, neither alone was sufficient, see Phase-0
clarification PROVENANCE), then forces N sequential "speak" turns (real
EasyCom audio per turn, cycling through the pool) -- deliberately MORE
generation-dense than a natural conversation, to stress-test the actual
risk factor (does repeated generate_interruptible() calls leak memory
turn-over-turn) as directly and quickly as possible, rather than spending
wall-clock time simulating realistic silence-heavy pacing.

Continuous tegrastats sampling (same background-thread pattern as
jetson_phase0_memory.py's full-tick test) spans the ENTIRE run, not just
before/after -- this is what actually answers "does it creep."

Usage:
    python3 jetson_sustained_conversation.py --n-turns 40
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, tag, device):
    module = module.to("cpu")
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
        print(f"[sustained]   int8-quantized on CPU: {tag}", flush=True)
    except Exception as e:
        print(f"[sustained]   INT8 QUANTIZATION FAILED for {tag}: {e!r}", flush=True)
    gc.collect()
    _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize()
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    return module


def tegra_used_mib():
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--joint-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_joint/best.pt"))
    p.add_argument("--decision-head-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class/best.pt"))
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--n-turns", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--post-turn-cleanup", action="store_true",
                    help="gc.collect+empty_cache+malloc_trim after EVERY generation (test the fix)")
    p.add_argument("--out", default=os.path.expanduser("~/jepa_omni_transfer/sustained_results.json"))
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[sustained] device={torch.cuda.get_device_name(0)}  post_turn_cleanup={args.post_turn_cleanup}", flush=True)

    from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
    from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig
    from models.m4_duplex_loop import DuplexLoop
    from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[sustained] loading whisper (int8)...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16)
    whisper = q_int8_cpu_then_move(whisper, "whisper", device)

    print("[sustained] loading M4b projector (frozen, small -- not quantized)...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location="cpu", weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()
    m4b_projector = m4b_projector.to(device)

    print("[sustained] loading Qwen2.5-1.5B (int8)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)
    llm = q_int8_cpu_then_move(llm, "qwen1.5b", device)

    print("[sustained] loading 3-class decision head...", flush=True)
    dh_ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = ThreeClassHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    duplex = DuplexLoop(None, None, m4b_projector, whisper, decision_head, llm, tokenizer, device)

    print("[sustained] building real EasyCom speak-segment pool...", flush=True)
    _, test_ticks = build_ticks()
    speak_ticks = [t for t in test_ticks if t.label3 == "speak"]
    ds = EasyComTurnTakingDataset(speak_ticks)

    # ---- continuous tegrastats sampling across the WHOLE run ----
    samples = []   # (t, used_mib)
    stop_evt = threading.Event()
    t_start = time.time()

    def _sampler():
        proc = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
        try:
            while not stop_evt.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                t_read = time.time()
                m = re.search(r"RAM (\d+)/(\d+)MB", line)
                if m:
                    samples.append((t_read - t_start, int(m.group(1))))
        finally:
            proc.terminate()

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()
    time.sleep(0.5)

    used0, total0 = tegra_used_mib()
    print(f"[sustained] pre-loop steady state: {used0}MiB / {total0}MiB", flush=True)

    turn_log = []
    print(f"[sustained] running {args.n_turns} sequential generation turns...", flush=True)
    for turn in range(args.n_turns):
        item = ds[turn % len(ds)]
        t0 = time.perf_counter()
        with torch.no_grad():
            hidden, valid_frames = whisper([item["waveform"]], [item["duration_sec"]], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                stoks, smask = m4b_projector(hidden.float(), valid_frames)
            sattn = (~smask).long()
            result = duplex.generate_interruptible(stoks, sattn, max_new_tokens=args.max_new_tokens)
        gen_ms = (time.perf_counter() - t0) * 1000.0

        if args.post_turn_cleanup:
            gc.collect()
            torch.cuda.empty_cache()
            _malloc_trim()

        used, total = tegra_used_mib()
        turn_log.append({"turn": turn, "t_s": time.time() - t_start, "used_mib": used,
                          "gen_ms": gen_ms, "n_tokens": result.n_tokens_generated})
        if (turn + 1) % 5 == 0 or turn == 0:
            print(f"[sustained] turn {turn+1}/{args.n_turns}  used={used}MiB  gen={gen_ms:.0f}ms  "
                  f"tokens={result.n_tokens_generated}", flush=True)

    time.sleep(0.5)
    stop_evt.set()
    sampler_thread.join(timeout=2)

    # ---- trend analysis ----
    xs = list(range(len(turn_log)))
    ys = [t["used_mib"] for t in turn_log]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope_mib_per_turn = cov / var_x if var_x > 0 else 0.0

    first_5_mean = sum(ys[:5]) / min(5, len(ys))
    last_5_mean = sum(ys[-5:]) / min(5, len(ys))

    headroom_at_end = total0 - ys[-1]
    results = {
        "n_turns": args.n_turns, "post_turn_cleanup": args.post_turn_cleanup,
        "pre_loop_used_mib": used0, "total_mib": total0,
        "first_5_turns_mean_used_mib": first_5_mean, "last_5_turns_mean_used_mib": last_5_mean,
        "delta_first_to_last_mib": last_5_mean - first_5_mean,
        "linear_trend_mib_per_turn": slope_mib_per_turn,
        "min_used_mib": min(ys), "max_used_mib": max(ys),
        "headroom_at_end_mib": headroom_at_end,
        "n_continuous_samples": len(samples),
        "turn_log": turn_log,
    }
    print("\n[sustained] === RESULTS ===")
    print(json.dumps({k: v for k, v in results.items() if k != "turn_log"}, indent=2))
    print(f"\n[sustained] first-5-turn mean={first_5_mean:.0f}MiB  last-5-turn mean={last_5_mean:.0f}MiB  "
          f"delta={last_5_mean-first_5_mean:+.0f}MiB  linear trend={slope_mib_per_turn:+.2f}MiB/turn")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sustained] wrote {args.out}")


if __name__ == "__main__":
    main()
