"""scripts/m4_echo_test.py — test the self-interruption fix: does the
decision head fire "speak" on the robot's OWN echoed speech (false
interruption), and does it still correctly fire on a REAL simultaneous
user interruption overlapping that echo?

Three conditions per test case:
  no_fix:    raw simulated mic signal (echo, and optionally + real interrupt)
  mic_gate:  mic zeroed while "TTS is playing" -- blocks everything, echo AND
             any real simultaneous interruption
  aec:       NLMS adaptive echo cancellation using the known reference

Usage:
    python scripts/m4_echo_test.py --n-cases 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.m4_speech_dataset import build_segments, EasyComSpeechDataset
from models.m4_speech import WhisperSpeechEncoder
from models.m4_decision_head import SpeakSilenceHead, DecisionHeadConfig
from models.m4_echo_cancellation import MicGate, PyAecCanceller, simulate_echo_path


@torch.no_grad()
def decision_on_waveform(whisper, decision_head, waveform: np.ndarray, device, threshold: float) -> bool:
    """Returns True if the decision head says SPEAK (i.e. would trigger an
    interruption under the simple binary policy: any detected speech halts
    generation)."""
    duration_sec = len(waveform) / 16000.0
    if duration_sec < 0.05:
        return False
    hidden, valid_frames = whisper([waveform.astype(np.float32)], [duration_sec], device)
    vf = int(valid_frames[0].item())
    speech_feat = hidden[0, :vf].float().mean(dim=0, keepdim=True)
    world_state = torch.zeros(1, decision_head.cfg.world_state_dim, device=device)
    logit = decision_head(world_state, speech_feat)
    prob = torch.sigmoid(logit).item()
    return prob > threshold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--decision-head-ckpt", default="checkpoints/m4_decision_head/best.pt")
    p.add_argument("--n-cases", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="checkpoints/m4_decision_head/echo_test_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print("[echo-test] loading frozen whisper + decision head...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    cfg = DecisionHeadConfig(**ckpt["cfg"])
    decision_head = SpeakSilenceHead(cfg).to(device)
    decision_head.load_state_dict(ckpt["state_dict"])
    decision_head.eval()
    threshold = ckpt["threshold"]

    print("[echo-test] loading EasyCom segments, building a long concatenated 'TTS session' stream...", flush=True)
    train_segs, test_segs = build_segments()
    all_segs = train_segs + test_segs
    rng.shuffle(all_segs)
    # NLMS needs real convergence time -- individual EasyCom clips are often
    # <1-2s, nowhere near enough excitation for a 1024-tap filter on real
    # (non-stationary, broadband) speech. Build one long reference stream by
    # concatenating many segments (a realistic stand-in for "the robot has
    # been talking for a while in this session"), run the canceller
    # CONTINUOUSLY (not reset per clip, matching how AEC actually runs in
    # production), then evaluate false-interruption on individual SEGMENTS
    # cut from the converged (post-warm-up) tail vs the unconverged head --
    # reporting both honestly rather than only the flattering one.
    n_pool = min(len(all_segs), args.n_cases * 6)
    pool = EasyComSpeechDataset(all_segs[:n_pool])
    concat_chunks, seg_bounds = [], []
    cursor = 0
    for i in range(n_pool):
        w = pool[i]["waveform"]
        concat_chunks.append(w)
        seg_bounds.append((cursor, cursor + len(w)))
        cursor += len(w)
    tts_stream = np.concatenate(concat_chunks).astype(np.float32)
    print(f"[echo-test] TTS stream length: {len(tts_stream)/16000:.1f}s from {n_pool} concatenated segments", flush=True)

    echo_stream = simulate_echo_path(tts_stream, sr=16000, delay_ms=15.0, attenuation=0.6,
                                      noise_level=0.01, rng=np_rng)
    canceller = PyAecCanceller(sr=16000, frame_size=256, filter_length=1024, enable_preprocess=True)
    residual_stream = canceller.process(reference=tts_stream, mic_signal=echo_stream)
    warmup_boundary = len(tts_stream) // 3   # first third = convergence warm-up, rest = converged
    erle_warmup = 10 * np.log10(np.mean(echo_stream[:warmup_boundary] ** 2) / (np.mean(residual_stream[:warmup_boundary] ** 2) + 1e-12))
    erle_converged = 10 * np.log10(np.mean(echo_stream[warmup_boundary:] ** 2) / (np.mean(residual_stream[warmup_boundary:] ** 2) + 1e-12))
    print(f"[echo-test] NLMS ERLE: warm-up third={erle_warmup:.2f}dB  converged two-thirds={erle_converged:.2f}dB", flush=True)

    results = {"self_echo_only": {"no_fix": 0, "mic_gate": 0, "aec_warmup": 0, "aec_converged": 0, "n_warmup": 0, "n_converged": 0},
               "self_echo_plus_real_interrupt": {"no_fix": 0, "mic_gate": 0, "aec_converged": 0, "n": 0},
               "nlms_erle_db": {"warmup_third": float(erle_warmup), "converged_two_thirds": float(erle_converged)}}

    # spread eval indices across the WHOLE stream (not just the first
    # n_cases segments, which would all fall inside the warm-up region) so
    # both warm-up and converged regions get real coverage
    eval_indices = sorted(rng.sample(range(len(seg_bounds)), min(args.n_cases, len(seg_bounds))))
    n_eval = len(eval_indices)
    for i in eval_indices:
        s0, s1 = seg_bounds[i]
        is_converged_region = s0 >= warmup_boundary
        echo_seg = echo_stream[s0:s1]
        residual_seg = residual_stream[s0:s1]
        gate = MicGate(); gate.is_playing = True

        d_no_fix = decision_on_waveform(whisper, decision_head, echo_seg, device, threshold)
        d_gate = False if not gate.should_run_decision() else decision_on_waveform(
            whisper, decision_head, echo_seg, device, threshold)
        d_aec = decision_on_waveform(whisper, decision_head, residual_seg, device, threshold)

        key = "n_converged" if is_converged_region else "n_warmup"
        aec_key = "aec_converged" if is_converged_region else "aec_warmup"
        results["self_echo_only"][key] += 1
        results["self_echo_only"]["no_fix"] += int(d_no_fix)
        results["self_echo_only"]["mic_gate"] += int(d_gate)
        results["self_echo_only"][aec_key] += int(d_aec)

        if is_converged_region:
            interrupt_item = pool[(i + n_eval // 2) % n_pool]
            interrupt_audio = interrupt_item["waveform"]
            min_len = min(len(echo_seg), len(interrupt_audio))
            mixed_no_fix = echo_seg[:min_len] + interrupt_audio[:min_len]
            mixed_residual = residual_seg[:min_len] + interrupt_audio[:min_len]   # AEC only removes the echo-correlated part
            gate2 = MicGate(); gate2.is_playing = True

            d2_no_fix = decision_on_waveform(whisper, decision_head, mixed_no_fix, device, threshold)
            d2_gate = False if not gate2.should_run_decision() else decision_on_waveform(
                whisper, decision_head, mixed_no_fix, device, threshold)
            d2_aec = decision_on_waveform(whisper, decision_head, mixed_residual, device, threshold)
            results["self_echo_plus_real_interrupt"]["n"] += 1
            results["self_echo_plus_real_interrupt"]["no_fix"] += int(d2_no_fix)
            results["self_echo_plus_real_interrupt"]["mic_gate"] += int(d2_gate)
            results["self_echo_plus_real_interrupt"]["aec_converged"] += int(d2_aec)

        if (i + 1) % 10 == 0:
            print(f"[echo-test] processed {i+1}/{n_eval}", flush=True)

    print("\n[echo-test] === Case A: self-echo ONLY (no real user speech) -- false-interruption rate ===", flush=True)
    a = results["self_echo_only"]
    n_w, n_c = max(1, a["n_warmup"]), max(1, a["n_converged"])
    print(f"[echo-test]   no_fix    : false-interruption rate = {(a['no_fix']/(n_w+n_c)):.3f}  "
          f"({a['no_fix']}/{n_w+n_c})", flush=True)
    print(f"[echo-test]   mic_gate  : false-interruption rate = {(a['mic_gate']/(n_w+n_c)):.3f}  "
          f"({a['mic_gate']}/{n_w+n_c})", flush=True)
    print(f"[echo-test]   aec (warm-up third, filter still converging)  : rate = {a['aec_warmup']/n_w:.3f}  "
          f"({a['aec_warmup']}/{n_w})", flush=True)
    print(f"[echo-test]   aec (converged two-thirds)                    : rate = {a['aec_converged']/n_c:.3f}  "
          f"({a['aec_converged']}/{n_c})", flush=True)

    print("\n[echo-test] === Case B: self-echo + REAL simultaneous interruption (converged region only) "
          "-- detection-preserved rate ===", flush=True)
    b = results["self_echo_plus_real_interrupt"]
    for cond, key in [("no_fix", "no_fix"), ("mic_gate", "mic_gate"), ("aec", "aec_converged")]:
        rate = b[key] / max(1, b["n"])
        print(f"[echo-test]   {cond:10s}: real-interruption STILL detected rate = {rate:.3f}  ({b[key]}/{b['n']})", flush=True)

    print(f"\n[echo-test] NLMS ERLE: warm-up={results['nlms_erle_db']['warmup_third']:.2f}dB  "
          f"converged={results['nlms_erle_db']['converged_two_thirds']:.2f}dB", flush=True)
    print("[echo-test] Summary: AEC needs a convergence warm-up period (production systems run continuously "
          "across a whole session, so this is a one-time cost, not per-utterance). After convergence, AEC "
          "should show LOW false-interruption in Case A AND HIGH detection-preserved in Case B. mic_gate shows "
          "LOW false-interruption in Case A but ALSO LOW (~0) detection in Case B -- it can't distinguish echo "
          "from a real interruption, it blocks everything.", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[echo-test] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
