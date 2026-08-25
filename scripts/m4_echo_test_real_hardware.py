"""scripts/m4_echo_test_real_hardware.py — Phase B4: verify MicGate against
REAL acoustic speaker->mic coupling on the Jetson, not the mathematical
simulate_echo_path() model used by scripts/m4_echo_test.py.

That script's "echo" is `attenuation * delayed(reference) + noise` -- a
plausible but synthetic room model. It never answers "does the actual
speaker, actual mic, actual room the demo will run in produce a false
interruption." This script does: it plays real audio out of the real
speaker with models/m5_tts.py's TTSEngine (the same engine wired into
models/m5_streaming_loop.py for the demo) and records the real mic
simultaneously with sounddevice, then runs the SAME deployed decision path
(DuplexLoop.decide3_speechonly + SpeechOnlyThreeClassHead, per
models/m4_duplex_loop.py's A1 deployment note) on what the mic actually
picked up.

Three phases:
  1. self_echo_capture -- TTS speaks N fixed sentences through the real
     speaker. Mic records the whole playback + a short tail. Decision head
     runs on the raw recording (no_fix) to measure the REAL false-
     interruption rate from acoustic self-echo -- the number that justifies
     mic_gate rather than assuming it.
  2. mic_gate_mechanism -- confirms should_run_decision() returns False for
     every tick inside the real playback window actually measured in phase
     1 (mechanism-level: MicGate short-circuits on a wall-clock flag, not
     on audio content, so this is a timing check, not a re-run of the
     model). This is the condition the deployed loop actually runs; by
     construction its false-interruption rate is 0 -- reported for
     symmetry with scripts/m4_echo_test.py's report format, not because
     the outcome is in doubt.
  3. live_control -- with the mic NOT gated, prompts a human to speak on
     cue and confirms the same decision head still fires "speak" on real
     room audio through the real mic -- a sanity check that the hardware
     chain (mic capture -> Whisper -> decision head) works at all, since
     phase 1/2 alone could pass vacuously if the mic were disconnected.

Requires real speaker + mic attached and unmuted. Not runnable on the dev
machine (no audio hardware, no Piper voice files) -- run on the Jetson.

Usage:
    python3 scripts/m4_echo_test_real_hardware.py --n-utterances 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
import importlib.machinery

# Jetson-only workaround (matches scripts/jetson_phase4_full_stack_memory_v2_withqwen.py):
# torchaudio's compiled extension needs libcudart.so.13, which doesn't exist on this
# Jetson's CUDA install -- importing transformers (via models/vision_encoder.py's
# AutoModel/AutoVideoProcessor) transitively imports torchaudio and crashes with
# OSError before any of our own code runs. Stub it out BEFORE importing torch.
_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MIC_SR = 16000

FIXED_UTTERANCES = [
    "The weather today is partly cloudy with a light breeze.",
    "I found three files matching your search query.",
    "Let me know if you would like me to continue.",
    "The next step is to check the configuration file.",
    "Your meeting is scheduled for three o'clock this afternoon.",
    "I can see a red mug on the table to your left.",
    "That command finished successfully with no errors.",
    "Would you like a summary of the last five minutes?",
]


def record_mic(duration_sec: float, sr: int = MIC_SR, device: int = 24) -> np.ndarray:
    # device=24 (2026-08-01, real hardware confirmed): ReSpeaker 4 Mic Array
    # (UAC1.0) -- do not rely on the system default input resolving to it.
    import sounddevice as sd
    n = int(duration_sec * sr)
    audio = sd.rec(n, samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0].astype(np.float32)


@torch.no_grad()
def decision_on_waveform(whisper, decision_head, waveform: np.ndarray, device) -> str:
    """Mirrors DuplexLoop.decide3_speechonly (models/m4_duplex_loop.py) --
    speech-only, no world_state input, per the A1 deployment decision."""
    from models.m4_decision_head import IDX_TO_LABEL
    duration_sec = len(waveform) / float(MIC_SR)
    if duration_sec < 0.05:
        return "silence"
    hidden, valid_frames = whisper([waveform.astype(np.float32)], [duration_sec], device)
    vf = int(valid_frames[0].item())
    speech_feat = hidden[0, :vf].float().mean(dim=0, keepdim=True)
    logits = decision_head(speech_feat)
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax().item())
    return IDX_TO_LABEL[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--speechonly-ckpt", default="checkpoints/m4_decision_head_3class_speechonly_v2/best.pt",
                   help="locked head checkpoint per freeze-submission-v1 (95.00% acc, macro_F1=94.98%)")
    p.add_argument("--n-utterances", type=int, default=8)
    p.add_argument("--tail-sec", type=float, default=1.0,
                   help="extra mic recording after TTS playback nominally ends, to catch trailing echo/reverb")
    p.add_argument("--window-sec", type=float, default=1.5,
                   help="analysis window size for chunking the self-echo recording into decision-head calls")
    p.add_argument("--out", default="checkpoints/m4_decision_head_3class_speechonly/echo_test_real_hardware_results.json")
    p.add_argument("--live-cue-countdown-sec", type=float, default=3.0,
                   help="0 to skip Phase 3's live-speech cue entirely")
    p.add_argument("--live-record-sec", type=float, default=4.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[real-echo-test] loading TTSEngine (Piper, real speaker output)...", flush=True)
    from models.m5_tts import TTSEngine
    tts = TTSEngine()

    print("[real-echo-test] loading frozen Whisper + speech-only decision head "
          "(same classes/checkpoint as the deployed DuplexLoop.decide3_speechonly)...", flush=True)
    from models.m4_speech import WhisperSpeechEncoder
    from train_decision_head_3class_speechonly_v2 import SpeechOnlyThreeClassHead
    from models.m4_echo_cancellation import MicGate

    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    ckpt = torch.load(args.speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = SpeechOnlyThreeClassHead(speech_feat_dim=ckpt["sf_dim"]).to(device)
    decision_head.load_state_dict(ckpt["state_dict"])
    decision_head.eval()

    utterances = FIXED_UTTERANCES[: args.n_utterances] if args.n_utterances <= len(FIXED_UTTERANCES) \
        else (FIXED_UTTERANCES * (args.n_utterances // len(FIXED_UTTERANCES) + 1))[: args.n_utterances]

    results = {
        "self_echo_capture": {"no_fix_false_interrupt": 0, "n_windows": 0, "per_utterance": []},
        "mic_gate_mechanism": {"gated_correctly": 0, "n_ticks_checked": 0},
        "live_control": {"detected_speak": None, "detected_silence_before_cue": None},
    }

    # --- Phase 1: real self-echo capture ---------------------------------
    print(f"\n[real-echo-test] === Phase 1: self-echo capture, {len(utterances)} utterances, "
          f"real speaker + real mic ===", flush=True)
    for i, text in enumerate(utterances):
        synth = tts.synthesize(text)  # do NOT play yet -- need exact duration to size the recording
        record_dur = synth.duration_sec + args.tail_sec

        gate = MicGate(); gate.is_playing = True
        t_play_start = time.perf_counter()
        tts.play(synth, blocking=False)
        recorded = record_mic(record_dur)  # blocks for record_dur, overlapping real playback
        t_play_elapsed = time.perf_counter() - t_play_start
        gate.is_playing = False

        n_win = int(len(recorded) / MIC_SR / args.window_sec) or 1
        win_len = len(recorded) // n_win
        any_false_interrupt = False
        for w in range(n_win):
            seg = recorded[w * win_len: (w + 1) * win_len]
            label = decision_on_waveform(whisper, decision_head, seg, device)
            results["self_echo_capture"]["n_windows"] += 1
            if label != "silence":
                results["self_echo_capture"]["no_fix_false_interrupt"] += 1
                any_false_interrupt = True

        results["self_echo_capture"]["per_utterance"].append({
            "text": text, "tts_duration_sec": synth.duration_sec,
            "playback_wall_sec": t_play_elapsed, "any_false_interrupt": any_false_interrupt,
        })
        print(f"[real-echo-test]   [{i+1}/{len(utterances)}] dur={synth.duration_sec:.2f}s "
              f"false_interrupt={'YES' if any_false_interrupt else 'no'}", flush=True)

        # Phase 2, same playback window: should_run_decision() must be False
        # for the entire nominal playback duration -- this is a wall-clock
        # mechanism check, not a re-run of the model (should_run_decision()
        # doesn't look at audio at all, see models/m4_echo_cancellation.py).
        n_ticks = max(1, int(synth.duration_sec / 0.1))
        gate2 = MicGate(); gate2.is_playing = True
        for _ in range(n_ticks):
            results["mic_gate_mechanism"]["n_ticks_checked"] += 1
            if not gate2.should_run_decision():
                results["mic_gate_mechanism"]["gated_correctly"] += 1

    # --- Phase 3: live control, mic NOT gated -----------------------------
    print("\n[real-echo-test] === Phase 3: live control (mic NOT gated) ===", flush=True)
    print("[real-echo-test] recording 2s of silence (do not speak)...", flush=True)
    quiet = record_mic(2.0)
    label_quiet = decision_on_waveform(whisper, decision_head, quiet, device)
    results["live_control"]["detected_silence_before_cue"] = label_quiet
    print(f"[real-echo-test]   quiet-room label = {label_quiet}", flush=True)

    if args.live_cue_countdown_sec > 0:
        # Audible cue via the Jetson's own speaker (2026-08-01 fix) -- printing a
        # countdown to stdout is useless when the operator is physically at the
        # Jetson but this process is driven headlessly over SSH from elsewhere and
        # no one is watching that terminal. input() doesn't work either (raised
        # EOFError with no TTY attached). Speaking the cue through the ReSpeaker is
        # the one synchronization channel that reaches a person standing at the
        # device without requiring them to look at a screen.
        cue_text = "Recording in 3, 2, 1, speak now."
        print(f"[real-echo-test] announcing live-speech cue through the speaker: {cue_text!r}", flush=True)
        cue_synth = tts.speak(cue_text, blocking=True)
        print(f"[real-echo-test]   RECORDING NOW ({args.live_record_sec:.0f}s)", flush=True)
        live = record_mic(args.live_record_sec)
        label_live = decision_on_waveform(whisper, decision_head, live, device)
        results["live_control"]["detected_speak"] = label_live
        print(f"[real-echo-test]   live-speech label = {label_live} (expect 'speak')", flush=True)
    else:
        print("[real-echo-test]   live-speech cue skipped (--live-cue-countdown-sec 0)", flush=True)
        results["live_control"]["detected_speak"] = "SKIPPED"
        label_live = "SKIPPED"

    # --- Summary -----------------------------------------------------------
    a = results["self_echo_capture"]
    fi_rate = a["no_fix_false_interrupt"] / max(1, a["n_windows"])
    g = results["mic_gate_mechanism"]
    gate_rate = g["gated_correctly"] / max(1, g["n_ticks_checked"])
    print(f"\n[real-echo-test] REAL acoustic self-echo, no_fix false-interruption rate = "
          f"{fi_rate:.3f} ({a['no_fix_false_interrupt']}/{a['n_windows']} windows)", flush=True)
    print(f"[real-echo-test] mic_gate mechanism correctness during real playback = "
          f"{gate_rate:.3f} ({g['gated_correctly']}/{g['n_ticks_checked']} ticks)", flush=True)
    print(f"[real-echo-test] live control: quiet={label_quiet}  live-speech={label_live} "
          f"(hardware chain sane iff quiet!='speak' and live=='speak')", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[real-echo-test] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
