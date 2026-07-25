"""scripts/easycom_characterize.py — characterize the downloaded EasyCom
dataset (structure, annotation format, usable turn-taking segments, and a
VERIFIED check of genuine multi-channel spatial audio -- channel count and
actual inter-channel difference, not assumed from documentation).

Acquisition/characterization only, per instruction -- does NOT train
anything.

Usage:
    python scripts/easycom_characterize.py --root /home/utkarsh/raid2-data/easycom/EasyComDataset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_report(root: str) -> None:
    print(f"\n=== directory structure under {root} ===", flush=True)
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            n_files = sum(len(files) for _, _, files in os.walk(full))
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, files in os.walk(full) for f in files)
            print(f"  {entry}/  ({n_files} files, {human_size(size)})", flush=True)
        else:
            print(f"  {entry}  ({human_size(os.path.getsize(full))})", flush=True)


def audio_channel_report(root: str, pattern_dirs, n_sample: int = 8) -> None:
    print(f"\n=== audio channel verification (n_sample={n_sample} files per dir) ===", flush=True)
    for d in pattern_dirs:
        full_dir = os.path.join(root, d)
        if not os.path.isdir(full_dir):
            print(f"  {d}: NOT FOUND", flush=True)
            continue
        wavs = glob.glob(os.path.join(full_dir, "**", "*.wav"), recursive=True)
        if not wavs:
            print(f"  {d}: no .wav files found", flush=True)
            continue
        print(f"  {d}: {len(wavs)} wav files total", flush=True)
        sample = wavs[:n_sample]
        for w in sample:
            info = sf.info(w)
            rel = os.path.relpath(w, full_dir)
            print(f"    {rel}: channels={info.channels}  samplerate={info.samplerate}  "
                  f"duration={info.duration:.1f}s  subtype={info.subtype}", flush=True)
            if info.channels > 1:
                # VERIFY genuine spatial audio: read a short window, check
                # inter-channel correlation and RMS difference. Mono-
                # duplicated channels would have correlation ~1.0 and
                # near-zero difference; real spatial audio should not.
                data, sr = sf.read(w, frames=min(info.frames, sr_frames := info.samplerate * 5))
                if data.ndim == 2 and data.shape[1] >= 2:
                    ch0, ch1 = data[:, 0], data[:, 1]
                    corr = float(np.corrcoef(ch0, ch1)[0, 1])
                    diff_rms = float(np.sqrt(np.mean((ch0 - ch1) ** 2)))
                    sig_rms = float(np.sqrt(np.mean(ch0 ** 2)) + 1e-9)
                    print(f"      ch0 vs ch1 (first 5s): correlation={corr:.4f}  "
                          f"diff_rms/sig_rms={diff_rms/sig_rms:.4f}  "
                          f"{'SUSPICIOUS (looks mono-duplicated)' if corr > 0.999 and diff_rms/sig_rms < 0.01 else 'genuinely distinct channels'}",
                          flush=True)


def annotation_report(root: str) -> None:
    print(f"\n=== annotation format + turn-taking segment count ===", flush=True)
    print("  actual layout: Main/Speech_Transcriptions/Session_N/<chunk>.json, "
          "each chunk = a list of utterance-level segments with Start_Frame, "
          "End_Frame, Participant_ID (speaker), Transcription, Target_of_Speech "
          "(addressee: a participant id, a list of ids, and/or the literal "
          "string 'Group'). This IS real addressee-labeled turn-taking data, "
          "richer than a binary talking-to-me signal.", flush=True)

    st_root = os.path.join(root, "Main", "Speech_Transcriptions")
    sessions = sorted(glob.glob(os.path.join(st_root, "Session_*")))
    print(f"  {len(sessions)} sessions with transcriptions found under Main/Speech_Transcriptions", flush=True)

    total_chunks = 0
    total_segments = 0
    total_turn_transitions = 0          # consecutive utterances, different SPEAKER (Participant_ID)
    total_group_addressed = 0           # Target_of_Speech includes "Group"
    total_individually_addressed = 0    # Target_of_Speech is only specific participant id(s)
    speaker_counts = Counter()
    per_session_segment_counts = []
    malformed = 0

    for sess in sessions:
        chunk_files = sorted(glob.glob(os.path.join(sess, "*.json")))
        total_chunks += len(chunk_files)
        session_segments = 0
        prev_speaker = None
        # sort chunks + within-chunk by Start_Frame for a session-ordered turn sequence
        all_entries = []
        for cf in chunk_files:
            try:
                with open(cf) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed += 1
                continue
            if not isinstance(data, list):
                malformed += 1
                continue
            all_entries.extend(data)
        all_entries.sort(key=lambda e: (e.get("Start_Frame", 0)))
        for e in all_entries:
            session_segments += 1
            speaker = e.get("Participant_ID")
            speaker_counts[speaker] += 1
            target = e.get("Target_of_Speech", [])
            target_list = target if isinstance(target, list) else [target]
            if "Group" in target_list:
                total_group_addressed += 1
            else:
                total_individually_addressed += 1
            if prev_speaker is not None and speaker != prev_speaker:
                total_turn_transitions += 1
            prev_speaker = speaker
        total_segments += session_segments
        per_session_segment_counts.append(session_segments)

    print(f"\n  total annotation chunk files: {total_chunks}", flush=True)
    print(f"  total utterance-level segments (usable turn-taking examples): {total_segments}", flush=True)
    print(f"  total speaker turn-transitions (consecutive utterances, different speaker): {total_turn_transitions}", flush=True)
    print(f"  group-addressed utterances: {total_group_addressed}  "
          f"({100*total_group_addressed/max(1,total_segments):.1f}%)", flush=True)
    print(f"  individually-addressed utterances: {total_individually_addressed}  "
          f"({100*total_individually_addressed/max(1,total_segments):.1f}%)", flush=True)
    print(f"  distinct speaker (Participant_ID) count across all sessions: {len(speaker_counts)}", flush=True)
    print(f"  malformed/unreadable annotation files: {malformed}", flush=True)
    if per_session_segment_counts:
        print(f"  segments per session: mean={np.mean(per_session_segment_counts):.1f}  "
              f"min={min(per_session_segment_counts)}  max={max(per_session_segment_counts)}", flush=True)

    # sample one raw file for the record
    if sessions:
        sample_chunks = sorted(glob.glob(os.path.join(sessions[0], "*.json")))
        if sample_chunks:
            with open(sample_chunks[0]) as f:
                sample_data = json.load(f)
            print(f"\n  sample chunk ({os.path.relpath(sample_chunks[0], root)}): "
                  f"{len(sample_data)} segments, first entry:", flush=True)
            print(f"  {json.dumps(sample_data[0], indent=2)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/utkarsh/raid2-data/easycom/EasyComDataset")
    args = p.parse_args()

    dir_report(args.root)
    audio_channel_report(args.root, [
        "Main/Glasses_Microphone_Array_Audio", "Main/Close_Microphone_Audio",
        "Extra/Glasses_Microphone_Array_Audio", "Extra/Close_Microphone_Audio",
    ])
    annotation_report(args.root)


if __name__ == "__main__":
    main()
