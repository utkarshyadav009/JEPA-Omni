"""data/m4_easycom_turntaking.py — real EasyCom turn-taking ticks for M4a
Phase-1a LoRA training.

SPEAK ticks: real utterances (same as data/m4_speech_dataset.py).
SILENCE ticks: real GAPS between consecutive utterances from the SAME
participant's Close_Microphone_Audio file -- i.e. genuine recorded silence/
background, not synthetic. Requires a gap of at least SILENCE_MIN_GAP_SEC;
the fed window is SILENCE_WINDOW_SEC drawn from the gap's midpoint (avoids
overlapping the tail/onset of the adjacent real utterances).

This is EasyCom-derived turn-taking signal for the M4b (speech-only)
single-stream task in Phase 1a's mixed training -- deliberately NOT
combined with M3/VGGSound in the same example (that's Phase 1b's job, the
full duplex loop). Session-level split matches data/m4_speech_dataset.py's
TEST_SESSIONS so there's no leakage between the two EasyCom-derived tasks.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.m4_speech_dataset import (EASYCOM_ROOT, VIDEO_FPS, TEST_SESSIONS, WHISPER_SR,
                                     _read_json_robust)

SILENCE_MIN_GAP_SEC = 1.0
SILENCE_WINDOW_SEC = 2.0

# 3-class extension (2026-07-25): a SPEAK tick is reclassified as
# BACKCHANNEL if its transcript, after stripping disfluency markers
# ([L]/[H]/[X]/[T]/[U]/[C] etc.) and punctuation, consists ENTIRELY of
# short acknowledgment tokens (<=2 words) -- or nothing at all (a bare
# laugh/nonverbal marker like "[L]"). Deliberately conservative: "yeah"
# followed by real content ("Yeah, I think...") is NOT a backchannel, it's
# a real turn that happens to start with an acknowledgment word. Mined on
# the full 4,482-segment pool: 899 (20.1%) classified as backchannel.
BACKCHANNEL_WORDS = {
    "yeah", "yep", "yup", "uh-huh", "uhhuh", "mm-hmm", "mmhmm", "mhm", "okay", "ok", "kay",
    "right", "alright", "gotcha", "cool", "wow", "oh", "ah", "huh", "sure", "yea",
}
_DISFLUENCY_RE = re.compile(r"\[[A-Za-z]+\]")


def is_backchannel(text: str) -> bool:
    stripped = _DISFLUENCY_RE.sub("", text).strip()
    stripped_nopunct = re.sub(r"[^\w\s\-]", "", stripped).strip().lower()
    if stripped_nopunct == "":
        return True   # laugh/nonverbal only, e.g. "[L]"
    words = stripped_nopunct.split()
    if len(words) > 2:
        return False
    return all(w in BACKCHANNEL_WORDS for w in words)


@dataclass
class TurnTakingTick:
    session: int
    is_speak: bool
    audio_path: str
    start_sec: float
    end_sec: float
    text: str   # transcription if speak, "" if silence
    label3: str = "silence"   # "speak" | "silence" | "backchannel" -- 3-class extension


def build_ticks(root: str = EASYCOM_ROOT) -> Tuple[List[TurnTakingTick], List[TurnTakingTick]]:
    st_root = os.path.join(root, "Speech_Transcriptions")
    cm_root = os.path.join(root, "Close_Microphone_Audio")
    sessions = sorted(glob.glob(os.path.join(st_root, "Session_*")), key=lambda p: int(p.rsplit("_", 1)[1]))

    train, test = [], []
    for sess_dir in sessions:
        sess_id = int(sess_dir.rsplit("_", 1)[1])
        cm_sess_dir = os.path.join(cm_root, f"Session_{sess_id}")
        available_audio = {}
        for wav in glob.glob(os.path.join(cm_sess_dir, "*.wav")):
            base = os.path.basename(wav)[:-4]
            chunk, tag = base.split("_Participant_ID_")
            available_audio[(chunk, int(tag))] = wav

        bucket = test if sess_id in TEST_SESSIONS else train

        # group segments by (chunk, participant) so gaps are computed within
        # the SAME 60s audio file for the SAME speaker
        by_key: Dict[Tuple[str, int], List[Dict]] = {}
        for cf in sorted(glob.glob(os.path.join(sess_dir, "*.json"))):
            chunk = os.path.basename(cf)[:-5]
            entries = _read_json_robust(cf)
            for e in entries:
                pid = e.get("Participant_ID")
                key = (chunk, pid)
                if key not in available_audio:
                    continue
                text = (e.get("Transcription") or "").strip()
                if not text:
                    continue
                start_sec = e["Start_Frame"] / VIDEO_FPS
                end_sec = e["End_Frame"] / VIDEO_FPS
                if end_sec <= start_sec:
                    continue
                by_key.setdefault(key, []).append({"start": start_sec, "end": end_sec, "text": text})

        for (chunk, pid), segs in by_key.items():
            segs.sort(key=lambda s: s["start"])
            audio_path = available_audio[(chunk, pid)]
            for s in segs:
                label3 = "backchannel" if is_backchannel(s["text"]) else "speak"
                bucket.append(TurnTakingTick(session=sess_id, is_speak=True, audio_path=audio_path,
                                              start_sec=s["start"], end_sec=s["end"], text=s["text"],
                                              label3=label3))
            for i in range(1, len(segs)):
                gap_start, gap_end = segs[i - 1]["end"], segs[i]["start"]
                gap_len = gap_end - gap_start
                if gap_len >= SILENCE_MIN_GAP_SEC:
                    mid = (gap_start + gap_end) / 2
                    w0 = max(gap_start, mid - SILENCE_WINDOW_SEC / 2)
                    w1 = min(gap_end, mid + SILENCE_WINDOW_SEC / 2)
                    bucket.append(TurnTakingTick(session=sess_id, is_speak=False, audio_path=audio_path,
                                                  start_sec=w0, end_sec=w1, text=""))
    return train, test


class EasyComTurnTakingDataset(Dataset):
    def __init__(self, ticks: List[TurnTakingTick]):
        self.ticks = ticks
        self._cache: Dict[str, Tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.ticks)

    def _load(self, path: str):
        if path not in self._cache:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._cache[path] = (audio, sr)
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[path]

    def __getitem__(self, idx: int) -> Dict:
        t = self.ticks[idx]
        audio, sr = self._load(t.audio_path)
        i0, i1 = max(0, int(t.start_sec * sr)), min(len(audio), int(t.end_sec * sr))
        clip = audio[i0:i1]
        if clip.size == 0:
            clip = np.zeros(int(0.02 * sr), dtype=np.float32)
        if sr != WHISPER_SR:
            import librosa
            clip = librosa.resample(clip, orig_sr=sr, target_sr=WHISPER_SR)
            sr = WHISPER_SR
        duration_sec = clip.shape[0] / sr
        return {"waveform": clip.astype(np.float32), "duration_sec": duration_sec,
                "is_speak": t.is_speak, "text": t.text, "session": t.session, "label3": t.label3}


if __name__ == "__main__":
    train, test = build_ticks()
    n_speak_train = sum(t.is_speak for t in train)
    n_speak_test = sum(t.is_speak for t in test)
    print(f"train: {len(train)} ticks ({n_speak_train} speak / {len(train)-n_speak_train} silence)")
    print(f"test:  {len(test)} ticks ({n_speak_test} speak / {len(test)-n_speak_test} silence)")
    print(f"train sessions={sorted(set(t.session for t in train))}  test sessions={sorted(set(t.session for t in test))}")
