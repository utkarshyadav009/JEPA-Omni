"""data/m4_speech_dataset.py — EasyCom Close_Microphone_Audio <-> Transcription
pairs for M4b speech-path training.

Session-level held-out split (NOT random segment split): with only 7
distinct speakers across 12 sessions, splitting at the segment level would
let the same speaker's voice/room acoustics appear in both train and test,
inflating held-out numbers on voice/acoustic familiarity rather than
genuine transcription-grounding generalization. Whole sessions are held out
instead.

Only Participant_IDs that have their OWN Close_Microphone_Audio file for a
given chunk are usable -- the AR-glasses wearer for that session has no
separate close-mic recording (their speech is captured via the glasses
array instead), so their utterances are skipped here (no clean per-speaker
audio target to train the ASR-alignment objective against).

15/338 annotation chunk files are latin-1 encoded, not UTF-8 (accented
characters in transcriptions) -- read with encoding="latin-1" throughout,
per the characterization run's finding.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

EASYCOM_ROOT = "/home/utkarsh/raid2-data/easycom/extracted/Main"
VIDEO_FPS = 20.0            # per EasyCom docs: MPEG4, 20fps -- Start_Frame/End_Frame are on this axis
TEST_SESSIONS = {10, 11, 12}   # held out WHOLE, per the session-level split requirement
WHISPER_SR = 16000


def _read_json_robust(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return json.load(f)


@dataclass
class SpeechSegment:
    session: int
    chunk: str
    participant_id: int
    start_sec: float
    end_sec: float
    transcription: str
    audio_path: str


def build_segments(root: str = EASYCOM_ROOT) -> Tuple[List[SpeechSegment], List[SpeechSegment]]:
    """Returns (train_segments, test_segments), split by whole session."""
    st_root = os.path.join(root, "Speech_Transcriptions")
    cm_root = os.path.join(root, "Close_Microphone_Audio")
    sessions = sorted(glob.glob(os.path.join(st_root, "Session_*")),
                       key=lambda p: int(p.rsplit("_", 1)[1]))

    train, test = [], []
    for sess_dir in sessions:
        sess_id = int(sess_dir.rsplit("_", 1)[1])
        cm_sess_dir = os.path.join(cm_root, f"Session_{sess_id}")
        # which (chunk, participant_id) pairs actually have close-mic audio
        available_audio = {}
        for wav in glob.glob(os.path.join(cm_sess_dir, "*.wav")):
            base = os.path.basename(wav)[:-4]
            chunk, tag = base.split("_Participant_ID_")
            available_audio[(chunk, int(tag))] = wav

        bucket = test if sess_id in TEST_SESSIONS else train
        for cf in sorted(glob.glob(os.path.join(sess_dir, "*.json"))):
            chunk = os.path.basename(cf)[:-5]
            entries = _read_json_robust(cf)
            for e in entries:
                pid = e.get("Participant_ID")
                key = (chunk, pid)
                if key not in available_audio:
                    continue   # this speaker is the glasses-wearer for this session -- no clean target audio
                text = (e.get("Transcription") or "").strip()
                if not text:
                    continue
                start_sec = e["Start_Frame"] / VIDEO_FPS
                end_sec = e["End_Frame"] / VIDEO_FPS
                if end_sec <= start_sec:
                    continue
                bucket.append(SpeechSegment(
                    session=sess_id, chunk=chunk, participant_id=pid,
                    start_sec=start_sec, end_sec=end_sec,
                    transcription=text, audio_path=available_audio[key],
                ))
    return train, test


class EasyComSpeechDataset(Dataset):
    def __init__(self, segments: List[SpeechSegment]):
        self.segments = segments
        self._audio_cache: Dict[str, np.ndarray] = {}   # per-file cache within a worker

    def __len__(self) -> int:
        return len(self.segments)

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        if path not in self._audio_cache:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._audio_cache[path] = (audio, sr)
            if len(self._audio_cache) > 8:   # small per-worker LRU-ish cap
                self._audio_cache.pop(next(iter(self._audio_cache)))
        return self._audio_cache[path]

    def __getitem__(self, idx: int) -> Dict:
        seg = self.segments[idx]
        audio, sr = self._load_audio(seg.audio_path)
        i0 = max(0, int(seg.start_sec * sr))
        i1 = min(len(audio), int(seg.end_sec * sr))
        clip = audio[i0:i1]
        if clip.size == 0:
            clip = np.zeros(int(0.02 * sr), dtype=np.float32)
        duration_sec = clip.shape[0] / sr

        if sr != WHISPER_SR:
            import librosa
            clip = librosa.resample(clip, orig_sr=sr, target_sr=WHISPER_SR)
            duration_sec = clip.shape[0] / WHISPER_SR

        return {
            "waveform": clip.astype(np.float32),
            "duration_sec": duration_sec,
            "text": seg.transcription,
            "session": seg.session, "chunk": seg.chunk, "participant_id": seg.participant_id,
        }


def m4b_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "waveforms": [b["waveform"] for b in batch],
        "durations_sec": [b["duration_sec"] for b in batch],
        "texts": [b["text"] for b in batch],
        "sessions": [b["session"] for b in batch],
        "chunks": [b["chunk"] for b in batch],
        "participant_ids": [b["participant_id"] for b in batch],
    }


if __name__ == "__main__":
    train, test = build_segments()
    print(f"train segments={len(train)}  test segments={len(test)}")
    print(f"train sessions={sorted(set(s.session for s in train))}")
    print(f"test sessions={sorted(set(s.session for s in test))}")
    ds = EasyComSpeechDataset(train[:4])
    for i in range(len(ds)):
        item = ds[i]
        print(item["session"], item["chunk"], item["participant_id"],
              f"{item['duration_sec']:.2f}s", repr(item["text"][:60]))
