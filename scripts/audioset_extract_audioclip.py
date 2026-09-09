"""scripts/audioset_extract_audioclip.py — AudioCLIP AUDIO-TOWER feature
extraction for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing reused VERBATIM from scripts/audioclip_retrieval.py:68-107
(SAMPLE_RATE=44100, AUDIO_OUT_LEN=220500 i.e. 5.0s, center_pad_or_crop) and
:146-147 (checkpoint load: `from model import AudioCLIP; AudioCLIP(pretrained=ckpt)`).
Audio-only forward reused from that same script's own usage pattern
(:184, `model(audio=wav_batch)`), which AudioCLIP.forward (model/audioclip.py)
supports natively — audio/image/text are each independently optional, and
`audio_features` is already L2-normalised internally
(`audio_features / audio_features.norm(dim=-1, keepdim=True)`) before being
returned, exactly mirroring the extra F.normalize the retrieval script applies
on top (idempotent).

AudioSet clips are 10s @ 48kHz; decoded mono, resampled to 44.1kHz, then
center-cropped to AudioCLIP's own 5.0s (220500-sample) eval window via the
model's own center_pad_or_crop convention (same function used for VGGSound's
~10s clips in the reference script, applied here to AudioSet's 10s clips —
identical situation, not a new approximation).

No token sequence: ESResNeXtFBSP (model/esresnet.py) is a CNN classifier
trunk ending in global pooling + FC to embed_dim; there is no intermediate
per-frame sequence the model itself exposes as an output. feat_tokens is
therefore omitted for this model, per the task's "never fabricate" rule.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/audioclip_<split>.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.audioset_baseline_common import iter_audioset_rows, decode_flac_mono
import scripts.audioclip_retrieval as ac  # gives us center_pad_or_crop, SAMPLE_RATE, AUDIO_OUT_LEN, and puts the AudioCLIP repo on sys.path

PREPROC_SRC = "scripts/audioclip_retrieval.py:68-99,146-147,184-187 (SAMPLE_RATE/AUDIO_OUT_LEN/center_pad_or_crop, AudioCLIP(pretrained=ckpt), model(audio=wav_batch))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default=os.path.join(
        "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/audioclip/repo",
        "assets", "AudioCLIP-Full-Training.pt"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    from model import AudioCLIP
    model = AudioCLIP(pretrained=a.ckpt).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    D = model.embed_dim

    ids, labs, feats = [], [], []
    bad = 0
    t0 = time.time()
    buf_w, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_w, buf_i, buf_l
        if not buf_w:
            return
        wav_batch = torch.stack(buf_w).unsqueeze(1).to(dev)  # (B,1,T)
        with torch.no_grad():
            ((a_feat, _, _), _), _ = model(audio=wav_batch)
            a_feat = F.normalize(a_feat, dim=-1)
        ids.extend(buf_i); labs.extend(buf_l)
        feats.append(a_feat.float().cpu().half())
        buf_w, buf_i, buf_l = [], [], []

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float()
            if sr != ac.SAMPLE_RATE:
                import torchaudio
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, ac.SAMPLE_RATE).squeeze(0)
            wav = ac.center_pad_or_crop(wav, ac.AUDIO_OUT_LEN)
        except Exception:
            bad += 1
            continue
        buf_w.append(wav); buf_i.append(row["video_id"]); buf_l.append(row["labels"])
        if len(buf_w) >= a.batch:
            flush()
            n = len(ids)
            if n % 1024 < a.batch:
                el = time.time() - t0
                print(f"[audioclip] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
    flush()

    feat_mean = torch.cat(feats) if feats else torch.zeros(0, D)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "model": "audioclip",
        "dim": D,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[audioclip] WROTE {a.out} n={len(ids)} D={D} bad={bad} "
          f"feat_tokens=NO {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
