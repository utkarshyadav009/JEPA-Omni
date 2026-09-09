"""scripts/audioset_extract_cavmae.py — CAV-MAE AUDIO-TOWER feature extraction
for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing and model reused VERBATIM (by import, no modification) from
scripts/cavmae_retrieval.py:
  - waveform_to_logmel (:100-136): kaldi.fbank(htk_compat=True, frame_length=25,
    frame_shift=10, num_mel_bins=128, sample_frequency=16000, use_log_fbank=True,
    dither=0.0), normalised with AUDIO_NORM_MEAN=-4.2677393 / AUDIO_NORM_STD=4.5689974
    (:53-59), padded/center-cropped to AUDIO_TIME_FRAMES=416 (:49).
  - CAVMAEEncoder.from_checkpoint (:284-299): loads audio_model.25.pth.
  - CAVMAEEncoder.encode_audio (:254-267): the model's own CLS-token pooling +
    L2-normalise, used verbatim for feat_mean.

feat_tokens: CAV-MAE's audio tower IS a transformer with a real pre-pool
token sequence (CLS + 16 register tokens + 208 audio patches = 225 tokens,
post blocks_a + blocks_u + norm_a — the exact tensor encode_audio slices
`[:, 0]` out of to get the CLS token). We reuse that same forward path
(duplicated here only because encode_audio returns just the CLS slice, not
the full sequence — the underlying blocks/norm calls are copied unchanged)
and adaptive-avg-pool the full 225-token sequence to 32 steps. This is a real
transformer output being pooled down, not a fabricated/tiled sequence.

AudioSet clips are 10s @ 48kHz stereo -> decoded mono -> resampled to 16kHz
(kaldi.fbank's own sample_frequency) exactly mirroring the reference script's
`AudioDecoder(vpath, sample_rate=16000)` step.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/cavmae_<split>.pt
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
from scripts.cavmae_retrieval import (
    CAVMAEEncoder, waveform_to_logmel, AUDIO_FREQ, AUDIO_TIME_FRAMES, EMBED_DIM,
)

PREPROC_SRC = ("scripts/cavmae_retrieval.py:100-136 (waveform_to_logmel), "
               ":284-299 (CAVMAEEncoder.from_checkpoint), "
               ":254-267 (encode_audio pooling; tokens = same forward path "
               "pre-CLS-slice, adaptive-avg-pooled to 32)")

N_TOK = 32


@torch.no_grad()
def encode_audio_with_tokens(model: CAVMAEEncoder, spec: torch.Tensor):
    """Duplicates CAVMAEEncoder.encode_audio's forward path (scripts/cavmae_retrieval.py:254-267)
    unchanged, only additionally returning the pre-CLS-slice full token sequence."""
    B = spec.shape[0]
    x = model.patch_embed_a(spec)
    x = x + model.pos_embed_a + model.modality_a
    cls = model.cls_token_a.expand(B, -1, -1)
    reg = model.register_tokens.unsqueeze(0).expand(B, -1, -1)
    x = torch.cat([cls, reg, x], dim=1)
    for blk in model.blocks_a:
        x = blk(x)
    for blk in model.blocks_u:
        x = blk(x)
    x = model.norm_a(x)                      # (B, 225, D) -- full sequence, real transformer output
    cls_mean = F.normalize(x[:, 0], dim=-1)   # identical to encode_audio's return value
    tokens = F.adaptive_avg_pool1d(x.transpose(1, 2), N_TOK).transpose(1, 2)  # (B, 32, D)
    return cls_mean, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="/home/utkarsh/models/cav-mae/audio_model.25.pth")
    a = ap.parse_args()
    dev = torch.device(a.device)

    model = CAVMAEEncoder.from_checkpoint(a.ckpt, device=str(dev))

    ids, labs, means, toks = [], [], [], []
    bad = 0
    t0 = time.time()
    buf_w, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_w, buf_i, buf_l
        if not buf_w:
            return
        spec_batch = torch.stack(buf_w).to(dev)
        cls_mean, tokens = encode_audio_with_tokens(model, spec_batch)
        ids.extend(buf_i); labs.extend(buf_l)
        means.append(cls_mean.float().cpu().half())
        toks.append(tokens.float().cpu().half())
        buf_w, buf_i, buf_l = [], [], []

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float()
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            spec = waveform_to_logmel(wav, sr=16000, n_mels=AUDIO_FREQ, target_frames=AUDIO_TIME_FRAMES)
        except Exception:
            bad += 1
            continue
        buf_w.append(spec); buf_i.append(row["video_id"]); buf_l.append(row["labels"])
        if len(buf_w) >= a.batch:
            flush()
            n = len(ids)
            if n % 1024 < a.batch:
                el = time.time() - t0
                print(f"[cavmae] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
    flush()

    feat_mean = torch.cat(means) if means else torch.zeros(0, EMBED_DIM)
    feat_tokens = torch.cat(toks) if toks else torch.zeros(0, N_TOK, EMBED_DIM)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "feat_tokens": feat_tokens.half(),
        "model": "cavmae",
        "dim": EMBED_DIM,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[cavmae] WROTE {a.out} n={len(ids)} D={EMBED_DIM} bad={bad} "
          f"feat_tokens=YES(32) {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
