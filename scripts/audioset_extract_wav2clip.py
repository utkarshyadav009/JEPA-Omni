"""scripts/audioset_extract_wav2clip.py — Wav2CLIP AUDIO-TOWER feature extraction
for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing reused VERBATIM from scripts/wav2clip_retrieval.py:120-155 —
Wav2CLIP's own audio tower (wav2clip.get_model(), scenario="frozen",
transform=True) run at BATCH SIZE 1, because ResNetExtractor.forward's internal
spectrogram normalisation (`wav2clip/model/resnet.py` ResNet.forward) computes
mean/std over the WHOLE input tensor, not per-sample — batching would leak
stats across clips (documented in that script's own module docstring and
confirmed by reading wav2clip/model/resnet.py directly: `torch.mean(x)` /
`torch.std(x)` over the full 4-D batch tensor).

Audio: raw waveform, any sample rate the model's own torchaudio.transforms
.Spectrogram(n_fft=512, hop_length=353) accepts — Wav2CLIP was trained on
VGGSound at 16kHz per the retrieval script's own docstring, so we resample
our 48kHz AudioSet decode to 16kHz mono exactly as the reference script's
decode_audio_16k_mono does (torchaudio resample), then feed raw waveform
directly into ResNetExtractor.forward — no separate spectrogram step ours,
it's computed inside the model.

No token sequence: ResNet18's own forward ends in AdaptiveAvgPool2d((1,1))
(global spatial+temporal pool) before the model even reaches Wav2CLIP's MLP
head — there is no intermediate per-time-step sequence to expose. feat_tokens
is therefore omitted for this model, per the task's "never fabricate" rule.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/wav2clip_<split>.pt
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

PREPROC_SRC = "scripts/wav2clip_retrieval.py:127-130,164-168 (wav2clip.get_model + audio_model(wav.unsqueeze(0)), batch=1)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-dir", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/wav2clip")
    a = ap.parse_args()
    dev = torch.device(a.device)

    import wav2clip
    os.environ.setdefault("TORCH_HOME", a.ckpt_dir)
    model = wav2clip.get_model(device=str(dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D = 512

    ids, labs, feats = [], [], []
    bad = 0
    n = 0
    t0 = time.time()

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float()
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        except Exception:
            bad += 1
            continue

        try:
            with torch.no_grad():
                emb = model(wav.unsqueeze(0).to(dev))  # (1, 512), batch=1 mandatory
            emb = emb.squeeze(0).float().cpu()
            emb = F.normalize(emb, dim=-1)  # matches retrieval script's final embedding (F.normalize before cosine sim)
        except Exception:
            bad += 1
            continue

        ids.append(row["video_id"])
        labs.append(row["labels"])
        feats.append(emb.half())
        n = len(ids)
        if n % 1024 == 0:
            el = time.time() - t0
            print(f"[wav2clip] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)

    feat_mean = torch.stack(feats) if feats else torch.zeros(0, D)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "model": "wav2clip",
        "dim": D,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[wav2clip] WROTE {a.out} n={len(ids)} D={D} bad={bad} "
          f"feat_tokens=NO {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
