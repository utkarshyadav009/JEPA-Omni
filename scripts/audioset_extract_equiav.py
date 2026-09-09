"""scripts/audioset_extract_equiav.py — EquiAV AUDIO-TOWER feature extraction
for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing and model reused VERBATIM (by import, no modification) from
scripts/equiav_retrieval.py:
  - wav_to_fbank (:114-129): mean-subtracted mono waveform ->
    torchaudio.compliance.kaldi.fbank(htk_compat=True, sample_frequency=16000,
    use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
    frame_shift=10), HEAD-padded/cropped to TARGET_FRAMES=1024, normalised
    (fbank - AUDIO_NORM_MEAN) / AUDIO_NORM_STD with AUDIO_NORM_MEAN=-4.346,
    AUDIO_NORM_STD=4.332, channel-repeated to (3,1024,128) exactly as
    decode_audio_video does (:132-140).
  - MainModel + checkpoint load (:180-187): `MainModel(gpu=1)`, state dict
    with "__M__." prefix stripped, strict=False.
  - forward_feat (:217, `model.forward_feat(a_batch, v_batch)`), F.normalize
    on the audio output (:218).

AUDIO-ONLY justification (verified by reading the model source, not assumed):
MainModel.forward_feat(a, v) (repo models/pt_EquiAV.py:489-537) computes the
audio branch as patch_embed_a(a) -> +pos_embed_a -> +modality_a -> blocks_a
-> norm_a -> a, entirely independent of `v` (the video branch runs the exact
same shape of computation on `v` alone, in parallel, never mixed with `a`).
`z_a_inv` (the returned audio embedding) is a pure function of `a`: it comes
from cross_attn_a(a, aug_encoder_a(sampled_aug_vec)) -> pred_a_inter, where
the "cross-attention" is against a self-sampled augmentation-parameter vector
(sample_t_a(a), also a function of `a` alone), never against `v`. We
therefore feed a zero-tensor placeholder for `v` (correct shape (3,224,224),
discarded) and use only the first forward_feat() return value.

feat_tokens: OMITTED for this model. `a` post-blocks_a/norm_a (repo
models/pt_EquiAV.py:501-502) is a real 512-token, 768-d sequence -- a real
transformer output, not fabricated. But the model's own pooled+projected
z_a_inv output (what feat_mean uses, and what the retrieval script's
recall@k is computed on) lives in a DIFFERENT 512-d projection space,
produced by pred_a_inter's MLP applied only after cross-attending the mean-
pooled sequence against a random augmentation draw (CrossAttentionBlock.
forward, models/pt_EquiAV.py:87-95) -- there is no part of the model's own
released design that projects individual per-patch tokens into that same
512-d space (pred_a_inter's own MLP is only ever applied post-cross-attention
-pooling in the official code path). Storing 768-d raw tokens next to a
512-d feat_mean would silently mix two different representation spaces under
one "dim" field; per the task's "never fabricate a sequence" rule, we omit
feat_tokens here rather than invent a projection the checkpoint was never
trained to produce.

torch.manual_seed(0) is set (as the reference script does) because
forward_feat samples one random augmentation vector per call -- EquiAV's
actual released design, not a bug.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/equiav_<split>.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# NOTE: deliberately do NOT put PROJECT_ROOT on sys.path in this script.
# equiav_retrieval.py's own repo checkout has a `models/` dir with NO
# __init__.py (a PEP 420 namespace-package portion); JEPA-Omni's own
# /models/ (audio_encoder.py etc.) DOES have an __init__.py. If PROJECT_ROOT
# were on sys.path, Python's import resolution would find the EquiAV repo's
# namespace-portion "models" first, keep scanning (namespace portions don't
# stop the search), then hit JEPA-Omni's regular "models" package later in
# sys.path and use THAT instead (a regular package found anywhere in the
# scan wins over an earlier namespace portion) -- silently importing the
# wrong "models.pt_EquiAV" (ModuleNotFoundError, caught and fixed while
# building this script). Avoiding PROJECT_ROOT on sys.path sidesteps this
# entirely: Python auto-adds this script's own directory (scripts/) to
# sys.path[0], so plain `import equiav_retrieval` / `import
# audioset_baseline_common` (no "scripts." package prefix) resolve directly
# against scripts/*.py without ever adding PROJECT_ROOT.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ (usually already sys.path[0], explicit for safety)

from audioset_baseline_common import iter_audioset_rows, decode_flac_mono
import equiav_retrieval as eq  # applies timm/numpy compat shims, defines wav_to_fbank, imports MainModel path

torch.manual_seed(0)

PREPROC_SRC = ("scripts/equiav_retrieval.py:114-129 (wav_to_fbank), "
               ":132-140 (channel-repeat to (3,1024,128)), "
               ":180-187 (MainModel + checkpoint load), "
               ":217-218 (forward_feat + F.normalize); audio-only forward "
               "verified from repo models/pt_EquiAV.py forward_feat:489-537 "
               "(a and v branches independent, v discarded); feat_tokens "
               "omitted (see module docstring: dim mismatch between raw "
               "768-d tokens and the 512-d projected feat_mean space)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/equiav/equiav_pretrained.pth")
    a = ap.parse_args()
    dev = torch.device(a.device)

    from models.pt_EquiAV import MainModel
    model = MainModel(gpu=1)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd_clean = {k.replace("__M__.", ""): v for k, v in sd.items()}
    result = model.load_state_dict(sd_clean, strict=False)
    print(f"EquiAV: loaded state_dict, missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}", flush=True)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D = eq.EMBED_DIM

    ids, labs, means = [], [], []
    bad = 0
    t0 = time.time()
    buf_a, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_a, buf_i, buf_l
        if not buf_a:
            return
        a_batch = torch.stack(buf_a).to(dev)
        v_batch = torch.zeros(a_batch.shape[0], 3, eq.IM_RES, eq.IM_RES, device=dev)  # discarded, see module docstring
        with torch.no_grad():
            z_a, _ = model.forward_feat(a_batch, v_batch)
            z_a = F.normalize(z_a, dim=-1)
        ids.extend(buf_i); labs.extend(buf_l)
        means.append(z_a.float().cpu().half())
        buf_a, buf_i, buf_l = [], [], []

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float()
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            fbank = eq.wav_to_fbank(wav)               # (1024, 128)
            audio_input = fbank.unsqueeze(0).repeat(3, 1, 1)  # (3,1024,128)
        except Exception:
            bad += 1
            continue
        buf_a.append(audio_input); buf_i.append(row["video_id"]); buf_l.append(row["labels"])
        if len(buf_a) >= a.batch:
            flush()
            n = len(ids)
            if n % 1024 < a.batch:
                el = time.time() - t0
                print(f"[equiav] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
    flush()

    feat_mean = torch.cat(means) if means else torch.zeros(0, D)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "model": "equiav",
        "dim": D,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[equiav] WROTE {a.out} n={len(ids)} D={D} bad={bad} "
          f"feat_tokens=NO {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
