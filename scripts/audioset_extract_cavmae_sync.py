"""scripts/audioset_extract_cavmae_sync.py — CAV-MAE Sync AUDIO-TOWER feature
extraction for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing and model loading reused VERBATIM (by import, no modification)
from scripts/cavmae_sync_retrieval.py:
  - wav2fbank (:122-135): kaldi.fbank(htk_compat=True, use_energy=False,
    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) on
    mean-subtracted mono waveform, padded/cut to BASE_TARGET_LEN=1024 frames.
  - map_frame_to_spectrogram (:112-119) + the 16-sync-segment crop/normalise
    convention (NORM_MEAN=-5.081, NORM_STD=4.4849, SEG_TARGET_LEN=416),
    reused from load_clip's audio half (:138-169) -- the video half of
    load_clip is NOT used here (no AudioSet video available; see below).
  - build_model (:172-183): CAVMAESync(audio_length=416,
    modality_specific_depth=11, num_register_tokens=8, cls_token=True,
    total_frame=16, contrastive_heads=False), state_dict loaded strictly-off
    (strict=False) exactly as the reference script does.

AUDIO-ONLY justification (verified by reading the model source, not assumed):
CAVMAESync.forward_feat(a, v) (repo src/models/cav_mae_sync.py:568-637) runs
the audio stream through patch_embed_a -> blocks_a -> blocks_u(a,'a') ->
norm_a and the visual stream through patch_embed_v -> blocks_v ->
blocks_u(v,'v') -> norm_v as two COMPLETELY INDEPENDENT branches -- there is
no cross-attention or any operation that mixes `a` and `v` anywhere in this
method. cls_a/ca (the audio outputs) are mathematically a pure function of
`a` alone. We therefore feed a zero-tensor placeholder for `v` (correct
shape, discarded) and never read cv/cls_v -- this is not an approximation of
the model's audio computation, it is exactly what the model already computes
for audio, verified from source rather than assumed from the paper.

feat_mean: for each clip, 16 sync-segment CLS embeddings (:191-193 in
embed_clip, L2-normalised per segment exactly as the reference script does)
are averaged across the 16 segments then re-L2-normalised -- this is the
natural per-clip pooling of the reference script's per-segment output (the
reference script itself only ever compares 16-segment tensors pairwise via
diagonal_mean, a retrieval-time-only aggregation across DIFFERENT clips; a
single per-clip vector requires pooling across the 16 segments of the SAME
clip, which we do here by mean).

feat_tokens: `ca` (:626, the local patch tokens per sync segment, i.e. the
16-sync-segment x 208-patch pre-CLS sequence) is a real transformer output,
not fabricated. Flattened across (16 segments x 208 patches = 3328 tokens)
and adaptive-avg-pooled to 32 steps.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/cavmae_sync_<split>.pt
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
import scripts.cavmae_sync_retrieval as cs  # imports repo, applies timm compat shim, defines wav2fbank/map_frame_to_spectrogram/build_model

PREPROC_SRC = ("scripts/cavmae_sync_retrieval.py:122-135 (wav2fbank), "
               ":112-119,151-167 (map_frame_to_spectrogram + 16-segment crop/normalise), "
               ":172-183 (build_model); audio-only forward verified from "
               "repo src/models/cav_mae_sync.py forward_feat:568-637 (a and v "
               "branches are independent, v discarded); feat_mean = mean over "
               "16 segment CLS embeddings, re-normalised; feat_tokens = ca "
               "patch tokens (16seg x 208patch=3328) adaptive-avg-pooled to 32")

N_TOK = 32


@torch.no_grad()
def embed_clips_audio_only(model, fbanks_batch: torch.Tensor, device):
    """fbanks_batch: (B*16, 416, 128) -- B clips' 16 sync segments concatenated
    along the pseudo-batch dim (forward_feat's blocks are row-independent, so
    grouping multiple clips into one real batch changes nothing numerically
    vs. calling it once per clip -- see module docstring on independence).
    Returns (cls_mean (B,768), tokens (B,32,768))."""
    B16 = fbanks_batch.shape[0]
    B = B16 // cs.TOTAL_FRAME
    a = fbanks_batch.to(device)
    v = torch.zeros(B16, 3, cs.IM_RES, cs.IM_RES, device=device)  # discarded, see module docstring
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        ca, cv, cls_a, cls_v = model.module.forward_feat(a, v)
    cls_a = F.normalize(cls_a.float(), dim=-1).view(B, cs.TOTAL_FRAME, -1)   # (B,16,768)
    ca = ca.float().view(B, cs.TOTAL_FRAME, ca.shape[-2], ca.shape[-1])       # (B,16,n_patch,768)
    cls_mean = F.normalize(cls_a.mean(1), dim=-1)             # (B,768) pooled across the 16 segments
    tok_seq = ca.reshape(B, -1, ca.shape[-1])                 # (B, 16*n_patch, 768)
    tokens = F.adaptive_avg_pool1d(tok_seq.transpose(1, 2), N_TOK).transpose(1, 2)  # (B,32,768)
    return cls_mean.cpu(), tokens.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/cavmae_sync/cav_mae_sync.pth")
    ap.add_argument("--batch-clips", type=int, default=8, help="clips per forward call (each contributes 16 pseudo-batch rows)")
    a = ap.parse_args()
    dev = torch.device(a.device)

    model = cs.build_model(a.ckpt, str(dev))
    D = 768

    ids, labs, means, toks = [], [], [], []
    bad = 0
    t0 = time.time()
    buf_fb, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_fb, buf_i, buf_l
        if not buf_fb:
            return
        fbanks_batch = torch.cat(buf_fb, dim=0)  # (B*16, 416, 128)
        cls_mean, tokens = embed_clips_audio_only(model, fbanks_batch, dev)
        ids.extend(buf_i); labs.extend(buf_l)
        means.append(cls_mean.half()); toks.append(tokens.half())
        buf_fb, buf_i, buf_l = [], [], []

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float().unsqueeze(0)   # (1, n_samples) @ native sr, mono
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, 16000)
                sr = 16000
            base_fbank = cs.wav2fbank(wav, sr)                # (1024, 128)
            spec_len = base_fbank.shape[0]
            fbanks = []
            for k in range(cs.TOTAL_FRAME):
                start, end = cs.map_frame_to_spectrogram(k, cs.TOTAL_FRAME, spec_len, cs.SEG_TARGET_LEN)
                seg = base_fbank[start:end, :]
                if seg.shape[0] < cs.SEG_TARGET_LEN:
                    seg = torch.nn.functional.pad(seg, (0, 0, 0, cs.SEG_TARGET_LEN - seg.shape[0]))
                seg = (seg - cs.NORM_MEAN) / cs.NORM_STD
                fbanks.append(seg)
            fbanks = torch.stack(fbanks)                      # (16, 416, 128)
        except Exception:
            bad += 1
            continue

        buf_fb.append(fbanks); buf_i.append(row["video_id"]); buf_l.append(row["labels"])
        if len(buf_fb) >= a.batch_clips:
            flush()
            n = len(ids)
            if n % 512 < a.batch_clips:
                el = time.time() - t0
                print(f"[cavmae_sync] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
    flush()

    feat_mean = torch.cat(means) if means else torch.zeros(0, D)
    feat_tokens = torch.cat(toks) if toks else torch.zeros(0, N_TOK, D)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "feat_tokens": feat_tokens.half(),
        "model": "cavmae_sync",
        "dim": D,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[cavmae_sync] WROTE {a.out} n={len(ids)} D={D} bad={bad} "
          f"feat_tokens=YES(32) {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
