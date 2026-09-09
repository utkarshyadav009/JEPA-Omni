"""scripts/audioset_extract_imagebind.py — ImageBind AUDIO-TOWER feature
extraction for AudioSet (bal_train / eval). EXTRACTION ONLY: no probe, no mAP.

Preprocessing constants and model reused VERBATIM (by import, no
modification) from scripts/imagebind_retrieval.py, which itself documents
(module docstring) that every constant is ImageBind's own default, read from
the official repo's imagebind/data.py, unmodified:
  - Audio: resample to 16kHz, ConstantClipsPerVideoSampler(clip_duration=2s,
    clips_per_video=3), each 2s sub-clip -> kaldi fbank (num_mel_bins=128,
    frame_shift=10ms, htk_compat=True, hanning window) padded/cropped to
    target_length=204, normalised mean=-4.268/std=9.138, 3 clip-embeddings
    mean-pooled by the model's own forward() (ndim>=5 multi-clip reduction).
  - Checkpoint: imagebind_model.imagebind_huge(pretrained=False) +
    state_dict load, strict=True (:105-107 of the reference script).

We call `ib_data.get_clip_timepoints` / `ib_data.waveform2melspec` /
`ib_data.ConstantClipsPerVideoSampler` directly (all module-level functions
imported unmodified from the official repo) on our own already-decoded
AudioSet waveform tensor instead of `ib_data.load_and_transform_audio_data`,
because that convenience wrapper only accepts a file PATH (it calls
`torchaudio.load(audio_path)` internally) and we have parquet-embedded FLAC
bytes, not files on disk. Every actual signal-processing step downstream of
the raw waveform is the identical, unmodified repo function.

AUDIO-ONLY: ImageBind's modalities are architecturally independent at
inference (imagebind_model.py ImageBindModel.forward:444-476 loops over
`inputs.items()` and processes each modality through its own
preprocessor/trunk/head/postprocessor with no cross-modality mixing) --
confirmed by reading the source, not assumed. We therefore never construct a
vision input at all (unlike CAV-MAE Sync / EquiAV, no video branch is
computed even as a discarded placeholder).

feat_tokens: the audio modality_trunk's own SimpleTransformer output (the
tensor `modality_heads[AUDIO]`'s `SelectElement(index=0)` step CLS-pools down
to a single vector for the final embedding) is captured here BEFORE that
pooling -- a real per-patch token sequence, not fabricated. We replicate
ImageBindModel.forward's per-modality body (:456-468) call-by-call using the
model's own already-loaded submodules (`model.modality_preprocessors`,
`model.modality_trunks`, `model.modality_heads`, `model.modality_postprocessors`
-- all `nn.ModuleDict` attributes on the same model instance the reference
script builds), which is orchestration of the existing modules in their
existing order, not new math. Trunk output across the 3 sub-clips is flattened
projected through the audio head's own LayerNorm+Linear (applied per-token,
see encode_audio_with_tokens docstring) into the 1024-d out_embed_dim space
that feat_mean lives in, and adaptive-avg-pooled to 32 steps.

Output: /mnt/Raid-Storage-2/utkarsh-data/audioset_feats/imagebind_<split>.pt
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
import scripts.imagebind_retrieval as ibr  # puts the ImageBind repo on sys.path, imports imagebind_model/ModalityType/ib_data

PREPROC_SRC = ("scripts/imagebind_retrieval.py:105-107 (imagebind_huge + "
               "strict state_dict load); repo imagebind/data.py "
               "get_clip_timepoints/waveform2melspec/ConstantClipsPerVideoSampler "
               "(called directly on our decoded waveform, see module docstring); "
               "repo imagebind_model.py ImageBindModel.forward:444-476 per-"
               "modality body replicated via the model's own submodules to "
               "additionally capture pre-head trunk tokens")

N_MEL = 128
TARGET_LEN = 204
CLIP_DUR = 2
CLIPS_PER_VIDEO = 3
SR = 16000
AUDIO_MEAN, AUDIO_STD = -4.268, 9.138
N_TOK = 32


def make_audio_tensor(wav_1xT: torch.Tensor) -> torch.Tensor:
    """wav_1xT: (1, n_samples) @ 16kHz mono. Returns (3, 1, 128, 204), matching
    ib_data.load_and_transform_audio_data's per-video output before the outer
    torch.stack(audio_outputs, dim=0) adds the batch dim."""
    clip_sampler = ibr.ib_data.ConstantClipsPerVideoSampler(
        clip_duration=CLIP_DUR, clips_per_video=CLIPS_PER_VIDEO)
    all_clips_timepoints = ibr.ib_data.get_clip_timepoints(clip_sampler, wav_1xT.size(1) / SR)
    all_clips = []
    for start, end in all_clips_timepoints:
        waveform_clip = wav_1xT[:, int(start * SR):int(end * SR)]
        melspec = ibr.ib_data.waveform2melspec(waveform_clip, SR, N_MEL, TARGET_LEN)  # (1,128,204)
        all_clips.append(melspec)
    normalize = ibr.ib_data.transforms.Normalize(mean=AUDIO_MEAN, std=AUDIO_STD)
    all_clips = [normalize(ac) for ac in all_clips]
    return torch.stack(all_clips, dim=0)  # (3,1,128,204)


@torch.no_grad()
def encode_audio_with_tokens(model, audio_batch: torch.Tensor, device):
    """audio_batch: (B,3,1,128,204). Returns (feat_mean (B,1024) L2-normalised,
    tokens (B,32,1024) pooled to 32 steps).
    Replicates ImageBindModel.forward's AUDIO body (imagebind_model.py:444-476).

    NOTE on dims: imagebind_huge() (repo imagebind_model.py:479-490) passes
    out_embed_dim=1024 (overriding the class default of 768), while
    audio_embed_dim stays at its default 768 -- so the audio TRUNK's raw
    token sequence is 768-d but the model's own final audio embedding
    (after modality_heads[AUDIO]'s Linear(768,1024)) is 1024-d. Storing raw
    768-d trunk tokens next to a 1024-d feat_mean would silently mix two
    representation spaces under one "dim" field (the same problem identified
    for EquiAV, where we chose to omit tokens instead). Here we avoid that by
    applying modality_heads[AUDIO]'s own LayerNorm + Linear sub-modules
    (head[0], head[2]) to EVERY token, not just the index-0 CLS token that
    SelectElement (head[1]) picks out for the model's own final embedding --
    both LayerNorm and Linear are per-token/row-independent operations
    (identical arithmetic per token), so this is exactly what the head
    already computes for the CLS token, extended uniformly to the rest of
    the sequence with the SAME trained weights -- not a fabricated
    projection, and it keeps tokens and feat_mean in the same 1024-d space.
    """
    B, S = audio_batch.shape[:2]
    x = audio_batch.reshape(B * S, *audio_batch.shape[2:]).to(device)
    prep = model.modality_preprocessors[ibr.ModalityType.AUDIO](audio=x)
    trunk_out = model.modality_trunks[ibr.ModalityType.AUDIO](**prep["trunk"])   # (B*S, N_tok, 768) real tokens
    head = model.modality_heads[ibr.ModalityType.AUDIO]                          # Sequential(LayerNorm, SelectElement(0), Linear(768,1024,bias=False))
    head_out = head(trunk_out, **prep["head"])                                   # (B*S, 1024) -- the model's own CLS-pooled embedding
    post_out = model.modality_postprocessors[ibr.ModalityType.AUDIO](head_out)   # (B*S, 1024) L2-norm * fixed logit scale
    post_out = post_out.reshape(B, S, -1).mean(dim=1)                            # (B,1024), matches model.forward's mean-pool
    feat_mean = F.normalize(post_out.float(), dim=-1)                            # strip the fixed logit-scale factor (direction-only), matches reference script's own extra F.normalize before cosine sim

    ln, lin = head[0], head[2]                                                    # skip head[1]=SelectElement(0), apply LN+Linear to every token
    tok_proj = lin(ln(trunk_out)).float()                                         # (B*S, N_tok, 1024) -- same weights, per-token
    D = tok_proj.shape[-1]
    tok = tok_proj.reshape(B, S, -1, D).reshape(B, -1, D)                        # (B, S*N_tok, 1024)
    tokens = F.adaptive_avg_pool1d(tok.transpose(1, 2), N_TOK).transpose(1, 2)   # (B,32,1024)
    return feat_mean.cpu(), tokens.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/imagebind/imagebind_huge.pth")
    a = ap.parse_args()
    dev = torch.device(a.device)

    model = ibr.imagebind_model.imagebind_huge(pretrained=False)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    result = model.load_state_dict(sd, strict=True)
    print(f"ImageBind: loaded strictly, missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}", flush=True)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D = 1024  # imagebind_huge's out_embed_dim (repo imagebind_model.py:487) -- see encode_audio_with_tokens docstring

    ids, labs, means, toks = [], [], [], []
    bad = 0
    t0 = time.time()
    buf_a, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_a, buf_i, buf_l
        if not buf_a:
            return
        audio_batch = torch.stack(buf_a)  # (B,3,1,128,204)
        feat_mean, tokens = encode_audio_with_tokens(model, audio_batch, dev)
        ids.extend(buf_i); labs.extend(buf_l)
        means.append(feat_mean.half()); toks.append(tokens.half())
        buf_a, buf_i, buf_l = [], [], []

    for row in iter_audioset_rows(a.split, limit=a.limit):
        try:
            w, sr = decode_flac_mono(row["audio_bytes"])
            wav = torch.from_numpy(w).float().unsqueeze(0)  # (1, n_samples)
            if sr != SR:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, SR)
            audio_tensor = make_audio_tensor(wav)  # (3,1,128,204)
        except Exception:
            bad += 1
            continue
        buf_a.append(audio_tensor); buf_i.append(row["video_id"]); buf_l.append(row["labels"])
        if len(buf_a) >= a.batch:
            flush()
            n = len(ids)
            if n % 512 < a.batch:
                el = time.time() - t0
                print(f"[imagebind] {a.split} n={n} {n/max(el,1e-9):.1f} clips/s bad={bad}", flush=True)
    flush()

    feat_mean = torch.cat(means) if means else torch.zeros(0, D)
    feat_tokens = torch.cat(toks) if toks else torch.zeros(0, N_TOK, D)
    out = {
        "ids": ids,
        "labels": labs,
        "feat_mean": feat_mean.half(),
        "feat_tokens": feat_tokens.half(),
        "model": "imagebind",
        "dim": D,
        "preprocessing_source": PREPROC_SRC,
        "split": a.split,
        "bad": bad,
    }
    torch.save(out, a.out)
    print(f"[imagebind] WROTE {a.out} n={len(ids)} D={D} bad={bad} "
          f"feat_tokens=YES(32) {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
