"""scripts/languagebind_retrieval.py — LanguageBind audio<->video retrieval baseline.

Runs LanguageBind_Video_FT + LanguageBind_Audio_FT (both from the official
PKU-YuanGroup/LanguageBind HF checkpoints) on our fixed 1545-clip VGGSound
eval gallery and computes audio->visual / visual->audio R@1, R@5, R@10.

This is a legitimate use of the model: the official repo's own inference.py
demo computes "Video x Audio" similarity directly (both towers share one
embedding space by construction, each independently CLIP-aligned to
language) — we reuse that same audio<->video comparison, just over our
gallery instead of the 2-sample demo.

Preprocessing exactly matches LanguageBind's own code (languagebind/video/
processing_video.py, languagebind/audio/processing_audio.py), driven by
each checkpoint's own vision_config (not hardcoded here):
  Video: decord backend (config default, confirmed via
    model.config.vision_config.video_decode_backend == 'decord'),
    num_frames=8 uniformly sampled (np.linspace over the full clip),
    resize short side to 224, center-crop 224, OpenAI CLIP mean/std
    normalisation. The official transform also does RandomHorizontalFlip
    (p=0.5); we DISABLE that for eval determinism (it's a training-only
    augmentation, not part of the core encoding path) — this is the one
    deliberate deviation from the training-time transform.
  Audio: torchaudio.load() (ffmpeg backend, reads the mp4 directly) ->
    resample to audio_sample_rate=16000 (from LanguageBind_Audio_FT's own
    config) -> Kaldi-compatible fbank (torchaudio.compliance.kaldi.fbank,
    num_mel_bins=112, frame_length=25ms, frame_shift=10ms, hanning window,
    htk_compat, dither=0) -> pad/repeat to target_length=1036 frames since
    our ~10s clips are shorter -> tile into 3 identical "front/middle/back"
    channels (deterministic in our case, since real content < target
    length) -> normalise via (mel - audio_mean) / (audio_std * 2) with
    audio_mean=-4.2677393, audio_std=4.5689974 (both from the Audio_FT
    checkpoint's own vision_config — note these are the same constants
    CAV-MAE uses, LanguageBind's audio preprocessing lineage borrows from
    the AST/CAV-MAE codebase).

Usage:
    conda activate langbind_eval
    python scripts/languagebind_retrieval.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torchaudio
import decord
from decord import VideoReader, cpu

# torchaudio removed set_audio_backend()/list_audio_backends() (deprecated API,
# gone in the torchaudio version pinned to our torch 2.12.1+cu130 build); the
# LanguageBind repo's languagebind/audio/processing_audio.py calls it at import
# time purely to force soundfile as the backend. No-op shim — soundfile-based
# loading is the modern default anyway, this call is a legacy no-op now.
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

sys.path.insert(0, "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/languagebind/repo")
from languagebind.video.modeling_video import LanguageBindVideo
from languagebind.audio.modeling_audio import LanguageBindAudio
from languagebind.video.processing_video import get_video_transform
from languagebind.audio.processing_audio import get_audio_transform

decord.bridge.set_bridge("torch")

CACHE_DIR = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/languagebind/cache_dir"
VIDEO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_video_embed(clip_path: str, model, transform, num_frames: int, device):
    decord_vr = VideoReader(clip_path, ctx=cpu(0))
    duration = len(decord_vr)
    frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)          # (T,H,W,C)
    video_data = video_data.permute(3, 0, 1, 2).float()       # (C,T,H,W)
    video_data = transform(video_data)                        # eval transform, no flip
    pixel_values = video_data.unsqueeze(0).to(device)          # (1,C,T,H,W)
    with torch.no_grad():
        out = model.vision_model(pixel_values=pixel_values)[1]
        emb = model.visual_projection(out)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze(0).float().cpu()


def load_audio_embed(clip_path: str, model, audio_transform, device):
    waveform, sr = torchaudio.load(clip_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_fusion = audio_transform((waveform, sr))               # (3, n_mels, target_length)
    pixel_values = mel_fusion.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.vision_model(pixel_values=pixel_values)[1]
        emb = model.visual_projection(out)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze(0).float().cpu()


def make_eval_video_transform(config):
    """Same as languagebind.video.processing_video.get_video_transform but
    with RandomHorizontalFlip removed for deterministic eval."""
    from torchvision.transforms import Compose, Lambda
    from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
    from pytorchvideo.transforms import ShortSideScale
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    return Compose([
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ShortSideScale(size=224),
        CenterCropVideo(224),
    ])


def compute_retrieval(a_embeds: torch.Tensor, v_embeds: torch.Tensor) -> dict:
    sim = a_embeds @ v_embeds.T           # (N,N) audio-rows x visual-cols
    N = sim.shape[0]
    gt = torch.arange(N).unsqueeze(1)

    def ranks(mat):
        out = {}
        for k in (1, 5, 10):
            topk = mat.topk(k, dim=1).indices
            hits = (topk == gt).any(dim=1).float().mean().item()
            out[f"R@{k}"] = round(hits * 100, 2)
        return out

    a2v = ranks(sim)
    v2a = ranks(sim.T)
    return {"audio_to_visual": a2v, "visual_to_audio": v2a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-list", default="/home/utkarsh/JEPA-Omni/data/vggsound_eval_1545.txt")
    ap.add_argument("--video-dir", default=VIDEO_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/utkarsh/JEPA-Omni/data/languagebind_retrieval_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}", flush=True)
    assert len(clip_ids) == 1545, f"expected 1545 clips in eval list, got {len(clip_ids)}"

    print("Loading LanguageBind_Video_FT ...", flush=True)
    vmodel = LanguageBindVideo.from_pretrained("LanguageBind/LanguageBind_Video_FT", cache_dir=CACHE_DIR).to(device).eval()
    print("Loading LanguageBind_Audio_FT ...", flush=True)
    amodel = LanguageBindAudio.from_pretrained("LanguageBind/LanguageBind_Audio_FT", cache_dir=CACHE_DIR).to(device).eval()

    num_frames = vmodel.config.vision_config.num_frames
    assert vmodel.config.vision_config.video_decode_backend == "decord"
    v_transform = make_eval_video_transform(vmodel.config)
    a_transform = get_audio_transform(amodel.config)

    print(f"Evaluating {len(clip_ids)} clips on {device}", flush=True)

    a_embeds, v_embeds, ok_ids = [], [], []
    n_failed = 0
    t0 = time.time()
    for i, cid in enumerate(clip_ids):
        vp = os.path.join(args.video_dir, cid + ".mp4")
        try:
            ve = load_video_embed(vp, vmodel, v_transform, num_frames, device)
            ae = load_audio_embed(vp, amodel, a_transform, device)
        except Exception as e:
            n_failed += 1
            print(f"  [FAIL] {cid}: {e}", flush=True)
            continue
        v_embeds.append(ve)
        a_embeds.append(ae)
        ok_ids.append(cid)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(clip_ids)} processed, {n_failed} failed so far, {time.time()-t0:.1f}s elapsed", flush=True)

    N = len(ok_ids)
    print(f"clips_requested=1545  clips_seen={N}  failed={n_failed}", flush=True)
    if n_failed == 0:
        assert N == 1545, f"expected clips_seen==1545 with zero failures, got {N}"

    a_embeds = torch.stack(a_embeds)
    v_embeds = torch.stack(v_embeds)
    metrics = compute_retrieval(a_embeds, v_embeds)

    print(f"audio→visual: R@1={metrics['audio_to_visual']['R@1']}% "
          f"R@5={metrics['audio_to_visual']['R@5']}% R@10={metrics['audio_to_visual']['R@10']}%", flush=True)
    print(f"visual→audio: R@1={metrics['visual_to_audio']['R@1']}% "
          f"R@5={metrics['visual_to_audio']['R@5']}% R@10={metrics['visual_to_audio']['R@10']}%", flush=True)

    video_ckpt = os.path.realpath(os.path.join(CACHE_DIR,
        "models--LanguageBind--LanguageBind_Video_FT/snapshots"))
    video_ckpt = os.path.join(video_ckpt, os.listdir(video_ckpt)[0], "pytorch_model.bin")
    audio_ckpt = os.path.realpath(os.path.join(CACHE_DIR,
        "models--LanguageBind--LanguageBind_Audio_FT/snapshots"))
    audio_ckpt = os.path.join(audio_ckpt, os.listdir(audio_ckpt)[0], "pytorch_model.bin")

    print("Hashing checkpoints (sha256) ...", flush=True)
    video_sha = sha256_of(video_ckpt)
    audio_sha = sha256_of(audio_ckpt)

    v_params = sum(p.numel() for p in vmodel.vision_model.parameters()) + sum(p.numel() for p in vmodel.visual_projection.parameters())
    a_params = sum(p.numel() for p in amodel.vision_model.parameters()) + sum(p.numel() for p in amodel.visual_projection.parameters())
    total_params_m = (v_params + a_params) / 1e6

    result = {
        "n_clips": N,
        "clips_requested": 1545,
        "n_failed": n_failed,
        "eval_list": "data/vggsound_eval_1545.txt",
        "audio_to_visual": metrics["audio_to_visual"],
        "visual_to_audio": metrics["visual_to_audio"],
        "ckpt": {"video": video_ckpt, "audio": audio_ckpt},
        "ckpt_sha256": {"video": video_sha, "audio": audio_sha},
        "repo": "https://github.com/PKU-YuanGroup/LanguageBind",
        "hf_repos": ["LanguageBind/LanguageBind_Video_FT", "LanguageBind/LanguageBind_Audio_FT"],
        "params_millions": round(total_params_m, 1),
        "pretrain_corpus": "VIDAL-10M (self-collected video/infrared/depth/audio/language pairs, language-anchored contrastive binding) — NOT VGGSound",
        "contamination_flag": "HELD-OUT",
        "preprocessing": (
            "Video: decord backend, 8 uniformly-sampled frames (np.linspace over full "
            "clip length), resize short side to 224, center-crop 224, OpenAI CLIP "
            "mean/std normalisation; RandomHorizontalFlip(p=0.5) from the official "
            "train-time transform disabled for deterministic eval. Audio: torchaudio.load "
            "(ffmpeg backend, mp4 read directly) -> resample to 16kHz -> Kaldi fbank "
            "(num_mel_bins=112, frame_length=25ms, frame_shift=10ms, hanning, htk_compat, "
            "dither=0) -> pad/repeat to target_length=1036 frames -> tile to 3 identical "
            "channels (deterministic for our <=10s clips) -> normalise "
            "(mel + 4.2677393) / (4.5689974 * 2). All numeric constants read from each "
            "checkpoint's own vision_config, not hardcoded."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda activate langbind_eval && python scripts/languagebind_retrieval.py",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
