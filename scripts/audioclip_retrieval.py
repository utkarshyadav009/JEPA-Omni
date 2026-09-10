"""scripts/audioclip_retrieval.py — AudioCLIP audio<->visual retrieval baseline.

Runs AudioCLIP (Guzhov et al., "AudioCLIP: Extending CLIP to Image, Text and
Audio", ICASSP 2022 — https://github.com/AndreyGuzhov/AudioCLIP) on the fixed
1545-clip VGGSound eval gallery (data/vggsound_eval_1545.txt) and computes:
    audio->visual  R@1, R@5, R@10
    visual->audio  R@1, R@5, R@10

Checkpoint: AudioCLIP-Full-Training.pt (the single combined checkpoint — text,
image (ResNet-50 CLIP) and audio (ESResNeXt-fbsp) towers all loaded from this
one file via `AudioCLIP(pretrained=<path>)`, per the model's own __init__).

Preprocessing matches the model's own official demo notebook
(AudioCLIP/demo/AudioCLIP.ipynb) and its released eval-time transform config
(AudioCLIP/protocols/audioclip-esc50.json, "test" transform list), NOT our own
CAV-MAE-style constants:

Audio:
  - torchcodec AudioDecoder resampled to 44100 Hz (SAMPLE_RATE, per the demo's
    `librosa.load(path, sr=SAMPLE_RATE)` with SAMPLE_RATE=44100), mono
    (channel-averaged if the source is multi-channel).
  - Raw waveform fed directly to `AudioCLIP.encode_audio` — AudioCLIP computes
    its own FBSP spectrogram internally (n_fft=2048, hop=561, win=1654,
    window='blackmanharris', per `model/audioclip.py`'s ESResNeXtFBSP args).
    No external mel-spectrogram step, unlike CAV-MAE.
  - Center-cropped/padded to 220500 samples (5.0s @ 44.1kHz) — this is the
    exact eval-time (`train: false`) window length AudioCLIP's own released
    protocol config uses for `utils.transforms.RandomCrop` /
    `RandomPadding` (both deterministic-center in eval mode). VGGSound clips
    run slightly under/over 5s; this reproduces the model's own eval-time
    windowing convention rather than inventing a new one. Padding uses the
    edge-sample mean (RandomPadding's own fill value), cropping is centered
    (RandomCrop's own `train=False` branch: `left = round(0.5*(len-out))`).

Visual:
  - Center video frame, decoded via torchcodec VideoDecoder, converted to PIL,
    then AudioCLIP's own image_transforms from the demo notebook: Resize(224,
    bicubic) -> CenterCrop(224) -> Normalize(CLIP mean/std
    (0.48145466,0.4578275,0.40821073)/(0.26862954,0.26130258,0.27577711)).

Both embeddings are L2-normalised exactly as `AudioCLIP.forward` does
internally, then compared by plain cosine similarity (equivalent to the
model's own `audio_features @ image_features.T`, without the learned
logit-scale factor, which only rescales and does not change the ranking).

Usage:
    conda run -n jepa-omni python scripts/audioclip_retrieval.py
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import torchvision as tv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOCLIP_REPO = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/audioclip/repo"
sys.path.insert(0, AUDIOCLIP_REPO)

SAMPLE_RATE = 44100
AUDIO_OUT_LEN = 220500          # 5.0s @ 44.1kHz — AudioCLIP's own eval-time window (protocols/audioclip-esc50.json)
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

image_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])


def center_pad_or_crop(wav: Tensor, out_len: int = AUDIO_OUT_LEN) -> Tensor:
    """Reproduces AudioCLIP's own eval-time (train=False) RandomPadding + RandomCrop
    (utils/transforms.py) as a single deterministic center transform."""
    n = wav.shape[-1]
    if n < out_len:
        left = int(round(0.5 * (out_len - n)))
        right = out_len - left - n
        pad_val_l = wav[..., 0].float().mean().to(wav.dtype)
        pad_val_r = wav[..., -1].float().mean().to(wav.dtype)
        wav = torch.cat([
            torch.full((left,), pad_val_l, dtype=wav.dtype),
            wav,
            torch.full((right,), pad_val_r, dtype=wav.dtype),
        ])
    elif n > out_len:
        left = int(round(0.5 * (n - out_len)))
        wav = wav[..., left:left + out_len]
    return wav


def decode_audio(video_path: str) -> Tensor:
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(video_path, sample_rate=SAMPLE_RATE)
    w = dec.get_all_samples().data
    w = w.mean(0) if w.shape[0] > 1 else w[0]
    return center_pad_or_crop(w.float())


def decode_center_frame(video_path: str) -> Tensor:
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(video_path, device="cpu")
    n = int(getattr(decoder.metadata, "num_frames", None) or len(decoder))
    idx = n // 2
    frame = decoder.get_frames_at(indices=[idx]).data[0]        # (3,H,W) uint8
    pil = tv.transforms.functional.to_pil_image(frame)
    return image_transforms(pil)


def recall_at_k(sim: Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(AUDIOCLIP_REPO, "assets", "AudioCLIP-Full-Training.pt"))
    parser.add_argument("--video-dir", default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    parser.add_argument("--eval-list", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "data", "audioclip_retrieval_results.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}", flush=True)
    assert len(clip_ids) == 1545, f"Eval list must have 1545 clips, got {len(clip_ids)}"

    from model import AudioCLIP
    model = AudioCLIP(pretrained=args.ckpt).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    ok_ids: List[str] = []
    audio_embs: List[Tensor] = []
    video_embs: List[Tensor] = []
    n_failed = 0
    fail_examples = []

    N = len(clip_ids)
    print(f"Evaluating {N} clips on {args.device}", flush=True)

    for i in range(0, N, args.batch_size):
        batch_ids = clip_ids[i:i + args.batch_size]
        wavs, frames, kept = [], [], []
        for vid in batch_ids:
            vpath = os.path.join(args.video_dir, vid + ".mp4")
            try:
                w = decode_audio(vpath)
                fr = decode_center_frame(vpath)
            except Exception as e:
                n_failed += 1
                if len(fail_examples) < 10:
                    fail_examples.append((vid, str(e)[:120]))
                continue
            wavs.append(w)
            frames.append(fr)
            kept.append(vid)

        if not kept:
            continue

        wav_batch = torch.stack(wavs).unsqueeze(1).to(args.device)   # (B,1,T)
        frame_batch = torch.stack(frames).to(args.device)            # (B,3,224,224)

        with torch.no_grad():
            ((a_feat, _, _), _), _ = model(audio=wav_batch)
            ((_, v_feat, _), _), _ = model(image=frame_batch)
            a_feat = F.normalize(a_feat, dim=-1)
            v_feat = F.normalize(v_feat, dim=-1)

        audio_embs.append(a_feat.cpu())
        video_embs.append(v_feat.cpu())
        ok_ids.extend(kept)

        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch_ids)}/{N}  (failed so far: {n_failed})", flush=True)

    n_ok = len(ok_ids)
    print(f"clips_seen={n_ok}  failed={n_failed}", flush=True)
    if n_failed:
        print(f"  fail examples: {fail_examples}", flush=True)

    if n_ok == 1545:
        print("clips_seen == 1545 (full-gallery OK)", flush=True)
    else:
        print(f"WARNING: clips_seen ({n_ok}) != 1545 (requested). Evaluating on intersection.", flush=True)

    audio_embs = torch.cat(audio_embs, dim=0)
    video_embs = torch.cat(video_embs, dim=0)

    sim = audio_embs @ video_embs.T   # (N,N) cosine sim, both already L2-normalised

    a2v = recall_at_k(sim, (1, 5, 10))
    v2a = recall_at_k(sim.T, (1, 5, 10))

    print("\naudio->visual:", " ".join(f"{k}={v}%" for k, v in a2v.items()))
    print("visual->audio:", " ".join(f"{k}={v}%" for k, v in v2a.items()))

    ckpt_sha256 = hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest()

    audio_params = sum(p.numel() for p in model.audio.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    result = {
        "n_clips": n_ok,
        "clips_requested": 1545,
        "eval_list": "data/vggsound_eval_1545.txt",
        "audio_to_visual": a2v,
        "visual_to_audio": v2a,
        "ckpt": args.ckpt,
        "ckpt_sha256": ckpt_sha256,
        "repo": "https://github.com/AndreyGuzhov/AudioCLIP",
        "params_millions": round(total_params / 1e6, 1),
        "audio_tower_params_millions": round(audio_params / 1e6, 1),
        "pretrain_corpus": "Audio tower (ESResNeXt-fbsp): AudioSet. Image+text towers: frozen/fine-tuned from OpenAI CLIP (ResNet-50), CLIP's original ~400M web image-text pairs.",
        "contamination_flag": "HELD-OUT",
        "preprocessing": (
            "Audio: torchcodec AudioDecoder resampled to 44100 Hz, mono; raw waveform "
            "center-cropped/padded to 220500 samples (5.0s), matching AudioCLIP's own "
            "eval-time RandomCrop/RandomPadding (protocols/audioclip-esc50.json, "
            "train:false); fed directly to AudioCLIP.encode_audio (internal FBSP "
            "spectrogram, n_fft=2048, hop=561, win=1654, window=blackmanharris, no "
            "external mel step). Visual: torchcodec center frame -> PIL -> "
            "Resize(224,bicubic) -> CenterCrop(224) -> Normalize(CLIP mean/std), matching "
            "AudioCLIP's demo/AudioCLIP.ipynb image_transforms exactly. Both embeddings "
            "L2-normalised (as AudioCLIP.forward does internally) before cosine similarity."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda run -n jepa-omni python scripts/audioclip_retrieval.py",
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
