"""Debug retrieval with a few real clips."""
import sys, torch, json, random, os
sys.path.insert(0, '/home/utkarsh/JEPA-Omni')
from scripts.cavmae_retrieval import (
    CAVMAEEncoder, waveform_to_logmel, decode_center_frame,
    recall_at_k, VIDEO_SIZE, AUDIO_TIME_FRAMES, EMBED_DIM
)
import torch.nn.functional as F

device = "cpu"
model = CAVMAEEncoder.from_checkpoint('/home/utkarsh/models/cav-mae/audio_model.25.pth', device=device)

# Load 20 test clips
with open('/home/utkarsh/JEPA-Omni/data/cavmae_eval_subset.txt') as f:
    clip_ids = [l.strip() for l in f][:20]

video_dir = '/home/utkarsh/data/vggsound'
import torchaudio

audio_embs = []
video_embs = []
for vid in clip_ids:
    vpath = os.path.join(video_dir, vid + ".mp4")
    # Audio
    try:
        wav, sr = torchaudio.load(vpath)
        if wav.shape[0] > 1: wav = wav.mean(0)
        else: wav = wav[0]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = waveform_to_logmel(wav)
        # CAV-MAE normalization: mean -4.27, std 4.57 (from dataset)
        spec = (spec - (-4.27)) / 4.57
    except Exception as e:
        print(f"  audio fail {vid}: {e}")
        spec = torch.zeros(1, 128, AUDIO_TIME_FRAMES)

    # Video
    frame = decode_center_frame(vpath)
    # ImageNet normalization
    mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(3, 1, 1)
    std  = torch.tensor([0.2290, 0.2240, 0.2250]).view(3, 1, 1)
    frame = (frame - mean) / std

    with torch.no_grad():
        a_emb = model.encode_audio(spec.unsqueeze(0))
        v_emb = model.encode_video(frame.unsqueeze(0))

    audio_embs.append(a_emb[0])
    video_embs.append(v_emb[0])

audio_embs = torch.stack(audio_embs)  # (20, 768)
video_embs = torch.stack(video_embs)  # (20, 768)

print("audio emb std (should be >0.01):", audio_embs.std(dim=0).mean().item())
print("video emb std:", video_embs.std(dim=0).mean().item())
print("audio-audio similarity matrix diagonal:", (audio_embs @ audio_embs.T).diag())
print("audio-video similarity diagonal:", (audio_embs @ video_embs.T).diag()[:5])
print("audio-video similarity off-diag [0,1]:", (audio_embs[0] @ video_embs[1]).item())

# Retrieval on 20 clips
sim = audio_embs @ video_embs.T
print("\nWith CAV-MAE normalization:")
print(recall_at_k(sim))
