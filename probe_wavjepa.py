"""probe_wavjepa.py — measure WavJEPA output shapes for a 10s clip.

Run BEFORE writing audio_encoder.py.  This confirms the real from_pretrained
call, output tensor shape, and token rate for both labhamlet/wavjepa-base and
labhamlet/wavjepa-nat-base.

Usage:
    python probe_wavjepa.py
"""
import time
import torch
from transformers import AutoModel, AutoFeatureExtractor

CLIP_DURATION_S = 10.0

# VGGSound / labhamlet models typically use 16 kHz mono audio
# nat-base is multi-channel (2ch), base is mono (1ch)
SAMPLE_RATE = 16_000

print("=" * 60)
print("WavJEPA probe: 10s clip at 16 kHz")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

n_samples = int(CLIP_DURATION_S * SAMPLE_RATE)

for repo, n_channels in [
    ("labhamlet/wavjepa-base", 1),
    ("labhamlet/wavjepa-nat-base", 2),
]:
    print(f"--- {repo} ---")
    t0 = time.time()

    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    load_time = time.time() - t0
    print(f"  loaded in {load_time:.1f}s")

    # Build dummy audio: (n_channels, n_samples) or (n_samples,) for mono
    if n_channels == 1:
        audio_np = torch.zeros(n_samples).numpy()
    else:
        audio_np = torch.zeros(n_channels, n_samples).numpy()

    # extractor expects numpy array (or list thereof)
    t1 = time.time()
    try:
        extracted = extractor(audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    except TypeError:
        # some extractors don't accept sampling_rate kwarg
        extracted = extractor(audio_np, return_tensors="pt")

    input_values = extracted["input_values"].to(device)
    print(f"  input_values shape: {tuple(input_values.shape)}")

    with torch.no_grad():
        out = model(input_values)

    infer_time = time.time() - t1

    # Inspect output — could be a tuple, BaseModelOutput, or plain tensor
    if isinstance(out, torch.Tensor):
        feat = out
    elif hasattr(out, "last_hidden_state"):
        feat = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        feat = out[0]
    else:
        feat = out[0]

    T, D = feat.shape[-2], feat.shape[-1]
    token_rate_hz = T / CLIP_DURATION_S

    print(f"  output shape:        {tuple(feat.shape)}")
    print(f"  T (tokens):          {T}")
    print(f"  D (dim):             {D}")
    print(f"  token rate:          {token_rate_hz:.1f} Hz  (for {CLIP_DURATION_S}s clip)")
    print(f"  infer time:          {infer_time:.2f}s")
    print(f"  out dtype:           {feat.dtype}")
    print()

print("DONE — copy token_rate numbers into models/audio_encoder.py and manifest.")
