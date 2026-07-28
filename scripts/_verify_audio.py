import sys; sys.path.insert(0, '/home/utkarsh/JEPA-Omni')
from scripts.extract_features_av import _decode_audio_raw
wav = _decode_audio_raw('/home/utkarsh/data/vggsound/OxPnZzn1_L8_000883.mp4')
print('audio shape:', wav.shape, 'dtype:', wav.dtype)
rms = wav.pow(2).mean().sqrt().item()
print(f'audio rms: {rms:.4f}')
print('PASS' if rms > 0.001 else 'FAIL: zeros returned')
