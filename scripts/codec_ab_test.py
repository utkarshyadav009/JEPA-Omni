"""A/B the SAME speech tokens decoded by the INT8 ONNX codec (streaming, fast)
vs the fp32 torch NeuCodec (the 'clean' path the user preferred). Isolates
whether the 'processed' sound on non-lexical vocables (Ooh/Whoa/Mmm) is a codec
artifact. Generates tokens ONCE per line, decodes both ways, saves both."""
import sys, pathlib, numpy as np, wave, torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
from models.m5_streaming_voice import StreamingVoice, _linear_overlap_add

GGUF = "/home/utkarsh/JEPA-Omni/checkpoints/bmo_neutts_emotion_v4_Q8_0.gguf"
OUT = pathlib.Path("/home/utkarsh/JEPA-Omni/data/emotion_spotcheck/codec_ab"); OUT.mkdir(parents=True, exist_ok=True)
LINES = {"excited_ooh": ("Ooh ooh! Beemo found a brand new adventure!", "excited"),
         "surprised_whoa": ("Whoa! Beemo did not see that coming!", "surprised"),
         "content_mmm": ("Mmm, Beemo feels cozy and calm.", "content")}

def save(p, w, sr=24000):
    x = np.clip(w, -1, 1); pcm = (x*32767).astype(np.int16)
    with wave.open(str(p), "wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())

print("loading StreamingVoice (INT8 ONNX) ...", flush=True)
sv = StreamingVoice(GGUF, device=None)
print("loading torch NeuCodec (bf16) for the clean decode ...", flush=True)
from neucodec import NeuCodec
codec = NeuCodec.from_pretrained("neuphonic/neucodec")
codec = codec.eval().to(torch.bfloat16).cuda()

def gen_tokens(text, emotion):
    toks = sv.llm.tokenize(sv._prompt(text, emotion), special=True, add_bos=False)
    sv.llm.reset(); sp = []
    for tid in sv.llm.generate(toks, temp=sv.TEMP, top_k=sv.TOP_K):
        if tid == sv._end: break
        if sv._sp0 <= tid <= sv._sp_hi: sp.append(tid - sv._sp0)
        if len(sp) >= 300: break
    return sp

def decode_torch(sp):
    codes = torch.tensor(sp, dtype=torch.long, device="cuda").view(1, 1, -1)
    with torch.no_grad():
        wav = codec.decode_code(codes)
    return np.asarray(wav.float().cpu()).reshape(-1)

for name, (text, emo) in LINES.items():
    sp = gen_tokens(text, emo)
    a_int8 = np.concatenate([sv._decode(sp)])  # ONNX int8
    a_fp = decode_torch(sp)                      # torch bf16
    save(OUT / f"{name}_INT8onnx.wav", sv._trim_and_fade(a_int8, 24000))
    save(OUT / f"{name}_torchBF16.wav", sv._trim_and_fade(a_fp, 24000))
    print(f"{name}: n_tok={len(sp)} int8={len(a_int8)/24000:.2f}s torch={len(a_fp)/24000:.2f}s", flush=True)
print("-> http://100.87.60.100:8901/codec_ab/", flush=True)
