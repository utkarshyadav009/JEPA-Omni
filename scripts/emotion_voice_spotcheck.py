"""Spot-check the emotion fine-tune with MOOD-APPROPRIATE lines (not one neutral
sentence) so prosody AND content combine -- the way emotions actually read in
conversation (per user feedback). Uses the exact production path (StreamingVoice
+ INT8 ONNX decoder). Saves wavs for listening + prints distinctness stats."""
import sys, argparse, pathlib, numpy as np, wave
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

# BMO-flavored, mood-fitting lines (use "Beemo" -- the model now knows it)
LINES = {
    "neutral":   "Okay! Let me help you with that.",
    "excited":   "Ooh ooh, Finn! A brand new adventure is starting, let's go!",
    "happy":     "Yay! That made Beemo so happy, thank you!",
    "content":   "Mmm, this is nice. Beemo feels cozy and calm right now.",
    "surprised": "Whoa?! Beemo did not see that coming at all!",
    "curious":   "Hmm, what is that? Beemo really wants to know how it works.",
    "tired":     "Beemo is so sleepy. Maybe just a little rest now.",
    "bored":     "There is nothing to do. Beemo is so bored.",
    "lonely":    "Finn? Jake? Beemo has been waiting here all alone.",
    "anxious":   "Oh no, oh no. Beemo is not sure this is safe.",
    "concerned": "Are you okay? Beemo is really worried about you.",
    "stressed":  "Too much, too much! Beemo needs everything to slow down!",
}

def save_wav(p, wav, sr=24000):
    x = np.clip(wav, -1, 1); pcm = (x*32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

def f0(wav, sr=24000):
    w = wav.astype(np.float64)
    if len(w) >= sr//2:
        step=sr//10; e=[np.sqrt(np.mean(w[i:i+step]**2)+1e-9) for i in range(0,len(w)-step,step)]
        c=int(np.argmax(e))*step; w=w[c:c+sr//2]
    w=w-w.mean()
    if np.sqrt(np.mean(w**2))<1e-4: return 0.0
    ac=np.correlate(w,w,"full")[len(w)-1:]; lo,hi=sr//400,sr//75
    if hi>=len(ac): return 0.0
    lag=lo+int(np.argmax(ac[lo:hi])); return sr/lag if lag else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/utkarsh/JEPA-Omni/checkpoints/bmo_neutts_emotion_v2_Q8_0.gguf")
    ap.add_argument("--out", default="/home/utkarsh/JEPA-Omni/data/emotion_spotcheck_v2")
    args = ap.parse_args()
    OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    from models.m5_streaming_voice import StreamingVoice
    print("Loading StreamingVoice on the v2 emotion GGUF ...", flush=True)
    sv = StreamingVoice(args.gguf, device=None)
    print(f"has_emotion={sv.has_emotion}", flush=True)
    print(f"{'mood':10s} {'dur_s':>6s} {'rms':>7s} {'f0_hz':>6s}  line", flush=True)
    for m, line in LINES.items():
        emo = None if m == "neutral" else m
        wav, dur = sv.synth(line, emotion=emo)
        rms = float(np.sqrt(np.mean(wav**2)+1e-12))
        save_wav(OUT / f"{m}.wav", wav)
        print(f"{m:10s} {dur:6.2f} {rms:7.4f} {f0(wav):6.0f}  {line[:40]}", flush=True)
    print(f"wavs -> {OUT}", flush=True)
