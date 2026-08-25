#!/usr/bin/env python3
"""bmo_showcase.py -- hardened showcase pipeline. Supersedes causal_duplex_pipeline.py
(kept as a fallback). Everything here is either measured on-device 2026-08-23 or a
documented invariant; see docs/SHOWCASE_PREP_2026-08-23.md.

LOAD ORDER IS NOT NEGOTIABLE (bmo_jetson_startup.py docstring, CLAUDE.md:124,
MEMORY_OPTIMIZATION_PLAN.md:103): llama.cpp GGUFs FIRST, torch/int8 perception SECOND.
Reversing it fragments Jetson unified memory and the GGUFs fail to allocate. Re-confirmed
2026-08-23: loading a GGUF *after* perception fails at Q8_0/Q6_K/Q4_K_M alike -- even a
484 MB model with 675 MiB free -- which is why LazyGGUFReasoningTier is not used anywhere.

Design rules this file enforces:
  * No stage may crash the loop. Every external call is wrapped; degraded > dead.
  * TTS streams; visemes are extracted PER CHUNK and scheduled against the audio device's
    real playback position, not wall clock (drift-free if generation stalls).
  * Exactly ONE mic gain stage.
  * HomeostaticState.update() is actually called, so mood is not a constant.
"""
from __future__ import annotations
import argparse, collections, gc, os, re, subprocess, sys, threading, time, traceback
from math import gcd
import numpy as np

sys.path.insert(0, "/home/bmo/bmo_production/pipeline")
sys.path.insert(0, "/home/bmo/bmo_production/scripts")

SR_MIC, SR_OUT = 16000, 24000
PROD = "/home/bmo/bmo_production"
G, TOK = f"{PROD}/models_gguf", f"{PROD}/tokenizers"
VAD_MODEL = "/home/bmo/sherpa_models/silero_vad.onnx"
SV_DIR = "/home/bmo/sherpa_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
VISEME_PATH = "/dev/shm/bmo_speech.txt"

# ONE gain stage. Previously MIC_GAIN=1.8 in the callback AND *1.5 in process_turn
# clipped twice (2.7x) and fed distorted audio to ASR.
MIC_GAIN, VOLUME = 2.0, 0.85


def log(m): print(m, flush=True)


def compact_memory():
    try:
        with open("/proc/sys/vm/compact_memory", "w") as f:
            f.write("1")
    except Exception:
        os.system("echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1")


def mem_avail():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"):
                return int(l.split()[1]) // 1024
    except Exception:
        pass
    return -1


# ───────────────────────── audio out ─────────────────────────
class UnifiedAudioEngine:
    """Continuous OutputStream + ring queue. Tracks REAL playback position so the
    viseme scheduler cannot drift from the audio when generation stalls."""

    def __init__(self, out_dev, sample_rate=SR_OUT):
        import sounddevice as sd
        self._sd = sd
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._q = collections.deque()
        self._played = 0            # frames actually handed to the device
        self._stream = None
        self.ok = False
        try:
            def cb(outdata, frames, t, status):
                out = np.zeros(frames, dtype=np.float32)
                pos = 0
                with self._lock:
                    while pos < frames and self._q:
                        c = self._q[0]
                        need = frames - pos
                        if len(c) <= need:
                            out[pos:pos + len(c)] = c; pos += len(c); self._q.popleft()
                        else:
                            out[pos:pos + need] = c[:need]; self._q[0] = c[need:]; pos += need
                    self._played += pos
                outdata[:, 0] = out * VOLUME
            self._stream = sd.OutputStream(device=out_dev, samplerate=sample_rate, channels=1,
                                           dtype="float32", blocksize=1200, callback=cb)
            self._stream.start(); self.ok = True
        except Exception as e:
            log(f"[audio] output unavailable ({e!r}) -- running silent")

    def played_sec(self):
        with self._lock:
            return self._played / float(self.sample_rate)

    def queued_sec(self):
        with self._lock:
            return sum(len(c) for c in self._q) / float(self.sample_rate)

    def push(self, audio, sr=SR_OUT):
        if audio is None or len(audio) == 0:
            return
        try:
            from scipy.signal import resample_poly
            if sr != self.sample_rate:
                g = gcd(sr, self.sample_rate)
                audio = resample_poly(audio.astype(np.float32), self.sample_rate // g, sr // g)
            with self._lock:
                self._q.append(np.asarray(audio, dtype=np.float32))
        except Exception as e:
            log(f"[audio] push failed: {e!r}")

    def busy(self):
        with self._lock:
            return bool(self._q)

    def reset_clock(self):
        with self._lock:
            self._played = 0

    def close(self):
        try:
            if self._stream: self._stream.stop(); self._stream.close()
        except Exception:
            pass


# ───────────────────────── streaming visemes ─────────────────────────
class StreamingVisemeExtractor:
    """Per-chunk causal formant->viseme extraction. The old AudioFormantExtractor ran over
    the FINISHED waveform, which forced synthesis to complete before playback (measured
    ~2900 ms to first audio). It is a 30 ms-frame / 15 ms-hop FFT and already causal; the
    only non-causal part was the MIN_DWELL merge, handled here by carrying one pending
    segment across chunk boundaries. Whole-utterance cost measured at 20-29 ms, so doing
    this incrementally is effectively free."""

    FRAME_MS, HOP_MS, MIN_DWELL_MS, SILENCE_RMS = 30, 15, 60, 0.018

    def __init__(self, sr=SR_OUT):
        self.sr = sr
        self.fsz = int(self.FRAME_MS * sr / 1000)
        self.hop = int(self.HOP_MS * sr / 1000)
        self.win = np.hanning(self.fsz).astype(np.float32)
        self._resid = np.zeros(0, dtype=np.float32)
        self._pending = None          # segment awaiting MIN_DWELL confirmation
        self._t = 0.0                 # absolute time of the next frame

    def _classify(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < self.SILENCE_RMS:
            return "mouth_phoneme_X", 0.0, rms
        fft = np.abs(np.fft.rfft(frame * self.win))
        freqs = np.fft.rfftfreq(self.fsz, 1.0 / self.sr)
        def band(a, b): return float(np.sum(fft[(freqs >= a) & (freqs < b)] ** 2))
        lo, ml, mh, hi = band(150, 700), band(700, 1600), band(1600, 3000), band(3000, 8000)
        tot = lo + ml + mh + hi + 1e-9
        rl, rml, rmh, rh = lo / tot, ml / tot, mh / tot, hi / tot
        cen = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-9))
        inten = min(1.2, max(0.4, rms / 0.08))
        if rh > 0.40 or (cen > 3200 and rh > 0.25):
            v = "CDGKNST" if cen > 4200 else "J"
        elif rl > 0.65:
            v = "U" if cen < 900 else "O"
        elif rmh > 0.35 or (cen > 1800 and rmh > rml):
            v = "EE"
        else:
            v = "AEI"
        return v, inten, rms

    def push(self, chunk):
        """Feed one audio chunk; return [(viseme, t_start, t_end, intensity)] ready to schedule."""
        out = []
        buf = np.concatenate([self._resid, np.asarray(chunk, dtype=np.float32)])
        i = 0
        while i + self.fsz <= len(buf):
            v, inten, _ = self._classify(buf[i:i + self.fsz])
            t0 = self._t
            self._t += self.HOP_MS / 1000.0
            if self._pending and self._pending[0] == v:
                self._pending[2] = t0 + self.HOP_MS / 1000.0
                self._pending[3] = max(self._pending[3], inten)
            else:
                if self._pending:
                    if (self._pending[2] - self._pending[1]) * 1000 >= self.MIN_DWELL_MS:
                        out.append(tuple(self._pending))
                    elif out:      # too short: absorb into the previous segment
                        prev = list(out[-1]); prev[2] = self._pending[2]; out[-1] = tuple(prev)
                self._pending = [v, t0, t0 + self.HOP_MS / 1000.0, inten]
            i += self.hop
        self._resid = buf[i:]
        return out

    def flush(self):
        out = []
        if self._pending:
            out.append(tuple(self._pending)); self._pending = None
        return out

    def reset(self):
        self._resid = np.zeros(0, dtype=np.float32); self._pending = None; self._t = 0.0


class VisemeScheduler:
    """Writes the current viseme to /dev/shm, driven by the audio device's REAL playback
    position (engine.played_sec()), so a generation stall cannot desync the mouth."""

    def __init__(self, engine, face_emotion=""):
        self.engine, self.face_emotion = engine, face_emotion
        self._tl, self._lock = [], threading.Lock()
        self._stop = threading.Event()
        self._th = None

    def add(self, segs):
        if segs:
            with self._lock:
                self._tl.extend(segs)

    def _write(self, v, inten):
        try:
            tmp = f"{VISEME_PATH}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(f"1 {v} {inten:.2f} {self.face_emotion}\n" if self.face_emotion
                        else f"1 {v} {inten:.2f}\n")
            os.replace(tmp, VISEME_PATH)
        except Exception:
            pass

    def start(self):
        def run():
            last = None
            while not self._stop.is_set():
                el = self.engine.played_sec()
                cur = None
                with self._lock:
                    for v, a, b, inten in self._tl:
                        if a <= el < b:
                            cur = (v, inten); break
                if cur and cur[0] != last:
                    self._write(cur[0], cur[1]); last = cur[0]
                time.sleep(0.010)
        self._stop.clear(); self._th = threading.Thread(target=run, daemon=True); self._th.start()

    def stop(self):
        self._stop.set()
        if self._th: self._th.join(timeout=0.3)
        try:
            tmp = f"{VISEME_PATH}.{os.getpid()}.tmp"
            with open(tmp, "w") as f: f.write("0 mouth_phoneme_X 0.0\n")
            os.replace(tmp, VISEME_PATH)
        except Exception:
            pass


# ───────────────────────── ASR ─────────────────────────
class SenseVoiceASR:
    """Chosen over Citrinet on recognition quality. NOTE: Citrinet was previously judged
    'bad' on audio that had passed through TWO clipping gain stages (2.7x) -- that ranking
    deserves a re-test on the single-gain path now that S2 is fixed."""

    def __init__(self):
        import sherpa_onnx
        self.r = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{SV_DIR}/model.int8.onnx", tokens=f"{SV_DIR}/tokens.txt",
            num_threads=2, use_itn=True, language="en", debug=False, provider="cuda")
        st = self.r.create_stream()
        st.accept_waveform(SR_MIC, np.zeros(SR_MIC // 2, np.float32))
        self.r.decode_stream(st)

    def transcribe(self, wav, sr=SR_MIC):
        try:
            st = self.r.create_stream()
            st.accept_waveform(sr, wav)
            self.r.decode_stream(st)
            raw = st.result.text or ""
        except Exception as e:
            log(f"[stt] failed: {e!r}"); return "", "NEUTRAL"
        m = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>", raw, re.I)
        emo = m.group(1).upper() if m else "NEUTRAL"
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emo


# ───────────────────────── the stack ─────────────────────────
_CLEAN1 = re.compile(r"<.*?>")
_CLEAN2 = re.compile(r"\[.*?\]|\(.*?\)|(\*.*?\*)")
_CLEAN3 = re.compile(r"[^\w\s.,?'\"-]")
# bmu/bmp/bno observed in v6 output ("BMU says hi!") -- the 350M tier mis-emits the
# name in several ways, so fold every near-miss to BMO before it is spoken or logged.
NAME_RE = re.compile(r"\b(be+ ?mo(re)?|bee? ?more|b ?more|d[- ]?mo|di?mo|mi?mo|imo|nemo|demo"
                     r"|bre?mo|vi?mo|bi?mo|be?mo|ee?mo|pee?mo|mee?mo|neemo"
                     r"|bmj|bmw|bmu|bmp|bno|bmo0)\b", re.I)


def clean_for_tts(t):
    t = _CLEAN1.sub("", t); t = _CLEAN2.sub("", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t); t = _CLEAN3.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


# Utterances that are ASKING about what BMO can perceive. On these the scene is pushed into
# the speaker prompt in its trained `perception_grounded` shape.
PERCEPTION_ASK = re.compile(
    r"\b(what (do|can) you (see|hear)|describe|what'?s (happening|going on|in the room)|"
    r"look around|what am i wearing|where are we|who (is|are) (here|there|i)|"
    r"what is that (noise|sound)|can you see)\b", re.I)

# ── enrolment ──────────────────────────────────────────────────────────────────────────
# BMO must NEVER invent a name. The whole "why does it think I'm Alice" incident came from a
# simulated enrolment persisting, and the corpus bug behind it was rows that addressed a
# stranger by an invented name. So a name is only ever taken from an EXPLICIT self-statement
# the person actually made -- never guessed from context, never inferred.
# The prefix is case-flexible (sentence-initial "I'm"/"My"), but the NAME must stay
# capitalised -- that is the actual signal separating "I'm Utkarsh" from "I'm tired". Using
# re.I on the whole pattern would make [A-Z] case-insensitive and destroy exactly that.
# SenseVoice runs with use_itn=True and does capitalise, so this holds on real transcripts;
# if it ever stops, enrolment simply does not fire, which is the safe direction to fail.
NAME_PATTERNS = [
    re.compile(r"\b[Mm]y name(?:'?s| is)\s+([A-Z][a-z]{1,15})\b"),
    re.compile(r"\b[Ii]'?m\s+([A-Z][a-z]{1,15})\b"),
    re.compile(r"\b[Ii] am\s+([A-Z][a-z]{1,15})\b"),
    re.compile(r"\b[Cc]all me\s+([A-Z][a-z]{1,15})\b"),
    re.compile(r"\b[Ii]t(?:'?s| is)\s+([A-Z][a-z]{1,15})\b"),
    re.compile(r"^\s*([A-Z][a-z]{1,15})\s*[.!]?\s*$"),      # bare reply to "what's your name?"
]
# words that look like names to a regex but are not
NOT_A_NAME = {"okay", "sorry", "sure", "yeah", "yes", "no", "hi", "hey", "hello", "good",
              "fine", "tired", "busy", "here", "back", "just", "well", "nothing", "nobody",
              "beemo", "bmo", "thanks", "thank", "right", "really", "actually", "still"}


def extract_name(text: str):
    for pat in NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1)
            if cand.lower() in NOT_A_NAME or len(cand) < 2:
                continue
            return cand
    return None


_PLACEHOLDER = re.compile(r"(\{[a-z_]+\}|\[[a-z_]+\]|<[a-z_]+>)", re.I)


def strip_placeholder(t: str) -> str:
    """Belt-and-braces for the {name}/[name] leak. Removing the token alone leaves
    "I'm , and you are...?" -- worse than the leak -- so remove the ADDRESS construct with
    its punctuation, which is exactly what scripts/fix_name_placeholders.py does to the
    corpus. Substituting a real name here would be the bug that produced the "why does it
    think I'm Alice" incident, so we never invent one."""
    PH = r"(?:\{[a-z_]+\}|\[[a-z_]+\]|<[a-z_]+>)"
    # "I'm [name], and you are...?" -> drop the whole self-introduction clause
    t = re.sub(rf"\bI'?m\s+{PH}\s*,?\s*", "", t, flags=re.I)
    # ", {name}" / " {name}" as a term of address -> drop it
    t = re.sub(rf",?\s*{PH}\s*", " ", t, flags=re.I)
    # re-capitalise if the clause removal left a lowercase sentence start
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r",\s*,", ",", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _directive_guard(d: str):
    """The deployed thinker (bmo_thinker_qwen3_v5) was fine-tuned to produce conversational
    ANSWERS, not directives -- asked for an instruction it returns a finished BMO line
    ("Alright! Let's press start on a nice jingle together."). Feeding that to the speaker as
    "[instruction] <a BMO line>" recreates exactly the paraphrase/non-sequitur failure the
    directive contract exists to remove. Until the thinker is retrained on a directive corpus,
    reject anything that reads as a spoken line rather than an instruction.
    Verified on-device 2026-08-23: v5 fails this guard, which is the correct outcome."""
    if not d:
        return None
    t = d.strip()
    low = t.lower()
    # first person / addressed to the user -> it is a line, not an instruction
    if re.search(r"\b(i|i'm|i'll|let's|we|my|beemo can|shall we)\b", low[:40]):
        return None
    if t.count("!") >= 1 and len(t) < 90:
        return None
    if low.startswith(("hey", "hi ", "hello", "oh ", "wow", "alright")):
        return None
    # a directive should read as an instruction verb
    if not re.match(r"^(ask|tell|offer|acknowledge|suggest|remind|encourage|greet|reassure|"
                    r"invite|comfort|answer|explain|check|avoid|do not|don't|keep|stay)\b", low):
        return None
    return t


class BmoShowcase:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.speaker = self.thinker = self.tts = self.asr = self.vad = None
        self._bc_clips = []
        self.perception = None
        self.stop_event = threading.Event()
        self.robot_speaking = threading.Event()
        self._directive = None
        self._thinker_lock = threading.Lock()
        self._last_emb = None
        self._awaiting_name = False
        self._asked_name_once = False

        from models.homeostatic_state import HomeostaticState
        self.state = HomeostaticState()
        self._last_turn_t = time.time()

        self._build()

    # ---- LOAD ORDER: llama.cpp GGUFs FIRST, torch perception SECOND. Do not reorder. ----
    def _build(self):
        import torch
        from transformers import AutoTokenizer
        a = self.args
        log(f"[boot] avail={mem_avail()} MiB")

        # 1/4 speaker
        compact_memory()
        from models.m4_cognitive_core import GGUFFastTier
        tok = AutoTokenizer.from_pretrained(f"{TOK}/lfm25_350m_tok")
        self.speaker = GGUFFastTier(f"{G}/{a.speaker}", tok, max_new_tokens=a.speaker_tokens,
                                    n_gpu_layers=-1)
        log(f"[boot] speaker {a.speaker}  avail={mem_avail()} MiB")

        # 2/4 thinker (Q4_K_M: measured to fit WITH camera; Q8_0 leaves 0 MiB and the
        #     camera then returns 0 frames). Eager and IN ORDER -- lazy loading fails.
        if a.thinker != "none":
            try:
                from models.m4_cognitive_core import GGUFReasoningTier
                rtok = AutoTokenizer.from_pretrained(f"{TOK}/qwen3_thinker_tok")
                self.thinker = GGUFReasoningTier(f"{G}/{a.thinker}", rtok,
                                                 max_new_tokens=a.thinker_tokens, n_gpu_layers=-1)
                log(f"[boot] thinker {a.thinker}  avail={mem_avail()} MiB")
            except Exception as e:
                log(f"[boot] thinker SKIPPED ({e!r}) -- continuing without one")
                self.thinker = None

        # 3/4 TTS
        from models.m5_streaming_voice import StreamingVoice
        # VOICE ORDER -- MEASURED 2026-08-23, n=32 utterances per voice, neutral only:
        #   bmo_neutts_nano_v1        0/32 non-EOS exits, duration 3.74-5.84s on a 52-char line
        #   bmo_neutts_emotion_nano   9/32 (28%) non-EOS, SAME line ranged 2.64-9.20s
        #   (a 30-char line hit the cap on MOST samples: median == max == 7.00s)
        # The emotion fine-tune degraded EOS calibration -- and that is at `neutral`, its
        # anchor mood. A voice that stretches a short sentence to 9s on stage is worse than
        # a voice without mood tokens, so plain Nano leads. Pass --voice
        # bmo_neutts_emotion_nano_Q8_0.gguf to override once EOS is recalibrated.
        cands = [a.voice] if a.voice else ["bmo_neutts_nano_v1_Q8_0.gguf",
                                           "bmo_neutts_nano_v2_Q8_0.gguf",
                                           "bmo_neutts_emotion_nano_Q8_0.gguf",
                                           "bmo_neutts_v5_Q8_0.gguf"]
        path = next((f"{G}/{c}" for c in cands if os.path.exists(f"{G}/{c}")), None)
        if path is None:
            raise SystemExit("no TTS GGUF found")
        self.tts = StreamingVoice(path, compact_fn=compact_memory)
        self.has_emotion = bool(getattr(self.tts, "has_emotion", False))
        log(f"[boot] voice {os.path.basename(path)} emotion={self.has_emotion}  avail={mem_avail()} MiB")

        # Load the thinking_filler WAVs directly rather than via PrebuiltVoiceBank, so they
        # go through OUR audio engine (see the note in process_turn).
        self._bc_clips = []
        try:
            import soundfile as sf, glob as _glob
            for f in sorted(_glob.glob(f"{PROD}/pipeline/assets/bmo_backchannels/thinking_filler*.wav")):
                w, sr = sf.read(f, dtype="float32")
                if w.ndim > 1: w = w.mean(axis=1)
                self._bc_clips.append((w, sr))
            log(f"[boot] backchannel clips: {len(self._bc_clips)}")
            if not self._bc_clips:
                log("[boot]   WARNING: no thinking_filler*.wav found -- latency will be bare")
        except Exception as e:
            log(f"[boot] backchannel clips skipped ({e!r})")

        # 4/4 ASR + VAD (onnxruntime, not llama.cpp -- safe here)
        self.asr = SenseVoiceASR()
        import sherpa_onnx
        vc = sherpa_onnx.VadModelConfig()
        vc.silero_vad.model = VAD_MODEL
        vc.silero_vad.threshold = a.vad_threshold
        vc.silero_vad.min_silence_duration = a.min_silence
        vc.silero_vad.min_speech_duration = 0.18
        vc.silero_vad.max_speech_duration = 15.0
        vc.sample_rate = SR_MIC; vc.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(vc, buffer_size_in_seconds=30)
        log(f"[boot] STT+VAD ready  avail={mem_avail()} MiB")

        # 5/5 PERCEPTION -- torch/int8, LAST. Never before the GGUFs (load-order rule).
        if a.perception:
            try:
                self._build_perception()
            except Exception:
                log("[boot] perception SKIPPED:\n" + traceback.format_exc())
                self.perception = None
        log(f"[boot] DONE  avail={mem_avail()} MiB")

    def _build_perception(self):
        import torch, torch.nn.functional as F
        from transformers import AutoModel
        from bmo_jetson_startup import q_int8_cpu_then_move
        dev = torch.device("cuda")
        P = f"{PROD}/pipeline/checkpoints"

        from models.vision_encoder import VisionEncoder
        ve = VisionEncoder(device="cpu", dtype=torch.bfloat16)
        ve.model = q_int8_cpu_then_move(ve.model, dev); ve.device_str = "cuda"
        from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO
        wj = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
        wj.model = q_int8_cpu_then_move(wj.model, dev); wj.device_str = "cuda"
        from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
        pred = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                            max_tdm_bins=512, dropout=0.0))
        ck = torch.load(f"{P}/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
                        map_location="cpu", weights_only=False)
        pred.load_state_dict(ck["model"], strict=True); del ck; gc.collect()
        pred = pred.to(torch.bfloat16)          # O1: -64 MiB resident, -231 transient, cos 0.999969
        pred = q_int8_cpu_then_move(pred, dev); pred.eval()

        from models.m5_perception_query import load_perception_query_engine
        from models.text_target import PreEncodedTextSpace
        qck = torch.load(f"{P}/qp_runD.pt", map_location="cpu", weights_only=False)
        qv = torch.load(f"{P}/query_vectors_siglip2_v2.pt", map_location="cpu", weights_only=False)
        tt = PreEncodedTextSpace(qv["text"], qv["emb"], device=str(dev))
        cand = torch.load(f"{P}/candidates_siglip2_v2.pt", map_location="cpu", weights_only=False)
        raw = F.normalize(cand["emb"].float(), dim=-1).to(dev)
        tp = qck.get("text_target_proj") or {}
        bank = (F.normalize(raw @ tp["weight"].float().to(dev).t() + tp["bias"].float().to(dev), dim=-1)
                if tp else raw)
        pq = load_perception_query_engine(f"{P}/qp_runD.pt", tt, dev,
                                          bank_emb=bank, bank_text=cand["text"], max_age_s=15.0)
        # REQUIRED. Without this, `cats` in _ask_perception is None and every category falls
        # back to a top-1 over the whole 1,482-tag bank -- so all six questions return the SAME
        # tag ("who: dim lighting; wearing: dim lighting; doing: dim lighting; ..."). Production
        # sets it in bmo_jetson_startup.py:398; this path omitted it and the composed scene was
        # silently useless. Caught in a live full-stack run, not by any offline check.
        pq.bank_category = cand.get("category", ["mined"] * len(cand["text"]))
        sig = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16)
        if hasattr(sig, "text_model"): del sig.text_model
        sig = sig.to(dev).eval()
        self._p = dict(ve=ve, wj=wj, pred=pred, pq=pq, sig=sig, dev=dev)

        # identity head + the two persistent stores
        self.ident = self.idmem = self.bmem = None
        if self.args.identity:
            try:
                from models.jepa_identity_head import IdentityHead, IdentityHeadConfig
                from models.jepa_memory import JepaMemory, MemoryConfig
                from models.bmo_memory import BmoMemory
                ick = torch.load(f"{P}/identity_head_joint.pt", map_location="cpu",
                                 weights_only=False)
                self.ident = IdentityHead(IdentityHeadConfig(in_dims=ick["dims"],
                                                             emb_dim=ick["emb_dim"])).to(dev)
                self.ident.load_state_dict(ick["head"]); self.ident.eval()
                # NOTE threshold 0.5 is the DEFAULT, not the calibrated 0.691-0.765 operating
                # point. calibrate_threshold() exists and has never been called. At 0.5 the
                # false-accept rate is higher than measured, i.e. more likely to call a
                # stranger by an enrolled name -- keep enrolments few and distinct until it
                # is calibrated on this camera and room.
                try:
                    self.idmem = JepaMemory.load(self.args.identity_store, device=str(dev))
                except Exception:
                    self.idmem = JepaMemory(MemoryConfig(threshold=0.5))
                self.bmem = BmoMemory(path=self.args.memory_store)
                log(f"[boot] identity ready ({len(self.idmem)} enrolled)")
            except Exception as e:
                log(f"[boot] identity SKIPPED ({e!r})")
                self.ident = None
        log(f"[boot] perception ready streams={pq.source_names}  avail={mem_avail()} MiB")

        # CAMERA. The CSI sensor is EXCLUSIVE -- only one process may open it. The face's
        # eye tracking needs it (motion_tracker -> /dev/shm/bmo_motion.txt) and so does
        # perception, so opening it here directly makes those mutually exclusive.
        # bmo_camera_hub.py exists to solve that: one nvarguscamerasrc, tee'd to a motion
        # branch and a 256x256 shmsink. MEASURED 2026-08-23: reading the hub socket costs
        # ~13 MiB and delivers 24/24 frames at 31.4 fps, vs ~290 MiB opening the sensor
        # directly. Prefer the hub; fall back to direct only if it is not running.
        import cv2
        HUB_SOCK = "/tmp/bmo_cam_perception.sock"
        self._cap = None
        # RETRY: the hub is a systemd service (Restart=always) and may not have reached
        # PLAYING when we boot, or may be mid-restart after its watchdog fired. A single
        # existence check races that and silently falls back to opening the CSI sensor
        # directly -- which would then make eye tracking impossible. Retry briefly first.
        for attempt in range(6):
            if os.path.exists(HUB_SOCK):
                gst = (f"shmsrc socket-path={HUB_SOCK} is-live=true ! "
                       "video/x-raw,format=BGRx,width=256,height=256,framerate=30/1 ! "
                       "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    self._cap = cap
                    log(f"[boot] camera via HUB socket (eye tracking stays alive)  "
                        f"avail={mem_avail()} MiB")
                    break
                cap.release()
            if attempt < 5:
                log(f"[boot]   hub socket not ready, retry {attempt+1}/5 ...")
                time.sleep(2)
        if self._cap is None:
            log("[boot] camera hub not running -> opening the CSI sensor DIRECTLY. "
                "Eye tracking cannot run at the same time.")
            gst = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,"
                   "height=720,framerate=30/1 ! nvvidconv flip-method=2 ! "
                   "video/x-raw,width=256,height=256,format=BGRx ! videoconvert ! "
                   "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
            self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            log(f"[boot] camera direct open={self._cap.isOpened()}  avail={mem_avail()} MiB")
        self.perception = self._ask_perception

    def _ask_perception(self, query):
        import torch, torch.nn.functional as F
        from models.world_state_builder import build_world_state_features
        from models.m5_motion_crop import siglip2_preprocess
        from PIL import Image as _Im
        p = self._p
        frames = []
        for _ in range(16):
            ok, f = self._cap.read()
            if ok: frames.append(f[:, :, ::-1].copy())
        if len(frames) < 16:
            return ""
        fr = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        wav = torch.from_numpy(np.zeros(SR_MIC * 10, np.float32) + 1e-4)
        ws = build_world_state_features(fr, wav, 10.0, p["ve"], p["wj"], None, 512, p["dev"])
        names = p["pq"].source_names; src = {}
        if "vision" in names: src["vision"] = ws.feats["vision"].float()
        if "ambient" in names: src["ambient"] = ws.feats["ambient"].float()
        if "m2" in names:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                src["m2"] = p["pred"].encode_pre_pool_tokens(
                    {k: v.float() for k, v in ws.feats.items()}, ws.tbins).float()
        if "scene" in names:
            idx = torch.linspace(0, fr.shape[0] - 1, 4).long()
            imgs = [_Im.fromarray(fr[i].permute(1, 2, 0).numpy()) for i in idx]
            with torch.no_grad():
                px = siglip2_preprocess(imgs).to(p["dev"]).to(torch.bfloat16)
                o = p["sig"].vision_model(pixel_values=px)
                o = o.pooler_output if hasattr(o, "pooler_output") else o
                src["scene"] = F.normalize(o.float(), dim=-1).unsqueeze(0)
        p["pq"].set_perception(src, None)

        # Identity rides on the SAME world-state features perception just built -- no second
        # encode. Vision+voice jointly (measured TAR@FAR1% 0.765 joint vs 0.694 voice-only,
        # 0.571 vision-only), so both streams are pooled.
        if self.ident is not None:
            try:
                from models.jepa_identity_head import stats_pool
                with torch.no_grad():
                    self._last_emb = self.ident({
                        "vision": stats_pool(ws.feats["vision"]),
                        "audio":  stats_pool(ws.feats["ambient"])})[0]
            except Exception as e:
                log(f"  [identity] embed failed: {e!r}"); self._last_emb = None

        # A single ask() returns ONE tag out of 1,482 -- "describe the room in detail" comes
        # back as 'a shadowy room', not a description. The retrieval interface cannot compose.
        # So ask PER CATEGORY and assemble, which is also the exact format the speaker's
        # directive rows were trained on ("wearing: ..; doing: ..; who: ..; where: ..").
        # Restricting each question to the tags that could possibly answer it matters: one
        # top-1 over the whole bank cannot answer "what are they wearing" AND "what are they
        # doing" at once.
        QUESTIONS = [
            ("who",      "people",     "Tell me in detail what the person is doing."),
            ("wearing",  "appearance", "Describe the room and setting in detail."),
            ("doing",    "action",     "Explain everything that happens, in order."),
            ("where",    "place",      "What does this place look like? Describe it fully."),
            ("lighting", "light",      "Describe the room and setting in detail."),
            ("hearing",  "sound",      "What do you hear?"),
        ]
        cats = getattr(p["pq"], "bank_category", None)
        parts = []
        for label, cat, q in QUESTIONS:
            try:
                if cats is not None:
                    idx = [i for i, c in enumerate(cats) if c == cat]
                    if not idx:
                        # An empty category must be loud, not skipped: a silently-vanishing
                        # question is how the `wearing` field disappeared for a whole run.
                        log(f"  [percep] category '{cat}' EMPTY in the candidate bank")
                        continue
                    # RESTRICTED retrieval. ask_topk() searches the WHOLE 1,482-tag bank and
                    # takes no category argument -- an earlier version computed `idx` and then
                    # called ask_topk anyway, so all six questions returned the same top-1
                    # ("who: dim lighting; wearing: dim lighting; ..."). Score against the
                    # category's rows only, mirroring jetson_real_demo.py's CAT_IDX path.
                    eng = p["pq"]
                    with torch.no_grad():
                        q_emb = eng.tt.encode_text_frozen_raw([q]).to(eng.device)
                        z_q = eng.qp(eng._sources, q_emb, eng._masks)
                        ids = torch.as_tensor(idx, device=eng.bank_emb.device)
                        sims = (z_q.to(eng.bank_emb.dtype) @ eng.bank_emb[ids].T)[0].float()
                        best = int(torch.argmax(sims))
                    txt = eng.bank_text[idx[best]]
                else:
                    a = p["pq"].ask(q)
                    txt = a.text if a is not None else None
                if txt:
                    parts.append(f"{label}: {txt}")
            except Exception as e:
                log(f"  [percep] {label} failed: {e!r}")
        if not parts:
            return ""
        self._last_scene = "; ".join(parts)
        return self._last_scene

    # ---- one turn ----
    def _mood(self):
        from models.homeostatic_state import homeostatic_to_mood_state
        try:
            return homeostatic_to_mood_state(self.state)
        except Exception:
            return {"energy": 0.5, "mood": "curious"}

    def _update_state(self, *, user_spoke, arousal):
        """S1 FIX. HomeostaticState.update() was never called in the previous pipeline, so
        the mood was permanently its constructed value and the 12-mood emotion voice would
        have rendered exactly ONE mood forever, looking consistent rather than broken."""
        now = time.time()
        dt = max(0.0, now - self._last_turn_t); self._last_turn_t = now
        try:
            self.state.update(dt_s=dt, user_speaking=user_spoke, user_present=True,
                              scene_embedding_drift=0.0, input_arousal_signal=float(arousal))
        except Exception as e:
            log(f"[state] update failed (mood will be static): {e!r}")

    def speak(self, text, mood):
        """Stream TTS -> audio device, extracting visemes PER CHUNK and scheduling them
        against real playback position. Returns time-to-first-audio in ms."""
        text = clean_for_tts(NAME_RE.sub("BMO", text))
        if not text:
            return None
        emotion = mood.get("mood") if (self.has_emotion and not self.args.no_emotion) else None
        if emotion and emotion in self.args.unsafe_moods:
            emotion = "neutral"     # 4/12 moods run to the length cap; keep them off stage
        vx = StreamingVisemeExtractor(SR_OUT)
        sched = VisemeScheduler(self.engine, face_emotion=self.args.face_emotion)
        self.engine.reset_clock(); sched.start()
        self.robot_speaking.set()
        t0 = time.perf_counter(); ttfa = None; n = 0
        try:
            for seg in self.tts.stream(text, emotion):
                if seg is None or len(seg) == 0:
                    continue
                seg = np.nan_to_num(np.asarray(seg, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000.0
                sched.add(vx.push(seg))
                self.engine.push(seg, SR_OUT)
                n += len(seg)
        except Exception as e:
            log(f"[tts] stream failed: {e!r}")
        sched.add(vx.flush())
        reason = getattr(self.tts, "last_exit_reason", "?")
        if reason not in ("EOS_TOKEN", "?"):
            log(f"  [tts] exit={reason} (bounded, not a hang)")
        while self.engine.busy() and not self.stop_event.is_set():
            time.sleep(0.02)
        sched.stop(); self.robot_speaking.clear()
        try: self.vad.reset()
        except Exception: pass
        return ttfa

    def think_async(self, transcript, mood, scene):
        """Thinker runs OFF the response path and produces a DIRECTIVE for the next turn.
        It never emits a spoken line -- that removes the paraphrase/non-sequitur failure by
        construction, because there is no competing utterance for the speaker to echo."""
        if self.thinker is None or self._thinker_lock.locked():
            return
        def run():
            with self._thinker_lock:
                try:
                    p = (f"[what Beemo can see] {scene}\n" if scene else "") + \
                        f"The person said: '{transcript}'\n" \
                        f"Think briefly, then reply with ONE short instruction telling Beemo " \
                        f"what to say next. Do not write Beemo's line yourself."
                    r = self.thinker.generate(p, mood)
                    d = (r.text or "").strip()
                    d = _directive_guard(d)
                    if d:
                        self._directive = d[:200]
                        log(f"  [thinker] directive: {self._directive!r}")
                    else:
                        log(f"  [thinker] REJECTED (looks like a spoken line, not a "
                            f"directive): {(r.text or '')[:80]!r}")
                except Exception as e:
                    log(f"  [thinker] failed: {e!r}")
        threading.Thread(target=run, daemon=True).start()

    def process_turn(self, seg):
        t_all = time.perf_counter()
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-9))
        t0 = time.perf_counter()
        text, emo = self.asr.transcribe(seg)
        stt_ms = (time.perf_counter() - t0) * 1000
        text = NAME_RE.sub("BMO", text)
        if len(text) < 2 or text.lower().strip(" .!?") in ("mm", "mmm", "um", "uh", "hmm"):
            return
        log(f"\n  YOU [{emo}]: {text}   (stt {stt_ms:.0f}ms, rms {rms:.4f})")

        self._update_state(user_spoke=True, arousal=1.0 if emo in ("ANGRY", "SURPRISED") else 0.3)
        mood = self._mood()

        if self._bc_clips and self.args.backchannel:
            # PrebuiltVoiceBank exposes play(category) which drives its OWN sounddevice
            # output (device=24) -- that would bypass this engine entirely and, later, keep
            # the clip out of the VAP reference channel. Push the raw WAV through our engine
            # instead. (An earlier version called bank.pick(), which does not exist; the
            # AttributeError was swallowed and the backchannel silently never fired.)
            try:
                import random
                w, sr = random.choice(self._bc_clips)
                self.engine.push(w, sr)
            except Exception as e:
                log(f"  [backchannel] failed: {e!r}")

        scene = ""
        if self.perception is not None:
            try: scene = self.perception("describe the room and the person") or ""
            except Exception: scene = ""

        # ── ENROLMENT ─────────────────────────────────────────────────────────────────
        # The machinery (enroll/query/save on JepaMemory, ensure/note_encounter on BmoMemory)
        # was complete and reboot-verified; the FLOW connecting them never existed -- the only
        # enroll() call sites in the tree were tests, one of which simulated "I'm Alice" and
        # left it persisted. This is that flow.
        known = None
        if self.ident is not None and getattr(self, "_last_emb", None) is not None:
            # 1. did they just tell us their name, having been asked?
            if self._awaiting_name:
                nm = extract_name(text)
                if nm:
                    try:
                        self.idmem.enroll(self._last_emb, nm)
                        self.idmem.save(self.args.identity_store)
                        self.bmem.ensure(nm, name=nm)
                        self.bmem.note_encounter(nm, summary=scene[:80], mood=mood.get("mood",""))
                        self.bmem.save(self.args.memory_store)
                        self._awaiting_name = False
                        known = nm
                        log(f"  [identity] ENROLLED '{nm}' ({len(self.idmem)} total, persisted)")
                    except Exception as e:
                        log(f"  [identity] enrol failed: {e!r}")
                else:
                    # they did not give a name. Ask ONCE, never nag -- pestering a stranger for
                    # a name is worse than not knowing it.
                    self._awaiting_name = False
            # 2. otherwise, who is this?
            if known is None:
                try:
                    label, score, why = self.idmem.query(self._last_emb)
                except Exception as e:
                    label, score, why = None, 0.0, f"error:{e!r}"
                if label:
                    known = label
                    try:
                        self.bmem.note_encounter(label, summary=scene[:80],
                                                 mood=mood.get("mood", ""))
                        self.bmem.save(self.args.memory_store)
                    except Exception:
                        pass
                    log(f"  [identity] recognised {label} ({score:.3f})")
                elif why == "empty_memory" or why == "below_threshold":
                    # a stranger. Ask for a name ONCE this session, and only ever store what
                    # they actually say -- never a guess.
                    if not self._asked_name_once:
                        self._awaiting_name = True
                        self._asked_name_once = True
                        log(f"  [identity] stranger ({why}) -> asking for a name")
                    else:
                        log(f"  [identity] stranger ({why}), already asked once")
                else:
                    log(f"  [identity] {why} ({score:.3f}) -- staying quiet")

        mem_line = ""
        if known and self.bmem is not None:
            try:
                mem_line = self.bmem.to_prompt_line(known, char_budget=160) or ""
            except Exception:
                mem_line = ""

        d = self._directive; self._directive = None
        # OFF BY DEFAULT. The deployed speaker (v5) has ZERO instruction-conditioned rows in
        # its corpus and the deployed thinker does not emit directives -- so this path is not
        # ready and enabling it blind is how the non-sequiturs come back. --use-directive to
        # experiment (pair it with speaker v6, which has the 372-row directive slice).
        # Directive prompt format MUST match the speaker's training data
        # (bmo_companion_corpus_v12.jsonl, speaker_directive slice):
        #   "You can see: wearing: ..; doing: ..; who: ..; where: ..; lighting: ..; hearing: ..
        #    . Your private thinking: <directive>"
        # An earlier version used "[instruction] .." -- a format the speaker has never seen.
        if d and self.args.use_directive:
            seen = scene if scene else "wearing: unknown; doing: unknown; who: one person"
            prompt = f"You can see: {seen}. Your private thinking: {d}\n{text}"
        elif self._awaiting_name:
            # trained shape, and the ONLY sanctioned way to get a name: ask for it
            prompt = (f"You can see: {scene}. Your private thinking: ask what their name is, "
                      f"because you have never met them\n{text}")
        elif scene and (PERCEPTION_ASK.search(text) or mem_line):
            # The speaker has 126 `perception_grounded` rows trained as
            #   "You can see: <scene>. <user utterance>" -> line
            # so when they ask what BMO can see, hand it the scene in exactly that shape. No
            # thinker tool-call needed -- this is a trained capability that simply was never
            # being fed. Also used whenever we have something remembered about this person.
            head = f"You can see: {scene}. " if scene else ""
            who = f"You remember: {mem_line} " if mem_line else ""
            prompt = f"{head}{who}{text}"
        else:
            prompt = text
        t0 = time.perf_counter()
        reply = ""
        for attempt in range(2):     # one cheap retry if a placeholder leaks (~150 ms)
            try:
                res = self.speaker.generate(prompt, mood)
                reply = (res.text or "").strip()
            except Exception as e:
                log(f"  [speaker] failed: {e!r}")
                reply = "Hmm, my circuits glitched. Say that again?"
                break
            if not _PLACEHOLDER.search(reply):
                break
            log(f"  [speaker] placeholder leak, retrying: {reply[:60]!r}")
        if _PLACEHOLDER.search(reply):
            reply = strip_placeholder(reply)
        llm_ms = (time.perf_counter() - t0) * 1000
        log(f"  BMO [{mood.get('mood')}]: {reply}   (llm {llm_ms:.0f}ms)")

        self.think_async(text, mood, scene)
        ttfa = self.speak(reply, mood)
        log(f"  [turn] stt {stt_ms:.0f} + llm {llm_ms:.0f} + ttfa {ttfa or -1:.0f} "
            f"= first audio ~{stt_ms + llm_ms + (ttfa or 0):.0f}ms | total "
            f"{(time.perf_counter()-t_all)*1000:.0f}ms | avail {mem_avail()} MiB")

    # ---- device selection + run loop ----
    def _devices(self):
        import sounddevice as sd
        try:
            subprocess.run(["pactl", "set-default-sink",
                            "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo"],
                           check=False, stderr=subprocess.DEVNULL)
            subprocess.run(["pactl", "set-default-source",
                            "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input"],
                           check=False, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        devs = sd.query_devices()
        in_idx = out_idx = pulse = None
        for i, d in enumerate(devs):
            n = d["name"].lower()
            if "pulse" in n: pulse = i
            if any(k in n for k in ("respeaker", "seeed", "mic array", "arrayuac", "usb audio")):
                if d["max_input_channels"] > 0 and in_idx is None:
                    in_idx = i
        if in_idx is None:
            in_idx = pulse
        out_idx = pulse if pulse is not None else 0
        return in_idx, out_idx

    def run(self):
        import sounddevice as sd
        in_dev, out_dev = self._devices()
        self.engine = UnifiedAudioEngine(out_dev)
        if in_dev is None:
            log("[mic] NO INPUT DEVICE FOUND -- refusing to start (would capture silence).")
            log("      plug in the ReSpeaker, or run with --selftest to check the stack.")
            return 2
        log(f"[mic] input={in_dev} ({sd.query_devices(in_dev)['name']})")
        log(f"[out] output={out_dev}")
        log("=" * 70)
        log(f"  BMO SHOWCASE READY   thinker={'on' if self.thinker else 'off'} "
            f"emotion={self.has_emotion and not self.args.no_emotion}")
        log(f"  avail={mem_avail()} MiB   Ctrl-C to quit")
        log("=" * 70)

        from scipy.signal import butter, sosfilt
        # NOTE: the fan is TONAL (38.6% of its energy at 808-812 Hz, the blade-pass
        # frequency). A 150 Hz highpass does nothing for it -- models/m5_fan_notch.py is
        # the right tool and is written but unwired. Highpass kept only for rumble/DC.
        sos = butter(4, 90.0, btype="highpass", fs=SR_MIC, output="sos")

        def mic_cb(indata, frames, t, status):
            if self.robot_speaking.is_set():
                return                          # no AEC yet -> hard gate (see docs S4)
            try:
                x = indata[:, 0].astype(np.float32) * MIC_GAIN     # ONE gain stage
                np.clip(x, -1.0, 1.0, out=x)
                self.vad.accept_waveform(sosfilt(sos, x).astype(np.float32))
            except Exception:
                pass

        try:
            with sd.InputStream(device=in_dev, channels=1, samplerate=SR_MIC,
                                blocksize=int(0.1 * SR_MIC), dtype="float32", callback=mic_cb):
                idle = time.time()
                while not self.stop_event.is_set():
                    while not self.vad.empty():
                        seg = np.asarray(self.vad.front.samples, dtype=np.float32)
                        self.vad.pop()
                        seg = np.clip(np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0), -1, 1)
                        if len(seg) < int(0.2 * SR_MIC) or \
                           float(np.sqrt(np.mean(seg ** 2) + 1e-9)) < 0.006:
                            continue
                        try:
                            self.process_turn(seg)
                        except Exception:
                            log("[turn] UNCAUGHT -- loop continues:\n" + traceback.format_exc())
                        idle = time.time()
                    if time.time() - idle > 30:
                        self._update_state(user_spoke=False, arousal=0.0); idle = time.time()
                    time.sleep(0.015)
        except KeyboardInterrupt:
            log("\nshutting down")
        finally:
            self.stop_event.set()
            if self.engine: self.engine.close()
        return 0

    # ---- headless self-test: proves the whole chain without a mic ----
    def selftest(self):
        self.engine = UnifiedAudioEngine(None)
        log("=" * 70); log("  SELFTEST (no mic, no speaker)"); log("=" * 70)
        ok = True
        turns = ["Hey BMO, what are you up to?",
                 "I've been staring at this bug for three hours.",
                 "Do you want to play a game?"]
        for i, u in enumerate(turns):
            mood = self._mood()
            before = dict(self.state.as_dict()) if hasattr(self.state, "as_dict") else {}
            self._update_state(user_spoke=True, arousal=0.3 + 0.2 * i)
            after = dict(self.state.as_dict()) if hasattr(self.state, "as_dict") else {}
            changed = before != after
            m2 = self._mood()
            t0 = time.perf_counter()
            res = self.speaker.generate(u, m2)
            llm = (time.perf_counter() - t0) * 1000
            reply = (res.text or "").strip()
            if _PLACEHOLDER.search(reply):
                log(f"     !! placeholder leak: {reply[:70]!r}")
                reply = strip_placeholder(reply)

            vx = StreamingVisemeExtractor(SR_OUT); segs = []; audio = []
            t0 = time.perf_counter(); ttfa = None
            emo = m2.get("mood") if (self.has_emotion and not self.args.no_emotion) else None
            if emo in self.args.unsafe_moods: emo = "neutral"
            for s in self.tts.stream(clean_for_tts(reply), emo):
                if s is None or len(s) == 0: continue
                if ttfa is None: ttfa = (time.perf_counter() - t0) * 1000
                segs.extend(vx.push(s)); audio.append(s)
            segs.extend(vx.flush())
            synth = (time.perf_counter() - t0) * 1000
            dur = sum(len(a) for a in audio) / SR_OUT
            gaps = sum(1 for a, b in zip(segs, segs[1:]) if b[1] - a[2] > 0.05)
            cov = sum(b - a for _, a, b, _ in segs) / max(dur, 1e-6)
            log(f"[t{i}] state_changed={changed} mood={m2.get('mood')} llm={llm:.0f}ms "
                f"ttfa={ttfa or -1:.0f}ms synth={synth:.0f}ms audio={dur:.2f}s "
                f"RTF={synth/1000/max(dur,1e-6):.2f} visemes={len(segs)} cov={cov:.2f} gaps={gaps} "
                f"exit={getattr(self.tts,'last_exit_reason','?')}")
            log(f"     BMO: {reply[:110]!r}")
            if ttfa is None or ttfa > 900: log(f"     !! TTFA too high"); ok = False
            if len(segs) < 3: log(f"     !! too few visemes"); ok = False
            if not changed: log(f"     !! homeostatic state did NOT change (S1 regression)"); ok = False
            exit_r = getattr(self.tts, "last_exit_reason", "?")
            if exit_r != "EOS_TOKEN":
                log(f"     !! TTS did not emit EOS (exit={exit_r}) -- voice EOS calibration")
                ok = False
            # this voice speaks ~8.3 chars/sec (measured); >1.3x expected means a stretch
            exp = max(len(clean_for_tts(reply)) / 8.3, 0.5)
            if dur / exp > 1.3:
                log(f"     !! utterance stretched {dur/exp:.2f}x expected ({dur:.2f}s)")
                ok = False
        if self.thinker is not None:
            t0 = time.perf_counter()
            try:
                r = self.thinker.generate(
                    "The person said: 'I'm tired.'\nThink, then give ONE short instruction "
                    "for what Beemo should say. Do not write the line yourself.", self._mood())
                log(f"[thinker] {(time.perf_counter()-t0)*1000:.0f}ms CoT="
                    f"{'yes' if r.reasoning else 'NO'} -> {(r.text or '')[:110]!r}")
                if not r.reasoning: log("     !! thinker emitted no <think>"); ok = False
            except Exception as e:
                log(f"[thinker] FAILED {e!r}"); ok = False
        # The face engine polls /dev/shm/bmo_speech.txt every 15 ms with
        # fscanf("%d %63s %f %63s"). Nothing above touches that path, so exercise it here --
        # otherwise a broken write would only ever show up as a still mouth on stage.
        try:
            sch = VisemeScheduler(self.engine, face_emotion=self.args.face_emotion)
            sch._write("AEI", 1.0)
            raw = open(VISEME_PATH).read().strip()
            parts = raw.split()
            good = len(parts) >= 3 and parts[0] == "1" and float(parts[2]) > 0
            log(f"[viseme] wrote {VISEME_PATH}: {raw!r} parse_ok={good}")
            if not good: log("     !! viseme file format wrong"); ok = False
            sch.stop()
            raw2 = open(VISEME_PATH).read().strip()
            log(f"[viseme] after stop: {raw2!r} (must start with 0)")
            if not raw2.startswith("0"): log("     !! mouth not closed on stop"); ok = False
        except Exception as e:
            log(f"[viseme] WRITE PATH FAILED: {e!r}"); ok = False

        # PERCEPTION VERIFICATION. Without this the selftest reported PASS while the camera
        # was dead (it never pulls a frame), and again while every category returned the SAME
        # tag. Assert on CONTENT, not on "did it return something".
        if self.perception is not None:
            try:
                t0 = time.perf_counter()
                scene = self.perception("describe the room and the person") or ""
                dt = (time.perf_counter() - t0) * 1000
                log(f"[percep] {dt:.0f}ms -> {scene[:150]!r}")
                fields = [f.split(":", 1) for f in scene.split(";") if ":" in f]
                vals = [v.strip() for _, v in fields]
                uniq = len(set(vals))
                log(f"[percep] {len(fields)} categories, {uniq} distinct answers")
                if len(fields) < 3:
                    log("     !! too few perception categories"); ok = False
                # all-identical answers means bank_category was not set and every question
                # collapsed to one top-1 over the whole bank
                if uniq <= 1:
                    log("     !! every category returned the SAME tag "
                        "(bank_category unset?)"); ok = False
                m = "/dev/shm/bmo_motion.txt"
                if os.path.exists(m):
                    age = time.time() - os.path.getmtime(m)
                    log(f"[motion] age={age:.3f}s content={open(m).read().strip()!r}")
                    if age > 2.0:
                        log("     !! motion file stale -- camera hub is dead"); ok = False
                else:
                    log("     !! no motion file -- camera hub not running"); ok = False
            except Exception as e:
                log(f"[percep] VERIFICATION FAILED: {e!r}"); ok = False

        log(f"\nSELFTEST {'PASS' if ok else 'FAIL'}   avail={mem_avail()} MiB")
        log("SELFTEST_DONE")
        return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    # v6, not v5. MEASURED 2026-08-23, n=20 prompts: v5 leaks a literal placeholder
    # ("I'm [name], and you are...?", "Playing a happy tune for you, {name}.") on 2/20 --
    # 10% of turns show a template artifact on stage. v6 leaks 0/20 and is faster
    # (147 vs 173 ms). v6 was shelved on an 8/12-vs-9/12 bake-off that the ledger itself
    # records as having false passes and being underpowered at n=6; a visible placeholder
    # is the worse failure for a live demo. v6 also carries the 372-row directive slice.
    p.add_argument("--speaker", default="bmo_lfm25_350m_v6_Q8_0.gguf")
    p.add_argument("--speaker-tokens", type=int, default=48)
    p.add_argument("--thinker", default="bmo_thinker_qwen3_v5_Q4_K_M.gguf",
                   help="Q4_K_M fits WITH the camera; Q8_0 does not. 'none' to disable.")
    p.add_argument("--thinker-tokens", type=int, default=200)
    p.add_argument("--voice", default=None)
    p.add_argument("--no-emotion", action="store_true")
    p.add_argument("--face-emotion", default="")
    p.add_argument("--backchannel", action="store_true", default=True)
    p.add_argument("--vad-threshold", type=float, default=0.45)
    p.add_argument("--min-silence", type=float, default=0.55)
    p.add_argument("--identity", action="store_true",
                   help="enable the identity head + enrolment flow (needs --perception)")
    p.add_argument("--identity-store", default="/home/bmo/bmo_identity.pt")
    p.add_argument("--memory-store", default="/home/bmo/bmo_memory.json")
    p.add_argument("--perception", action="store_true",
                   help="load the JEPA perception stack + camera (AFTER the GGUFs)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--use-directive", action="store_true",
                   help="inject the thinker directive into the speaker prompt "
                        "(needs a directive-trained speaker, e.g. v6)")
    # 4/12 moods ran to the length cap on-device (thin training data); keep them off stage.
    p.add_argument("--unsafe-moods", nargs="*",
                   default=["surprised", "anxious", "lonely", "bored"])
    a = p.parse_args()
    try:
        bmo = BmoShowcase(a)
    except Exception:
        log("BOOT FAILED:\n" + traceback.format_exc()); return 3
    return bmo.selftest() if a.selftest else bmo.run()


if __name__ == "__main__":
    sys.exit(main())
