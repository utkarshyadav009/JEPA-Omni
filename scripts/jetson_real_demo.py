"""scripts/jetson_real_demo.py — the REAL end-to-end test. No fixtures, no fake names.

WHAT WAS WRONG WITH THE PREVIOUS TEST, and why its output was misleading:
  * It called `memory.enroll(emb, "Alice")` to SIMULATE someone answering "I'm Alice", then
    printed `[IDENTITY] Alice` -- which reads as recognition and is not. Nobody was
    recognised. There is no Alice.
  * It asked perception exactly one question ("describe the room in detail") and printed the
    single top-1 tag out of 1,482. One tag cannot say what someone is wearing AND doing.
  * It hardcoded "The user just walked in" into the thinker prompt every tick, so the thinker
    dutifully said "I see you just walked in" no matter what perception reported.

THIS SCRIPT:
  * loads EVERY component including TTS and STT, to prove the whole stack fits;
  * asks perception SEPARATE questions per category -- what they are WEARING, what they are
    DOING, WHERE this is, WHO is present -- using the category labels already in
    candidates_siglip2.pt, so clothing is retrieved from clothing tags rather than competing
    with 1,372 other phrases;
  * never invents an identity: unknown stays unknown;
  * prints the thinker's instruction and the speaker's line adjacently so it is obvious
    whether the speaker followed the thinker;
  * tees everything to a log file for live tailing.

Usage on the Jetson:
    python3 scripts/jetson_real_demo.py --rounds 6 --log ~/bmo_demo.log
    # watch live from anywhere:  tail -f ~/bmo_demo.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

LOG = None


def say(msg: str = "") -> None:
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n"); LOG.flush()


def avail() -> float:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    return float(out.splitlines()[1].split()[-1])


class MicThread:
    """10s rolling 16kHz mono ring buffer off the ReSpeaker 4-mic array.

    ADDED 2026-08-16. The previous version of this script passed
    `torch.zeros(16000 * 10)` to `build_world_state_features` -- so WavJEPA-base
    (645 MiB resident) and M2's entire audio branch ran on SILENCE every round,
    and the query predictor's `ambient` source contributed a constant vector to
    every answer it produced. Half the trained pipeline was loaded and fed nothing.
    The hardware was there the whole time: card 0, `ReSpeaker 4 Mic Array (UAC1.0)`,
    6 input channels. Mixed to mono here exactly as `_decode_audio_raw` does, since
    that is what WavJEPA-base was trained on."""

    def __init__(self, window_sec: float = 10.0, sr: int = 16000, device_hint: str = "ReSpeaker",
                 denoise: bool = False, snr_gate: float = 2.0):
        self.sr, self.window_sec = sr, window_sec
        self.buf = np.zeros(int(sr * window_sec), dtype=np.float32)
        self.n_frames = 0
        self._lock = threading.Lock()
        self._stream = None
        self.device_hint = device_hint
        self.rms = 0.0
        # DENOISE DEFAULTS OFF -- measured, not assumed. Spectral subtraction against the
        # fan profile made the audio percept WORSE on the real device (2026-08-16): the
        # `hearing` answer "an alarm beeping" rose 0.451 -> 0.478 and "glass breaking"
        # appeared as runner-up, which are exactly how a listener describes musical-noise
        # artefacts. The fan is stationary AND tonal, so subtraction leaves residual
        # narrowband peaks that WavJEPA faithfully reports as beeping. The un-processed
        # signal at least yields the truthful "a fan humming". Keep the code -- it is the
        # right tool once the mic is not 10cm from the intake -- but do not default to it.
        self.denoise = denoise
        self.snr_gate = snr_gate    # multiple of the calibrated floor rms below which the
                                    # audio percept is declared uninformative
        self.floor_rms = 0.0
        self.noise_mag = None       # measured fan profile, set by calibrate()
        self.use_ch = None          # channel subset, chosen by calibrate()
        # TACH-DRIVEN NOTCH (2026-08-16). The fan was proved TONAL, not broadband, by
        # differencing two commanded fan speeds -- 38.6% of its energy sits in one peak at
        # 808-812 Hz, and 9 blades predicts BPF = rpm/60*9 = 810.4 Hz to within 1.9 Hz. So the
        # frequency to remove does not have to be estimated from the audio at all: the
        # tachometer says what the fan is about to sound like. 33.4 dB on the tone, 0.0 dB on
        # speech, 1.6 ms per 10 s buffer.
        self.notch = None
        self._rpm = None
        try:
            from models.m5_fan_notch import FanNotch, FanNotchConfig
            self.notch = FanNotch(FanNotchConfig(sr=sr))
        except Exception as e:
            say(f"  [mic] fan notch unavailable ({type(e).__name__})")

    def _fan_rpm(self):
        try:
            from models.m5_tools import power_status
            st = power_status() or {}
            return (st.get("fan") or {}).get("rpm")
        except Exception:
            return None

    def above_floor(self) -> bool:
        """Is there anything in the room louder than the Jetson's own fans?

        Without this, a constant fan produces a confident wrong percept every round and the
        thinker REASONS ABOUT IT -- all 8 rounds of the 2026-08-16 run opened with "I hear a
        faint alarm beeping", and the speaker echoed it into every line. A wrong perception
        does not stay in perception; it propagates through the whole stack."""
        return self.floor_rms > 0 and self.rms > self.floor_rms * self.snr_gate

    def start(self) -> bool:
        try:
            import sounddevice as sd
        except ImportError:
            say("  [mic] sounddevice not installed -- audio will be SILENT")
            return False
        idx, ch = None, 1
        for i, d in enumerate(sd.query_devices()):
            if self.device_hint.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                idx, ch = i, min(d["max_input_channels"], 6)
                break
        if idx is None:
            say(f"  [mic] no input device matching '{self.device_hint}' -- audio will be SILENT")
            return False

        def cb(indata, frames, t, status):
            sel = indata if self.use_ch is None else indata[:, self.use_ch]
            mono = sel.mean(axis=1).astype(np.float32)   # mix down, as _decode_audio_raw does
            with self._lock:
                n = len(mono)
                if n >= len(self.buf):
                    self.buf[:] = mono[-len(self.buf):]
                else:
                    self.buf[:-n] = self.buf[n:]
                    self.buf[-n:] = mono
                self.n_frames += n
                self.rms = float(np.sqrt(np.mean(mono ** 2)))

        # The array runs at 16kHz natively (UAC1.0), which is also WavJEPA's rate --
        # no resample needed, so nothing here can silently change the sample rate
        # out from under the encoder.
        say(f"  [mic] {sd.query_devices()[idx]['name']} — {ch}ch @ {self.sr}Hz, {self.window_sec:.0f}s window")
        # Calibrate BEFORE opening the persistent stream: this is an ALSA hw: device and a
        # second concurrent capture on it is not guaranteed to open.
        self._calibrate(sd, idx, ch)
        self._stream = sd.InputStream(device=idx, channels=ch, samplerate=self.sr,
                                      dtype="float32", blocksize=1600, callback=cb)
        self._stream.start()
        return True

    def _calibrate(self, sd, idx: int, ch: int, sec: float = 3.0) -> None:
        """Measure the room floor -- which on this build is mostly the Jetson's own fans,
        because the array sits ~10cm from the intake.

        Two things come out of one 3s recording:
          * WHICH CHANNELS TO USE. Per-channel probe 2026-08-16: ch5 is dead silence (the
            playback/loopback channel) and averaging it in only attenuates; ch1 and ch4 run
            ~4.5dB quieter than ch0/2/3, presumably facing away from the intake. This is a
            plain 4-mic array -- a hypothesis that ch0 was an XMOS beamformed/denoised
            output was TESTED AND FALSIFIED (ch0 was the LOUDEST, not the cleanest), so
            there is no on-board DSP to fall back on and the cleanup has to happen here.
          * A NOISE MAGNITUDE PROFILE for spectral subtraction. The fan is broadband, not
            rumble (measured lo<300Hz / 300-4k energy ratio ~0.15), so a high-pass filter
            does not touch it and full-band profile subtraction is the applicable tool.

        Requires the room to be quiet for these 3 seconds -- if someone talks, their voice
        gets learned as noise. Stated in the log so a bad calibration is visible."""
        drop = []
        try:
            say(f"  [mic] calibrating noise floor — stay quiet for {sec:.0f}s...")
            x = sd.rec(int(self.sr * sec), samplerate=self.sr, channels=ch,
                       dtype="float32", device=idx)
            sd.wait()
        except Exception as e:
            say(f"  [mic] calibration failed ({type(e).__name__}), using all channels raw")
            return

        rms = np.sqrt((x ** 2).mean(axis=0))
        live = [c for c in range(ch) if rms[c] > 1e-6]           # drop dead/loopback channels
        if live:
            thresh = np.median(rms[live]) * 1.02                  # keep the quieter half
            keep = [c for c in live if rms[c] <= thresh] or live
            self.use_ch = keep
            drop = [c for c in range(ch) if c not in keep]
        say(f"  [mic] per-channel rms {np.array2string(rms, precision=5)} "
            f"→ using {self.use_ch}, dropping {drop}")

        if self.use_ch:
            mono = x[:, self.use_ch].mean(axis=1).astype(np.float32)
            self.floor_rms = float(np.sqrt((mono ** 2).mean()))
            if self.denoise:
                self.noise_mag = self._avg_mag(mono)
            say(f"  [mic] floor rms={self.floor_rms:.5f}  gate={self.floor_rms*self.snr_gate:.5f}"
                f"  denoise={'on' if self.denoise else 'off'}")

    _NFFT, _HOP = 512, 128

    def _avg_mag(self, x: np.ndarray) -> np.ndarray:
        w = np.hanning(self._NFFT).astype(np.float32)
        n = 1 + max(0, (len(x) - self._NFFT)) // self._HOP
        acc = np.zeros(self._NFFT // 2 + 1, dtype=np.float32)
        for i in range(n):
            seg = x[i * self._HOP: i * self._HOP + self._NFFT]
            acc += np.abs(np.fft.rfft(seg * w))
        return acc / max(n, 1)

    def _subtract(self, x: np.ndarray, over: float = 1.5, floor: float = 0.05) -> np.ndarray:
        """Textbook magnitude spectral subtraction with an over-subtraction factor and a
        spectral floor (the floor is what keeps it from producing musical-noise artefacts
        that WavJEPA would then dutifully describe)."""
        w = np.hanning(self._NFFT).astype(np.float32)
        out = np.zeros(len(x) + self._NFFT, dtype=np.float32)
        wsum = np.zeros_like(out)
        n = 1 + max(0, (len(x) - self._NFFT)) // self._HOP
        for i in range(n):
            s = i * self._HOP
            seg = x[s: s + self._NFFT]
            if len(seg) < self._NFFT:
                break
            S = np.fft.rfft(seg * w)
            mag = np.abs(S)
            clean = np.maximum(mag - over * self.noise_mag, floor * mag)
            rec = np.fft.irfft(clean * np.exp(1j * np.angle(S)), n=self._NFFT)
            out[s: s + self._NFFT] += rec * w
            wsum[s: s + self._NFFT] += w ** 2
        return (out[:len(x)] / np.maximum(wsum[:len(x)], 1e-8)).astype(np.float32)

    def get(self) -> torch.Tensor:
        with self._lock:
            raw = self.buf.copy()
        if self.notch is not None:
            self._rpm = self._fan_rpm()
            if self._rpm:
                raw = self.notch(raw, self._rpm)
        if self.denoise and self.noise_mag is not None:
            return torch.from_numpy(self._subtract(raw))
        return torch.from_numpy(raw)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop(); self._stream.close()


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--secs-between", type=float, default=3.0)
    ap.add_argument("--fast-gguf", default="bmo_lfm25_350m_v5_Q8_0.gguf")
    ap.add_argument("--thinker-gguf", default="bmo_thinker_qwen3_v5_Q8_0.gguf")
    ap.add_argument("--query-ckpt", default="qp_runD.pt")
    # _v2 (2026-08-16): the deployed candidates_siglip2.pt predates the APPEARANCE tags,
    # so it has 0 `appearance` entries and the "what are they wearing" question had nothing
    # to retrieve from. v2 = same 1,482 tags but with the 110 colour x garment appearance
    # phrases in their own category.
    ap.add_argument("--candidates", default="candidates_siglip2_v2.pt")
    ap.add_argument("--query-vectors", default="query_vectors_siglip2_v2.pt")
    ap.add_argument("--identity-ckpt", default="identity_head_joint.pt")
    ap.add_argument("--tts-gguf", default="bmo_neutts_nano_v1_Q8_0.gguf",
                    help="NANO (241 MB), not Air (765 MB) -- the memory budget was computed "
                         "for Nano and Air does not fit alongside both v5 LLMs")
    ap.add_argument("--with-tts", action="store_true", default=False,
                    help="OFF by default. TTS+STT+everything loads (proven: all 7 resident, 280 MiB free) but leaves too little headroom for the CAMERA to get its NVMM buffers -- at 73 MiB free it opened and returned flat frames.")
    ap.add_argument("--with-stt", action="store_true", default=False)
    # 180, not 90: the CSI module is mounted upside down in the chassis. At 90 the stack
    # confidently reported `who: a person lying down (+0.71)` for a seated person -- the
    # highest-confidence answer in the whole set, and wrong.
    ap.add_argument("--rotate", type=int, default=180)
    # A/B the tach-driven fan notch. It measures 33.4 dB on the 810 Hz blade-pass tone and
    # 0.0 dB on speech, but its effect on the `hearing` PERCEPT is unverified -- the only test
    # so far ran in a room sitting at the noise floor (rms 0.0018 vs 0.0016), where there was
    # nothing to hear and WavJEPA returned confident arbitrary answers either way. Run this
    # pair WITH SOUND IN THE ROOM (talk, play music) at the same fan speed:
    #     python3 scripts/jetson_real_demo.py --rounds 4 --no-notch --log ~/hear_off.log
    #     python3 scripts/jetson_real_demo.py --rounds 4            --log ~/hear_on.log
    # and compare the `hearing` top-2. Fan RPM is printed each round so a speed change between
    # the two runs is visible rather than silently confounding the comparison.
    ap.add_argument("--no-notch", action="store_true",
                    help="disable the fan blade-pass notch (A/B control arm)")
    ap.add_argument("--gain", type=float, default=1.8)
    ap.add_argument("--log", default=os.path.expanduser("~/bmo_demo.log"))
    args = ap.parse_args()

    LOG = open(args.log, "w")
    t_start = time.time()
    base = avail()
    say("=" * 78)
    say("BMO REAL DEMO — every component, no fixtures, no invented identities")
    say("=" * 78)

    def mem(tag):
        a = avail()
        say(f"  [load] {tag:36s} {base - a:7.1f} MiB used   {a:7.1f} free")
        return a

    # ---- LLMs first: load order is load-bearing on this device ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, GGUFReasoningTier
    fast = GGUFFastTier(f"{PROD}/models_gguf/{args.fast_gguf}",
                        AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/lfm25_350m_tok"),
                        max_new_tokens=48, n_gpu_layers=-1)
    mem(f"speaker ({args.fast_gguf.split('_Q8')[0]})")
    thinker = GGUFReasoningTier(f"{PROD}/models_gguf/{args.thinker_gguf}",
                                AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/qwen3_thinker_tok"),
                                max_new_tokens=320, n_gpu_layers=-1)
    mem(f"thinker ({args.thinker_gguf.split('_Q8')[0]})")

    # TTS IS A GGUF, SO IT LOADS HERE -- WITH THE OTHER GGUFs, BEFORE THE TORCH STACK.
    # The first version of this script loaded it LAST and it failed after 5 retries with
    # ~697 MiB free. Two causes, both mine:
    #   1. it loaded NeuTTS-**Air** (765 MB) when the whole memory budget was computed for
    #      NeuTTS **Nano** (241 MB) -- a 524 MB error;
    #   2. it loaded AFTER V-JEPA2/WavJEPA/M2/SigLIP2, which CLAUDE.md documents as the
    #      failure mode: llama.cpp GGUFs must be allocated before the torch/int8 perception
    #      stack fragments Jetson's unified memory. "Failed after 5 tries" with
    #      retry+compaction is exactly what that looks like.
    tts = None
    if args.with_tts:
        try:
            from models.m5_streaming_voice import StreamingVoice
            tts = StreamingVoice(f"{PROD}/models_gguf/{args.tts_gguf}")
            mem(f"TTS ({args.tts_gguf.split('_Q8')[0]})")
        except Exception as e:
            say(f"  [load] TTS FAILED: {type(e).__name__}: {str(e)[:90]}")

    device = torch.device("cuda")
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from scripts.jetson_core_pipeline_test import q_int8, open_camera, CaptureThread, _prep

    vision = VisionEncoder(device="cpu"); vision.model = q_int8(vision.model, device)
    vision.device_str = str(device); mem("V-JEPA2 ViT-L int8")   # device_str, NOT device
    wb = AudioEncoder("labhamlet/wavjepa-base", n_channels=1, device="cpu")
    wb.model = q_int8(wb.model, device); wb.device_str = str(device); mem("WavJEPA base")

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(f"{PROD}/pipeline/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
                                  map_location=device, weights_only=False)["model"], strict=True)
    m2.eval(); mem("M2 predictor")

    from models.jepa_identity_head import IdentityHead, IdentityHeadConfig, stats_pool
    from models.jepa_memory import JepaMemory, MemoryConfig
    ick = torch.load(f"{PROD}/pipeline/checkpoints/{args.identity_ckpt}", map_location="cpu", weights_only=False)
    ident = IdentityHead(IdentityHeadConfig(in_dims=ick["dims"], emb_dim=ick["emb_dim"])).to(device)
    ident.load_state_dict(ick["head"]); ident.eval()
    idmem = JepaMemory(MemoryConfig(threshold=0.5))
    mem("identity head")

    from transformers import AutoModel
    sig = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16)
    if hasattr(sig, "text_model"):
        del sig.text_model
    sig = sig.to(device).eval(); mem("SigLIP2 vision tower")

    from models.m5_motion_crop import siglip2_preprocess
    from models.query_predictor import QueryPredictor, QueryPredictorConfig
    ck = torch.load(f"{PROD}/pipeline/checkpoints/{args.query_ckpt}", map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]
    sd = ck["query_predictor"]
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768, "scene": 768}
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=int(sd["q_proj.weight"].shape[1]),
                                             shared_dim=int(sd["head.weight"].shape[0]))).to(device)
    qp.load_state_dict(sd); qp.eval()

    cand = torch.load(f"{PROD}/pipeline/checkpoints/{args.candidates}", map_location="cpu", weights_only=False)
    W = ck["text_target_proj"]["weight"].float().to(device)
    B = ck["text_target_proj"]["bias"].float().to(device)
    C_raw = F.normalize(cand["emb"].float(), dim=-1).to(device)
    C_prj = F.normalize(C_raw @ W.t() + B, dim=-1)          # into the predictor's learned space
    C_txt, C_cat = cand["text"], cand.get("category", ["mined"] * len(cand["text"]))
    qv = torch.load(f"{PROD}/pipeline/checkpoints/{args.query_vectors}", map_location="cpu", weights_only=False)
    QV = {t: F.normalize(v.float(), dim=-1).to(device) for t, v in zip(qv["text"], qv["emb"])}
    mem("query predictor + candidates")

    stt = None
    if args.with_stt:
        try:
            from models.m4_speech import MoonshineSpeechEncoder
            stt = MoonshineSpeechEncoder()          # takes no device kwarg
            mem("STT (Moonshine)")
        except Exception as e:
            say(f"  [load] STT FAILED: {type(e).__name__}: {str(e)[:90]}")

    mic = MicThread(window_sec=10.0, sr=16000)
    if args.no_notch:
        mic.notch = None
        say("  [mic] fan notch DISABLED (--no-notch control arm)")
    mic_ok = mic.start()
    mem("mic ring buffer")

    cap = open_camera()
    capture = CaptureThread(cap, window_sec=10.0, fps=6.4, rotate=args.rotate, gain=args.gain)
    capture.start()
    t0 = time.time()
    while capture.n_read < 16 and time.time() - t0 < 12:
        time.sleep(0.1)
    after = mem("camera + ring buffer")
    say("")
    say(f"  ALL COMPONENTS LOADED — {base - after:.0f} MiB used, {after:.0f} MiB free")
    say(f"  TTS: {'loaded' if tts else 'NOT loaded'}   STT: {'loaded' if stt else 'NOT loaded'}"
        f"   MIC: {'live' if mic_ok else 'SILENT (audio branch is meaningless)'}")
    say("=" * 78)

    from models.world_state_builder import build_world_state_features

    class Ad:
        def compute_pre_pool(self, feats, tbins):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return m2.encode_pre_pool_tokens({k: v.float() for k, v in feats.items()}, tbins)
    ad = Ad()

    # ASK PER CATEGORY. One top-1 over 1,482 tags cannot answer "what are they wearing" AND
    # "what are they doing" -- the categories already exist in the candidates file, so
    # restrict each question to the tags that could possibly answer it.
    #
    # `hearing` ADDED 2026-08-16. `gpt_sound_acoustic` ("What do you hear?") is one of the
    # six fields the query predictor was actually TRAINED on, over VGGSound audio, and the
    # candidate file carries 34 `sound` tags -- the audio question was supported end to end
    # and simply never asked. Perception was visual-only for no reason.
    QUESTIONS = [
        ("wearing",  "appearance", "What is the person wearing?"),
        ("doing",    "action",     "What is happening?"),
        ("who",      "people",     "What is happening?"),
        ("where",    "place",      "Describe the room and setting in detail."),
        ("lighting", "light",      "Describe the room and setting in detail."),
        ("hearing",  "sound",      "What do you hear?"),
    ]
    CAT_IDX = {c: torch.tensor([i for i, x in enumerate(C_cat) if x == c], device=device)
               for _, c, _ in QUESTIONS}
    # FAIL LOUD on an empty category. The previous version did `if len(ids)==0: continue`,
    # so when the DEPLOYED candidates file turned out to predate the appearance tags the
    # `wearing` line just never printed and the run looked like it had only ever asked four
    # questions. A silently-skipped question is indistinguishable from a question that was
    # never written.
    for label, cat, _ in QUESTIONS:
        if len(CAT_IDX[cat]) == 0:
            raise SystemExit(
                f"candidates file '{args.candidates}' has NO '{cat}' tags, so the "
                f"'{label}' question cannot be answered. Present categories: "
                f"{sorted(set(C_cat))}. Rebuild with scripts/build_candidate_vocab.py.")

    def qvec(q):
        if q in QV:
            return QV[q]
        k = min(QV, key=lambda t: -len(set(t.lower().split()) & set(q.lower().split())))
        return QV[k]

    for r in range(args.rounds):
        say("")
        say(f"───────── round {r+1}/{args.rounds} ─────────")
        frames = capture.get_window(16)
        wav = mic.get()
        ws = build_world_state_features(frames, wav, 10.0, vision, wb, None, 512, device)
        srcs = {}
        if "vision" in names:  srcs["vision"] = ws.feats["vision"].float()
        if "ambient" in names: srcs["ambient"] = ws.feats["ambient"].float()
        if "m2" in names:      srcs["m2"] = ad.compute_pre_pool(ws.feats, ws.tbins).float()
        if "scene" in names:
            idx = torch.linspace(0, frames.shape[0] - 1, 4).long()
            from PIL import Image as _Im
            imgs = [_Im.fromarray(frames[i].permute(1, 2, 0).numpy()) for i in idx]
            with torch.no_grad():
                px = siglip2_preprocess(imgs).to(device).to(torch.bfloat16)
                o = sig.vision_model(pixel_values=px)
                o = o.pooler_output if hasattr(o, "pooler_output") else o
                srcs["scene"] = F.normalize(o.float(), dim=-1).unsqueeze(0)

        seen = {}
        with torch.no_grad():
            for label, cat, q in QUESTIONS:
                z = F.normalize(qp(srcs, qvec(q).unsqueeze(0), None).float(), dim=-1)
                ids = CAT_IDX[cat]
                if len(ids) == 0:
                    continue
                sims = (z @ C_prj[ids].T)[0]
                top = sims.topk(min(2, len(ids)))
                seen[label] = [(C_txt[ids[i].item()], float(v))
                               for v, i in zip(top.values, top.indices)]

        loud = mic.above_floor()
        nt = ("" if mic.notch is None or not mic._rpm else
              f" notch@{mic.notch.bpf(mic._rpm):.0f}Hz(fan {mic._rpm:.0f}rpm)")
        say(f"  PERCEPTION (asked per category, top-2 each)   "
            f"mic_rms={mic.rms:.4f} floor={mic.floor_rms:.4f} "
            f"{'ABOVE' if loud else 'at'} floor{nt}:")
        for label, hits in seen.items():
            tag = "" if (label != "hearing" or loud) else "   [below floor — not sent to thinker]"
            say(f"     {label:9s} " + "  |  ".join(f"{t} ({s:+.3f})" for t, s in hits) + tag)

        # identity: no fixtures. unknown means unknown.
        with torch.no_grad():
            emb = ident({"vision": stats_pool(ws.feats["vision"]),
                         "audio": stats_pool(ws.feats["ambient"])})
        who, conf, reason = idmem.query(emb[0])
        say(f"  IDENTITY   {who or 'UNKNOWN'} (conf={conf:.3f}, {reason}, {len(idmem)} enrolled)")

        # Drop `hearing` from what the thinker is told when the room is at the fan floor.
        # Reporting it in the log stays useful for debugging; feeding it to the thinker is
        # what turns a noise artefact into an invented alarm the speaker then announces.
        scene_line = "; ".join(f"{k}: {v[0][0]}" for k, v in seen.items()
                               if k != "hearing" or loud)
        if not loud:
            scene_line += "; hearing: nothing but the room's own noise"
        t = time.time()
        tr = thinker.generate(
            f"You can see: {scene_line}. You do not know this person's name. "
            f"Decide what to say to them right now, and why.",
            {"energy": 0.6, "mood": "curious"})
        think = getattr(tr, "text", str(tr))
        reason = getattr(tr, "reasoning", None)
        think_ms = (time.time() - t) * 1000
        say(f"  THINKER ({think_ms:.0f} ms)")
        # Print the CoT and the answer SEPARATELY. Printing only the answer hid the fact
        # that the thinker's answer is itself a finished BMO line -- i.e. that the thinker
        # was doing the speaker's job -- and made the speaker look like it was echoing.
        say(f"     reasoning: {(reason or '(model emitted no <think> block)')[:400]}")
        say(f"     answer:    {think[:200]}")

        # CONDITION THE SPEAKER ON THE REASONING, NOT THE ANSWER. Handing it the answer
        # gave it a complete utterance and asked it to produce an utterance, so the only
        # available move was paraphrase. The reasoning is the instruction; the line is the
        # speaker's to write.
        t = time.time()
        instr = reason or think
        fr = fast.generate(f"You can see: {scene_line}. Your private thinking: {instr[:260]} "
                           f"Now say one short line out loud to them.",
                           {"energy": 0.6, "mood": "curious"})
        said = getattr(fr, "text", str(fr))
        say(f"  SPEAKER ({(time.time()-t)*1000:.0f} ms)")
        say(f"     \"{said}\"")
        say(f"  free: {avail():.0f} MiB")
        time.sleep(args.secs_between)

    capture.stop(); cap.release(); mic.stop()
    say("")
    say("=" * 78)
    say(f"DONE — {time.time()-t_start:.0f}s total, {avail():.0f} MiB free at exit")
    LOG.close()


if __name__ == "__main__":
    main()
