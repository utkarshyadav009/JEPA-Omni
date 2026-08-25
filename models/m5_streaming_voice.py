"""models/m5_streaming_voice.py -- production streaming BMO voice.

The real deployed voice path: NeuTTS GGUF backbone (llama.cpp) -> sampled
speech-token generation -> seam-free chunked NeuCodec decode -> audio, yielded
incrementally so first-audio lands in ~0.3-0.5s while the rest streams behind
playback. Self-contained (inlines phonemizer + overlap-add), no neutts package
dependency.

Real lessons baked in (all learned the hard way this session, 2026-08-07):
  - MUST sample (temp~0.7, top_k=50). Greedy (temp=0) makes neural-codec TTS
    loop forever and never emit <|SPEECH_GENERATION_END|>.
  - Speech tokens come out of the LOW-LEVEL Llama.generate() iterator; the
    high-level completion API silently suppresses the <|speech_N|> special
    tokens (they're contiguous from id SP0, so idx = tid - SP0).
  - NeuCodec must be moved to bf16 on CPU BEFORE .to("cuda") (fp32->cuda
    reproducibly crashes with a CUDACachingAllocator NVML assertion here).
  - Seam-free audio needs _linear_overlap_add cross-fade of overlapping
    decoded windows (naive per-chunk decode clicks at boundaries).
  - Leading breath/silence trim + short fades clean up the clip start.
  - Hard length cap is a required safety net (sampling reduces but doesn't
    100% eliminate runaways on odd inputs like onomatopoeia).

Emotion-ready: pass emotion="excited" to prepend an emotion control token
once the emotion fine-tune (custom <|EXCITED|>-style tokens) lands; until
then leave it None and the base model synthesizes neutrally.
"""
from __future__ import annotations

import ctypes
import gc
import re
import time
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

# Guard a known llama-cpp-python teardown race, not a leak: Llama.__del__ can
# fire during interpreter shutdown after llama_cpp's own module-level ctypes
# function bindings (e.g. llama_model_free) have already been cleared, so
# _internals.py's free_model() callback raises "TypeError: 'NoneType' object
# is not callable" calling the (now-None) binding rather than failing on
# self.model being None (which it already guards). Harmless -- the OS
# reclaims the memory at process exit either way -- but noisy. Patched here
# (not in site-packages, so it survives reinstalls) rather than relying on
# every call site remembering to close() the Llama explicitly before exit.
try:
    from llama_cpp import Llama as _Llama
    if not getattr(_Llama, "_bmo_del_guarded", False):
        _orig_llama_del = _Llama.__del__
        def _guarded_llama_del(self):
            try:
                _orig_llama_del(self)
            except Exception:
                pass
        _Llama.__del__ = _guarded_llama_del
        _Llama._bmo_del_guarded = True
except Exception:
    pass


# BMW: the smaller 350M fast tier occasionally mis-emits "BMW" for "BMO"; in
# BMO's domain that's virtually always the misgeneration, so fold it to Beemo too.
_BMO_RE = re.compile(r"\b(?:BMO|BMW)\b", re.IGNORECASE)

# Typographic Unicode -> ASCII. GPT-OSS / Fish emit "smart" punctuation (curly
# quotes, em/en dashes, the NON-BREAKING hyphen U+2011, ellipsis) that render as
# confusing glyphs mid-word ("brand<U+2011>new") and can trip tokenization /
# phonemization. Fold them to plain ASCII everywhere text is used.
_UNI_MAP = {
    "’": "'", "‘": "'", "‛": "'",       # single quotes
    "“": '"', "”": '"',                        # double quotes
    "–": "-", "‑": "-", "‒": "-",         # en / non-breaking / figure dash
    "—": " - ",                                      # em dash (spaced, don't glue words)
    "…": "...",                                      # ellipsis
    " ": " ", " ": " ", " ": " ",         # nbsp / thin spaces
}
_UNI_RE = re.compile("|".join(re.escape(k) for k in _UNI_MAP))
_WS_RE = re.compile(r"[ \t]{2,}")


def ascii_normalize(text: str) -> str:
    """Fold typographic Unicode punctuation to ASCII and collapse doubled spaces."""
    text = _UNI_RE.sub(lambda m: _UNI_MAP[m.group()], text)
    return _WS_RE.sub(" ", text)


def normalize_bmo_text(text: str) -> str:
    """BMO is pronounced "Beemo" (+ ASCII-normalize typographic punctuation).
    Two reasons the Beemo part matters, one root cause:
      1. espeak pronounces the all-caps token "BMO" letter-by-letter ("bee-em-oh")
         instead of the character's actual name.
      2. The dataset-prep acronym filter (\\b[A-Z]{2,}\\b) REJECTS "BMO", which
         silently deleted most self-referential lines and decimated the
         self-referential moods (lonely 132->19, happy 111->39). "Beemo" is not
         all-caps, so those lines survive.
    Applied at BOTH training-prep time (so the saved text + phonemes are clean)
    and inference time here (so the LLM's "BMO" is spoken the same way it trained)."""
    return ascii_normalize(_BMO_RE.sub("Beemo", text))


@dataclass
class SpeakResult:
    """Return type of StreamingVoice.speak(). Exposes ATTRIBUTES for the live
    conversational loop (models/m5_streaming_loop.py reads .duration_sec and
    .synth_latency_sec) while staying tuple-unpackable as (wav, duration_sec,
    ttfa_ms) so the older `wav, dur, ttfa = speak(...)` smoke-test call sites
    keep working unchanged. Integration gap found + fixed 2026-08-07: the loop
    was written against models/m5_tts.py's result object, not a bare tuple."""
    wav: "np.ndarray"
    duration_sec: float
    ttfa_ms: float
    synth_latency_sec: float

    def __iter__(self):  # back-compat: wav, dur, ttfa = speak(...)
        yield self.wav
        yield self.duration_sec
        yield self.ttfa_ms


# encodec-style triangular-window overlap-add (from facebookresearch/encodec)
def _linear_overlap_add(frames, stride: int, power: float = 1.0):
    assert len(frames)
    dtype = frames[0].dtype
    total = max(stride * i + f.shape[-1] for i, f in enumerate(frames))
    sum_w = np.zeros(total, dtype=dtype)
    out = np.zeros(total, dtype=dtype)
    offset = 0
    for f in frames:
        L = f.shape[-1]
        t = np.linspace(0, 1, L + 2, dtype=dtype)[1:-1]
        w = (0.5 - np.abs(t - 0.5)) ** power
        out[offset:offset + L] += w * f
        sum_w[offset:offset + L] += w
        offset += stride
    assert sum_w.min() > 0
    return out / sum_w


class StreamingVoice:
    CHUNK, LOOKFWD, LOOKBACK, OVERLAP, HOP = 25, 5, 50, 1, 480
    TEMP, TOP_K, MAX_TOK = 0.7, 50, 2000
    # Length-cap calibration, MEASURED on-device 2026-08-23 (n=36 utterances,
    # Air v5 + Nano v1, 3 lines x 6 stochastic samples each). A pure
    # tokens-per-char model is WRONG for this voice: short utterances carry
    # fixed per-utterance overhead (leading breath, trailing silence), so
    # tokens/char is ~9 at 23 chars but ~4 at 73 chars. A linear 7/char cap
    # clipped real speech on 7/36 samples, pinning 3/6 of Nano's short line at
    # exactly the cap. Intercept + slope fits the measured envelope:
    #   observed genuine maxima -> 23ch:216 tok  58ch:240 tok  73ch:398 tok
    #   this cap                -> 23ch:315      58ch:490      73ch:565
    # i.e. 1.4-2.0x headroom over anything observed, while still bounding a
    # non-terminating generation to 6-11s instead of the 40.0s measured at
    # MAX_TOK. Re-verify these constants if the TTS backbone is retrained --
    # last_exit_reason=="LENGTH_CAP_FIRED" makes a clip visible, not silent.

    CAP_BASE, CAP_PER_CHAR = 200, 5

    def __init__(self, gguf_path: str, *, codec_repo: str = "neuphonic/neucodec",
                 language: str = "en-us", device=None, max_load_attempts: int = 5,
                 compact_fn=None, codec_device: str = "cpu"):
        # codec_device default "cpu" (Task 180, 2026-08-07): moving NeuCodec
        # (fp32, 823M) to CUDA reproducibly hits a CUDACachingAllocator NVML
        # assertion in the FULL stack -- the 3 llama.cpp models already hold
        # the CUDA context and torch's allocator conflicts. CPU fp32 decode
        # sidesteps it entirely AND is the exact codec path the user
        # preferred by ear (v6 "clean") over the bf16-GPU path (v5
        # "processed"). Decode is per-25-token-chunk so CPU cost lands
        # mostly on time-to-first-audio, masked by the thinking_filler
        # backchannel anyway.
        import torch
        from llama_cpp import Llama
        self.torch = torch
        self._Llama = Llama
        self.stride = self.CHUNK * self.HOP
        self.sample_rate = 24000
        self._device = device  # sounddevice output index; None = no playback

        # backbone with the reliability retry wrapper (see bmo_jetson_startup)
        last = None
        self.llm = None
        for a in range(1, max_load_attempts + 1):
            try:
                # n_ctx=2048 measured unsafe even for a single utterance: MAX_TOK=2000
                # generation cap leaves only 48 tokens for the prompt, but real
                # phonemized prompts already run 81-108 tokens on ordinary sentences
                # (measured on the two BMO test presets) -- a longer utterance that
                # actually reached the safety cap would overflow n_ctx within one
                # call, independent of the cross-call KV-cache bug below. 2560 gives
                # 560 tokens of prompt headroom above MAX_TOK, comfortably more than
                # any observed prompt, without a large extra KV-cache memory cost on
                # this 7.4GB Jetson.
                self.llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=2560,
                                 logits_all=False, verbose=False)
                break
            except Exception as e:  # noqa
                last = e
                if compact_fn and a < max_load_attempts:
                    compact_fn()
        if self.llm is None:
            raise RuntimeError(f"StreamingVoice backbone load failed after {max_load_attempts} tries") from last

        self._sp0 = self.llm.tokenize(b"<|speech_0|>", special=True, add_bos=False)[0]
        self._sp_hi = self.llm.tokenize(b"<|speech_65535|>", special=True, add_bos=False)[0]
        self._end = self.llm.tokenize(b"<|SPEECH_GENERATION_END|>", special=True, add_bos=False)[0]

        # Emotion capability self-detection (Task 179, 2026-08-07): only the
        # emotion fine-tune has the <|MOOD|> control tokens in its vocab. On a
        # model WITHOUT them (e.g. the base v5 voice), "<|NEUTRAL|>" tokenizes
        # to several BPE text tokens (len > 1) instead of one special token --
        # prepending it would inject garbage and perturb the output. So detect
        # once and make speak(emotion=...) a NO-OP when the tokens aren't there.
        # This lets the live loop always pass the mood; it's harmless on v5.
        self.has_emotion = len(self.llm.tokenize(b"<|NEUTRAL|>", special=True, add_bos=False)) == 1

        # NeuCodec INT8 ONNX DECODER (Task 180, 2026-08-07): the decisive fix.
        # Decode-only (all StreamingVoice needs), CPU via onnxruntime.
        # Measured on the Jetson: ~54ms per 27-token chunk (matches the GPU
        # torch codec's ~51ms!), full-utterance RTF ~0.08 (13x realtime) --
        # vs the fp32-torch-CPU codec's RTF ~2.0 (unusable for streaming),
        # and with NO GPU-allocator conflict with the 3 resident llama.cpp
        # models (the torch NeuCodec's .to('cuda') hit a CUDACachingAllocator
        # NVML assertion in the full stack). This is faster AND more robust.
        # MEASURED 2026-08-15 on the Jetson, same model file, same decode:
        #     default ORT session            414 MiB   106 ms warm
        #     enable_cpu_mem_arena=False       1 MiB   105 ms warm    <- this
        #     ...plus intra_op_num_threads=2   0 MiB   198 ms         <- do NOT, 2x slower
        # ONNX Runtime's default CPU arena pre-allocates ~408 MiB this decoder never uses.
        # That is most of the entire TTS+STT memory budget, for nothing.
        #
        # NeuCodecOnnxDecoder builds its own session WITHOUT that flag, and freeing it
        # afterwards only returns ~80 MiB (glibc does not hand arena pages back), so the
        # session has to be built correctly the FIRST time. Its decode_code() is just
        # `session.run(None, {"codes": codes})[0].astype(np.float32)` plus shape validation,
        # so owning the session directly costs nothing in behaviour.
        import numpy as _np
        import onnxruntime as _ort
        from huggingface_hub import hf_hub_download

        _onnx_path = hf_hub_download("neuphonic/neucodec-onnx-decoder-int8", "model.onnx")
        _so = _ort.SessionOptions()
        _so.enable_cpu_mem_arena = False          # the 413 MiB line
        _sess = _ort.InferenceSession(_onnx_path, sess_options=_so,
                                      providers=["CPUExecutionProvider"])

        class _ArenaFreeNeuCodecDecoder:
            """Drop-in for NeuCodecOnnxDecoder's decode path, arena disabled."""
            sample_rate = 24_000

            def __init__(self, session):
                self.session = session

            def decode_code(self, codes):
                if not isinstance(codes, _np.ndarray):
                    raise ValueError("`Codes` should be an np.array.")
                if len(codes.shape) != 3 or codes.shape[1] != 1:
                    raise ValueError("`Codes` should be of shape [B, 1, F].")
                return self.session.run(None, {"codes": codes})[0].astype(_np.float32)

        self.codec = _ArenaFreeNeuCodecDecoder(_sess)
        self._codec_dev = "onnx-cpu"

        # phonemizer (espeak); matches finetune_bmo_neutts.py exactly
        import phonemizer
        self._g2p = phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=True,
            words_mismatch="ignore", language_switch="remove-flags")

    def _phones(self, text: str) -> str:
        return " ".join(self._g2p.phonemize([normalize_bmo_text(text)])[0].split())

    def _prompt(self, text: str, emotion: Optional[str]) -> bytes:
        emo = f"<|{emotion.upper()}|>" if (emotion and self.has_emotion) else ""
        return (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{self._phones(text)}"
                f"<|TEXT_PROMPT_END|>\nassistant:{emo}<|SPEECH_GENERATION_START|>").encode("utf-8")

    def _decode(self, idx):
        codes = np.array(idx, dtype=np.int32)[None, None, :]
        return np.asarray(self.codec.decode_code(codes)).reshape(-1).astype(np.float32)

    @staticmethod
    def _trim_and_fade(wav, sr):
        w = int(0.01 * sr)
        if len(wav) < 4 * w:
            return wav
        n = len(wav) // w
        rms = np.array([np.sqrt(np.mean(wav[i*w:(i+1)*w] ** 2) + 1e-9) for i in range(n)])
        thr = 0.06 * rms.max()
        onset = next((i for i, r in enumerate(rms) if r > thr), 0)
        wav = wav[max(0, onset * w - int(0.02 * sr)):].copy()
        f = int(0.008 * sr)
        if len(wav) > 2 * f:
            wav[:f] *= np.linspace(0, 1, f); wav[-f:] *= np.linspace(1, 0, f)
        return wav

    def stream(self, text: str, emotion: Optional[str] = None) -> Generator[np.ndarray, None, None]:
        """Yields audio chunks (float32, 24kHz) as they're decoded. First
        yield is time-to-first-audio."""
        # Full clear, not a sequence-scoped removal: kv_cache_seq_rm(-1, 0, -1)
        # (seq_id=-1 clamps to 0 internally in llama-cpp-python 0.3.34, and the
        # call reports success) measurably failed to fully release the cache
        # across calls -- reproduced live, second alternating preset call
        # crashed with "failed to find a memory slot for batch of size 1" /
        # RuntimeError: llama_decode returned 1. kv_cache_clear() (unconditional,
        # not sequence-scoped) verified clean over 5 rounds x 2 presets (10 calls)
        # with no growth in per-call token counts. self.llm.reset() alone is not
        # enough either -- it only zeroes the Python-side n_tokens bookkeeping;
        # for a non-recurrent model it never touches the underlying C-level cache.
        self.llm.reset()
        try:
            self.llm._ctx.kv_cache_clear()
        except Exception:
            pass
        toks = self.llm.tokenize(self._prompt(text, emotion), special=True, add_bos=False)
        # Length-aware safety cap, REINSTATED 2026-08-23 on top of this file's
        # n_ctx/kv_cache fixes (it had been removed when MAX_TOK went 350->2000
        # to diagnose the EOS problem). Measured on-device that day, n=18 lines
        # per model: Air v5 fails to emit <|SPEECH_GENERATION_END|> on 3/18
        # utterances and runs to the 2000-token cap, producing 40.0s of audio in
        # 35.9s for a 23-CHARACTER line. Nano v1 is 1/18, worst case 4.3s. The
        # hard MAX_TOK is the wrong guard because it is length-independent: it
        # bounds the damage at 40s no matter how short the text. This bounds it
        # at ~2x the natural speaking rate for THIS text instead.
        # 7 speech tokens/char at HOP=480/24kHz (=50 tok/s) is ~7.1 chars/s of
        # audio, against ~14 chars/s of natural speech -> ~2x headroom, verified
        # not to clip any of the six benchmarked lines on either backbone.
        # MAX_TOK stays as the absolute ceiling; last_exit_reason distinguishes
        # the two so a legitimate clip is visible rather than silent.
        tok_cap = min(self.MAX_TOK, self.CAP_BASE + self.CAP_PER_CHAR * len(text.strip()))
        self.last_exit_reason = "UNKNOWN"
        self.last_token_count = 0
        self.last_tok_cap = tok_cap
        sp, cache = [], []
        nt = ns = 0
        for tid in self.llm.generate(toks, temp=self.TEMP, top_k=self.TOP_K):
            if tid == self._end:
                self.last_exit_reason = "EOS_TOKEN"
                break
            if self._sp0 <= tid <= self._sp_hi:
                sp.append(tid - self._sp0)
            if len(sp) - nt >= self.CHUNK + self.LOOKFWD:
                ts = max(nt - self.LOOKBACK - self.OVERLAP, 0)
                te = nt + self.CHUNK + self.LOOKFWD + self.OVERLAP
                ss = (nt - ts) * self.HOP
                se = ss + (self.CHUNK + 2 * self.OVERLAP) * self.HOP
                cache.append(self._decode(sp[ts:te])[ss:se])
                proc = _linear_overlap_add(cache, stride=self.stride)
                ne = len(cache) * self.stride
                yield proc[ns:ne]
                ns = ne; nt += self.CHUNK
            if len(sp) >= tok_cap:
                self.last_exit_reason = ("LENGTH_CAP_FIRED" if tok_cap < self.MAX_TOK
                                          else "SAFETY_CAP_FIRED")
                print(f"*** StreamingVoice: {self.last_exit_reason} ({tok_cap} tokens, "
                      f"{len(text.strip())} chars) before EOS! ***")
                break
        self.last_token_count = len(sp)
        if len(sp) > nt:
            rem = len(sp) - nt
            ts = max(len(sp) - (self.LOOKBACK + self.OVERLAP + rem), 0)
            ss = max(0, (len(sp) - ts - rem - self.OVERLAP) * self.HOP)
            cache.append(self._decode(sp[ts:])[ss:])
            proc = _linear_overlap_add(cache, stride=self.stride)
            yield proc[ns:]

    def synth(self, text: str, emotion: Optional[str] = None, min_rms: float = 0.02, max_retries: int = 2):
        """Full (non-streaming) synth -> (wav float32, duration_sec). Guards
        against the rare near-silent/'huffing' fluke of stochastic sampling
        (temp=0.7 occasionally emits a degenerate low-energy take, seen ~1/8 on
        some emotion tokens): if the result's RMS is below min_rms, re-generate
        (up to max_retries). Cheap, and eliminates the exact artifact heard on
        an unlucky bored/concerned sample."""
        wav = np.zeros(1, np.float32)
        for _ in range(max_retries + 1):
            chunks = list(self.stream(text, emotion))
            wav = self._trim_and_fade(np.concatenate(chunks), self.sample_rate) if chunks else np.zeros(1, np.float32)
            if float(np.sqrt(np.mean(wav ** 2) + 1e-12)) >= min_rms:
                break
        return wav, len(wav) / self.sample_rate

    def speak(self, text: str, emotion: Optional[str] = None, blocking: bool = False) -> "SpeakResult":
        """Stream + play via sounddevice as chunks arrive (best-effort; if no
        device configured, just synthesizes). Returns a SpeakResult (also
        unpackable as wav, duration_sec, ttfa_ms). `emotion` is a homeostatic
        mood name (e.g. "excited"); it prepends the matching <|MOOD|> control
        token so the SAME state that shapes the LLM's words shapes the voice."""
        t0 = time.perf_counter()
        ttfa = None
        collected = []
        sd = None
        if self._device is not None:
            try:
                import sounddevice as sd  # noqa
            except Exception:
                sd = None
        for seg in self.stream(text, emotion):
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000.0
            collected.append(seg)
            if sd is not None:
                try:
                    sd.play(seg.astype(np.float32), samplerate=self.sample_rate, device=self._device)
                    if blocking:
                        sd.wait()
                except Exception:
                    sd = None
        wav = self._trim_and_fade(np.concatenate(collected), self.sample_rate) if collected else np.zeros(1, np.float32)
        synth_latency = time.perf_counter() - t0
        return SpeakResult(wav, len(wav) / self.sample_rate, (ttfa or 0.0), synth_latency)
