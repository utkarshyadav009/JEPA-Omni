"""scripts/jetson_m2m3_live_caption.py — smaller milestone than the full
M4/M5 duplex demo (jetson_m5_live_demo.py), deliberately scoped down per
direct instruction: just M2 (joint AV predictor, LOCKED checkpoint) + M3
(vision-language connector, LOCKED checkpoint) running live, no decision
head, no turn-taking, no MicGate, no backchannel, no interruption. The
model looks and listens through the real camera/mic on a fixed cadence
and says what it sees -- a live version of M3's captioning path, nothing
else in the loop.

Also fixes the repetition-loop problem seen in jetson_m5_live_demo.py's
first live run ("What kind of guard? What kind of guard? ...") -- that
script's generation went through DuplexLoop.generate_interruptible, which
does plain greedy argmax decoding with no repetition control (needed for
its manual step-by-step KV-cache loop to support mid-generation
interruption). This script has no interruption requirement at all, so it
uses the LLM's own .generate() directly with repetition_penalty and
no_repeat_ngram_size -- HF's well-tested decoding controls, not a
hand-rolled fix.

Usage:
    python3 scripts/jetson_m2m3_live_caption.py --interval-sec 8
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import types
import importlib.machinery

_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_AMBIENT_T = 1024
# The EXACT tag format M3 was trained/evaluated with (train_m3.py's
# GRANULARITY_TAGS, confirmed by direct read 2026-08-01) -- a short
# imperative "Task: ...\n" tag placed AFTER the M3 latents, tokenized with
# add_special_tokens=False, NOT a chat-style question. A natural-language
# question ("Describe what you currently see and hear...") is a prompt
# format the model was never trained on -- the first live run using that
# format produced empty strings, hallucinated fake dialogue, and meta-
# commentary instead of grounded descriptions. gpt_summary_detailed is the
# closest of the five trained tags to "describe what you see and hear."
# gpt_action_brief, not gpt_summary_detailed -- a short, action-focused tag
# fits a continuously-updating live status much better than a "short
# paragraph" tag being truncated at 14 tokens (which would cut off mid-sentence).
PROMPT_TEXT = "Task: state the main physical action in one short sentence.\n"


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, device):
    # Same proven sequence as scripts/jetson_phase4_full_stack_memory_v2_withqwen.py
    # and (after the 2026-08-01 fix) scripts/jetson_m5_live_demo.py -- the
    # malloc_trim() calls around the GPU move are load-bearing on this
    # Jetson's unified memory, not cosmetic (see jetson_m5_live_demo.py's
    # comment for the reproduced crash this avoids).
    import gc
    module = module.to("cpu")
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[live-caption]   int8 quant failed (running fp/bf16 instead): {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return module


class SimpleAVBuffer:
    """Just the two rolling buffers LiveCameraCapture/LiveMicCapture need to
    push into -- no decision head, no MicGate, no DuplexLoop. Reuses the
    same buffer classes models/m5_streaming_loop.py already validated
    (uniform 64-frame sampling from a partially-filled window, etc.)."""

    def __init__(self, window_sec: float, fps: float, audio_sr: int):
        from models.m5_streaming_loop import RollingVideoBuffer, RollingAudioBuffer
        self.video_buf = RollingVideoBuffer(window_sec, fps)
        self.audio_buf = RollingAudioBuffer(window_sec, audio_sr)   # ambient (WavJEPA) window

    def ingest_video_frame(self, frame: torch.Tensor) -> None:
        self.video_buf.push(frame)

    def ingest_audio_chunk(self, chunk) -> None:
        self.audio_buf.push(chunk)


class PerceptionThread:
    """Runs perceive (build_world_state_features -> pre-pool -> M3) in a
    tight background loop, continuously overwriting a cached soft prompt.
    The main generation loop reads whatever's freshest instead of blocking
    on a fresh perceive every round -- this is the fix for "generation
    always waits ~2.5s for perception first" (2026-08-01 feedback: "make it
    causal ... constant continuous stream ... no intervals"). This is NOT
    true causal/frame-by-frame encoding -- V-JEPA2/WavJEPA are windowed
    encoders, not causal streaming models, and making them causal would
    need retraining (out of scope, FREEZE forbids backbone/architecture
    changes). What this DOES do: decouples the ~2.5s perceive cost from
    the ~1-2s generate cost so they overlap instead of serializing, which
    is the real, honest latency ceiling available without retraining."""

    def __init__(self, buf, vision_enc, wavjepa_base, wavjepa_nat, predictor,
                 m3_connector, max_tdm_bins, device):
        self.buf = buf
        self.vision_enc = vision_enc
        self.wavjepa_base = wavjepa_base
        self.wavjepa_nat = wavjepa_nat
        self.predictor = predictor
        self.m3_connector = m3_connector
        self.max_tdm_bins = max_tdm_bins
        self.device = device
        self._lock = threading.Lock()
        self._soft_prompt = None
        self._n_updates = 0
        self._last_perceive_ms = None
        self._stop = threading.Event()
        self._thread = None

    def _run(self) -> None:
        from models.world_state_builder import build_world_state_features
        while not self._stop.is_set():
            video_window = self.buf.video_buf.get_window()
            ambient_window = self.buf.audio_buf.get_window()
            if video_window is None or ambient_window is None:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                audio_t = torch.from_numpy(ambient_window).float()
                true_dur = audio_t.shape[0] / 16000.0
                result = build_world_state_features(video_window, audio_t, true_dur, self.vision_enc,
                                                      self.wavjepa_base, self.wavjepa_nat,
                                                      self.max_tdm_bins, self.device)
                pre_pool = self.predictor.encode_pre_pool_tokens(result.feats, result.tbins)
                if pre_pool.shape[1] > MAX_AMBIENT_T:
                    pre_pool = pre_pool[:, :MAX_AMBIENT_T]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    soft_prompt = self.m3_connector(pre_pool)
            with self._lock:
                self._soft_prompt = soft_prompt
                self._n_updates += 1
                self._last_perceive_ms = (time.perf_counter() - t0) * 1000.0

    def get_latest(self):
        with self._lock:
            return self._soft_prompt, self._n_updates, self._last_perceive_ms

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--m3-ckpt", default="checkpoints/m3_multigran_richcaption_v2/last.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--camera-index", type=int, default=4)
    p.add_argument("--mic-device", type=int, default=24)
    p.add_argument("--tts-device", type=int, default=24)
    p.add_argument("--speak", action="store_true", default=True)
    p.add_argument("--no-speak", dest="speak", action="store_false", help="print only, don't play TTS")
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--interval-sec", type=float, default=0.0,
                   help="extra gap AFTER each round on top of perceive+generate latency -- "
                        "0 means back-to-back rounds, the perceive+generate cost itself is the "
                        "pacing (2026-08-01: full 50-token paragraphs took ~13s/round and felt "
                        "nothing like live; short phrases looped back-to-back feel much closer "
                        "to a continuously-updating status than a fixed 8s gap plus a slow paragraph)")
    p.add_argument("--n-rounds", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="as short as still produces a real phrase -- the dominant latency lever "
                        "(~200ms/token on this Jetson's Qwen2.5-1.5B-int8, measured in A1)")
    p.add_argument("--no-speak-blocking", dest="speak_blocking", action="store_false", default=True,
                   help="don't wait for TTS playback to finish before starting the next round's "
                        "generation -- speech and perception/generation overlap (may talk over "
                        "itself if generation is faster than the previous phrase's playback)")
    p.add_argument("--repetition-penalty", type=float, default=1.15,
                   help="matches scripts/m4_joint_eval.py's validated generate() call")
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live-caption] device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)

    print("[live-caption] loading V-JEPA2 ViT-L (int8)...", flush=True)
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, device)
    vision_enc.device_str = str(device)

    print("[live-caption] loading WavJEPA-base/nat (int8)...", flush=True)
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, device)
    wavjepa_base.device_str = str(device)
    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, device)
    wavjepa_nat.device_str = str(device)

    print("[live-caption] loading locked M2 predictor (int8)...", flush=True)
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, device)
    predictor.eval()

    print("[live-caption] loading Qwen2.5-1.5B-Instruct (int8)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)
    llm = q_int8_cpu_then_move(llm, device)
    llm.eval()

    print("[live-caption] loading locked M3 connector...", flush=True)
    from models.m3_connector import M3Connector, M3ConnectorConfig
    m3ckpt = torch.load(args.m3_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    tts = None
    if args.speak:
        print("[live-caption] loading real TTS (Piper)...", flush=True)
        from models.m5_tts import TTSEngine
        tts = TTSEngine(device=args.tts_device)

    buf = SimpleAVBuffer(args.window_sec, fps=6.4, audio_sr=16000)

    print(f"[live-caption] starting live camera (index {args.camera_index}) + mic (device {args.mic_device})...", flush=True)
    from models.m5_live_capture import LiveCameraCapture, LiveMicCapture
    cam = LiveCameraCapture(buf, device_index=args.camera_index, target_fps=6.4)
    mic = LiveMicCapture(buf, sample_rate=16000, chunk_sec=0.25, device=args.mic_device)
    cam.start()
    mic.start()

    # prompt tokens, embedded once, prepended to every soft prompt (gives the
    # LLM something to condition the caption on beyond raw perception tokens)
    # add_special_tokens=False matches train_m3.py's GRANULARITY_TAGS tokenization exactly
    prompt_ids = tokenizer(PROMPT_TEXT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)

    print(f"[live-caption] warming up ({args.window_sec:.0f}s, filling the buffer)...", flush=True)
    time.sleep(args.window_sec)

    print("[live-caption] starting background perception thread (decoupled from generation)...", flush=True)
    perception = PerceptionThread(buf, vision_enc, wavjepa_base, wavjepa_nat, predictor,
                                   m3_connector, predictor_cfg.max_tdm_bins, device)
    perception.start()
    while perception.get_latest()[0] is None:
        time.sleep(0.05)

    last_text = None
    for round_i in range(args.n_rounds):
        soft_prompt_m3, n_perceives, last_perceive_ms = perception.get_latest()

        with torch.no_grad():
            inputs_embeds = torch.cat([soft_prompt_m3.to(prompt_embeds.dtype), prompt_embeds], dim=1)
            attn = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)

            t1 = time.perf_counter()
            out_ids = llm.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ms = (time.perf_counter() - t1) * 1000.0
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        changed = text != last_text
        print(f"[live-caption] round {round_i+1}/{args.n_rounds}  generate={gen_ms:.0f}ms  "
              f"(perception thread: {n_perceives} updates so far, last took {last_perceive_ms:.0f}ms)  "
              f"{'' if changed else '(unchanged) '}-> {text!r}", flush=True)

        if tts is not None and text and changed:
            tts.speak(text, blocking=args.speak_blocking)
        last_text = text

        time.sleep(max(0.0, args.interval_sec))

    perception.stop()
    cam.stop()
    mic.stop()
    print("[live-caption] DONE", flush=True)


if __name__ == "__main__":
    main()
