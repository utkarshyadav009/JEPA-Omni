"""scripts/jetson_describe_demo.py — "BMO looks at the room and says what it sees."

The narrowest useful end-to-end demo of the north-star perception path on real hardware:
CSI camera -> V-JEPA2 + WavJEPA -> M2 -> QueryPredictor -> retrieval -> fast-tier LLM
phrasing -> TTS. **No STT, no VAD, no decision head, no M3 connector** -- this build is
deliberately not a conversation, it is BMO describing its own view, which is the capability
the JEPA-memory track exists to deliver.

SCOPE NOTE, stated because it differs from `build_bmo_stack()`: that function also loads
Moonshine (STT) + the 3-class decision head + the M3 connector. Those are excluded here by
request. Loading them would only make the memory picture tighter, so the numbers below are
a FLOOR for the describe-only pipeline, not a measurement of the full conversational stack.

PRECISION — no new quantization was introduced (explicit instruction):
  * GGUFs (fast tier, TTS)  : Q8_0, exactly as already shipped on the device
  * perception (ViT-L, WavJEPA, M2) : int8 weight-only via `q_int8_cpu_then_move`, which is
    the EXISTING production load path (since Phase 0), not something added to force a fit
  * EmbeddingGemma          : bf16 (TextTarget's own default)
  * QueryPredictor + bank   : fp32 weights / fp16 bank, as trained and shipped
If it does not fit, the levers (smaller TTS, further quantization) are deliberately NOT
applied here -- the point of this run is to find out.

LOAD ORDER is load-bearing on this device and is preserved: llama.cpp GGUFs FIRST, then the
torch/int8 perception stack. Reversing it fragments Jetson unified memory badly enough that
the GGUFs fail to load (documented in bmo_jetson_startup.py's module docstring).

Usage (after `sudo bash ~/bmo_production/scripts/jetson_preflight.sh`):
    python3 ~/jetson_describe_demo.py --rounds 3 --speak
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

MEM_LOG: List[Dict] = []


def _trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def mem(tag: str) -> float:
    """Real available MiB from /proc/meminfo -- the same signal jetson_preflight gates on."""
    with open("/proc/meminfo") as f:
        info = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in f)}
    total = info["MemTotal"] / 1024
    avail = info["MemAvailable"] / 1024
    used = total - avail
    prev = MEM_LOG[-1]["avail"] if MEM_LOG else None
    delta = (prev - avail) if prev is not None else 0.0
    MEM_LOG.append({"tag": tag, "used": used, "avail": avail, "delta": delta})
    print(f"[mem] {tag:32s} used={used:7.0f}MiB avail={avail:7.0f}MiB  (+{delta:6.0f})", flush=True)
    return avail


def q_int8_cpu_then_move(module, device):
    """The PROVEN production sequence (see bmo_jetson_startup.py): quantize on CPU, trim,
    then move. malloc_trim around the GPU move is load-bearing on this device's unified
    memory -- an NvMap contiguity issue, not a totals problem."""
    module = module.to("cpu")
    gc.collect(); _trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[warn] int8 quantize skipped ({e!r}) -- loading at native precision", flush=True)
    gc.collect(); _trim()
    return module.to(device)


# ── camera ────────────────────────────────────────────────────────────────
def _frame_is_degenerate(bgr) -> bool:
    """A CSI sensor opened on plain V4L2 returns UNCONVERTED data that OpenCV happily
    reports as a successful read -- in practice a flat green image. That silently turned
    a whole demo run into 'describe this blank frame' (caught 2026-08-14: three identical
    nonsense captions). Per-channel spatial std is the cheap tell: a real scene has
    texture, a format-mismatch frame is nearly constant within each channel."""
    import numpy as np
    stds = [float(bgr[:, :, c].std()) for c in range(bgr.shape[2])]
    return max(stds) < 8.0


def open_camera(index: int = 0):
    """CSI camera. Tries nvarguscamerasrc FIRST -- that is the correct path for a Jetson
    CSI sensor -- and only falls back to V4L2 (for a USB webcam) if GStreamer is
    unavailable. Every candidate is validated with a real frame, because "opened" and
    "returns usable pixels" are not the same thing on this hardware."""
    import cv2
    gst = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,"
           "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
           "video/x-raw,format=BGR ! appsink drop=1 max-buffers=2")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        for _ in range(5):                      # first frames after sensor start can be blank
            ok, f = cap.read()
            if ok and not _frame_is_degenerate(f):
                print(f"[cam] GStreamer nvarguscamerasrc  frame={f.shape}", flush=True)
                return cap
        cap.release()
        print("[cam] nvarguscamerasrc opened but frames were degenerate", flush=True)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        ok, f = cap.read()
        if ok and not _frame_is_degenerate(f):
            print(f"[cam] V4L2 /dev/video{index}  frame={f.shape}", flush=True)
            return cap
        cap.release()
        raise RuntimeError(
            f"/dev/video{index} opened on V4L2 but returned a degenerate (flat) frame -- "
            "this is the CSI-sensor-without-nvargus case; do not treat it as usable input")
    raise RuntimeError("could not open the camera on nvarguscamerasrc or V4L2")


def grab_window(cap, n_frames: int, res: int = 256, spread_sec: float = 2.0,
                rotate: int = 0, gain: float = 1.0) -> torch.Tensor:
    """Capture n_frames spread over spread_sec -> (T,3,res,res) uint8, as V-JEPA2 expects.

    `rotate` matters: this CSI camera is physically mounted SIDEWAYS, and V-JEPA2 was
    trained on upright video, so an unrotated feed is out-of-distribution input dressed up
    as a valid one. `gain` brightens a dim room -- also a distribution question, not a
    cosmetic one."""
    import cv2
    ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}
    frames = []
    interval = spread_sec / max(1, n_frames)
    for _ in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("camera read failed mid-window")
        if rotate in ROT:
            bgr = cv2.rotate(bgr, ROT[rotate])
        if gain != 1.0:
            bgr = cv2.convertScaleAbs(bgr, alpha=gain, beta=0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(h, w)
        rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        time.sleep(interval)
    return torch.stack(frames, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--query", default="Describe the room and setting in detail.")
    ap.add_argument("--rotate", type=int, default=90, choices=[0, 90, -90, 180],
                    help="camera is mounted sideways; 90=CCW (default). Confirm visually.")
    ap.add_argument("--gain", type=float, default=1.8, help="brightness gain for a dim room")
    ap.add_argument("--save-frame", default="", help="write the first captured frame for inspection")
    ap.add_argument("--bank", default="perception_bank_max_fp16.pt", help="bank file under pipeline/checkpoints")
    ap.add_argument("--int8-text", action="store_true",
                    help="int8 (linear-only) the query encoder. Measured quality-neutral: "
                         "encoder-output cosine 0.9998, retrieval field 0.939->0.939, ~105 MiB saved. "
                         "Embedding-table int8 is NOT used -- it was measured to break the encoder.")
    ap.add_argument("--with-thinker", action="store_true", help="also load the Qwen3-0.6B thinker GGUF")
    ap.add_argument("--with-identity", action="store_true", help="also load the joint AV identity head")
    ap.add_argument("--out", default=os.path.expanduser("~/jetson_describe_demo_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    results: Dict = {"config": vars(args), "stages": [], "rounds": []}
    mem("00_start")
    torch.cuda.init(); mem("01_cuda_context")

    # ---- STEP 1: llama.cpp GGUFs FIRST (load order is load-bearing) ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier
    t0 = time.time()
    fast_tok = AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/lfm25_350m_tok")
    fast = GGUFFastTier(f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf", fast_tok,
                        max_new_tokens=48, n_gpu_layers=-1)
    results["fast_tier_load_s"] = time.time() - t0
    mem("02_fast_tier_v2_gguf")

    thinker = None
    if args.with_thinker:
        from models.m4_cognitive_core import GGUFReasoningTier
        t0 = time.time()
        think_tok = AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/qwen3_thinker_tok")
        thinker = GGUFReasoningTier(f"{PROD}/models_gguf/bmo_thinker_qwen3_v3_Q8_0.gguf",
                                    think_tok, n_gpu_layers=-1)
        results["thinker_load_s"] = time.time() - t0
        mem("02b_thinker_v3_gguf")

    voice = None
    if args.speak:
        from models.m5_streaming_voice import StreamingVoice
        t0 = time.time()
        for attempt in range(5):          # the documented 5x retry+compaction wrapper
            try:
                voice = StreamingVoice(f"{PROD}/models_gguf/bmo_neutts_emotion_Q8_0.gguf")
                break
            except Exception as e:
                print(f"[tts] attempt {attempt+1}/5 failed: {e!r}", flush=True)
                gc.collect(); _trim(); torch.cuda.empty_cache()
                os.system("echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1")
                time.sleep(2)
        results["tts_load_s"] = time.time() - t0
        results["tts_loaded"] = voice is not None
        mem("03_tts_emotion_voice")

    # ---- STEP 2: torch perception stack (int8, production path) ----
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor

    t0 = time.time()
    vision = VisionEncoder(device="cpu")
    vision.model = q_int8_cpu_then_move(vision.model, device)
    vision.device_str = str(device)
    mem("04_vitl_int8")
    wav_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wav_base.model = q_int8_cpu_then_move(wav_base.model, device)
    wav_base.device_str = str(device)
    mem("05_wavjepa_base_int8")
    wav_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wav_nat.model = q_int8_cpu_then_move(wav_nat.model, device)
    wav_nat.device_str = str(device)
    mem("06_wavjepa_nat_int8")

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg)
    m2.load_state_dict(torch.load(
        f"{PROD}/pipeline/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
        map_location="cpu", weights_only=False)["model"], strict=True)
    m2 = q_int8_cpu_then_move(m2, device).eval()
    results["perception_load_s"] = time.time() - t0
    mem("07_m2_predictor_int8")

    # ---- STEP 3: perception query engine ----
    from models.text_target import TextTarget
    from models.m5_perception_query import load_perception_query_engine
    t0 = time.time()
    if args.int8_text:
        from models.quantized_text_encoder import quantize_text_encoder
        tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False, device="cpu")
        tt.base = quantize_text_encoder(tt.base, do_linear=True, do_embedding=False).to(device)
        tt.proj = tt.proj.to(device); tt.device_str = str(device)
        mem("08_embeddinggemma_int8")
    else:
        tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False, device=str(device))
        mem("08_embeddinggemma_bf16")
    eng = load_perception_query_engine(
        f"{PROD}/pipeline/checkpoints/best.pt", tt, device,
        bank_emb_path=f"{PROD}/pipeline/checkpoints/{args.bank}",
        max_age_s=1e9)
    results["query_engine_load_s"] = time.time() - t0
    avail_after_load = mem("09_query_predictor+bank")
    results["stages"] = MEM_LOG.copy()
    results["avail_after_full_load_MiB"] = avail_after_load

    ident = None
    if args.with_identity:
        from models.jepa_identity_head import IdentityHead, IdentityHeadConfig
        t0 = time.time()
        ick = torch.load(f"{PROD}/pipeline/checkpoints/head_joint.pt", map_location="cpu",
                         weights_only=False)
        ident = IdentityHead(IdentityHeadConfig(in_dims=ick["dims"], emb_dim=ick["emb_dim"]))
        ident.load_state_dict(ick["head"]); ident.eval().to(device)
        results["identity_head_load_s"] = time.time() - t0
        mem("09b_identity_head")

    class _M2Adapter:                     # what update_from_features expects
        def compute_pre_pool(self, feats, tbins):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return m2.encode_pre_pool_tokens({k: v.float() for k, v in feats.items()}, tbins)
    adapter = _M2Adapter()

    # ---- STEP 4: the actual demo loop ----
    from models.world_state_builder import build_world_state_features
    cap = open_camera()
    mem("10_camera_open")
    SR = 16000
    silent = torch.zeros(int(10.0 * SR))   # no mic on this device -- real zero-valued audio.
                                            # WavJEPA still runs a REAL forward (its latency is
                                            # measured); only the CONTENT is absent.
    for r in range(args.rounds):
        print(f"\n───────── round {r+1}/{args.rounds} ─────────", flush=True)
        lat = {}
        t0 = time.time()
        frames = grab_window(cap, args.frames, rotate=args.rotate, gain=args.gain)
        lat["capture_ms"] = (time.time()-t0)*1000
        if args.save_frame and r == 0:
            import cv2 as _cv
            _cv.imwrite(args.save_frame,
                        _cv.cvtColor(frames[0].permute(1, 2, 0).numpy(), _cv.COLOR_RGB2BGR))
        t0 = time.time()
        ws = build_world_state_features(frames, silent, 10.0, vision, wav_base, wav_nat, 512, device)
        lat["perception_ms"] = (time.time()-t0)*1000
        t0 = time.time(); eng.update_from_features(ws.feats, ws.tbins, adapter)
        lat["m2_prepool_ms"] = (time.time()-t0)*1000
        t0 = time.time(); ans = eng.ask(args.query); lat["query_ms"] = (time.time()-t0)*1000
        seen = ans.text if ans else "(nothing confident)"
        print(f"  [perception] {seen[:160]}", flush=True)
        t0 = time.time()
        fr = fast.generate(f"You are looking at this scene: {seen} Say what you see, in one short sentence.",
                           {"energy": 0.6, "mood": "curious"})
        lat["llm_ms"] = (time.time()-t0)*1000
        said = getattr(fr, "text", str(fr))
        print(f"  [BMO] {said}", flush=True)
        if voice is not None:
            t0 = time.time()
            try:
                sr_ = voice.speak(said, emotion="curious", blocking=True)
                lat["tts_ttfa_ms"] = getattr(sr_, "ttfa_ms", None)
                lat["tts_total_ms"] = (time.time()-t0)*1000
            except Exception as e:
                print(f"  [tts] failed: {e!r}", flush=True)
        lat["total_ms"] = sum(v for k, v in lat.items() if k.endswith("_ms") and v)
        print("  " + "  ".join(f"{k}={v:.0f}" for k, v in lat.items() if v), flush=True)
        results["rounds"].append({"perception": seen, "spoken": said, "latency": lat})

    cap.release()
    mem("11_after_rounds")
    if thinker is not None:
        t0 = time.time()
        tr = thinker.generate("Describe in one sentence what you would say about a room "
                              "with a desk and a chair.", {"energy": 0.6, "mood": "curious"})
        results["thinker_generate_ms"] = (time.time() - t0) * 1000
        print(f"  [thinker] {getattr(tr,'text',str(tr))[:120]}  "
              f"({results['thinker_generate_ms']:.0f}ms)", flush=True)
        mem("12_after_thinker")
    results["mem_log"] = MEM_LOG
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
