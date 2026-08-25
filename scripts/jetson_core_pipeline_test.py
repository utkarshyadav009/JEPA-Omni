"""scripts/jetson_core_pipeline_test.py — does the CORE companion pipeline fit and run on the
Jetson, and does it say the right thing?

SCOPE (per instruction): face engine stays UP (it owns the display + CSI camera). **No TTS,
no STT, no VAD** — this is about whether perception -> thinker -> speaker works and fits, not
about voice I/O. TTS is separately broken on this device anyway (`import onnxruntime` aborts).

THE CHAIN UNDER TEST
    CSI camera ─► V-JEPA2 + WavJEPA(+/-nat) ─► M2 ─┐
                                                   ├─► QueryPredictor ─► retrieved description
    CSI camera ─► SigLIP2-base (scene stream) ─────┘                              │
                                                                                  ▼
                                                                    THINKER (Qwen3-0.6B, reasons)
                                                                                  │
                                                                                  ▼
                                                             SPEAKER / fast tier (LFM2.5-350M)
                                                                    the line BMO would say

WHY BOTH nat VARIANTS. The stream ablation said dropping WavJEPA-nat was free (R@1 0.566 vs
0.564); the congruence eval then showed it costs 4.7 points of audio-following (0.562 vs
0.609). nat's price is a measured 469 ms/tick on this device, so the trade has to be priced
on real hardware, not inferred. This script runs the identical pipeline with and without it.

MEMORY IS THE REAL QUESTION. The describe stack already ran with only 827 MiB free before
SigLIP2 and the thinker were added. Each component is therefore loaded in the documented
load-bearing order (llama.cpp GGUFs FIRST, then torch/int8 perception) with a MemAvailable
reading after every step, so a failure names the exact component rather than "it OOMed".
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import threading
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")

MEM: List[Dict] = []


def _trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def mem(tag: str) -> float:
    with open("/proc/meminfo") as f:
        d = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in f)}
    avail = d["MemAvailable"] / 1024
    used = d["MemTotal"] / 1024 - avail
    delta = (MEM[-1]["avail"] - avail) if MEM else 0.0
    MEM.append({"tag": tag, "used": used, "avail": avail, "delta": delta})
    print(f"[mem] {tag:34s} used={used:7.0f} avail={avail:7.0f}  (+{delta:6.0f})", flush=True)
    return avail


def q_int8(module, device):
    module = module.to("cpu"); gc.collect(); _trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[warn] int8 skipped ({e!r})", flush=True)
    gc.collect(); _trim()
    return module.to(device)


def open_camera():
    import cv2
    # Pipeline notes (NVIDIA Jetson guidance + measured):
    #  * stay in memory:NVMM until the last possible element -- every hop to system RAM is
    #    a DMA copy. nvvidconv is the one unavoidable exit point for an OpenCV appsink.
    #  * `queue` between source and a slower consumer prevents NVMM buffer exhaustion
    #    (Argus "producer frame drop" when the consumer is slow to release buffers).
    #  * max-buffers=1 drop=1 sync=false => always the FRESHEST frame, never a queued
    #    backlog. max-buffers=2 lets one stale frame sit in front of the live one.
    #  * BGRx -> BGR is a pure channel drop, so do it in numpy instead of paying a CPU
    #    `videoconvert` per frame.
    gst = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,"
           "framerate=30/1 ! queue leaky=downstream max-size-buffers=2 ! "
           "nvvidconv ! video/x-raw,format=BGRx ! "
           "appsink drop=1 max-buffers=1 sync=false")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("CSI camera would not open via nvarguscamerasrc")
    for _ in range(5):
        ok, f = cap.read()
        if ok and max(float(f[:, :, c].std()) for c in range(3)) >= 8.0:
            return cap
    raise RuntimeError("camera returned only degenerate (flat) frames")


_CHARACTER_NAMES = re.compile(r"\b(finn|jake|princess bubblegum|marceline|bmo)\b", re.I)


def sanitize_thought(thought: str) -> str:
    """Strip Adventure Time character names out of the thinker's text before the speaker
    sees it. MEASURED, not cosmetic (scripts/jetson_chain_probe3.py, greedy decoding; the
    ONLY delta between these two runs was the thought's trailing two words):

        "...keep the mood bright for Finn and Jake."  -> "Finn, Jake - just how you like it!"
        (names stripped)                              -> "Finn, you look a bit tired - maybe
                                                          I can queue a gentle lullaby while
                                                          you rest."   <- GROUNDED

    The 350M speaker keys on the most recent salient ENTITY in its prompt, and the thinker
    habitually signs off with character names it learned from the v9 companion corpus. Those
    names are persona artifacts, not scene content, so to the speaker they are pure noise --
    and they hijack the entire line.

    WHAT THIS DOES NOT FIX: the speaker still ADDRESSES the user as "Finn", because that
    comes from its own LoRA corpus, not from its input. No prompt change removes it. Only
    the identity head can supply a real name -- and it is trained (TAR@FAR1% 0.765 on 884
    unseen identities) but NOT wired into build_bmo_stack or m5_streaming_loop. Until it is,
    the honest options are: no vocative at all, or a name from enrolment."""
    return re.sub(r"\s+", " ", _CHARACTER_NAMES.sub("them", thought)).strip()


def _prep(bgrx, rotate, gain, res, cv2):
    ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}
    bgr = bgrx[:, :, :3]                       # BGRx -> BGR: channel drop, no videoconvert
    if rotate in ROT:
        bgr = cv2.rotate(bgr, ROT[rotate])
    if gain != 1.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=gain, beta=0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; sq = min(h, w)
    rgb = rgb[(h - sq) // 2:(h - sq) // 2 + sq, (w - sq) // 2:(w - sq) // 2 + sq]
    return torch.from_numpy(cv2.resize(rgb, (res, res))).permute(2, 0, 1)


class CaptureThread(threading.Thread):
    """Producer/consumer ring buffer -- the fix for the 905 ms 'capture' leg.

    THE BUG THIS REPLACES: grab() read n_frames SYNCHRONOUSLY with `time.sleep(0.05)`
    between them. 16 frames x 50 ms = **800 ms of pure sleep**, which was ~88% of the
    measured 905 ms capture time. It was not the camera: a correctly-formed Jetson
    nvarguscamerasrc pipeline is 10-30 ms glass-to-glass.

    The sleep was not gratuitous, though, and deleting it alone would be WRONG: V-JEPA2 is
    handed true_window_dur_sec=10.0, and 16 back-to-back frames at 30 fps span 0.53 s.
    Claiming that is a 10 s window corrupts the temporal bins the model reads.

    The correct fix is to decouple capture from inference, which is also exactly the
    producer/consumer model NVIDIA's own libargus samples use: a background thread fills a
    deque continuously, and inference uniformly samples n_frames spanning the real window
    from whatever is buffered. Capture at inference time becomes a memory read (~0 ms) AND
    the 10 s temporal span becomes true rather than asserted.

    This mirrors models/m5_streaming_loop.py::RollingVideoBuffer, which production already
    uses -- so this makes the benchmark measure the deployed architecture instead of a
    synchronous stand-in."""

    def __init__(self, cap, window_sec: float, fps: float, rotate: int, gain: float,
                 res: int = 256, motion: bool = True):
        super().__init__(daemon=True)
        import cv2
        self._cv2 = cv2
        # ONE camera client serving three consumers. The CSI sensor is EXCLUSIVE -- with
        # face_engine/motion_tracker holding it, a second nvarguscamerasrc client fails with
        # "Failed to create CaptureSession" (and fails silently: isOpened() True, read()
        # False). So motion tracking has to live HERE, in the process that owns the camera,
        # or the eyes and perception can never run at the same time. Bonus: motion_tracker's
        # measured 133 MiB RSS is not spent.
        from models.m5_motion_crop import MotionCentroid
        self.motion = MotionCentroid() if motion else None
        self.cap = cap
        self.rotate, self.gain, self.res = rotate, gain, res
        self.fps = fps
        self.buf = deque(maxlen=max(4, int(window_sec * fps)))
        self.window_sec = window_sec
        self._stop_evt = threading.Event()
        self.n_read = 0

    def run(self) -> None:
        # Throttle to the BUFFER fill rate, not the sensor rate. The camera runs at 30 fps
        # but a 10 s window only needs enough frames to sample 16 from, and holding
        # 10 s x 30 fps = 300 frames of 256x256x3 costs ~59 MiB for no benefit -- which
        # matters when the end-of-run headroom is 170 MiB. StreamingConfig.video_fps = 6.4
        # (64 frames / 10 s) is the production figure; matching it cuts the buffer ~5x.
        #
        # NOTE this sleep is NOT the bug that was just removed. That one was on the
        # INFERENCE path (16 x 50 ms blocking every tick). This one is on a background
        # producer thread and blocks nothing -- it is what makes the window span real time.
        period = 1.0 / max(1e-6, self.fps)
        nxt = time.time()
        while not self._stop_evt.is_set():
            ok, f = self.cap.read()          # keeps the pipeline drained (max-buffers=1)
            if not ok:
                continue
            now = time.time()
            if now < nxt:
                continue                      # discard: fresher frame will arrive
            nxt = now + period
            frame = _prep(f, self.rotate, self.gain, self.res, self._cv2)
            if self.motion is not None:
                # (C,H,W) -> (H,W,C) for the differencer; also refreshes
                # /dev/shm/bmo_motion.txt so the face engine's eyes track while we run
                self.motion.push(frame.permute(1, 2, 0).numpy())
            self.buf.append(frame)
            self.n_read += 1

    def stop(self) -> None:
        self._stop_evt.set(); self.join(timeout=2.0)

    def get_window(self, n_frames: int):
        """Uniformly sample n_frames across the buffered window (same pattern as
        RollingVideoBuffer.get_window)."""
        frames = list(self.buf)
        if not frames:
            return None
        idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
        return torch.stack([frames[i] for i in idx], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-nat", action="store_true", help="load WavJEPA-nat (the 469 ms leg)")
    ap.add_argument("--query-ckpt", default="best.pt", help="under pipeline/checkpoints")
    ap.add_argument("--bank", default="candidates_siglip2.pt",
                    help="candidate set: candidates_siglip2.pt (1372 tags, 2 MiB) or "
                         "bank_runD_fp16.pt (121k captions, 227 MiB)")
    ap.add_argument("--query-vectors", default="query_vectors_siglip2.pt",
                    help="pre-encoded question vectors; with these the device needs NO text "
                         "encoder at all (models/text_target.py::PreEncodedTextSpace)")
    ap.add_argument("--text-mode", default="preencoded",
                    choices=["preencoded", "embeddinggemma"],
                    help="'preencoded' = no text encoder resident (the whole point). "
                         "'embeddinggemma' reproduces the 578 MiB configuration that OOM'd, "
                         "for A/B only.")
    ap.add_argument("--proj-from-ckpt", action="store_true", default=True,
                    help="apply the checkpoint's trained text proj to the candidate set at "
                         "load time (needed when the candidates are stored RAW, e.g. tags)")
    ap.add_argument("--identity-ckpt", default="",
                    help="e.g. head_joint.pt -- loads the identity head + JepaMemory and "
                         "measures BOTH its memory cost and its effect on what BMO says")
    ap.add_argument("--identity-threshold", type=float, default=0.5)
    ap.add_argument("--identity-max-age", type=float, default=30.0,
                    help="force an identity refresh after this long, even under continuous "
                         "motion (guards a silent person-swap)")
    ap.add_argument("--identity-bank-path", default=os.path.expanduser("~/bmo_identity.pt"),
                    help="persisted JepaMemory centroids -- WHO someone is. Without this, "
                         "profiles persist but nobody is ever recognised again.")
    ap.add_argument("--memory-path", default=os.path.expanduser("~/bmo_memory.json"),
                    help="persistent per-person profiles (models/bmo_memory.py)")
    ap.add_argument("--memory-char-budget", type=int, default=180,
                    help="hard cap on the memory line injected into the 512-token context")
    ap.add_argument("--identity-retry-unknown", type=float, default=8.0,
                    help="retry a weak/unknown identity sooner than a confident one")
    ap.add_argument("--thinker-gguf", default="bmo_thinker_qwen3_v5_Q8_0.gguf",
                    help="thinker GGUF. v5 = corpus v5c (0 cartoon) and the only version "
                         "that gets the uncertainty branch right; v6 regressed behaviourally "
                         "despite a better val_loss.")
    ap.add_argument("--fast-gguf", default="bmo_lfm25_350m_v5_Q8_0.gguf",
                    help="speaker GGUF under models_gguf/. v3 = trained on corpus v10c "
                         "(0 cartoon refs, name+perception categories)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--scene-frames", type=int, default=4)
    ap.add_argument("--rotate", type=int, default=90)
    ap.add_argument("--gain", type=float, default=1.8)
    ap.add_argument("--out", default=os.path.expanduser("~/jetson_core_pipeline_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    R: Dict = {"config": vars(args), "rounds": []}
    mem("00_start"); torch.cuda.init(); mem("01_cuda_ctx")

    # ---- STEP 1: llama.cpp GGUFs FIRST (load order is load-bearing on this device) ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, GGUFReasoningTier
    t0 = time.time()
    fast = GGUFFastTier(f"{PROD}/models_gguf/{args.fast_gguf}",
                        AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/lfm25_350m_tok"),
                        max_new_tokens=48, n_gpu_layers=-1)
    R["fast_load_s"] = time.time() - t0; mem("02_speaker_fast_tier")
    t0 = time.time()
    thinker = GGUFReasoningTier(f"{PROD}/models_gguf/{args.thinker_gguf}",
                                AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/qwen3_thinker_tok"),
                                n_gpu_layers=-1)
    R["thinker_load_s"] = time.time() - t0; mem("03_thinker")

    # ---- STEP 2: torch perception ----
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    vision = VisionEncoder(device="cpu"); vision.model = q_int8(vision.model, device)
    vision.device_str = str(device); mem("04_vitl_int8")
    wb = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wb.model = q_int8(wb.model, device); wb.device_str = str(device); mem("05_wavjepa_base")
    wn = None
    if args.use_nat:
        wn = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
        wn.model = q_int8(wn.model, device); wn.device_str = str(device); mem("06_wavjepa_nat")
    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg)
    m2.load_state_dict(torch.load(f"{PROD}/pipeline/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
                                  map_location="cpu", weights_only=False)["model"], strict=True)
    m2 = q_int8(m2, device).eval(); mem("07_m2")

    # ---- STEP 2b: identity head + memory ----
    # Marginal cost ONLY: it consumes the SAME ViT-L and WavJEPA token streams the
    # perception tick already produced, so no extra encoder loads. 8.15M params.
    ident = memory = None
    if args.identity_ckpt:
        from models.jepa_identity_head import IdentityHead, IdentityHeadConfig, stats_pool
        from models.jepa_memory import JepaMemory, MemoryConfig
        ick = torch.load(f"{PROD}/pipeline/checkpoints/{args.identity_ckpt}",
                         map_location="cpu", weights_only=False)
        ident = IdentityHead(IdentityHeadConfig(in_dims=ick["dims"], emb_dim=ick["emb_dim"])).to(device)
        ident.load_state_dict(ick["head"]); ident.eval()
        # BOTH halves must persist or memory is useless. Measured 2026-08-15: persisting
        # only the profiles meant run 2 started with an EMPTY identity bank, reported
        # `empty_memory`, and could not reach Alice's profile at all -- every person is a
        # stranger after a reboot, and the profile is only ever found by coincidence.
        #   JepaMemory  -> WHO this is   (identity centroids)      ~1 KB/person
        #   BmoMemory   -> WHAT we know  (name, facts, episodes)   ~0.4 KB/person
        if os.path.exists(args.identity_bank_path):
            memory = JepaMemory.load(args.identity_bank_path, device=str(device))
            print(f"[info] identity bank loaded: {len(memory)} enrolled "
                  f"from {args.identity_bank_path}", flush=True)
        else:
            memory = JepaMemory(MemoryConfig(threshold=args.identity_threshold))
            print("[info] identity bank EMPTY (cold start)", flush=True)
        from models.bmo_memory import BmoMemory
        bmem = BmoMemory(args.memory_path)
        print(f"[info] persistent memory: {bmem.stats()} from {args.memory_path}", flush=True)
        from models.m5_identity_schedule import IdentitySchedule
        sched = IdentitySchedule(max_age_s=args.identity_max_age,
                                 retry_unknown_s=args.identity_retry_unknown)
        R["identity"] = {"ckpt": args.identity_ckpt, "dims": ick["dims"],
                         "params_M": sum(p.numel() for p in ident.parameters()) / 1e6}
        mem("07b_identity_head")
        print(f"[info] identity head loaded: {R['identity']['params_M']:.2f}M params, "
              f"memory starts EMPTY (cold start = stranger)", flush=True)

    # ---- STEP 3: SigLIP2 scene encoder ----
    from transformers import AutoModel, AutoProcessor
    t0 = time.time()
    # Load on CPU, drop the text tower, THEN move to GPU. Ordering matters: doing
    # .to(device) first makes the GPU pay the full 716 MiB peak (both towers) before the
    # 538 MiB text half is freed, and the caching allocator does not necessarily hand that
    # back. This way the GPU never materialises it at all.
    sig = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16)
    # DROP THE TEXT TOWER. It is 282.3M params / 538 MiB -- 3x the vision tower's
    # 92.9M / 177 MiB, because of a 256k-token Gemma vocab embedding -- and on-device it is
    # dead weight: every question and every candidate was pre-encoded offline.
    # Verified numerically equivalent before doing this: vision_model(px).pooler_output vs
    # get_image_features(px) gives cosine 1.0, which matters because the cached scene
    # features (and therefore the trained predictor) were built with get_image_features.
    import gc as _gc
    if hasattr(sig, "text_model"):
        del sig.text_model
        _gc.collect()
    sig = sig.to(device).eval()
    torch.cuda.empty_cache()
    # NOT AutoProcessor: it costs 327-500 MiB for a resize+normalise (measured), which is
    # more than the vision tower itself. models/m5_motion_crop.py::siglip2_preprocess
    # reproduces it to EMBEDDING cosine 0.99993 in ~6 lines. See that docstring.
    from models.m5_motion_crop import siglip2_preprocess
    R["siglip_load_s"] = time.time() - t0; mem("08_siglip2_scene")

    # ---- STEP 4: query predictor + candidates ----
    # THE POINT OF THIS RUN: no text encoder is resident. EmbeddingGemma (578 MiB) is gone,
    # and SigLIP2's own text tower (538 MiB) never loads either -- only its VISION tower
    # (177 MiB) is needed, because every question and every candidate was pre-encoded
    # offline. The trained projection is applied here as a matmul, not by running a model.
    from models.m5_perception_query import load_perception_query_engine
    ckpt_path = f"{PROD}/pipeline/checkpoints/{args.query_ckpt}"
    if args.text_mode == "embeddinggemma":
        from models.text_target import TextTarget
        tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False,
                        device=str(device))
        mem("09_embeddinggemma")
    else:
        from models.text_target import PreEncodedTextSpace
        qv = torch.load(f"{PROD}/pipeline/checkpoints/{args.query_vectors}",
                        map_location="cpu", weights_only=False)
        tt = PreEncodedTextSpace(qv["text"], qv["emb"], device=str(device))
        mem("09_preencoded_queries_NO_TEXT_ENCODER")

    bank_path = f"{PROD}/pipeline/checkpoints/{args.bank}"
    bank_d = torch.load(bank_path, map_location="cpu", weights_only=False)
    b_emb = bank_d["emb"].float()
    _ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _tp = _ck.get("text_target_proj") or {}
    if args.proj_from_ckpt and _tp and b_emb.shape[1] == _tp["weight"].shape[1]:
        # candidates stored RAW -> project them into the learned space, offline
        b_emb = torch.nn.functional.normalize(
            b_emb @ _tp["weight"].float().t() + _tp["bias"].float(), dim=-1)
        print(f"[info] applied trained proj {tuple(_tp['weight'].shape)} to candidates",
              flush=True)
    del _ck
    eng = load_perception_query_engine(ckpt_path, tt, device,
                                       bank_emb_path=None,
                                       bank_captions=None,
                                       max_age_s=1e9,
                                       bank_emb=b_emb.to(torch.float16),
                                       bank_text=bank_d["text"])
    avail = mem("10_query_predictor+candidates")
    R["avail_after_load_MiB"] = avail
    R["streams"] = eng.source_names
    print(f"[info] streams={eng.source_names} bank={len(eng.bank_text)} nat={args.use_nat}", flush=True)

    class Ad:
        def compute_pre_pool(self, feats, tbins):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return m2.encode_pre_pool_tokens({k: v.float() for k, v in feats.items()}, tbins)
    ad = Ad()

    from models.world_state_builder import build_world_state_features
    cap = open_camera(); mem("11_camera")
    # Ring buffer warm-up. Fill one full window BEFORE round 1 so the frames actually span
    # 10 s -- which is what build_world_state_features is told (true_window_dur_sec=10.0).
    # Paid ONCE at startup, not per tick; in production the buffer is already warm.
    capture = CaptureThread(cap, window_sec=10.0, fps=6.4, rotate=args.rotate, gain=args.gain)
    capture.start()
    _t = time.time()
    while capture.n_read < args.frames and time.time() - _t < 12.0:
        time.sleep(0.1)
    print(f"[info] capture thread warm: {capture.n_read} frames in {time.time()-_t:.1f}s",
          flush=True)
    silent = torch.zeros(160000)          # no mic on this build; WavJEPA still does a real pass

    for r in range(args.rounds):
        print(f"\n───── round {r+1}/{args.rounds} ─────", flush=True)
        L = {}
        t0 = time.time(); frames = capture.get_window(args.frames)
        L["capture_ms"] = (time.time() - t0) * 1000
        if frames is None:
            raise RuntimeError("capture buffer empty")
        t0 = time.time()
        # wn is None without --use-nat => audio_mode="base", matching how qp_runD was
        # trained. Do NOT substitute wb here: it is 1-channel and would be handed the
        # 2-channel tensor nat expects.
        ws = build_world_state_features(frames, silent, 10.0, vision, wb, wn, 512, device)
        L["perception_ms"] = (time.time() - t0) * 1000
        srcs = {}
        if "vision" in eng.source_names:  srcs["vision"] = ws.feats["vision"].float()
        if "ambient" in eng.source_names: srcs["ambient"] = ws.feats["ambient"].float()
        if "m2" in eng.source_names:
            t0 = time.time(); srcs["m2"] = ad.compute_pre_pool(ws.feats, ws.tbins).float()
            L["m2_ms"] = (time.time() - t0) * 1000
        if "scene" in eng.source_names:
            t0 = time.time()
            idx = torch.linspace(0, frames.shape[0] - 1, args.scene_frames).long()
            imgs = [frames[i].permute(1, 2, 0).numpy() for i in idx]
            with torch.no_grad():
                from PIL import Image as _Im
                px = {"pixel_values": siglip2_preprocess(
                    [_Im.fromarray(a) for a in imgs]).to(device).to(torch.bfloat16)}
                o = sig.vision_model(**px)      # == get_image_features (cos 1.0), no text tower
                o = o.pooler_output if hasattr(o, "pooler_output") else o
                srcs["scene"] = F.normalize(o.float(), dim=-1).unsqueeze(0)
            L["siglip_ms"] = (time.time() - t0) * 1000
        eng.set_perception(srcs, None)
        t0 = time.time(); ans = eng.ask("Describe the room and setting in detail.")
        L["query_ms"] = (time.time() - t0) * 1000
        seen = ans.text if ans else "(nothing confident)"
        print(f"  [PERCEPTION] {seen[:150]}", flush=True)

        t0 = time.time()
        tr = thinker.generate(f"You can see: {seen} The user just walked in. What should you say "
                              f"and why?", {"energy": 0.6, "mood": "curious"})
        L["thinker_ms"] = (time.time() - t0) * 1000
        think = getattr(tr, "text", str(tr))
        print(f"  [THINKER]    {think[:180]}", flush=True)

        t0 = time.time()
        # ── IDENTITY: who am I talking to? ──
        who, conf, reason = None, 0.0, "disabled"
        if ident is not None:
            t0 = time.time()
            mo_now = capture.motion.last if capture.motion else None
            run, why = sched.should_run(bool(mo_now and mo_now.moving))
            if not run:
                # AMORTIZED: reuse the cached answer. The expensive path costs 218-239 ms
                # (a second V-JEPA2 forward on the crop); "who is here" does not change at
                # tick rate, so recomputing it every tick is pure redundancy.
                dec = sched.cached()
                who, conf, reason = dec.who, dec.confidence, f"{dec.reason}|cached({dec.age_s:.0f}s)"
                L["identity_ms"] = (time.time() - t0) * 1000
                L["identity_ran"] = 0
                print(f"  [IDENTITY]   {who or 'unknown'} (conf={conf:.3f}, {reason}) "
                      f"[skipped: {why}]", flush=True)
                R.setdefault("rounds_identity", []).append(
                    {"who": who, "conf": conf, "reason": reason, "ran": False, "why": why})
            else:
              with torch.no_grad():
                # The head was trained on VoxCeleb2 FACE crops, so a whole 1280x720 frame is
                # out of distribution. The motion centroid localises the person for free --
                # see models/m5_motion_crop.py for why this is a heuristic, not a detector.
                mo = mo_now
                if mo is not None and mo.moving:
                    from models.m5_motion_crop import crop_around
                    crop = crop_around(frames[frames.shape[0] // 2], mo.cx, mo.cy, out_size=256)
                    L["crop_cx"], L["crop_cy"] = round(mo.cx, 3), round(mo.cy, 3)
                    vfeat = vision.encode(crop.unsqueeze(0).unsqueeze(0).to(device))
                    idf = {"vision": stats_pool(vfeat),
                           "audio": stats_pool(ws.feats["ambient"])}
                else:
                    idf = {"vision": stats_pool(ws.feats["vision"]),
                           "audio": stats_pool(ws.feats["ambient"])}
                emb = ident(idf)
              who, conf, reason = memory.query(emb[0])
              sched.commit(who, conf, reason)
              L["identity_ms"] = (time.time() - t0) * 1000
              L["identity_ran"] = 1
              print(f"  [IDENTITY]   {who or 'unknown'} (conf={conf:.3f}, {reason}) "
                    f"[ran: {why}]", flush=True)
              R.setdefault("rounds_identity", []).append(
                  {"who": who, "conf": conf, "reason": reason, "ran": True, "why": why})

        clean = sanitize_thought(think)
        # THE CONTRACT. Recognised -> the speaker is TOLD the name and uses it. Not
        # recognised -> BMO introduces itself and asks, instead of inventing a name. That
        # second branch is the whole point: the head's honest 1%-FAR split on unseen people
        # is 51.5% recognised / 48.1% "I don't know you" / 0.4% wrong, so "I don't know you"
        # is the COMMON case and has to produce good behaviour, not a fallback stumble.
        mem_line = ""
        if ident is not None and who:
            t0m = time.time()
            mem_line = bmem.to_prompt_line(who, char_budget=args.memory_char_budget)
            L["memory_ms"] = (time.time() - t0m) * 1000     # expected ~0: it is a dict lookup
            if mem_line:
                print(f"  [MEMORY]     {mem_line}", flush=True)
        if ident is None:
            task = "Say one short friendly line to the person."
        elif who:
            task = ((mem_line + " ") if mem_line else "") + (
                    f"You recognise this person: their name is {who}. Greet {who} by name in "
                    f"one short friendly line.")
        else:
            task = ("You do NOT recognise this person and do not know their name. Do not "
                    "guess or use any name. Introduce yourself as BMO in one short friendly "
                    "line and ask what their name is.")
        fr = fast.generate(f"You see: {seen[:200]} Your thought: {clean[:200]} {task}",
                           {"energy": 0.6, "mood": "curious"})
        L["speaker_ms"] = (time.time() - t0) * 1000
        said = getattr(fr, "text", str(fr))
        print(f"  [BMO SAYS]   {said}", flush=True)
        L["total_ms"] = sum(v for v in L.values() if isinstance(v, float))
        print("  " + "  ".join(f"{k}={v:.0f}" for k, v in L.items()), flush=True)
        R["rounds"].append({"perception": seen, "thinker": think, "said": said,
                            "identity": {"who": who, "conf": conf, "reason": reason},
                            "latency": L})
        # Simulate the user answering "I'm Alice" after the first meeting, so the SAME run
        # exercises both branches: cold-start stranger, then recognised on later ticks.
        if ident is not None and who:
            bmem.ensure(who, who)
            bmem.note_encounter(who, summary=seen[:60], mood="curious")
        if ident is not None and who is None and r == 0 and "emb" in dir() and len(memory) == 0:
            memory.enroll(emb[0], "Alice")
            bmem.ensure("Alice", "Alice")
            # DO NOT write the perception tag as a FACT. Tried it (2026-08-15) and it
            # produced "You know Alice: was a closet." -> "Hi there, closet queen!". The tag
            # vocabulary describes the SCENE ("a closet", "dim lighting"), not the person,
            # and even person-ish tags ("a person sitting") are transient states rather than
            # durable attributes. Observations belong in the EPISODE ring, which
            # note_encounter() already writes. See models/bmo_memory.py.
            sched._stamp = 0.0        # enrolment changes the answer -> force a re-query
            print("  [ENROL]      stored this person as 'Alice' (simulating them answering)",
                  flush=True)

    capture.stop(); cap.release(); mem("12_end")
    if ident is not None:
        bmem.save()
        memory.save(args.identity_bank_path)
        R["memory"] = bmem.stats()
        R["identity_bank"] = len(memory)
        print(f"[info] identity bank saved: {len(memory)} enrolled -> "
              f"{args.identity_bank_path}", flush=True)
        print(f"[info] persistent memory saved: {bmem.stats()} -> {args.memory_path}", flush=True)
        R["identity_schedule"] = sched.stats()
        print(f"[info] identity schedule: {sched.stats()}", flush=True)
    R["mem_log"] = MEM
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
