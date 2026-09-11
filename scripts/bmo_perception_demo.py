#!/usr/bin/env python3
"""bmo_perception_demo.py -- four-stream perception, no autoregressive model in the loop.

    camera -> RollingVideoBuffer (10 s)
              |-> 16 frames @256  -> V-JEPA2 ViT-L (fpc64 ckpt)  [motion / what is happening]
              |->  8 frames @full -> SigLIP2 base-patch16-224     [scene / what things are]
    mic    -> RollingAudioBuffer (10 s) -> WavJEPA-base           [sound / what it sounds like]
                          |
            build_world_state_features -> M2 (AVJepaPredictor) -> World-State (1024-d)
                          |
            query predictor qp_runD, ALL FOUR sources -> retrieval vs candidates_siglip2_v3

ONE decode per round. The V-JEPA2 path resizes to 256; the SigLIP2 path hands frames to
AutoProcessor at buffer resolution so its own 224 resize is the only one -- double-resizing
would diverge from how the cached scene features were built (verified live-vs-cached at
worst cosine 0.999999).
"""
from __future__ import annotations

import hashlib, json, os, queue, re, sys, threading, time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bmo/bmo_production/pipeline")
sys.path.insert(0, "/home/bmo/bmo_production/scripts")

PROD = "/home/bmo/bmo_production/pipeline"
P = f"{PROD}/checkpoints"

# ── locked checkpoints; sha256 printed at startup and rendered on screen ──────
LOCKED = {
    "m2":     (f"{P}/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt", "e1a8231ec9fbae6c"),
    "qp":     (f"{P}/qp_runD.pt",                                          "85635302da4fd837"),
    "bank":   (f"{P}/candidates_siglip2_v3.pt",                            None),
    "qvec":   (f"{P}/query_vectors_siglip2_v2.pt",                         None),
}
VJEPA_REPO  = "facebook/vjepa2-vitl-fpc64-256"
SIGLIP_REPO = "google/siglip2-base-patch16-224"

WINDOW_SEC     = 10.0
VIDEO_FPS      = 6.4          # 64 frames / 10 s, matches CLIP_DURATION_S
N_VISION       = 16           # m5_streaming_loop.StreamingConfig.n_vision_frames
N_SCENE        = 8            # extract_siglip2_scene_vgg.py --frames default
BUF_W, BUF_H   = 448, 336     # >224 so SigLIP2's processor only ever downscales; 4:3 like the sensor
VJEPA_RES      = 256
AUDIO_SR       = 16000
MAX_TDM_BINS   = 512
ROT180         = True         # BMO's camera is mounted inverted (verified against a 4-way grid)
HISTORY        = 6

GST = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1640,height=1232,"
       "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
       "video/x-raw,format=BGR ! appsink drop=1 max-buffers=2")

# label -> (bank category, one of the 30 TRAINED query phrasings, column)
# The query predictor knows only a 3x2 intent grid; the CATEGORY RESTRICTION does the
# discriminating and the question only steers ranking within it (bmo_showcase.py:1305-1311).
QUESTIONS = [
    ("where",    "place",       "What does this place look like? Describe it fully.",     "PERSON"),
    ("objects",  "object",      "Tell me in detail what the surroundings look like.",     "PERSON"),
    ("lighting", "light",       "Describe the room and setting in detail.",               "PERSON"),
    ("who",      "people",      "Tell me in detail what the person is doing.",            "PERSON"),
    ("wearing",  "appearance",  "Describe the room and setting in detail.",               "PERSON"),
    ("view",     "camera",      "Briefly, what am I looking at?",                         "PERSON"),
    ("animal",   "animal",      "Summarize the scene in one sentence.",                   "PERSON"),
    ("doing",    "action",      "Explain everything that happens, in order.",             "ACTION"),
    ("posture",  "posture",     "Tell me in detail what the person is doing.",            "ACTION"),
    ("holding",  "held_object", "Give me a detailed account of what is being done.",      "PERSON"),
    ("looks",    "expression",  "Walk me through step by step what happens.",             "PERSON"),
    ("hearing",  "sound",       "What do you hear?",                                      "SOUND"),
]
# animal (5 tags) and camera (6 tags) are excluded from the RENDER, not the bank: with so
# few candidates a forced top-1 always asserts something, and both were observed firing
# falsely ("a cat on the sofa" with no cat). They still occupy their bank rows.
# Recalibrated IN THE DEMO ROOM, camera in its final position (98-round empty + 97-round
# live). The empty condition was the same room with NO PERSON, so these floors separate
# PERSON-PRESENT from PERSON-ABSENT -- not scene-present from scene-absent. That is why
# where/lighting/view have negative gaps and are dropped: the room is equally there in both.
# The demo claim this supports is "step out of frame and PERSON goes silent" -- the system
# reporting whether a person is there, not whether the sensor is blocked.
RENDER_EXCLUDE = {"animal", "view", "objects", "hearing", "where", "lighting",
                  "posture", "who", "looks"}
# Floors from the 92-round empty-room calibration: live p05 vs empty p95 left a clean gap
# for these only. Everything else overlapped or inverted, so nothing else is gated -- an
# ungated field makes no silence claim. SOUND is cut entirely: two sessions showed the
# ambient embedding does not track audio energy (r=+0.048 in-case, r=-0.029 with nat), so
# nothing on that stream can be displayed honestly.
# floor = midpoint of (empty p95, live p05). Only categories with a real gap are gated.
# doing/who/objects had gaps of 0.006-0.010 -- real but too thin to hold on camera, so
# `doing` is shown UNGATED (score, no silence claim) and who/objects are not shown.
# Recalibrated on the REAL demo condition (60-round step-in/out run), not the blank wall.
# A blank wall is a much stronger "nothing" than an empty room, and it inflated `looks`'
# apparent separation: blank-wall p95 0.0176 vs person-absent-room p95 0.1335. Measured on
# the condition we actually demo, `looks` has a NEGATIVE gap and is dropped.
# GATING CRITERION: a gap is necessary but NOT sufficient. A field is gated only if its gap
# is wider than its round-to-round noise, i.e. stable enough to hold across consecutive
# rounds under continuous conditions. `doing` HAS a measured gap (0.0229, larger than
# `wearing`'s 0.0147) but that gap sits INSIDE its own noise band: gated, it flickered
# —/present/— on consecutive rounds with the subject sitting still in frame, and cut a
# correct answer ("someone is watching a screen" at +0.1494). It is therefore shown
# ungated -- score, no silence claim. Gap size alone is the wrong criterion.
FLOORS = {"wearing": 0.1519, "holding": 0.1143}
GAPS = {"wearing": (0.1446, 0.1593), "holding": (0.0900, 0.1385)}
UNGATED_MEASURED = {"doing": (0.1489, 0.1718)}   # gap 0.0229, measured but not gated
COLUMNS = ["PERSON", "ACTION"]
COL_SRC = {"PERSON": "gated", "ACTION": "ungated"}
# --no-gate shows ALL 12 categories with raw scores and no floors. Used OFF-SITE: the floors
# are absent-p95/present-p05 measured in ONE room under ONE light and have no basis anywhere
# else. Showing them ungated is the honest option, and is itself a finding worth narrating.
NOGATE_COLUMNS = ["SCENE", "PERSON", "ACTION", "SOUND"]
NOGATE_COL = {
    "where": "SCENE", "lighting": "SCENE", "objects": "SCENE", "view": "SCENE",
    "animal": "SCENE", "who": "PERSON", "wearing": "PERSON", "holding": "PERSON",
    "looks": "PERSON", "doing": "ACTION", "posture": "ACTION", "hearing": "SOUND",
}
NOGATE_SRC = {"SCENE": "SigLIP2", "PERSON": "SigLIP2+M2", "ACTION": "V-JEPA2", "SOUND": "WavJEPA"}


def sha16(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def mem_avail() -> int:
    return int(os.popen("free -m").read().splitlines()[1].split()[6])


def mem_used() -> int:
    return int(os.popen("free -m").read().splitlines()[1].split()[2])


# ── buffers ──────────────────────────────────────────────────────────────────
class RollingVideoBuffer:
    """Frames at BUF_W x BUF_H. get_window(n) uniformly samples across the FULL buffered
    window with np.linspace -- same pattern as m5_streaming_loop.RollingVideoBuffer, so a
    partially filled buffer still yields a valid input rather than an error."""

    def __init__(self, window_sec: float, fps: float):
        self.max_frames = int(window_sec * fps)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self._stamps: Deque[float] = deque(maxlen=64)

    def fps_est(self) -> float:
        """Actual achieved capture rate. In low light the CSI sensor lengthens exposure and
        the real framerate can fall below the requested 6.4 fps, starving the window -- so it
        is measured, not assumed."""
        with self._lock:
            st = list(self._stamps)
        if len(st) < 2:
            return 0.0
        return (len(st) - 1) / max(1e-6, st[-1] - st[0])

    def push(self, f: np.ndarray) -> None:
        with self._lock:
            self._buf.append(f)
            self._stamps.append(time.time())

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def get_window(self, n: int) -> Optional[List[np.ndarray]]:
        with self._lock:
            fr = list(self._buf)
        if not fr:
            return None
        idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
        return [fr[i] for i in idx]


class RollingAudioBuffer:
    def __init__(self, window_sec: float, sr: int):
        self.max_samples = int(window_sec * sr)
        self._buf: Deque[np.ndarray] = deque()
        self._n = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buf.append(chunk)
            self._n += len(chunk)
            while self._buf and self._n - len(self._buf[0]) >= self.max_samples:
                self._n -= len(self._buf.popleft())

    def get_window(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._n == 0:
                return None
            full = np.concatenate(list(self._buf))
        return full[-self.max_samples:]


# ── the perception engine ────────────────────────────────────────────────────
class Perception:
    def __init__(self, log, audio_mode="base"):
        self.audio_mode = audio_mode
        self.log = log
        self.stage_ms: Dict[str, List[float]] = {}
        from bmo_jetson_startup import q_int8_cpu_then_move
        self.q8 = q_int8_cpu_then_move
        dev = torch.device("cuda")
        self.dev = dev

        a0 = mem_avail()
        _t0 = time.perf_counter(); _tl = [_t0]
        def _lap(name):
            now = time.perf_counter()
            d = now - _tl[0]; _tl[0] = now
            log(f"    [load] {name:<22} {d:6.2f} s   (cumulative {now - _t0:6.2f} s)")
        from models.vision_encoder import VisionEncoder
        self.ve = VisionEncoder(device="cpu", dtype=torch.bfloat16)
        self.ve.model = self.q8(self.ve.model, dev); self.ve.device_str = "cuda"
        log(f"  vision  V-JEPA2 ViT-L int8   avail={mem_avail()} MiB")
        _lap("V-JEPA2 ViT-L int8")

        from transformers import AutoModel, AutoProcessor
        sig = AutoModel.from_pretrained(SIGLIP_REPO, dtype=torch.bfloat16)
        if hasattr(sig, "text_model"):
            del sig.text_model          # queries are pre-encoded; text tower is dead weight
        import gc; gc.collect()
        self.sig = sig.to(dev).eval()
        self.sig_proc = AutoProcessor.from_pretrained(SIGLIP_REPO)
        log(f"  scene   SigLIP2 base bf16    avail={mem_avail()} MiB")
        _lap("SigLIP2 base bf16")

        from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
        self.wj = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
        self.wj.model = self.q8(self.wj.model, dev); self.wj.device_str = "cuda"
        self.wj_nat = None
        if audio_mode == "mean":
            # M2's TRAINING default is base+nat averaged. nat is fed DUPLICATED MONO inside
            # world_state_builder (never real multichannel) -- that is what M2 saw, and
            # world_state_builder:158-160 is explicit that feeding real stereo would be a
            # different distribution from training.
            self.wj_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
            self.wj_nat.model = self.q8(self.wj_nat.model, dev); self.wj_nat.device_str = "cuda"
            log(f"  ambient WavJEPA base+nat int8 avail={mem_avail()} MiB  (audio_mode=mean, M2 training default)")
        else:
            log(f"  ambient WavJEPA-base int8    avail={mem_avail()} MiB   (nat NOT loaded)")
        _lap("WavJEPA int8")

        from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
        m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                          max_tdm_bins=MAX_TDM_BINS, dropout=0.0))
        ck = torch.load(LOCKED["m2"][0], map_location="cpu", weights_only=False)
        m2.load_state_dict(ck["model"], strict=True); del ck; gc.collect()
        m2 = m2.to(torch.bfloat16)
        self.m2 = self.q8(m2, dev); self.m2.eval()
        log(f"  m2      AVJepaPredictor int8 avail={mem_avail()} MiB")
        _lap("M2 predictor int8")

        from models.m5_perception_query import load_perception_query_engine
        from models.text_target import PreEncodedTextSpace
        qck = torch.load(LOCKED["qp"][0], map_location="cpu", weights_only=False)
        qv = torch.load(LOCKED["qvec"][0], map_location="cpu", weights_only=False)
        tt = PreEncodedTextSpace(qv["text"], qv["emb"], device=str(dev))
        cand = torch.load(LOCKED["bank"][0], map_location="cpu", weights_only=False)
        raw = F.normalize(cand["emb"].float(), dim=-1).to(dev)
        self.bank_raw = raw                                  # PANEL B: SigLIP2 joint space
        tp = qck.get("text_target_proj") or {}
        bank = (F.normalize(raw @ tp["weight"].float().to(dev).t()
                            + tp["bias"].float().to(dev), dim=-1) if tp else raw)
        self.pq = load_perception_query_engine(LOCKED["qp"][0], tt, dev,
                                               bank_emb=bank, bank_text=cand["text"],
                                               max_age_s=1e9)
        self.pq.bank_category = cand.get("category", ["mined"] * len(cand["text"]))
        self.bank_text = cand["text"]
        self.pq.qp = self.q8(self.pq.qp, dev); self.pq.qp.eval()
        del qck, cand, raw, tp; gc.collect(); torch.cuda.empty_cache()
        log(f"  qp      qp_runD int8 + bank  avail={mem_avail()} MiB  "
            f"({len(self.bank_text)} tags, {len(set(self.pq.bank_category))} categories)")
        _lap("query predictor + bank")
        log(f"  TOTAL perception footprint  {a0 - mem_avail()} MiB")
        log(f"    [load] TOTAL MODEL LOAD        {time.perf_counter() - _t0:6.2f} s")

        self.cat_idx = {}
        for _, cat, _, _ in QUESTIONS:
            self.cat_idx[cat] = torch.as_tensor(
                [i for i, c in enumerate(self.pq.bank_category) if c == cat], device=dev)

        # Time the encoders where they are ACTUALLY called (inside world_state_builder),
        # rather than running a second forward pass just to measure -- that would double the
        # ViT-L cost and misreport the round.
        def _wrap(obj, meth, key):
            orig = getattr(obj, meth)
            def timed(*aa, **kk):
                t0 = time.perf_counter()
                out = orig(*aa, **kk)
                torch.cuda.synchronize()
                self.stage_ms.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)
                return out
            setattr(obj, meth, timed)
        # stash encoder outputs so the probe can compare base-only vs base+nat on the
        # IDENTICAL audio window -- no second forward pass, no second noise session
        self.enc_out = {}
        def _wrap_stash(obj, meth, key, stash):
            orig = getattr(obj, meth)
            def timed(*aa, **kk):
                t0 = time.perf_counter()
                out = orig(*aa, **kk)
                torch.cuda.synchronize()
                self.stage_ms.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)
                self.enc_out[stash] = out
                return out
            setattr(obj, meth, timed)
        _wrap(self.ve, "encode", "vjepa2_16f")
        _wrap_stash(self.wj, "encode", "wavjepa_10s", "base")
        if self.wj_nat is not None:
            _wrap_stash(self.wj_nat, "encode", "wavjepa_nat_10s", "nat")

        from models.world_state_builder import build_world_state_features, assert_staircase
        self._build_ws = build_world_state_features
        self._assert_staircase = assert_staircase
        self.q_emb_cache: Dict[str, torch.Tensor] = {}
        self.debug = False          # flipped by the 'd' key; when False nothing is stashed
        self.view_only = False      # --record-view: stash the 8 scene frames only
        self.dbg: Dict = {}

        class _M2Adapter:
            def __init__(self, m2):
                self.m2 = m2
            def compute_pre_pool(self, feats, tbins):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.m2.encode_pre_pool_tokens(
                        {k: v.float() for k, v in feats.items()}, tbins)
        self.adapter = _M2Adapter(self.m2)

    def _t(self, name, t0):
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        self.stage_ms.setdefault(name, []).append(dt)
        return dt

    @torch.no_grad()
    def _scene(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Replicates scripts/extract_siglip2_scene_vgg.py:126-135 exactly.
        Frames arrive at buffer resolution; AutoProcessor does the ONLY resize."""
        imgs = [f[:, :, ::-1] for f in frames]                       # BGR -> RGB, HWC uint8
        px = self.sig_proc(images=imgs, return_tensors="pt").to(self.dev)
        if self.debug:
            # POST-resize pixel values exactly as the model receives them, de-normalised
            # back to viewable RGB. This is the double-resize fix made visible.
            pv = px["pixel_values"].detach().float().cpu()
            ip = getattr(self.sig_proc, "image_processor", self.sig_proc)
            mean = torch.tensor(getattr(ip, "image_mean", [0.5] * 3)).view(1, 3, 1, 1)
            std = torch.tensor(getattr(ip, "image_std", [0.5] * 3)).view(1, 3, 1, 1)
            self.dbg["scene_px"] = ((pv * std + mean).clamp(0, 1) * 255).to(torch.uint8)
        px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
              for k, v in px.items()}
        o = self.sig.get_image_features(**px)
        o = o.pooler_output if hasattr(o, "pooler_output") else o
        return F.normalize(o.float(), dim=-1)                        # (K, 768)

    @torch.no_grad()
    def round(self, vbuf: RollingVideoBuffer, abuf: RollingAudioBuffer) -> Optional[Dict]:
        import cv2
        t_round = time.perf_counter()

        t0 = time.perf_counter()
        fr_v = vbuf.get_window(N_VISION)
        fr_s = vbuf.get_window(N_SCENE)
        wav = abuf.get_window()
        if fr_v is None or wav is None:
            return None
        vid = np.stack([cv2.resize(f, (VJEPA_RES, VJEPA_RES)) for f in fr_v])   # (16,H,W,3) BGR
        vid = torch.from_numpy(vid[:, :, :, ::-1].copy()).permute(0, 3, 1, 2)   # (16,3,256,256) uint8 RGB
        audio = torch.from_numpy(wav.astype(np.float32))
        ms_decode = self._t("decode", t0)
        if self.debug and not self.view_only:
            # the REAL 256x256 RGB tensor V-JEPA2 receives, not a pre-resize buffer frame
            self.dbg["vjepa_in"] = vid.permute(0, 2, 3, 1).contiguous().numpy().copy()
            self.dbg["vis_frames"] = [f.copy() for f in fr_v]     # the 16 handed to V-JEPA2
            self.dbg["scene_frames"] = [f.copy() for f in fr_s]
            self.dbg["rms"] = float(np.sqrt(np.mean(wav[-16000:] ** 2)))

        t0 = time.perf_counter()
        scene = self._scene(fr_s).unsqueeze(0)                                  # (1,8,768)
        ms_siglip = self._t("siglip2_8f", t0)

        t0 = time.perf_counter()
        ws = self._build_ws(vid, audio, WINDOW_SEC, self.ve, self.wj, self.wj_nat,
                            MAX_TDM_BINS, self.dev)
        ms_build = self._t("world_state_build", t0)
        ms_vjepa = self.stage_ms["vjepa2_16f"][-1]      # recorded by the wrapper, real call
        ms_wavjepa = self.stage_ms["wavjepa_10s"][-1]
        if "wavjepa_nat_10s" in self.stage_ms:
            ms_wavjepa += self.stage_ms["wavjepa_nat_10s"][-1]

        n_temp = ws.vision_raw_ntok // 256
        self._assert_staircase(ws.tbins["vision"][0].cpu(), MAX_TDM_BINS, n_temp=n_temp, n_spat=16)

        t0 = time.perf_counter()
        self.pq.update_from_features(ws.feats, ws.tbins, self.adapter, scene=scene)
        ms_m2 = self._t("m2_fusion", t0)

        # ── PANEL A: query-conditioned, top-1 per bank category ──
        t0 = time.perf_counter()
        eng = self.pq
        panelA = {}
        for label, cat, q, col in QUESTIONS:
            ids = self.cat_idx[cat]
            if ids.numel() == 0:
                continue
            if q not in self.q_emb_cache:
                self.q_emb_cache[q] = eng.tt.encode_text_frozen_raw([q]).to(eng.device)
            z_q = eng.qp(eng._sources, self.q_emb_cache[q], eng._masks)
            sims = (z_q.to(eng.bank_emb.dtype) @ eng.bank_emb[ids].T)[0].float()
            b = int(torch.argmax(sims))
            panelA[label] = (self.bank_text[int(ids[b])], float(sims[b]), col)
            if self.debug:
                k = min(5, sims.numel())
                tv, ti = torch.topk(sims, k)
                self.dbg.setdefault("top5", {})[label] = [
                    (self.bank_text[int(ids[j])], float(v)) for v, j in zip(tv, ti)]
        ms_qp = self._t("query_predictor", t0)

        if self.debug:
            m2t = eng._sources["m2"]                       # (1, N, 1024) World-State tokens
            ws_vec = m2t[0].float().mean(0)                # pooled 1024-d World-State
            self.dbg["shapes"] = {
                "vision": tuple(ws.feats["vision"].shape), "n_temp": n_temp,
                "ambient": tuple(ws.feats["ambient"].shape),
                "scene": tuple(scene.shape[1:]),
                "m2_tokens": tuple(m2t.shape),
                "ws_mean": float(ws_vec.mean()), "ws_std": float(ws_vec.std()),
                "ws_l2": float(ws_vec.norm()),
            }

        # ── PANEL B: UNCONDITIONED baseline. Mean-pooled SigLIP2 scene vector vs the SAME
        #    bank in its raw joint space. No query, no query predictor. ──
        t0 = time.perf_counter()
        pooled = F.normalize(scene[0].mean(0, keepdim=True), dim=-1)
        sims_b = (pooled @ self.bank_raw.T)[0]
        v, i = torch.topk(sims_b, 5)
        panelB = [(self.bank_text[int(j)], float(s)) for s, j in zip(v, i)]
        ms_retr = self._t("retrieval_uncond", t0)

        total = (time.perf_counter() - t_round) * 1000.0
        self.stage_ms.setdefault("ROUND", []).append(total)
        amb = ws.feats["ambient"]                      # (1, T, 768) straight from WavJEPA
        amb_vec = amb[0].float().mean(0).cpu()          # (768,) pooled, pre-M2, pre-qp
        # both arms from the SAME audio window
        bv = self.enc_out["base"][0].float().mean(0).cpu()
        mv = bv
        if "nat" in self.enc_out:
            nv = self.enc_out["nat"][0].float()
            bf = self.enc_out["base"][0].float()
            if bf.shape[0] == nv.shape[0]:
                mv = ((bf + nv) * 0.5).mean(0).cpu()
        return {
            "t": time.time(), "A": panelA, "B": panelB, "amb_vec": amb_vec,
            "amb_base": bv, "amb_mean": mv,
            "lat": {"decode": ms_decode, "vjepa2_16f": ms_vjepa, "siglip2_8f": ms_siglip,
                    "wavjepa_10s": ms_wavjepa, "m2_fusion": ms_m2,
                    "query_predictor": ms_qp, "retrieval_uncond": ms_retr, "ROUND": total},
            "n_temp": n_temp, "vision_ntok": ws.vision_raw_ntok, "ambient_ntok": ws.ambient_ntok,
            "buf_depth": len(vbuf), "buf_max": vbuf.max_frames, "cap_fps": vbuf.fps_est(),
            "dbg": dict(self.dbg) if self.debug else None,
        }


# ── display ──────────────────────────────────────────────────────────────────
CLR = "\033[2J\033[H"
DIM, B, R, G, Y, C, M, RST = ("\033[2m", "\033[1m", "\033[31m", "\033[32m",
                              "\033[33m", "\033[36m", "\033[35m", "\033[0m")
HL = "\033[1;33m"          # changed since last refresh


# Every bank tag is a full sentence ("a person holding a mug") but the row label already
# says "holding", so the prefix is redundant AND it was consuming the width that made tags
# truncate mid-word. Stripped for DISPLAY only -- the literal bank string is what gets
# scored, and the debug view's top-5 still prints it in full.
_PREFIX = {
    "wearing":  ["a person wearing ", "a person in ", "a person with "],
    "holding":  ["a person holding "],
    "looks":    ["a person looking ", "a person with ", "a person "],
    "who":      ["a person ", "an "],
    "posture":  ["a person "],
    "doing":    ["someone is "],
}


def _short(label: str, txt: str) -> str:
    for pre in _PREFIX.get(label, []):
        if txt.startswith(pre):
            return txt[len(pre):] or txt
    return txt


def _fit(s: str, n: int) -> str:
    s = s if len(s) <= n else s[: n - 1] + "…"
    return s.ljust(n)


def render(state: Dict) -> str:
    rounds: List[Dict] = state["rounds"]
    prov: Dict = state["prov"]
    out = [CLR]
    W = 118
    out.append(B + C + "  BMO PERCEPTION  " + RST + DIM +
               "│  four streams → M2 world-state → query predictor → tag retrieval" + RST)
    out.append(B + G + "  NO AUTOREGRESSIVE MODEL IN THIS LOOP" + RST + DIM +
               "   │  no LLM, no TTS, no STT — retrieval only" + RST)
    if state.get("nogate"):
        out.append(B + Y + "  FLOORS DISABLED — thresholds are room-calibrated" + RST + DIM +
                   "   every category shown with its raw score; no silence claim anywhere" + RST)
    else:
        out.append(DIM + "  PERSON is gated — below its floor (calibrated in THIS room) a field "
                         "shows — rather than guess.  ACTION is ungated: score only, "
                         "no silence claim." + RST)
    out.append(DIM + "  " + "─" * W + RST)

    if not rounds:
        out.append("\n  " + Y + "warming up — filling the 10 s rolling buffers…" + RST + "\n")
        return "\n".join(out)

    cur = rounds[-1]
    prev = rounds[-2] if len(rounds) > 1 else None
    nogate = state.get("nogate", False)
    cols = NOGATE_COLUMNS if nogate else COLUMNS
    colw = 56

    hdr = "  "
    for col in cols:
        lbl = f"{col}  ({NOGATE_SRC[col]})" if nogate else f"{col}  ({COL_SRC[col]})"
        hdr += B + _fit(lbl, colw) + RST
    out.append(hdr)
    out.append(DIM + "  " + "─" * W + RST)

    floor = state.get("floor")
    need = state.get("stability", 0)
    run: Dict[str, int] = {}
    if need:
        for label, _, _, _ in QUESTIONS:
            n = 0
            for r in reversed(rounds):
                if label in r["A"] and r["A"][label][0] == cur["A"].get(label, (None,))[0]:
                    n += 1
                else:
                    break
            run[label] = n
    rows: Dict[str, List[str]] = {c: [] for c in cols}
    for label, cat, q, col in QUESTIONS:
        if label not in cur["A"]:
            continue
        if nogate:
            col = NOGATE_COL.get(label, "SCENE")
        elif label in RENDER_EXCLUDE:
            continue
        txt, score, _ = cur["A"][label]
        txt = _short(label, txt)
        cat_floor = None if nogate else FLOORS.get(label)
        below = (cat_floor is not None and score < cat_floor) or \
                (floor is not None and score < floor) or \
                (need and run.get(label, 0) < need)
        if below:
            # SILENCE is the demonstration: a category with nothing to say says nothing.
            rows[col].append(f"  {DIM}{label:<9}{RST}{DIM}{_fit('—', colw - 19)}{RST}"
                             f"{DIM}{score:+.3f}{RST}")
            continue
        changed = prev is not None and label in prev["A"] and prev["A"][label][0] != txt
        mark = HL + "● " + RST if changed else "  "
        body = (HL if changed else "") + _fit(txt, colw - 19) + (RST if changed else "")
        rows[col].append(f"{mark}{DIM}{label:<9}{RST}{body}{DIM}{score:+.3f}{RST}")

    for i in range(max(len(v) for v in rows.values())):
        line = "  "
        for col in cols:
            line += (rows[col][i] if i < len(rows[col]) else " " * colw)
            line += "  "
        out.append(line)

    out.append("")
    out.append(DIM + "  HISTORY — last %d refreshes (● = changed)" % HISTORY + RST)
    for r in rounds[-HISTORY:][::-1]:
        ts = time.strftime("%H:%M:%S", time.localtime(r["t"]))
        bits = []
        for lb in ("where", "lighting", "doing"):
            if lb in r["A"]:
                t_, sc_, _ = r["A"][lb]
                t_ = _short(lb, t_)
                _cf = FLOORS.get(lb)
                shown = "—" if (_cf is not None and sc_ < _cf) else _fit(t_, 24).strip()
                bits.append(f"{DIM}{lb}={RST}{shown}")
        out.append(f"  {DIM}{ts}{RST}  " + f"{DIM}│{RST} ".join(bits) +
                   f"   {DIM}{r['lat']['ROUND']:.0f} ms{RST}")

    out.append("")
    out.append(DIM + "  PANEL B — UNCONDITIONED baseline: SigLIP2 scene vector → same bank, "
                     "no query conditioning, top-5" + RST)
    for txt, s in cur["B"]:
        out.append(f"    {DIM}{s:+.3f}{RST}  {txt}")

    lat = cur["lat"]
    out.append("")
    out.append(DIM + "  " + "─" * W + RST)
    out.append("  " + B + "LATENCY  " + RST +
               f"{DIM}decode{RST} {lat['decode']:.0f} {DIM}│ V-JEPA2 16f{RST} {lat['vjepa2_16f']:.0f} "
               f"{DIM}│ SigLIP2 8f{RST} {lat['siglip2_8f']:.0f} {DIM}│ WavJEPA{RST} {lat['wavjepa_10s']:.0f} "
               f"{DIM}│ M2{RST} {lat['m2_fusion']:.0f} {DIM}│ QP{RST} {lat['query_predictor']:.0f} "
               f"{DIM}│ retr{RST} {lat['retrieval_uncond']:.0f}   "
               + B + f"ROUND {lat['ROUND']:.0f} ms = {1000.0/lat['ROUND']:.2f} Hz" + RST)
    _errs = state.get("errors", 0)
    if _errs:
        out.append("  " + Y + f"⚠ {_errs} round(s) failed and were skipped — display continued" + RST)
    out.append("  " + B + "MEMORY   " + RST +
               f"used {state['mem_used']} / 7620 MiB   peak {state['mem_peak']} MiB   "
               f"{DIM}vision_tok={cur['vision_ntok']} n_temp={cur['n_temp']} "
               f"ambient_tok={cur['ambient_ntok']}{RST}")
    _bd = cur.get("buf_depth"); _cf = cur.get("cap_fps") or 0.0
    if _bd is not None:
        _warn = _bd < cur.get("buf_max", 64) or _cf < 5.5
        out.append("  " + B + "CAPTURE  " + RST + (Y if _warn else "") +
                   f"buffer {_bd}/{cur.get('buf_max')} frames   camera {_cf:.2f} fps" +
                   (RST if _warn else "") +
                   DIM + "   (target 6.40 — low light lengthens exposure)" + RST)
    out.append("  " + DIM + "press 'd' for the debug view (frames · shapes · full ranking)" + RST)
    out.append("  " + DIM + f"m2 {prov['m2']}  │  qp {prov['qp']}  │  bank {prov['bank_name']} "
                            f"({prov['n_tags']} tags / {prov['n_cats']} cats)  │  "
                            f"{prov['device']}  {prov['power']}" + RST)
    return "\n".join(out)


def _thumb(img, w=26, h=13):
    """One frame as ANSI half-blocks: each char cell is two vertically stacked pixels
    (fg = upper, bg = lower), so a h-row strip shows 2h pixel rows."""
    import cv2
    sm = cv2.resize(img, (w, h * 2))
    out = []
    for r in range(h):
        line = ""
        for c in range(w):
            t = sm[2 * r, c]; b = sm[2 * r + 1, c]
            line += ("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm\u2580"
                     % (t[2], t[1], t[0], b[2], b[1], b[0]))
        out.append(line + RST)
    return out


def _strip(frames, label, per_row=4, w=26, h=13, tags=None):
    """A row-wrapped strip of thumbnails with an index/timestamp caption under each."""
    lines = [DIM + "  " + label + RST]
    for r0 in range(0, len(frames), per_row):
        chunk = frames[r0:r0 + per_row]
        thumbs = [_thumb(f, w, h) for f in chunk]
        for row in range(h):
            lines.append("  " + " ".join(t[row] for t in thumbs))
        cap = "  "
        for i, _ in enumerate(chunk):
            txt = tags[r0 + i] if tags else "f%d" % (r0 + i)
            cap += txt.ljust(w + 1)
        lines.append(DIM + cap + RST)
    return lines


def render_debug(state: Dict) -> str:
    rounds = state["rounds"]
    out = [CLR, B + C + "  DEBUG VIEW" + RST + DIM +
           "   what the models actually receive   ·   press 'd' to return" + RST, ""]
    if not rounds or not rounds[-1].get("dbg"):
        out.append("  " + Y + "waiting for a round with debug enabled…" + RST)
        return "\n".join(out)
    d = rounds[-1]["dbg"]
    ts = time.strftime("%H:%M:%S", time.localtime(rounds[-1]["t"]))
    win = WINDOW_SEC
    if "vis_frames" in d:
        n = len(d["vis_frames"])
        caps = ["-%.1fs" % (win - i * win / max(1, n - 1)) for i in range(n)]
        out += _strip(d["vis_frames"], "VISION — the %d frames handed to V-JEPA2 "
                      "(linspace over the %.0f s window, @%d before resize to %d)"
                      % (n, win, BUF_W, VJEPA_RES), tags=caps)
        out.append("")
    if "scene_px" in d:
        pv = d["scene_px"]
        fr = [pv[i].permute(1, 2, 0).numpy()[:, :, ::-1] for i in range(pv.shape[0])]
        caps = ["%dx%d" % (pv.shape[3], pv.shape[2])] * pv.shape[0]
        out += _strip(fr, "SCENE — the %d frames as SigLIP2's AutoProcessor delivers them "
                      "(POST-resize, de-normalised; the ONLY resize on this path)"
                      % pv.shape[0], tags=caps)
        out.append("")
    sh = d.get("shapes", {})
    if sh:
        out.append(DIM + "  SHAPES + STATS  " + ts + RST)
        out.append("    vision   %-22s n_temp=%-3d" % (sh["vision"], sh["n_temp"]))
        out.append("    ambient  %-22s rms=%.5f" % (sh["ambient"], d.get("rms", 0.0)))
        out.append("    scene    %-22s" % (sh["scene"],))
        out.append("    m2 toks  %-22s" % (sh["m2_tokens"],))
        out.append("    " + B + "World-State  mean=%+.4f  std=%.4f  L2=%.3f" %
                   (sh["ws_mean"], sh["ws_std"], sh["ws_l2"]) + RST)
        out.append("")
    t5 = d.get("top5", {})
    if t5:
        out.append(DIM + "  FULL RANKING — top-5 per displayed category "
                         "(floor shown where one applies)" + RST)
        for label, _, _, _ in QUESTIONS:
            if label in RENDER_EXCLUDE or label not in t5:
                continue
            fl = FLOORS.get(label)
            out.append("    " + B + label + RST +
                       (DIM + "   floor %.4f" % fl + RST if fl else DIM + "   ungated" + RST))
            for i, (txt, sc) in enumerate(t5[label]):
                cut = fl is not None and sc < fl
                mark = (R + " below floor" + RST) if (cut and i == 0) else ""
                out.append("      %d. %s%-42s%s %+.4f%s"
                           % (i + 1, (DIM if cut else ""), txt[:42], RST, sc, mark))
    return "\n".join(out)


_ANSI = re.compile(r"\033\[([0-9;]*)m")


def ansi_to_html(txt: str) -> str:
    """Convert the rendered ANSI screen to HTML spans. Handles the 24-bit fg/bg used by the
    debug thumbnails plus the handful of SGR codes the main view uses."""
    out, fg, bg, bold, dim = [], None, None, False, False
    def open_span():
        st = []
        if fg: st.append("color:rgb(%d,%d,%d)" % fg)
        if bg: st.append("background:rgb(%d,%d,%d)" % bg)
        if bold: st.append("font-weight:700")
        if dim: st.append("opacity:.55")
        return "<span style=\"%s\">" % ";".join(st) if st else "<span>"
    pos = 0
    out.append(open_span())
    for m in _ANSI.finditer(txt):
        chunk = txt[pos:m.start()]
        out.append(chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        pos = m.end()
        codes = [c for c in m.group(1).split(";") if c != ""] or ["0"]
        i = 0
        while i < len(codes):
            c = int(codes[i])
            if c == 0: fg = bg = None; bold = dim = False
            elif c == 1: bold = True
            elif c == 2: dim = True
            elif 30 <= c <= 37:
                fg = [(0,0,0),(200,60,60),(60,180,90),(200,180,60),
                      (80,130,220),(190,90,190),(70,190,200),(220,220,220)][c-30]
            elif c == 38 and i+4 < len(codes) and codes[i+1] == "2":
                fg = (int(codes[i+2]), int(codes[i+3]), int(codes[i+4])); i += 4
            elif c == 48 and i+4 < len(codes) and codes[i+1] == "2":
                bg = (int(codes[i+2]), int(codes[i+3]), int(codes[i+4])); i += 4
            i += 1
        out.append("</span>" + open_span())
    out.append(txt[pos:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    out.append("</span>")
    return "".join(out)


_FONT = None
def ansi_to_image(txt: str, fs: int = 22):
    """Render one ANSI screen to a PIL image. Runs on the DISPLAY thread, never the round
    loop, so recording cannot affect achieved Hz."""
    from PIL import Image, ImageDraw, ImageFont
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", fs)
    bb = _FONT.getbbox("M")
    cw, ch = bb[2] - bb[0] + 1, int(fs * 1.32)
    lines = txt.replace(CLR, "").split("\n")
    cols = max((len(_ANSI.sub("", l)) for l in lines), default=80)
    W, H = cols * cw + 16, len(lines) * ch + 16
    img = Image.new("RGB", (W, H), (11, 13, 16))
    dr = ImageDraw.Draw(img)
    PAL = [(0,0,0),(200,60,60),(60,180,90),(200,180,60),(80,130,220),(190,90,190),(70,190,200),(220,220,220)]
    for li, line in enumerate(lines):
        x, y = 8, 8 + li * ch
        fg, bg, bold, dim = (221, 227, 234), None, False, False
        pos = 0
        for m in _ANSI.finditer(line):
            seg = line[pos:m.start()]; pos = m.end()
            if seg:
                col = tuple(int(c * (0.55 if dim else 1.0)) for c in fg)
                if bg: dr.rectangle([x, y, x + len(seg) * cw, y + ch], fill=bg)
                dr.text((x, y), seg, font=_FONT, fill=col)
                x += len(seg) * cw
            codes = [c for c in m.group(1).split(";") if c != ""] or ["0"]
            i = 0
            while i < len(codes):
                c = int(codes[i])
                if c == 0: fg, bg, bold, dim = (221,227,234), None, False, False
                elif c == 1: bold = True
                elif c == 2: dim = True
                elif 30 <= c <= 37: fg = PAL[c-30]
                elif c == 38 and i+4 < len(codes): fg = (int(codes[i+2]),int(codes[i+3]),int(codes[i+4])); i += 4
                elif c == 48 and i+4 < len(codes): bg = (int(codes[i+2]),int(codes[i+3]),int(codes[i+4])); i += 4
                i += 1
        seg = line[pos:]
        if seg:
            col = tuple(int(c * (0.55 if dim else 1.0)) for c in fg)
            if bg: dr.rectangle([x, y, x + len(seg) * cw, y + ch], fill=bg)
            dr.text((x, y), seg, font=_FONT, fill=col)
    return img


def compose_frame(scr: str, dbg: Dict, fs: int = 22):
    """Video frame = the REAL model inputs at native resolution, above the display.
    V-JEPA2's 16x256x256 and SigLIP2's 8x224x224 are pasted as actual images, not terminal
    half-blocks -- so the recording shows exactly what the encoders received next to what
    was predicted from it."""
    from PIL import Image, ImageDraw, ImageFont
    panel = ansi_to_image(scr, fs)
    W = max(panel.width, 8 * 256 + 9 * 4)
    rows = []

    def montage(arr, n_per_row, cell, title):
        n = len(arr)
        r = (n + n_per_row - 1) // n_per_row
        h = 22 + r * (cell + 4)
        im = Image.new("RGB", (W, h), (11, 13, 16))
        d = ImageDraw.Draw(im)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
        except Exception:
            f = None
        d.text((6, 3), title, font=f, fill=(150, 160, 172))
        for i in range(n):
            fr = Image.fromarray(arr[i]).resize((cell, cell), Image.NEAREST)
            x = 4 + (i % n_per_row) * (cell + 4)
            y = 22 + (i // n_per_row) * (cell + 4)
            im.paste(fr, (x, y))
        return im

    if dbg.get("vjepa_in") is not None and not dbg.get("_view_only"):
        a = dbg["vjepa_in"]
        rows.append(montage(a, 8, 256,
                            "V-JEPA2 INPUT — the %d frames at native 256x256, linspace over "
                            "the 10 s window (this is exactly what the encoder receives)" % len(a)))
    if dbg.get("scene_px") is not None:
        pv = dbg["scene_px"]
        arr = [pv[i].permute(1, 2, 0).numpy() for i in range(pv.shape[0])]
        rows.append(montage(arr, 8, 224,
                            ("CAMERA — the %d frames the perception window actually contains, "
                             "at native 224x224 (SigLIP2 input, post-AutoProcessor)") % len(arr)))
    H = sum(r.height for r in rows) + panel.height
    out = Image.new("RGB", (W, H), (11, 13, 16))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.paste(panel, (0, y))
    return out


def start_web(state, port, log):
    """Serve the live display over HTTP so it can be watched from a phone (via cloudflared)
    while the demonstrator is out of frame. Renders from the same `state` the terminal uses,
    on the WEB thread -- the round loop is untouched."""
    import http.server, socketserver
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            try:
                body = render_debug(state) if state.get("debug") else render(state)
            except Exception as e:
                body = "render error: %r" % (e,)
            html = ("<!doctype html><meta charset=utf-8>"
                    "<meta name=viewport content='width=device-width,initial-scale=.45'>"
                    "<meta http-equiv=refresh content=1>"
                    "<title>BMO perception</title>"
                    "<body style='margin:0;background:#0b0d10;color:#dde3ea'>"
                    "<pre style='font:11px/1.15 ui-monospace,Menlo,Consolas,monospace;"
                    "white-space:pre;margin:0;padding:8px'>%s</pre>"
                    % ansi_to_html(body.replace(CLR, "")))
            b = html.encode("utf-8", "replace")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
    srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"  web view on http://0.0.0.0:{port}  (auto-refresh 1 s)")
    return srv


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until ctrl-C")
    ap.add_argument("--provenance", default="/home/bmo/PROVENANCE_perception_demo.json")
    ap.add_argument("--no-ui", action="store_true", help="log lines instead of the redraw UI")
    ap.add_argument("--floor", type=float, default=None,
                    help="confidence floor; below it a category renders '—'")
    ap.add_argument("--calibrate", default="", metavar="LABEL",
                    help="record per-category top-1 similarity every round under this label")
    ap.add_argument("--calib-out", default="/home/bmo/calib.json")
    ap.add_argument("--null-video", action="store_true",
                    help="feed featureless frames (simulates a covered lens)")
    ap.add_argument("--null-audio", action="store_true", help="feed digital silence")
    ap.add_argument("--sound-log", default="", metavar="FILE",
                    help="one line per round: time, SOUND tag, score, audio RMS")
    ap.add_argument("--no-gate", action="store_true", dest="no_gate",
                    help="disable all floors, show every category with its raw score. "
                         "Use OFF-SITE: floors are calibrated to one room and one light.")
    ap.add_argument("--record-view", action="store_true",
                    help="record ONE row of camera frames above the display (8x224). "
                         "~1/4 the memory of --record-frames and a far better aspect ratio "
                         "for slides; use this unless you specifically want both encoder strips.")
    ap.add_argument("--record-frames", action="store_true",
                    help="record the REAL 256x256 / 224x224 encoder inputs above the display. "
                         "Buffers raw frames, so keep the clip short (<=90 s).")
    ap.add_argument("--record-fontsize", type=int, default=22,
                    help="px font for recorded frames; larger = crisper video, no runtime cost")
    ap.add_argument("--record", default="", metavar="MP4",
                    help="record the display to an mp4 (frames rendered on the display "
                         "thread; the round loop is untouched)")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT",
                    help="serve the live display over HTTP (watch from a phone)")
    ap.add_argument("--debug-view", action="store_true",
                    help="start with the debug view on (toggle any time with 'd')")
    ap.add_argument("--audio-mode", default="base", choices=["base", "mean"],
                    help="'base' = WavJEPA-base only (qp_runD ckpt cfg). "
                         "'mean' = base+nat averaged (M2's training default, +~470 ms)")
    ap.add_argument("--audio-probe", default="", metavar="FILE",
                    help="log pooled WavJEPA embedding motion vs RMS each round")
    ap.add_argument("--stability", type=int, default=0, metavar="N",
                    help="render a tag only once it has repeated N rounds running "
                         "(temporal-stability gate; 0 = off)")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    log("[boot] checkpoint provenance")
    prov_sha = {}
    for k, (path, expect) in LOCKED.items():
        h = sha16(path)
        prov_sha[k] = h
        ok = "" if expect is None else ("  OK" if h == expect else "  !! MISMATCH expected " + expect)
        log(f"   {k:6} {os.path.basename(path):38} sha256:{h}{ok}")
        if expect is not None and h != expect:
            raise SystemExit(f"checkpoint {k} sha256 mismatch -- refusing to run")

    _boot0 = time.perf_counter()
    log("[boot] loading encoders (int8 where production does)")
    per = Perception(log, audio_mode=a.audio_mode)

    import cv2
    import sounddevice as sd

    vbuf = RollingVideoBuffer(WINDOW_SEC, VIDEO_FPS)
    abuf = RollingAudioBuffer(WINDOW_SEC, AUDIO_SR)
    stop = threading.Event()

    held = {}

    def cam_thread():
        cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
        held["cap"] = cap
        if not cap.isOpened():
            log("!! camera failed to open"); stop.set(); return
        for _ in range(30):
            cap.read()                                   # 3A converges once
        period = 1.0 / VIDEO_FPS
        nxt = time.time()
        while not stop.is_set():
            ok, f = cap.read()
            if ok:
                if ROT180:
                    f = cv2.rotate(f, cv2.ROTATE_180)
                fr = cv2.resize(f, (BUF_W, BUF_H))
                if a.null_video:
                    fr = np.full_like(fr, 128)
                vbuf.push(fr)
            nxt += period
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()
        cap.release()

    def mic_thread():
        # Device selection is retried across candidates: the raw ALSA card can be briefly
        # held by PulseAudio as a previous run releases it (observed: -9985 Device
        # unavailable), which silently cost a 160 s measurement run. Try the ReSpeaker
        # directly, then pulse, then the default.
        cands = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and "ReSpeaker" in d["name"]:
                    cands.append(i)
        except Exception:
            pass
        cands += ["pulse", "default", None]
        st = None
        for attempt in range(3):
            for dev in cands:
                try:
                    st = sd.InputStream(device=dev, samplerate=AUDIO_SR, channels=1,
                                        dtype="float32", blocksize=1600)
                    st.start()
                    held["mic"] = st
                    log(f"  mic OPEN on device={dev!r} (attempt {attempt+1})")
                    break
                except Exception:
                    st = None
            if st is not None:
                break
            time.sleep(1.0)
        if st is None:
            # A measurement run must NOT quietly become a silence run.
            if a.audio_probe or a.sound_log:
                log("!! mic UNAVAILABLE and this is a MEASUREMENT run -- aborting rather "
                    "than logging silence as if it were audio")
                stop.set()
                return
            log("!! mic unavailable -- feeding silence (demo continues, audio is NOT live)")
            state["mic_ok"] = False
            while not stop.is_set():
                abuf.push(np.zeros(1600, np.float32)); time.sleep(0.1)
            return
        state["mic_ok"] = True
        try:
            while not stop.is_set():
                chunk, _ = st.read(1600)
                abuf.push(np.zeros(1600, np.float32) if a.null_audio else chunk[:, 0].copy())
        except Exception as e:
            log(f"!! mic READ failed mid-run: {e!r}")
            stop.set()
        finally:
            try:
                st.stop(); st.close()
            except Exception:
                pass

    calib: Dict[str, List[float]] = {}
    aprobe = open(a.audio_probe, "w", buffering=1) if a.audio_probe else None
    if aprobe:
        aprobe.write("# t\trms\tdist_base\tdist_mean\thearing_tag\n")
    amb_hist: Deque[torch.Tensor] = deque(maxlen=30)
    amb_base: List[torch.Tensor] = []
    base_ref: List[torch.Tensor] = []
    mean_ref: List[torch.Tensor] = []
    slog = open(a.sound_log, "w", buffering=1) if a.sound_log else None
    if slog:
        slog.write("# t\thearing_tag\tscore\taudio_rms\twhere_tag\tdoing_tag\n")
    state = {"rounds": [], "floor": a.floor, "stability": a.stability, "mic_ok": None,
             "nogate": a.no_gate,
             "errors": 0, "mem_used": mem_used(), "mem_peak": mem_used(),
             "prov": {"m2": prov_sha["m2"], "qp": prov_sha["qp"],
                      "bank_name": os.path.basename(LOCKED["bank"][0]),
                      "n_tags": len(per.bank_text),
                      "n_cats": len(set(per.pq.bank_category)),
                      "device": "Jetson Orin Nano 8GB",
                      "power": os.popen("nvpmodel -q 2>/dev/null | head -1").read().strip() or "MAXN_SUPER"}}

    def percep_thread():
        # A single bad round must NOT take the display down mid-demo. Log it, count it,
        # keep going. Only a sustained run of failures (something structurally broken, not
        # a transient) stops the loop. assert_staircase() failures still surface here --
        # they are logged loudly and counted, never silently swallowed.
        fails = 0
        while not stop.is_set():
            try:
                r = per.round(vbuf, abuf)
                if r is not None:
                    if a.calibrate:
                        for lb, (txt, sc, col) in r["A"].items():
                            calib.setdefault(lb, []).append(sc)
                    if aprobe is not None:
                        w = abuf.get_window()
                        rms = float(np.sqrt(np.mean(w[-16000:] ** 2))) if w is not None else 0.0
                        def _d(vec, ref):
                            vv = F.normalize(vec, dim=-1)
                            if len(ref) < 30:
                                ref.append(vv)
                            if len(ref) < 5:
                                return float("nan")
                            b = F.normalize(torch.stack(ref).mean(0), dim=-1)
                            return 1.0 - float(torch.dot(vv, b))
                        d_base = _d(r["amb_base"], base_ref)
                        d_mean = _d(r["amb_mean"], mean_ref)
                        v = F.normalize(r["amb_vec"], dim=-1)
                        # FIXED baseline: the first 30 rounds (the opening quiet stretch).
                        # ROLLING baseline: the previous 30 rounds, excluding this one.
                        if len(amb_base) < 30:
                            amb_base.append(v)
                        df = dr = float("nan")
                        if len(amb_base) >= 5:
                            b = F.normalize(torch.stack(amb_base).mean(0), dim=-1)
                            df = 1.0 - float(torch.dot(v, b))
                        if len(amb_hist) >= 5:
                            b2 = F.normalize(torch.stack(list(amb_hist)).mean(0), dim=-1)
                            dr = 1.0 - float(torch.dot(v, b2))
                        amb_hist.append(v)
                        aprobe.write("%s\t%.5f\t%.6f\t%.6f\t%s\n" % (
                            time.strftime("%H:%M:%S"), rms, d_base, d_mean,
                            r["A"].get("hearing", ("—",))[0]))
                    if slog is not None:
                        w = abuf.get_window()
                        rms = float(np.sqrt(np.mean(w[-16000:] ** 2))) if w is not None else 0.0
                        h = r["A"].get("hearing", ("—", 0.0, ""))
                        wh = r["A"].get("where", ("—", 0.0, ""))
                        dg = r["A"].get("doing", ("—", 0.0, ""))
                        slog.write("%s\t%s\t%+.4f\t%.5f\t%s\t%s\n" % (
                            time.strftime("%H:%M:%S"), h[0], h[1], rms, wh[0], dg[0]))
                    state["rounds"].append(r)
                    state["mem_used"] = mem_used()
                    state["mem_peak"] = max(state["mem_peak"], state["mem_used"])
                fails = 0
            except Exception as e:
                fails += 1
                state["errors"] = state.get("errors", 0) + 1
                import traceback
                log(f"!! round failed ({fails} consecutive, {state['errors']} total): {e!r}")
                log("   " + traceback.format_exc().strip().replace("\n", "\n   "))
                if fails >= 20:
                    log("!! 20 consecutive failures -- something is structurally broken, stopping")
                    stop.set()
                time.sleep(0.5)

    threads = [threading.Thread(target=cam_thread, daemon=True),
               threading.Thread(target=mic_thread, daemon=True),
               threading.Thread(target=percep_thread, daemon=True)]
    log(f"[boot] CUDA+import+load complete at {time.perf_counter() - _boot0:6.2f} s")
    log("[boot] filling 10 s buffers before the first round…")
    threads[0].start(); threads[1].start()
    time.sleep(WINDOW_SEC + 1.0)
    threads[2].start()

    # non-blocking key toggle; restored on exit
    import select as _sel, termios as _tm, tty as _tty
    _fd = sys.stdin.fileno()
    _old = None
    try:
        _old = _tm.tcgetattr(_fd); _tty.setcbreak(_fd)
    except Exception:
        pass
    per.debug = a.debug_view
    state["debug"] = a.debug_view
    if a.serve:
        start_web(state, a.serve, log)

    # Recording does ZERO image work during the run: it buffers the rendered ANSI text
    # (microseconds) and renders every frame to PNG after the loop exits. PIL holds the GIL,
    # and doing it inline measurably slowed the perception thread (970 -> 1110 ms/round);
    # even on a worker thread it cost ~65 ms. Deferring it entirely keeps the Hz shown IN the
    # recording equal to the Hz the system actually achieves -- which matters, because the
    # video is the artefact the panel sees.
    rec_frames: List = []
    REC_FRAMES_CAP = 200
    rec_every = 2                       # capture every 2nd refresh -> 2 fps
    rec_i = 0
    if a.record:
        log(f"  recording -> {a.record}  (buffered; rendered after the run)"
            + ("  [+real encoder inputs, cap %d frames]" % REC_FRAMES_CAP
               if a.record_frames else
               "  [+camera view row, cap %d frames]" % REC_FRAMES_CAP if a.record_view else ""))
        if (a.record_frames or a.record_view) and not a.debug_view:
            per.debug = True          # frame stash lives behind the debug flag
            per.view_only = a.record_view and not a.record_frames
            state["debug"] = False    # …but keep the NORMAL display on screen

    t_start = time.time()
    try:
        while not stop.is_set():
            try:
                if _old is not None and _sel.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "d":
                        state["debug"] = not state["debug"]
                        per.debug = state["debug"]
                        if not per.debug:
                            per.dbg = {}
                    elif ch.lower() == "q":
                        break
            except Exception:
                pass
            if a.no_ui:
                if state["rounds"]:
                    r = state["rounds"][-1]
                    log(" | ".join(f"{k}={v:.0f}" for k, v in r["lat"].items())
                        + f" | buf={r.get('buf_depth')}/{r.get('buf_max')}"
                        + f" | cam_fps={r.get('cap_fps', 0):.2f}")
                time.sleep(1.0)
            else:
                _scr = render_debug(state) if state.get("debug") else render(state)
                sys.stdout.write(_scr); sys.stdout.flush()
                if a.record and state["rounds"]:
                    rec_i += 1
                    if rec_i % rec_every == 0:
                        if a.record_frames or a.record_view:
                            # raw uint8 copies: ~4 MB/frame, hence the cap
                            if len(rec_frames) < REC_FRAMES_CAP:
                                dd = state["rounds"][-1].get("dbg") or {}
                                rec_frames.append((_scr, {
                                    "vjepa_in": None if a.record_view else dd.get("vjepa_in"),
                                    "scene_px": dd.get("scene_px"),
                                    "_view_only": a.record_view}))
                        else:
                            rec_frames.append(_scr)    # text only -- no PIL, no GIL hit
                time.sleep(0.25)
            if a.seconds and (time.time() - t_start) > a.seconds:
                break
    except KeyboardInterrupt:
        pass
    try:
        if _old is not None:
            _tm.tcsetattr(_fd, _tm.TCSADRAIN, _old)
    except Exception:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
    try:
        if held.get("cap") is not None:
            held["cap"].release()
    except Exception:
        pass
    try:
        sd.stop()
    except Exception:
        pass
    # closed AFTER the threads are joined, or percep_thread writes to a closed file
    if slog is not None:
        slog.close()
    if aprobe is not None:
        aprobe.close()
    time.sleep(0.3)

    if a.record and rec_frames:
        import subprocess, tempfile
        log(f"  rendering {len(rec_frames)} buffered frames…")
        fdir = tempfile.mkdtemp(prefix="bmorec_")
        n = 0
        for item in rec_frames:
            try:
                if isinstance(item, tuple):
                    img = compose_frame(item[0], item[1], a.record_fontsize)
                else:
                    img = ansi_to_image(item, a.record_fontsize)
                img.save(os.path.join(fdir, "f%06d.png" % n)); n += 1
            except Exception as e:
                log(f"!! frame {n} render failed: {e!r}")
        r = subprocess.run(["ffmpeg", "-y", "-framerate", "2", "-i",
                            os.path.join(fdir, "f%06d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.record],
                           capture_output=True)
        if r.returncode == 0:
            log(f"  WROTE {a.record} ({os.path.getsize(a.record)/1e6:.1f} MB, "
                f"{n} frames @2fps = {n/2:.0f} s of video)")
        else:
            log("!! ffmpeg failed: " + r.stderr.decode()[-400:])
        import shutil; shutil.rmtree(fdir, ignore_errors=True)

    # ── PROVENANCE ──
    st = per.stage_ms
    summary = {k: {"n": len(v), "mean_ms": float(np.mean(v)), "p50_ms": float(np.median(v)),
                   "p95_ms": float(np.percentile(v, 95)) if len(v) > 1 else float(v[0]),
                   "max_ms": float(np.max(v)), "min_ms": float(np.min(v))}
               for k, v in st.items() if v}
    prov = {
        "device": "NVIDIA Jetson Orin Nano 8GB (bmo-desktop)",
        "power_mode": os.popen("nvpmodel -q 2>/dev/null").read().strip(),
        "gpu_clock_hz": open("/sys/devices/platform/bus@0/17000000.gpu/devfreq/"
                             "17000000.gpu/cur_freq").read().strip()
                        if os.path.exists("/sys/devices/platform/bus@0/17000000.gpu/devfreq/"
                                          "17000000.gpu/cur_freq") else None,
        "cpu_governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip(),
        "total_mem_mib": 7620,
        "peak_used_mib": state["mem_peak"],
        "checkpoints": {k: {"path": v[0], "sha256_16": prov_sha[k]} for k, v in LOCKED.items()},
        "encoders": {"vision": VJEPA_REPO + " (int8)", "scene": SIGLIP_REPO + " (bf16)",
                     "ambient": "labhamlet/wavjepa-base (int8), audio_mode=base, nat NOT loaded"},
        "config": {"window_sec": WINDOW_SEC, "n_vision_frames": N_VISION,
                   "n_scene_frames": N_SCENE, "buffer_res": [BUF_W, BUF_H],
                   "vjepa_res": VJEPA_RES, "max_tdm_bins": MAX_TDM_BINS,
                   "bank": os.path.basename(LOCKED["bank"][0]),
                   "n_tags": len(per.bank_text), "n_categories": len(set(per.pq.bank_category))},
        "gating": {
            "method": "per-category floor = midpoint of (person-absent p95, person-present p05)",
            "calibrated_on": "the step-in/out transition itself -- the exact condition "
                             "demonstrated -- NOT a blank wall. A blank wall is a much stronger "
                             "'nothing' than an empty room and inflated apparent separation: "
                             "for `looks`, blank-wall p95 was 0.0176 vs 0.1335 person-absent-"
                             "with-room, so a blank-wall floor never fired. `looks` has a "
                             "NEGATIVE gap on the real condition and is not displayed.",
            "claim_supported": "PERSON goes silent when the person leaves frame -- the system "
                               "reporting whether a person is present, not whether the sensor "
                               "is blocked. Covering the lens is NOT what these floors detect.",
            "floors": {k: {"floor": v, "absent_p95": GAPS[k][0], "present_p05": GAPS[k][1],
                           "gap": round(GAPS[k][1] - GAPS[k][0], 4)} for k, v in FLOORS.items()},
            "criterion": "A measured gap is NECESSARY but NOT SUFFICIENT. A field is gated "
                         "only if its gap is stable enough to hold across CONSECUTIVE rounds "
                         "under continuous conditions -- i.e. wider than its own round-to-round "
                         "noise band. `doing` has a measured gap of 0.0229 (larger than "
                         "`wearing`'s 0.0147) but was NOT gated: inside its noise band it "
                         "flickered on/off between consecutive rounds with the subject sitting "
                         "still in frame, and cut a correct answer ('someone is watching a "
                         "screen', +0.1494, below a 0.1603 floor). Gap size alone is the wrong "
                         "criterion; gap-vs-noise is the criterion.",
            "measured_but_ungated": {"doing": {"absent_p95": 0.1489, "present_p05": 0.1718,
                                               "gap": 0.0229,
                                               "reason": "gap inside round-to-round noise band"}},
            "dropped_no_separation": {
                "looks": -0.0049, "who": -0.0381, "posture": -0.0297,
                "where": -0.0285, "lighting": -0.1590, "view": -0.2866,
                "hearing": -0.0346, "animal": -0.0232},
            "transition_lag_rounds": "~9-10 rounds in each direction. This is the 10 s rolling "
                                     "video buffer flushing the person out of (or back into) the "
                                     "window -- the window must turn over before the features "
                                     "change. It is the buffer length, NOT a defect and NOT "
                                     "model latency; a single round is 0.97 s.",
        },
        "audio_status": "NOT DISPLAYED. Two sessions showed the pooled WavJEPA embedding does "
                        "not track audio energy (Pearson r=+0.048 base-only, r=-0.029 base+nat) "
                        "and sound tags fire on digital silence ('an alarm beeping' at +0.367 "
                        "with RMS exactly 0). A 10 s pooled window is the wrong instrument for "
                        "sub-second events. The ambient STREAM remains in the pipeline (M2 was "
                        "trained with it); only the display claim is withdrawn.",
        "capture_health": {
            "note": "buffer depth and achieved camera fps per round. A starved window shows "
                    "up HERE, not in latency -- if the CSI sensor lengthens exposure in low "
                    "light the real framerate falls and the 10 s window carries fewer frames, "
                    "while round time is unchanged.",
            "buf_depth_min": min((r.get("buf_depth", 0) for r in state["rounds"]), default=None),
            "buf_depth_max": max((r.get("buf_depth", 0) for r in state["rounds"]), default=None),
            "buf_target": int(WINDOW_SEC * VIDEO_FPS),
            "cam_fps_min": round(min((r.get("cap_fps", 0.0) for r in state["rounds"]), default=0.0), 2),
            "cam_fps_mean": round(sum(r.get("cap_fps", 0.0) for r in state["rounds"])
                                  / max(1, len(state["rounds"])), 2),
            "cam_fps_target": VIDEO_FPS,
            "starved_rounds": sum(1 for r in state["rounds"]
                                  if r.get("buf_depth", 99) < int(WINDOW_SEC * VIDEO_FPS)),
        },
        "rounds_completed": len(state["rounds"]),
        "stages_ms": summary,
        "achieved_hz": (1000.0 / summary["ROUND"]["p50_ms"]) if "ROUND" in summary else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if a.calibrate:
        blob = {"label": a.calibrate, "null_video": a.null_video, "null_audio": a.null_audio,
                "rounds": len(state["rounds"]),
                "per_category": {k: {"n": len(v), "mean": float(np.mean(v)),
                                     "p05": float(np.percentile(v, 5)),
                                     "p50": float(np.median(v)),
                                     "p95": float(np.percentile(v, 95)),
                                     "min": float(np.min(v)), "max": float(np.max(v)),
                                     "raw": [round(float(x), 5) for x in v]}
                                 for k, v in calib.items()}}
        try:
            existing = json.load(open(a.calib_out))
        except Exception:
            existing = {}
        existing[a.calibrate] = blob
        with open(a.calib_out, "w") as f:
            json.dump(existing, f, indent=2)
        print("\n[calib] label=%s rounds=%d -> %s" % (a.calibrate, len(state["rounds"]), a.calib_out))
        for k in sorted(calib):
            v = calib[k]
            print("   %-9s n=%-4d mean=%+.4f  p05=%+.4f  p50=%+.4f  p95=%+.4f  min=%+.4f max=%+.4f"
                  % (k, len(v), np.mean(v), np.percentile(v, 5), np.median(v),
                     np.percentile(v, 95), np.min(v), np.max(v)))

    with open(a.provenance, "w") as f:
        json.dump(prov, f, indent=2)
    print("\n" + "=" * 96)
    print("PER-STAGE LATENCY  (n / mean / p50 / p95 / max, ms)")
    order = ["decode", "vjepa2_16f", "siglip2_8f", "wavjepa_10s", "world_state_build",
             "m2_fusion", "query_predictor", "retrieval_uncond", "ROUND"]
    for k in order:
        if k in summary:
            s = summary[k]
            print("  %-20s n=%-4d mean=%8.1f  p50=%8.1f  p95=%8.1f  max=%8.1f"
                  % (k, s["n"], s["mean_ms"], s["p50_ms"], s["p95_ms"], s["max_ms"]))
    if "ROUND" in summary:
        print("\n  achieved %.3f Hz (p50)   peak memory %d / 7620 MiB"
              % (1000.0 / summary["ROUND"]["p50_ms"], state["mem_peak"]))
    print("  provenance -> %s" % a.provenance)


if __name__ == "__main__":
    main()
