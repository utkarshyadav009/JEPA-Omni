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

import hashlib, json, os, sys, threading, time
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
    ("where",    "place",       "What does this place look like? Describe it fully.",     "SCENE"),
    ("objects",  "object",      "Tell me in detail what the surroundings look like.",     "SCENE"),
    ("lighting", "light",       "Describe the room and setting in detail.",               "SCENE"),
    ("who",      "people",      "Tell me in detail what the person is doing.",            "SCENE"),
    ("wearing",  "appearance",  "Describe the room and setting in detail.",               "SCENE"),
    ("view",     "camera",      "Briefly, what am I looking at?",                         "SCENE"),
    ("animal",   "animal",      "Summarize the scene in one sentence.",                   "SCENE"),
    ("doing",    "action",      "Explain everything that happens, in order.",             "ACTION"),
    ("posture",  "posture",     "Tell me in detail what the person is doing.",            "ACTION"),
    ("holding",  "held_object", "Give me a detailed account of what is being done.",      "ACTION"),
    ("looks",    "expression",  "Walk me through step by step what happens.",             "ACTION"),
    ("hearing",  "sound",       "What do you hear?",                                      "SOUND"),
]
COLUMNS = ["SCENE", "ACTION", "SOUND"]
COL_SRC = {"SCENE": "SigLIP2", "ACTION": "V-JEPA2", "SOUND": "WavJEPA"}


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

    def push(self, f: np.ndarray) -> None:
        with self._lock:
            self._buf.append(f)

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
    def __init__(self, log):
        self.log = log
        self.stage_ms: Dict[str, List[float]] = {}
        from bmo_jetson_startup import q_int8_cpu_then_move
        self.q8 = q_int8_cpu_then_move
        dev = torch.device("cuda")
        self.dev = dev

        a0 = mem_avail()
        from models.vision_encoder import VisionEncoder
        self.ve = VisionEncoder(device="cpu", dtype=torch.bfloat16)
        self.ve.model = self.q8(self.ve.model, dev); self.ve.device_str = "cuda"
        log(f"  vision  V-JEPA2 ViT-L int8   avail={mem_avail()} MiB")

        from transformers import AutoModel, AutoProcessor
        sig = AutoModel.from_pretrained(SIGLIP_REPO, dtype=torch.bfloat16)
        if hasattr(sig, "text_model"):
            del sig.text_model          # queries are pre-encoded; text tower is dead weight
        import gc; gc.collect()
        self.sig = sig.to(dev).eval()
        self.sig_proc = AutoProcessor.from_pretrained(SIGLIP_REPO)
        log(f"  scene   SigLIP2 base bf16    avail={mem_avail()} MiB")

        from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO
        self.wj = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
        self.wj.model = self.q8(self.wj.model, dev); self.wj.device_str = "cuda"
        log(f"  ambient WavJEPA-base int8    avail={mem_avail()} MiB   (nat NOT loaded)")

        from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
        m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                          max_tdm_bins=MAX_TDM_BINS, dropout=0.0))
        ck = torch.load(LOCKED["m2"][0], map_location="cpu", weights_only=False)
        m2.load_state_dict(ck["model"], strict=True); del ck; gc.collect()
        m2 = m2.to(torch.bfloat16)
        self.m2 = self.q8(m2, dev); self.m2.eval()
        log(f"  m2      AVJepaPredictor int8 avail={mem_avail()} MiB")

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
        log(f"  TOTAL perception footprint  {a0 - mem_avail()} MiB")

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
        _wrap(self.ve, "encode", "vjepa2_16f")
        _wrap(self.wj, "encode", "wavjepa_10s")

        from models.world_state_builder import build_world_state_features, assert_staircase
        self._build_ws = build_world_state_features
        self._assert_staircase = assert_staircase
        self.q_emb_cache: Dict[str, torch.Tensor] = {}

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

        t0 = time.perf_counter()
        scene = self._scene(fr_s).unsqueeze(0)                                  # (1,8,768)
        ms_siglip = self._t("siglip2_8f", t0)

        t0 = time.perf_counter()
        ws = self._build_ws(vid, audio, WINDOW_SEC, self.ve, self.wj, None,
                            MAX_TDM_BINS, self.dev)
        ms_build = self._t("world_state_build", t0)
        ms_vjepa = self.stage_ms["vjepa2_16f"][-1]      # recorded by the wrapper, real call
        ms_wavjepa = self.stage_ms["wavjepa_10s"][-1]

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
        ms_qp = self._t("query_predictor", t0)

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
        return {
            "t": time.time(), "A": panelA, "B": panelB,
            "lat": {"decode": ms_decode, "vjepa2_16f": ms_vjepa, "siglip2_8f": ms_siglip,
                    "wavjepa_10s": ms_wavjepa, "m2_fusion": ms_m2,
                    "query_predictor": ms_qp, "retrieval_uncond": ms_retr, "ROUND": total},
            "n_temp": n_temp, "vision_ntok": ws.vision_raw_ntok, "ambient_ntok": ws.ambient_ntok,
        }


# ── display ──────────────────────────────────────────────────────────────────
CLR = "\033[2J\033[H"
DIM, B, R, G, Y, C, M, RST = ("\033[2m", "\033[1m", "\033[31m", "\033[32m",
                              "\033[33m", "\033[36m", "\033[35m", "\033[0m")
HL = "\033[1;33m"          # changed since last refresh


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
    out.append(DIM + "  " + "─" * W + RST)

    if not rounds:
        out.append("\n  " + Y + "warming up — filling the 10 s rolling buffers…" + RST + "\n")
        return "\n".join(out)

    cur = rounds[-1]
    prev = rounds[-2] if len(rounds) > 1 else None
    colw = 38

    hdr = "  "
    for col in COLUMNS:
        hdr += B + _fit(f"{col}  ({COL_SRC[col]})", colw) + RST
    out.append(hdr)
    out.append(DIM + "  " + "─" * W + RST)

    rows: Dict[str, List[str]] = {c: [] for c in COLUMNS}
    for label, cat, q, col in QUESTIONS:
        if label not in cur["A"]:
            continue
        txt, score, _ = cur["A"][label]
        changed = prev is not None and label in prev["A"] and prev["A"][label][0] != txt
        mark = HL + "● " + RST if changed else "  "
        body = (HL if changed else "") + _fit(txt, colw - 13) + (RST if changed else "")
        rows[col].append(f"{mark}{DIM}{label:<9}{RST}{body}")

    for i in range(max(len(v) for v in rows.values())):
        line = "  "
        for col in COLUMNS:
            line += (rows[col][i] if i < len(rows[col]) else " " * colw)
            line += "  "
        out.append(line)

    out.append("")
    out.append(DIM + "  HISTORY — last %d refreshes (● = changed)" % HISTORY + RST)
    for r in rounds[-HISTORY:][::-1]:
        ts = time.strftime("%H:%M:%S", time.localtime(r["t"]))
        bits = []
        for lb in ("where", "doing", "hearing"):
            if lb in r["A"]:
                bits.append(f"{DIM}{lb}={RST}{_fit(r['A'][lb][0], 26).strip()}")
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
    out.append("  " + B + "MEMORY   " + RST +
               f"used {state['mem_used']} / 7620 MiB   peak {state['mem_peak']} MiB   "
               f"{DIM}vision_tok={cur['vision_ntok']} n_temp={cur['n_temp']} "
               f"ambient_tok={cur['ambient_ntok']}{RST}")
    out.append("  " + DIM + f"m2 {prov['m2']}  │  qp {prov['qp']}  │  bank {prov['bank_name']} "
                            f"({prov['n_tags']} tags / {prov['n_cats']} cats)  │  "
                            f"{prov['device']}  {prov['power']}" + RST)
    return "\n".join(out)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until ctrl-C")
    ap.add_argument("--provenance", default="/home/bmo/PROVENANCE_perception_demo.json")
    ap.add_argument("--no-ui", action="store_true", help="log lines instead of the redraw UI")
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

    log("[boot] loading encoders (int8 where production does)")
    per = Perception(log)

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
                vbuf.push(cv2.resize(f, (BUF_W, BUF_H)))
            nxt += period
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()
        cap.release()

    def mic_thread():
        dev = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and "ReSpeaker" in d["name"]:
                dev = i; break
        ch = 1
        try:
            with sd.InputStream(device=dev, samplerate=AUDIO_SR, channels=ch,
                                dtype="float32", blocksize=1600) as st:
                held["mic"] = st
                while not stop.is_set():
                    chunk, _ = st.read(1600)
                    abuf.push(chunk[:, 0].copy())
        except Exception as e:
            log(f"!! mic failed ({e!r}) -- pushing silence so WavJEPA still runs a REAL forward")
            while not stop.is_set():
                abuf.push(np.zeros(1600, np.float32)); time.sleep(0.1)

    state = {"rounds": [], "mem_used": mem_used(), "mem_peak": mem_used(),
             "prov": {"m2": prov_sha["m2"], "qp": prov_sha["qp"],
                      "bank_name": os.path.basename(LOCKED["bank"][0]),
                      "n_tags": len(per.bank_text),
                      "n_cats": len(set(per.pq.bank_category)),
                      "device": "Jetson Orin Nano 8GB",
                      "power": os.popen("nvpmodel -q 2>/dev/null | head -1").read().strip() or "MAXN_SUPER"}}

    def percep_thread():
        while not stop.is_set():
            try:
                r = per.round(vbuf, abuf)
            except Exception as e:
                log(f"!! perception round FAILED: {e!r}")
                stop.set(); raise
            if r is not None:
                state["rounds"].append(r)
                state["mem_used"] = mem_used()
                state["mem_peak"] = max(state["mem_peak"], state["mem_used"])

    threads = [threading.Thread(target=cam_thread, daemon=True),
               threading.Thread(target=mic_thread, daemon=True),
               threading.Thread(target=percep_thread, daemon=True)]
    log("[boot] filling 10 s buffers before the first round…")
    threads[0].start(); threads[1].start()
    time.sleep(WINDOW_SEC + 1.0)
    threads[2].start()

    t_start = time.time()
    try:
        while not stop.is_set():
            if a.no_ui:
                if state["rounds"]:
                    r = state["rounds"][-1]
                    log(" | ".join(f"{k}={v:.0f}" for k, v in r["lat"].items()))
                time.sleep(1.0)
            else:
                sys.stdout.write(render(state)); sys.stdout.flush()
                time.sleep(0.25)
            if a.seconds and (time.time() - t_start) > a.seconds:
                break
    except KeyboardInterrupt:
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
    time.sleep(0.3)

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
        "rounds_completed": len(state["rounds"]),
        "stages_ms": summary,
        "achieved_hz": (1000.0 / summary["ROUND"]["p50_ms"]) if "ROUND" in summary else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
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
