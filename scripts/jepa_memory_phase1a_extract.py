"""scripts/jepa_memory_phase1a_extract.py — Phase 1A of the JEPA-memory track
(see JEPA_MEMORY_PLAN.md).

THE QUESTION: does the FROZEN M2 representation carry person identity at all?
If it does not, the north-star "recognize people by their joint visual+audio
JEPA embedding" idea does not work in its current form, and we learn that for
a few GPU-minutes instead of after building a whole memory subsystem.

This script only EXTRACTS embeddings. Scoring/gating lives in
scripts/jepa_memory_phase1a_eval.py so the eval can be re-run (with different
splits/controls) without paying the decode cost again.

── Feature streams extracted per person-window (all frozen, no training) ──
  vitl_crop        (1024)  mean of the spatially-pooled V-JEPA2 ViT-L tokens
  wavjepa          ( 768)  mean of the WavJEPA base+nat mean (M2's ambient input)
  m2_world_state   (1024)  AVJepaPredictor.encode_world_state  (the deployed WSV)
  m2_prepool_mean  (1024)  mean of AVJepaPredictor.encode_pre_pool_tokens
  moonshine        ( 416)  mean of MoonshineSpeechEncoder states (already resident
                           on the Jetson at 37ms/turn -- zero added deploy latency)
  z_p              (1536)  the embedding predictor's output == the CAPTION-ALIGNED
                           space. This is the G6 CONTROL: caption-InfoNCE is trained
                           to be identity-INVARIANT (two different people doing the
                           same thing get the same caption), so z_p is EXPECTED to
                           lose. If it wins, §4a of the plan is wrong and we want to
                           know that.

── Real EasyCom gotchas this script handles (all verified directly, 2026-08-10) ──
1. **Burned-in participant legend.** Frames are 2123x1080, NOT 1920x1080: the left
   LEGEND_W=203 px (2123-1920) is a burned-in legend showing EVERY participant's
   photo, identical in every frame of a session. Cropping into it would literally
   read the answer key AND leak session identity. Every crop is hard-clamped to
   x >= LEGEND_W and the clamp is asserted, not assumed.
2. **Bounding boxes are in RAW 2123-wide coordinates** (verified empirically by
   cropping with and without a +203 offset and looking at the result -- raw is
   correct). Do NOT "correct" them by subtracting the legend width.
3. **Participant IDs are per-session SLOTS, not global identities.** Verified by
   md5'ing all 84 Participant_Photos: slot 1 is one real recurring person (same
   image in all 12 sessions), slot 2 is an anonymized silhouette (the glasses
   WEARER -- never appears in any bounding box, no close-mic), and a black "X"
   image marks unused slots. The other 28 photos are singletons. So the global
   identity key is (session, pid) for guests, and the single string "P1" for the
   one cross-session identity.
4. **Only guests have Close_Microphone_Audio** (the 28 singletons). p1 has none,
   so p1's audio comes from the 6-channel Glasses_Microphone_Array (mixed down),
   which is a different acoustic channel -- flagged per-row via `audio_src` so the
   eval never silently compares close-mic against array-mic.
5. torchcodec decodes every frame up to the last requested index (~17ms/frame at
   1 thread), so per-chunk cost is FIXED regardless of how many windows we take.
   => decode each chunk ONCE and emit all its windows. num_ffmpeg_threads=16
   measured 10x faster (20.2s -> 2.0s per chunk).

Usage (4-way shard, one per GPU):
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/jepa_memory_phase1a_extract.py \
        --shard-idx $i --num-shards 4 --out-dir /dev/shm/jepa_mem_p1a &
    done; wait
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.world_state_builder import build_world_state_features

EASYCOM_ROOT = "/home/utkarsh/raid2-data/easycom/extracted/Main"
VIDEO_FPS = 20.0
LEGEND_W = 203            # burned-in participant-photo legend column (gotcha 1)
FRAME_W = 2123
MAIN_W = FRAME_W - LEGEND_W   # 1920, the real egocentric view
WINDOW_S = 10.0           # matches extract_features_av.CLIP_DURATION_S / M2 training
N_FRAMES = 16             # production StreamingConfig.n_vision_frames
CROP_RES = 256            # V-JEPA2 ViT-L native
BOX_EXPAND = 1.8          # face box -> head+shoulders
MIN_BOX_PX = 60           # reject windows where the person is too small to be identifiable
MIN_BOX_FRAMES = 12       # of N_FRAMES, how many must actually have a box
AUDIO_SR = 16000
M2_MAX_TDM_BINS = 512


# ──────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────
def _chunk_key(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


def build_manifest() -> List[Dict]:
    """One entry per (session, chunk). Each carries the per-frame boxes and the
    set of participants with usable audio, so the worker decodes each chunk once."""
    out: List[Dict] = []
    for sess in range(1, 13):
        bb_dir = os.path.join(EASYCOM_ROOT, "Face_Bounding_Boxes", f"Session_{sess}")
        vid_dir = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{sess}")
        cm_dir = os.path.join(EASYCOM_ROOT, "Close_Microphone_Audio", f"Session_{sess}")
        arr_dir = os.path.join(EASYCOM_ROOT, "Glasses_Microphone_Array_Audio", f"Session_{sess}")
        for bb_path in sorted(glob.glob(os.path.join(bb_dir, "*.json"))):
            ck = _chunk_key(bb_path)
            mp4 = os.path.join(vid_dir, f"{ck}.mp4")
            if not os.path.exists(mp4):
                continue
            close_mics = {}
            for wav in glob.glob(os.path.join(cm_dir, f"{ck}_Participant_ID_*.wav")):
                pid = int(wav.rsplit("_Participant_ID_", 1)[1][:-4])
                close_mics[pid] = wav
            out.append({
                "session": sess, "chunk": ck, "mp4": mp4, "bb": bb_path,
                "close_mics": close_mics,
                "array": os.path.join(arr_dir, f"{ck}.wav"),
            })
    return out


def _boxes_by_frame(bb_path: str) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    """frame_number -> {pid: (x1,y1,x2,y2)}"""
    out: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    for fr in json.load(open(bb_path)):
        d = {}
        for p in fr.get("Participants", []):
            pid = p.get("Participant_ID")
            if pid is None or pid < 0:
                continue
            d[pid] = (p["x1"], p["y1"], p["x2"], p["y2"])
        out[int(fr["Frame_Number"])] = d
    return out


# ──────────────────────────────────────────────────────────────────────────
# Crop
# ──────────────────────────────────────────────────────────────────────────
def _crop_person(frame: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
    """frame (3,H,W) uint8 -> (3,CROP_RES,CROP_RES) uint8, square, legend-safe.

    The x-clamp to >= LEGEND_W is the load-bearing line: without it a crop near
    the left edge would include the burned-in participant-photo legend, which
    would make any identity result meaningless."""
    _, H, W = frame.shape
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * BOX_EXPAND
    X1 = int(round(cx - side / 2.0)); X2 = int(round(cx + side / 2.0))
    Y1 = int(round(cy - side / 2.0)); Y2 = int(round(cy + side / 2.0))
    X1 = max(LEGEND_W, X1); X2 = min(W, X2)
    Y1 = max(0, Y1);        Y2 = min(H, Y2)
    assert X1 >= LEGEND_W, "crop leaked into the burned-in participant legend"
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None
    patch = frame[:, Y1:Y2, X1:X2].float().unsqueeze(0)
    patch = F.interpolate(patch, size=(CROP_RES, CROP_RES), mode="bilinear", align_corners=False)
    return patch[0].clamp(0, 255).to(torch.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────────────────────────────────
def _load_audio_16k(path: str, mixdown: bool = False) -> Optional[np.ndarray]:
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    w = w.mean(axis=1) if (mixdown or w.shape[1] > 1) else w[:, 0]
    if sr != AUDIO_SR:
        g = int(round(sr / AUDIO_SR))
        w = resample_poly(w, 1, g) if g * AUDIO_SR == sr else resample_poly(w, AUDIO_SR, sr)
    return np.asarray(w, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────
class Stack:
    def __init__(self, device: torch.device, m2_ckpt: str, pred_ckpt: str):
        from models.vision_encoder import VisionEncoder
        from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
        from models.m4_speech import MoonshineSpeechEncoder
        from models.predictor import Predictor

        self.device = device
        self.vision = VisionEncoder(device=str(device))
        self.base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
        self.nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
        self.moon = MoonshineSpeechEncoder().to(device).eval()

        cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                           max_tdm_bins=M2_MAX_TDM_BINS, dropout=0.0)
        self.m2 = AVJepaPredictor(cfg).to(device)
        ck = torch.load(m2_ckpt, map_location=device, weights_only=False)
        self.m2.load_state_dict(ck["model"], strict=True)
        self.m2.eval()
        for p in self.m2.parameters():
            p.requires_grad_(False)

        self.pred = Predictor(in_dim=cfg.d_model, shared_dim=1536, mode="mlp").to(device)
        pck = torch.load(pred_ckpt, map_location=device, weights_only=False)
        self.pred.load_state_dict(pck["predictor"], strict=True)
        self.pred.eval()
        for p in self.pred.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, frames: torch.Tensor, wav: np.ndarray, dur_s: float) -> Dict[str, torch.Tensor]:
        """frames (T,3,H,W) uint8, wav (n,) float32 @16k -> {stream: (D,) float32 cpu}"""
        wav_t = torch.from_numpy(wav)
        res = build_world_state_features(
            frames=frames, audio=wav_t, true_window_dur_sec=dur_s,
            vision_encoder=self.vision, base_encoder=self.base, nat_encoder=self.nat,
            max_tdm_bins=M2_MAX_TDM_BINS, device=self.device,
        )
        feats, tbins = res.feats, res.tbins
        out = {
            "vitl_crop": feats["vision"].mean(dim=1)[0],
            "wavjepa": feats["ambient"].mean(dim=1)[0],
            "m2_world_state": self.m2.encode_world_state(feats, tbins)[0],
        }
        pre = self.m2.encode_pre_pool_tokens(feats, tbins)
        out["m2_prepool_mean"] = pre.mean(dim=1)[0]
        out["z_p"] = self.pred(pre)[0]
        hidden, _ = self.moon([wav], [dur_s], self.device)
        out["moonshine"] = hidden.float().mean(dim=1)[0]
        return {k: v.detach().float().cpu() for k, v in out.items()}


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="/dev/shm/jepa_mem_p1a")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--pred-ckpt", default="checkpoints/m2_embed_predictor_mlp_ddp_gradcache_bs16384/best.pt")
    ap.add_argument("--ffmpeg-threads", type=int, default=16)
    ap.add_argument("--cpu-threads", type=int, default=32,
                    help="torch.set_num_threads -- guards the oversubscription bug this repo already hit")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all (debug knob)")
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchcodec.decoders import VideoDecoder

    manifest = build_manifest()
    manifest = [m for i, m in enumerate(manifest) if i % args.num_shards == args.shard_idx]
    if args.max_chunks:
        manifest = manifest[: args.max_chunks]
    print(f"[p1a:{args.shard_idx}] {len(manifest)} chunks to process", flush=True)

    stack = Stack(device, args.m2_ckpt, args.pred_ckpt)
    print(f"[p1a:{args.shard_idx}] encoders loaded", flush=True)

    rows: List[Dict] = []
    embs: Dict[str, List[torch.Tensor]] = {}
    n_win = n_skip = 0
    t0 = time.time()

    for ci, m in enumerate(manifest):
        try:
            boxes = _boxes_by_frame(m["bb"])
            dec = VideoDecoder(m["mp4"], device="cpu", num_ffmpeg_threads=args.ffmpeg_threads)
            n_total = dec.metadata.num_frames
        except Exception as e:
            print(f"[p1a:{args.shard_idx}] SKIP chunk {m['session']}/{m['chunk']}: {e!r}", flush=True)
            continue

        win_frames = int(round(WINDOW_S * VIDEO_FPS))          # 200
        n_windows = max(1, n_total // win_frames)              # 6 for a 60s chunk

        # who is present in this chunk: guests with close-mic + p1 (array audio)
        pids = sorted(set(list(m["close_mics"].keys()) + [1]))

        # one decode pass for every frame this chunk needs
        need: List[int] = []
        win_idx: Dict[int, List[int]] = {}
        for w in range(n_windows):
            lo, hi = w * win_frames, min(n_total, (w + 1) * win_frames) - 1
            idx = torch.linspace(lo, hi, N_FRAMES).long().tolist()
            win_idx[w] = idx
            need.extend(idx)
        need_sorted = sorted(set(need))
        try:
            decoded = dec.get_frames_at(indices=need_sorted).data       # (n,3,H,W) uint8
        except Exception as e:
            print(f"[p1a:{args.shard_idx}] DECODE FAIL {m['session']}/{m['chunk']}: {e!r}", flush=True)
            continue
        pos = {f: i for i, f in enumerate(need_sorted)}

        audio_cache: Dict[int, Optional[np.ndarray]] = {}
        for pid in pids:
            if pid in m["close_mics"]:
                audio_cache[pid] = _load_audio_16k(m["close_mics"][pid])
            else:
                audio_cache[pid] = _load_audio_16k(m["array"], mixdown=True)

        for w in range(n_windows):
            idx = win_idx[w]
            for pid in pids:
                wav_full = audio_cache.get(pid)
                if wav_full is None:
                    n_skip += 1; continue

                # gather this person's boxes across the window's sampled frames
                sel_boxes, sel_frames = [], []
                for f in idx:
                    b = boxes.get(f + 1, {}).get(pid)     # Frame_Number is 1-indexed
                    if b is not None:
                        sel_boxes.append(b); sel_frames.append(f)
                if len(sel_boxes) < MIN_BOX_FRAMES:
                    n_skip += 1; continue
                med_side = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in sel_boxes]))
                if med_side < MIN_BOX_PX:
                    n_skip += 1; continue

                # nearest-available box for frames where this person had none
                bmap = dict(zip(sel_frames, sel_boxes))
                crops = []
                ok = True
                for f in idx:
                    b = bmap.get(f) or bmap[min(bmap, key=lambda g: abs(g - f))]
                    c = _crop_person(decoded[pos[f]], b)
                    if c is None:
                        ok = False; break
                    crops.append(c)
                if not ok:
                    n_skip += 1; continue
                frames_t = torch.stack(crops, 0)                       # (16,3,256,256) uint8

                a0 = int(w * WINDOW_S * AUDIO_SR)
                a1 = min(len(wav_full), int((w + 1) * WINDOW_S * AUDIO_SR))
                wav = wav_full[a0:a1]
                if len(wav) < AUDIO_SR:            # <1s of audio, not usable
                    n_skip += 1; continue
                dur = len(wav) / AUDIO_SR

                try:
                    e = stack.embed(frames_t, wav, dur)
                except Exception as ex:
                    print(f"[p1a:{args.shard_idx}] EMBED FAIL s{m['session']} {m['chunk']} p{pid}: {ex!r}", flush=True)
                    n_skip += 1; continue

                for k, v in e.items():
                    embs.setdefault(k, []).append(v)
                rows.append({
                    "session": m["session"], "chunk": m["chunk"], "window": w, "pid": pid,
                    # global identity key: p1 is ONE person across all sessions; guests
                    # are single-session, so their key must include the session.
                    "identity": "P1" if pid == 1 else f"S{m['session']}_P{pid}",
                    "is_cross_session_identity": pid == 1,
                    "audio_src": "close_mic" if pid in m["close_mics"] else "array_mixdown",
                    "median_box_px": med_side,
                    "n_boxes": len(sel_boxes),
                })
                n_win += 1

        if (ci + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[p1a:{args.shard_idx}] {ci+1}/{len(manifest)} chunks  windows={n_win} "
                  f"skipped={n_skip}  {el/(ci+1):.2f}s/chunk", flush=True)

    out = {
        "rows": rows,
        "emb": {k: torch.stack(v, 0) for k, v in embs.items()},
        "config": {
            "window_s": WINDOW_S, "n_frames": N_FRAMES, "crop_res": CROP_RES,
            "box_expand": BOX_EXPAND, "min_box_px": MIN_BOX_PX,
            "min_box_frames": MIN_BOX_FRAMES, "legend_w": LEGEND_W,
            "m2_ckpt": args.m2_ckpt, "pred_ckpt": args.pred_ckpt,
        },
    }
    path = os.path.join(args.out_dir, f"shard{args.shard_idx}.pt")
    torch.save(out, path)
    print(f"[p1a:{args.shard_idx}] DONE windows={n_win} skipped={n_skip} -> {path} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
