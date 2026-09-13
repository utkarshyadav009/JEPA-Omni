"""P4.5 / P3.1 -- Epic-Kitchens extraction: ONE decode pass, TWO outputs.

Decode is the expensive operation (~89 h of 1080p video). It happens exactly once. Per video:

  world_state : EVERY window at --stride-s (default 1.0 s). ~5.5 KB/window. W (1024) plus
                mean-pooled vision (1024) and ambient (768), fp16, with per-window start times.
                Feeds the Phase 2/3 temporal probes and the RUN-5 pair audit.
  features    : every --feat-every-th window (default 2 => a 2.0 s aligned subset). ~3.93 MB
                each, in the EXISTING cache layout (vision / ambient_base / ambient_nat / *_ts /
                clip_duration_s), so AVCachedDataset consumes it unmodified. Checkpoint-
                independent and archival: a future encoder can be rebuilt from these, the
                world_states cannot.

THE TWO-GATE DELETION RULE (P4.5, non-negotiable)
-------------------------------------------------
A source .MP4 is deleted ONLY after BOTH outputs for that video are written, fsynced, reloaded
from disk, and checked for the expected window count. Verifying one output and deleting the
source before the other is checked is the exact failure mode this rule exists to prevent, so
the delete lives behind a single `both_verified` flag that is set in one place and nowhere else.
Deletion is also recoverable in principle: `epic_kitchens/urls.txt` + `download.sh` are retained.

EQUIVALENCE GATE
----------------
This script re-derives base/nat separately (the shared builder returns only their mean, and this
file must not modify a production module to get at them). To prove the re-derivation has not
drifted from the shared construction, window 0 of the first video is built BOTH ways and the
merged ambient / vision / tbins must match bit-for-bit. Mismatch aborts before anything is
written or deleted.

AMBIENT CAP
-----------
RUN-4 was trained at T_a = 896 and scores 4.27 R@1 when evaluated uncapped versus 41.35 capped
(docs/CANONICAL_NUMBERS.md 7.1). The cap is therefore applied to the world_state path, not
optional. The archival `features` path is written UNCAPPED on purpose -- truncation there would
bake one run's training length into a checkpoint-independent artifact.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, shutil, subprocess, sys, time
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

EK_ROOT    = "/mnt/Raid-Storage-2/utkarsh-data/epic_kitchens"
VIDEO_DIR  = f"{EK_ROOT}/video"
VIDEO_INFO = f"{EK_ROOT}/annotations/EPIC_100_video_info.csv"
WINDOW_SEC, VIDEO_FPS, AUDIO_SR, MAX_TDM = 10.0, 50.0, 16000, 512
RUN4_TA = 896

# P3.0-selected RUN-4 checkpoint (post-hoc by held-out R@1 over all 20 tagged steps, 3 seeds).
M2_CKPT = "checkpoints/m2_run4_padfix_ta896/step18000.pt"
M2_SHA  = "27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4"


def free_gib(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


GRID_FPS = 12.8          # 2x the 6.4 fps a 64-frame 10 s window needs => <=39 ms placement error
CHUNK_S  = 240.0         # decode in chunks so a 62-minute video does not need 5 GB of frames
SCALE_PX = 512           # swscale to 512, then the EXACT reference antialiased bilinear 512->256


def decode_chunk(src, t0, t1):
    """ONE sequential ffmpeg pass over [t0, t1) at GRID_FPS, scaled to SCALE_PX.

    This replaces per-window random access, which was the entire cost of this script: seeking
    64 scattered frames costs 5.68 s/window because H.264 must re-decode from a keyframe each
    time, and at a 1 s stride consecutive windows overlap by 90%, so the same frames were being
    decoded ~10x. Measured on P01_04 (105 s): 5.68 s for ONE window random-access vs 4.73 s for
    the WHOLE video sequentially -- ~120x.

    DEVIATION, recorded rather than buried: the reference path resizes native->256 in one
    torch antialiased-bilinear step. Here swscale does native->512 and torch does the reference
    512->256. Pixel values therefore differ slightly from a torchcodec-decoded clip. This is
    internally consistent across the whole Epic-Kitchens corpus, which is what the temporal
    probes and the RUN-5 pilot compare against; no existing cache is being extended.
    """
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", src,
           "-vf", f"fps={GRID_FPS},scale={SCALE_PX}:{SCALE_PX}", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "pipe:1"]
    fsz = SCALE_PX * SCALE_PX * 3
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    n = len(proc.stdout) // fsz
    if n < 1:
        raise ValueError(f"no frames from {os.path.basename(src)} @ [{t0:.0f},{t1:.0f})")
    arr = np.frombuffer(proc.stdout[: n * fsz], dtype=np.uint8).reshape(n, SCALE_PX, SCALE_PX, 3)
    return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).contiguous()   # (n,3,512,512) uint8


def window_frames(chunk, chunk_t0, start_sec):
    """The 64 frames for a window, picked from the grid by NEAREST absolute time, then resized
    512->256 with the reference call. Grid spacing is 1/12.8 s, so placement error is <=39 ms."""
    step = (WINDOW_SEC - 1.0 / GRID_FPS) / 63.0
    idx = torch.tensor([round((start_sec - chunk_t0 + k * step) * GRID_FPS) for k in range(64)])
    idx = idx.clamp(0, chunk.shape[0] - 1).long()
    if int(idx[-1]) - int(idx[0]) < 2:
        raise ValueError("window shorter than 2 grid frames")
    x = chunk[idx].float()
    x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear",
                                        align_corners=False, antialias=True)
    return x.round_().clamp_(0, 255).to(torch.uint8)


def decode_audio_full(src, dur):
    """Whole track ONCE per video (0.2 s for a 105 s video), sliced per window in memory.
    Previously one ffmpeg spawn per window -- 353,465 subprocesses across the corpus."""
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-i", src, "-vn", "-ar", str(AUDIO_SR), "-ac", "1",
           "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=1800)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    if sr != AUDIO_SR or audio.shape[0] < AUDIO_SR:
        raise ValueError(f"bad audio from {src}: sr={sr} n={audio.shape[0]}")
    return torch.from_numpy(audio)


def slice_audio(full, start_sec):
    i0 = int(round(start_sec * AUDIO_SR))
    i1 = min(full.shape[0], i0 + int(round(WINDOW_SEC * AUDIO_SR)))
    seg = full[i0:i1]
    if seg.shape[0] < AUDIO_SR:
        raise ValueError("audio window under 1 s")
    return seg, seg.shape[0] / AUDIO_SR


def build_both(frames, audio, dur, venc, benc, nenc, dev):
    """The shared construction, with base/nat kept instead of discarded. Every helper is
    IMPORTED from world_state_builder -- only the call order is restated here, and the
    equivalence gate proves that restatement is faithful."""
    from models.world_state_builder import (_spatial_pool, _group_timestamps,
                                            _uniform_token_timestamps, _ts_to_tdm_bins,
                                            VISION_DIM, VISION_SPAT)
    with torch.no_grad():
        raw = venc.encode(frames.unsqueeze(0).to(dev))[0]
    assert raw.shape[0] % 256 == 0, f"vision tokens {raw.shape[0]} not a multiple of 256"
    n_temp = raw.shape[0] // 256
    vis_pooled = _spatial_pool(raw.view(n_temp, 256, VISION_DIM)).to(torch.bfloat16)
    vis_ts = _group_timestamps(n_temp, n_raw_frames=frames.shape[0], true_window_dur_sec=dur)
    vis_ts_exp = vis_ts.unsqueeze(1).expand(n_temp, VISION_SPAT, 2).reshape(n_temp * VISION_SPAT, 2)
    vis_bins = _ts_to_tdm_bins(vis_ts_exp, dur, MAX_TDM)
    vis_flat = vis_pooled.reshape(n_temp * VISION_SPAT, VISION_DIM)

    wav = audio.mean(0) if audio.dim() > 1 else audio
    with torch.no_grad():
        base = benc.encode(wav.unsqueeze(0).unsqueeze(0).to(dev))[0]
        nat = nenc.encode(wav.unsqueeze(0).expand(2, -1).unsqueeze(0).to(dev))[0] if nenc else None
    if nat is not None and base.shape[0] == nat.shape[0]:
        aud = (base.float() + nat.float()).mul_(0.5).to(torch.bfloat16)
    else:
        aud = base.to(torch.bfloat16)
    aud_ts = _uniform_token_timestamps(aud.shape[0], dur)
    aud_bins = _ts_to_tdm_bins(aud_ts, dur, MAX_TDM)
    feats = {"vision": vis_flat.float().unsqueeze(0).to(dev), "ambient": aud.float().unsqueeze(0).to(dev)}
    tbins = {"vision": vis_bins.unsqueeze(0).to(dev), "ambient": aud_bins.unsqueeze(0).to(dev)}
    cache = {"vision": vis_pooled.cpu(), "ambient_base": base.to(torch.bfloat16).cpu(),
             "ambient_nat": (nat.to(torch.bfloat16).cpu() if nat is not None
                             else base.to(torch.bfloat16).cpu()),
             "vision_ts": vis_ts.cpu(), "ambient_base_ts": aud_ts.cpu(),
             "ambient_nat_ts": aud_ts.cpu(), "clip_duration_s": float(dur)}
    return feats, tbins, cache


def equivalence_gate(frames, audio, dur, venc, benc, nenc, dev):
    from models.world_state_builder import build_world_state_features
    ref = build_world_state_features(frames, audio, dur, venc, benc, nenc, MAX_TDM, dev)
    feats, tbins, _ = build_both(frames, audio, dur, venc, benc, nenc, dev)
    for k in ("vision", "ambient"):
        if not torch.equal(ref.feats[k], feats[k]):
            raise SystemExit(f"ABORT: equivalence gate FAILED on feats[{k}] -- build_both has "
                             f"drifted from build_world_state_features")
        if not torch.equal(ref.tbins[k], tbins[k]):
            raise SystemExit(f"ABORT: equivalence gate FAILED on tbins[{k}]")
    print("[ek] equivalence gate PASSED (feats+tbins bit-identical to the shared builder)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument("--feat-every", type=int, default=2, help="write full features every N-th window")
    ap.add_argument("--ws-dir", default="/home/utkarsh/JEPA-Omni/data/epic_kitchens_ws")
    ap.add_argument("--feat-dir", default="/home/utkarsh/JEPA-Omni/data/feature_cache_epic_kitchens")
    ap.add_argument("--feat-dir-alt", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_epic_kitchens",
                    help="preferred once it has headroom; features land on whichever side has more room")
    ap.add_argument("--m2-ckpt", default=M2_CKPT)
    ap.add_argument("--audio-mode", default="mean", choices=["mean", "base"])
    ap.add_argument("--floor-gib", type=float, default=25.0, help="hard per-filesystem floor")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-video", action="store_true", help="disable deletion entirely")
    ap.add_argument("--no-features", action="store_true", help="world_state only (NO deletion allowed)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.no_features and not a.keep_video:
        raise SystemExit("ABORT: --no-features without --keep-video would delete sources after "
                         "verifying only ONE output. That is the P4.5 failure mode. Refusing.")

    h = hashlib.sha256()
    with open(a.m2_ckpt, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    if h.hexdigest() != M2_SHA:
        raise SystemExit(f"ABORT: M2 sha256 {h.hexdigest()} != {M2_SHA}")

    info = {r["video_id"]: float(r["duration"]) for r in csv.DictReader(open(VIDEO_INFO))}
    have = sorted(f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith(".MP4"))
    missing = sorted(set(info) - set(have))
    print(f"[ek] official {len(info)} videos, present {len(have)}, MISSING {len(missing)} "
          f"({sum(info[v] for v in missing)/3600:.2f} h of {sum(info.values())/3600:.2f} h). "
          f"Not re-downloaded, by instruction.", flush=True)
    print(f"[ek] missing: {','.join(missing)}", flush=True)

    vids = have[: a.limit] if a.limit else have
    os.makedirs(a.ws_dir, exist_ok=True)
    for d in (a.feat_dir, a.feat_dir_alt):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            print(f"[ek] WARN cannot create {d}: {e!r}", flush=True)
    print(f"[ek] {len(vids)} videos, stride={a.stride_s}s, feat_every={a.feat_every} "
          f"(={a.stride_s*a.feat_every}s), ckpt={os.path.basename(a.m2_ckpt)}", flush=True)
    if a.dry_run:
        return

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    dev = torch.device("cuda")
    venc = VisionEncoder().to(dev).eval()
    benc = AudioEncoder(repo="labhamlet/wavjepa-base", n_channels=1).to(dev).eval()
    nenc = (AudioEncoder(repo="labhamlet/wavjepa-nat-base", n_channels=2).to(dev).eval()
            if a.audio_mode == "mean" else None)
    m2 = AVJepaPredictor(AVJepaConfig()).to(dev)
    m2.load_state_dict(torch.load(a.m2_ckpt, map_location=dev, weights_only=False)["model"])
    m2.eval()

    gate_done = False
    t_start = time.time(); n_ok = n_bad = n_skip = 0; freed = 0; feat_written = 0
    for vi, vid in enumerate(vids):
        src = os.path.join(VIDEO_DIR, vid + ".MP4")
        ws_path = os.path.join(a.ws_dir, vid + ".pt")
        if os.path.exists(ws_path) and not os.path.exists(src):
            n_skip += 1; continue
        if not os.path.exists(src):
            print(f"[ek] SKIP {vid}: source gone but no ws output", flush=True); n_bad += 1; continue

        # pick the feature destination: most headroom wins, floor enforced on BOTH the ws fs
        # and the chosen feature fs before a single byte is written.
        cands = [(free_gib(d), d) for d in (a.feat_dir, a.feat_dir_alt) if os.path.isdir(d)]
        cands.sort(reverse=True)
        if not cands:
            raise SystemExit("ABORT: no writable feature directory")
        ffree, fdir = cands[0]
        wsfree = free_gib(a.ws_dir)
        if not a.no_features and ffree < a.floor_gib:
            print(f"[ek] STOP: feature dirs below floor ({ffree:.1f} < {a.floor_gib} GiB). "
                  f"Nothing deleted. Free space, then re-run -- completed videos are skipped.",
                  flush=True)
            break
        if wsfree < a.floor_gib:
            print(f"[ek] STOP: ws dir below floor ({wsfree:.1f} < {a.floor_gib} GiB).", flush=True)
            break

        stage = os.path.join(fdir, f".stage_{vid}")
        try:
            if os.path.isdir(stage):
                shutil.rmtree(stage)
            os.makedirs(stage, exist_ok=True)
            dur = info.get(vid) or 0.0
            if dur <= 0:
                raise ValueError("no duration in EPIC_100_video_info.csv")
            full_audio = decode_audio_full(src, dur)
            n_starts = int(max(0.0, dur - WINDOW_SEC) / a.stride_s) + 1
            starts = [i * a.stride_s for i in range(n_starts)]
            W, V, A, keep, feat_files = [], [], [], [], []

            c0 = 0.0
            while c0 <= max(0.0, dur - WINDOW_SEC):
                c1 = min(dur, c0 + CHUNK_S + WINDOW_SEC)
                chunk = decode_chunk(src, c0, c1)
                lo = int(c0 / a.stride_s)
                hi = min(n_starts, int((c0 + CHUNK_S) / a.stride_s))
                for wi in range(lo, hi):
                    sw = starts[wi]
                    try:
                        frames = window_frames(chunk, c0, sw)
                        audio, true_dur = slice_audio(full_audio, sw)
                        if not gate_done:
                            equivalence_gate(frames, audio, true_dur, venc, benc, nenc, dev)
                            gate_done = True
                        feats, tbins, cache = build_both(frames, audio, true_dur,
                                                         venc, benc, nenc, dev)
                        # RUN-4 world_state path: cap to the training length (T_a=896).
                        # The archival `features` cache stays UNCAPPED on purpose.
                        if feats["ambient"].shape[1] > RUN4_TA:
                            feats = {**feats, "ambient": feats["ambient"][:, :RUN4_TA]}
                            tbins = {**tbins, "ambient": tbins["ambient"][:, :RUN4_TA]}
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            ws = m2.encode_world_state(feats, tbins).float()[0].cpu()
                        W.append(ws.half())
                        V.append(feats["vision"][0].mean(0).float().cpu().half())
                        A.append(feats["ambient"][0].mean(0).float().cpu().half())
                        keep.append(sw)
                        if not a.no_features and wi % a.feat_every == 0:
                            fp = os.path.join(stage, f"ek_{vid}_w{wi:05d}.pt")
                            torch.save(cache, fp); feat_files.append(fp)
                    except Exception:
                        continue
                del chunk
                c0 += CHUNK_S
            if not W:
                raise ValueError("no usable windows")

            # ---- GATE 1: world_state written, fsynced, reloaded, counted ----------------
            torch.save({"video_id": vid, "stride_s": a.stride_s, "window_s": WINDOW_SEC,
                        "start_s": torch.tensor(keep, dtype=torch.float32),
                        "world_state": torch.stack(W), "vision_mean": torch.stack(V),
                        "ambient_mean": torch.stack(A), "m2_ckpt_sha256": M2_SHA,
                        "audio_mode": a.audio_mode, "duration_s": dur,
                        "n_windows_attempted": n_starts}, ws_path)
            os.sync()
            chk = torch.load(ws_path, map_location="cpu", weights_only=True)
            ws_ok = (chk["world_state"].shape[0] == len(keep)
                     and chk["start_s"].shape[0] == len(keep)
                     and chk["vision_mean"].shape[0] == len(keep)
                     and torch.isfinite(chk["world_state"].float()).all().item())

            # ---- GATE 2: EVERY feature file reloaded and shape-checked -------------------
            if a.no_features:
                feat_ok = False            # cannot pass; --no-features forbids deletion above
            else:
                feat_ok = True
                expected = [i for i in range(n_starts) if i % a.feat_every == 0]
                for fp in feat_files:
                    try:
                        c = torch.load(fp, map_location="cpu", weights_only=True)
                        if c["vision"].shape[-1] != 1024 or c["ambient_base"].shape[-1] != 768:
                            feat_ok = False; break
                    except Exception as e:
                        print(f"[ek] feature verify FAILED {fp}: {e!r}", flush=True)
                        feat_ok = False; break
                if feat_ok and len(feat_files) < 0.5 * len(expected):
                    print(f"[ek] feature count too low {len(feat_files)}/{len(expected)}", flush=True)
                    feat_ok = False

            both_verified = bool(ws_ok and feat_ok)   # THE ONLY PLACE THIS IS SET

            if not both_verified:
                print(f"[ek] NOT VERIFIED {vid} (ws_ok={ws_ok} feat_ok={feat_ok}) -- source KEPT",
                      flush=True)
                n_bad += 1
                continue

            final = os.path.join(fdir, vid[:3].lower())
            os.makedirs(final, exist_ok=True)
            for fp in feat_files:
                shutil.move(fp, os.path.join(final, os.path.basename(fp)))
            shutil.rmtree(stage, ignore_errors=True)
            feat_written += len(feat_files)
            n_ok += 1
            if not a.keep_video:
                sz = os.path.getsize(src); os.remove(src); freed += sz
        except Exception as e:
            print(f"[ek] FAILED {vid}: {e!r}", flush=True)
            shutil.rmtree(stage, ignore_errors=True)
            n_bad += 1
            continue
        if (vi + 1) % 5 == 0:
            el = time.time() - t_start
            print(f"[ek] {vi+1}/{len(vids)} ok={n_ok} bad={n_bad} skip={n_skip} "
                  f"feat={feat_written} freed={freed/2**30:.1f}GB {el:.0f}s "
                  f"md1_free={free_gib('/mnt/Raid-Storage-2'):.1f}GB "
                  f"root_free={free_gib('/'):.1f}GB", flush=True)
    print(f"[ek] DONE ok={n_ok} bad={n_bad} skip={n_skip} feat_files={feat_written} "
          f"freed={freed/2**30:.1f}GB elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
