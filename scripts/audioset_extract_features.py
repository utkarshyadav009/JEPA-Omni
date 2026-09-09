"""Extract frozen features for the audio-only AudioSet probe (Part B).

Two variants, both from the LOCKED checkpoint's own construction path:
  (i)  ambient  -- WavJEPA-base tokens, the audio-only constituent.
  (ii) world_state -- M2's attentive-pooled 1024-d World-State with the VISION STREAM
       ZEROED. This is an AUDIO-ONLY FORWARD PASS THROUGH A FUSION MODEL. It is NOT an
       A-V number and must never be reported as one: AudioSet video is not obtainable
       from any public mirror, so the vision slot carries zeros, not content.

Token sequences are average-pooled to N_TOK=32 steps so the attentive probe has a
sequence to attend over without storing 996x768 per clip (55 GB across both splits).
The linear probe uses the mean over the full sequence, not the pooled one.
"""
import argparse, glob, io, json, os, sys, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.world_state_builder import assert_staircase, _uniform_token_timestamps
from data.av_cached_dataset import _ts_to_tdm_bins
from scripts.extract_features_av import _vision_ts

SR, DUR, NSAMP = 16000, 10.0, 160000
N_TOK = 32
MAX_TDM = 512
VIS_TEMP, VIS_SPAT = 32, 16          # matches the frozen training cache geometry (32,16,1024)
M2_CKPT = os.path.join(PROJECT_ROOT, "checkpoints",
                       "m2_run2_vggsound197k_ego4d134k_neg200", "step19000.pt")


def decode(b):
    w, sr = sf.read(io.BytesIO(b), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)                                  # mono EXACTLY as _decode_audio_raw does
    t = torch.from_numpy(w).unsqueeze(0)
    if sr != SR:
        import torchaudio
        t = torchaudio.functional.resample(t, sr, SR)
    t = t.squeeze(0)
    if t.numel() < NSAMP:
        t = F.pad(t, (0, NSAMP - t.numel()))
    return t[:NSAMP].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["bal_train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")

    wj = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(dev))
    cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                       max_tdm_bins=MAX_TDM, dropout=0.0)
    m2 = AVJepaPredictor(cfg).to(dev)
    ck = torch.load(M2_CKPT, map_location="cpu", weights_only=False)
    m2.load_state_dict(ck["model"], strict=True); m2.eval()
    for p in m2.parameters():
        p.requires_grad_(False)

    # zeroed vision stream, same geometry and staircase binning as training
    vts = _vision_ts(n_temp=VIS_TEMP, dur=DUR, n_frames=VIS_TEMP * 2)
    vts_exp = vts.unsqueeze(1).expand(VIS_TEMP, VIS_SPAT, 2).reshape(-1, 2)
    vbins1 = _ts_to_tdm_bins(vts_exp, DUR, MAX_TDM)
    assert_staircase(vbins1, MAX_TDM, n_temp=VIS_TEMP, n_spat=VIS_SPAT)
    print("[extract] staircase gate PASSED on the zeroed vision stream "
          "(n_temp=%d n_spat=%d -> %d tokens)" % (VIS_TEMP, VIS_SPAT, vbins1.numel()), flush=True)

    files = sorted(glob.glob("/mnt/Raid-Storage-2/utkarsh-data/audioset_hf/data/%s/*.parquet" % a.split))
    ids, labs, amb_m, amb_t, ws_v, ws_t = [], [], [], [], [], []
    n = 0; t0 = time.time(); bad = 0
    buf_w, buf_i, buf_l = [], [], []

    def flush():
        nonlocal buf_w, buf_i, buf_l
        if not buf_w:
            return
        wav = torch.stack(buf_w).unsqueeze(1).to(dev)          # (B,1,NSAMP)
        with torch.no_grad():
            amb = wj.encode(wav)                                # (B,T,768)
            B, T, _ = amb.shape
            am = amb.float().mean(1)                            # linear-probe feature
            at = F.adaptive_avg_pool1d(amb.float().transpose(1, 2), N_TOK).transpose(1, 2)
            ats = _uniform_token_timestamps(T, DUR)
            abins = _ts_to_tdm_bins(ats, DUR, MAX_TDM).to(dev).unsqueeze(0).expand(B, -1)
            feats = {"vision": torch.zeros(B, VIS_TEMP * VIS_SPAT, 1024, device=dev),
                     "ambient": amb.float()}
            tb = {"vision": vbins1.to(dev).unsqueeze(0).expand(B, -1), "ambient": abins}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                w_s = m2.encode_world_state(feats, tb).float()      # (B,1024) UN-normalised
                pre = m2.encode_pre_pool_tokens(feats, tb).float()  # (B,S,1024)
            wt = F.adaptive_avg_pool1d(pre.transpose(1, 2), N_TOK).transpose(1, 2)
        ids.extend(buf_i); labs.extend(buf_l)
        amb_m.append(am.cpu().half()); amb_t.append(at.cpu().half())
        ws_v.append(w_s.cpu().half()); ws_t.append(wt.cpu().half())
        buf_w, buf_i, buf_l = [], [], []

    for fp in files:
        for b in pq.ParquetFile(fp).iter_batches(batch_size=64):
            for d in b.to_pylist():
                try:
                    buf_w.append(decode(d["audio"]["bytes"]))
                except Exception:
                    bad += 1; continue
                buf_i.append(d["video_id"]); buf_l.append(list(d["labels"]))
                if len(buf_w) >= a.batch:
                    flush(); n = len(ids)
                    if n % 1024 < a.batch:
                        el = time.time() - t0
                        print("[extract] %s %d  %.1f clips/s  bad=%d" % (a.split, n, n / max(el, 1e-9), bad), flush=True)
            if a.limit and len(ids) >= a.limit:
                break
        if a.limit and len(ids) >= a.limit:
            break
    flush()

    torch.save({"ids": ids, "labels": labs,
                "ambient_mean": torch.cat(amb_m), "ambient_tokens": torch.cat(amb_t),
                "world_state": torch.cat(ws_v), "world_state_tokens": torch.cat(ws_t),
                "n_tok": N_TOK, "vision_zeroed": True,
                "m2_ckpt": M2_CKPT, "split": a.split, "bad": bad}, a.out)
    print("[extract] WROTE %s  n=%d  bad=%d  %.1f min" % (a.out, len(ids), bad, (time.time()-t0)/60), flush=True)


if __name__ == "__main__":
    main()
