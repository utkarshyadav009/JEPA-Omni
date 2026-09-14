"""P4.15 step 10 -- MEASURE the temporal receptive field of V-JEPA2 and WavJEPA.

Until now the claim "one window's receptive field is the whole 10 s" was derived from
CONFIGURATION (fpc64 over a 10 s window; WavJEPA at 100 Hz over the same window). That is an
argument, not a measurement, and it is load-bearing: it is the sole reason Delta < 10 s is
refused as a prediction horizon at any stride, on any corpus.

Method -- a causal influence matrix, not an appeal to architecture:
  1. encode a real clip, keep the output tokens as a baseline
  2. for each input time-slice j, corrupt ONLY that slice and re-encode
  3. record the relative L2 change induced at every OUTPUT time-group i
  => M[i][j] = how much input slice j moves output group i.

Reading it:
  * a LOCAL encoder gives a banded M -- influence concentrated near i == j
  * a GLOBAL encoder gives a dense M -- every slice moves every group
The quantity that decides the Delta floor is the fraction of influence that is NON-LOCAL: if
output group 0 moves when the last input slice is corrupted, then a target one slice away shares
information with the query and is not a prediction target.

Corruption is per-slice Gaussian noise at the clip's own scale (a mean-fill would be a weaker
probe: for some inputs the mean IS roughly the content).
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

WINDOW_S, AUDIO_SR = 10.0, 16000


def rel_change(a, b):
    return (a - b).norm().item() / max(a.norm().item(), 1e-8)


def vision_rf(venc, frames, dev, n_slices=16):
    """frames: (64,3,256,256) uint8. 64 frames, tubelet 2 => 32 output temporal groups."""
    x = frames.unsqueeze(0).to(dev)
    with torch.no_grad():
        base = venc.encode(x)[0].float()                     # (32*256, 1024)
    n_tok = base.shape[0]
    n_temp = n_tok // 256
    base_g = base.view(n_temp, 256, -1)
    T = frames.shape[0]
    per = T // n_slices
    M = torch.zeros(n_temp, n_slices)
    g = torch.Generator(device="cpu").manual_seed(0)
    for j in range(n_slices):
        pert = frames.clone().float()
        sl = slice(j * per, (j + 1) * per)
        noise = torch.randn(pert[sl].shape, generator=g) * pert.float().std() + pert.float().mean()
        pert[sl] = noise.clamp(0, 255)
        with torch.no_grad():
            out = venc.encode(pert.to(torch.uint8).unsqueeze(0).to(dev))[0].float().view(n_temp, 256, -1)
        for i in range(n_temp):
            M[i, j] = rel_change(out[i], base_g[i])
    return M, n_temp


def audio_rf(aenc, wav, dev, n_slices=40, n_channels=1):
    w = wav.unsqueeze(0).unsqueeze(0) if n_channels == 1 else wav.unsqueeze(0).expand(2, -1).unsqueeze(0)
    with torch.no_grad():
        base = aenc.encode(w.to(dev))[0].float()             # (T_a, 768)
    T_a = base.shape[0]
    per_out = T_a / n_slices
    per_in = wav.shape[0] // n_slices
    M = torch.zeros(n_slices, n_slices)
    g = torch.Generator(device="cpu").manual_seed(0)
    for j in range(n_slices):
        pert = wav.clone()
        sl = slice(j * per_in, (j + 1) * per_in)
        pert[sl] = torch.randn(pert[sl].shape, generator=g) * wav.std()
        pw = pert.unsqueeze(0).unsqueeze(0) if n_channels == 1 else pert.unsqueeze(0).expand(2, -1).unsqueeze(0)
        with torch.no_grad():
            out = aenc.encode(pw.to(dev))[0].float()
        for i in range(n_slices):
            a, b = int(i * per_out), int((i + 1) * per_out)
            M[i, j] = rel_change(out[a:b], base[a:b])
    return M, T_a


def summarise(M, name, window_s=WINDOW_S):
    n_out, n_in = M.shape
    Mn = M / M.max().clamp_min(1e-8)
    diag_band = torch.zeros_like(Mn, dtype=torch.bool)
    for i in range(n_out):
        j = int(round(i * (n_in - 1) / max(n_out - 1, 1)))
        for d in (-1, 0, 1):
            if 0 <= j + d < n_in:
                diag_band[i, j + d] = True
    local = Mn[diag_band].sum().item()
    total = Mn.sum().item()
    # the decisive number: does the FIRST output group respond to the LAST input slice?
    far = Mn[0, -1].item()
    near = Mn[0, 0].item()
    slice_s = window_s / n_in
    print(f"\n=== {name} ===")
    print(f"  output groups {n_out}, input slices {n_in} ({slice_s:.3f} s each)")
    print(f"  non-local share of influence : {100*(1-local/total):.1f}%")
    print(f"  M[first_out, last_in]        : {far:.4f}  (vs self-slice {near:.4f}, ratio {far/max(near,1e-8):.3f})")
    print(f"  min over ALL cells           : {Mn.min().item():.4f}  (0 would mean a true blind spot)")
    print(f"  row 0 (earliest output group), normalised:")
    print("   ", " ".join(f"{v:.2f}" for v in Mn[0].tolist()))
    print(f"  row -1 (latest output group), normalised:")
    print("   ", " ".join(f"{v:.2f}" for v in Mn[-1].tolist()))
    return {"n_out": n_out, "n_in": n_in, "slice_s": slice_s,
            "nonlocal_share": 1 - local / total, "first_out_last_in": far,
            "first_out_self": near, "min_cell": Mn.min().item(),
            "row_first": Mn[0].tolist(), "row_last": Mn[-1].tolist(),
            "matrix": Mn.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p4_receptive_field.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder

    # a REAL clip, not noise: receptive field on out-of-distribution input is not the claim
    src = a.clip or "/mnt/Raid-Storage-2/utkarsh-data/epic_kitchens/video/" + \
          sorted(os.listdir("/mnt/Raid-Storage-2/utkarsh-data/epic_kitchens/video"))[0]
    print(f"[rf] clip: {src}")
    import importlib.util
    sp = importlib.util.spec_from_file_location("ek", "scripts/temporal_probe/extract_epic_kitchens.py")
    ek = importlib.util.module_from_spec(sp); sp.loader.exec_module(ek)
    chunk = ek.decode_chunk(src, 0.0, WINDOW_S)
    frames = ek.window_frames(chunk, 0.0, 0.0)
    wav = ek.decode_audio_full(src, WINDOW_S)[: int(WINDOW_S * AUDIO_SR)]
    print(f"[rf] frames {tuple(frames.shape)}  audio {tuple(wav.shape)}")

    res = {}
    venc = VisionEncoder().to(dev).eval()
    Mv, n_temp = vision_rf(venc, frames, dev)
    res["vjepa2"] = summarise(Mv, "V-JEPA2 ViT-L (fpc64, 10 s window)")
    del venc; torch.cuda.empty_cache()

    import statistics
    for key, repo, ch in [("wavjepa_base", "labhamlet/wavjepa-base", 1),
                          ("wavjepa_nat", "labhamlet/wavjepa-nat-base", 2)]:
        enc = AudioEncoder(repo=repo, n_channels=ch).to(dev).eval()
        Ma, T_a = audio_rf(enc, wav, dev, n_channels=ch)
        res[key] = summarise(Ma, f"{repo} (10 s window, {T_a} tokens)")
        Mn = Ma / Ma.max()
        n = Mn.shape[0]; sl = WINDOW_S / n
        reach = [max(j for j in range(n) if Mn[i, j] > 0.01) - min(j for j in range(n) if Mn[i, j] > 0.01) + 1
                 for i in range(n) if (Mn[i] > 0.01).any()]
        res[key]["rf_span_median_s"] = statistics.median(reach) * sl
        res[key]["rf_span_max_s"] = max(reach) * sl
        res[key]["frac_cells_exactly_zero"] = (Mn <= 0).float().mean().item()
        print(f"  MEASURED receptive field: median {res[key]['rf_span_median_s']:.2f} s, "
              f"max {res[key]['rf_span_max_s']:.2f} s, {100*res[key]['frac_cells_exactly_zero']:.0f}% cells exactly 0")
        del enc; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[rf] wrote {a.out}")
