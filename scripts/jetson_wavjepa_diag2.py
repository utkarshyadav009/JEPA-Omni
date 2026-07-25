"""scripts/jetson_wavjepa_diag2.py — clean, single-process v1-vs-v2
comparison of torchao Int8WeightOnlyConfig on the REAL WavJEPA-base model
(not a toy Linear stack), to remove cross-process tegrastats noise as a
confound in the earlier full-pipeline before/after comparison.
"""
import ctypes
import gc
import re
import subprocess
import time

import torch
import torch.nn as nn


def trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def tegra():
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/", line)
    return int(m.group(1)) if m else None


def load_fresh_model():
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, _find_snapshot, _import_snapshot_pkg
    import importlib, os
    snap_dir = _find_snapshot(WAVJEPA_BASE_REPO)
    pkg = _import_snapshot_pkg(snap_dir)
    model_mod = importlib.import_module(f"{pkg}.modeling_wavjepa")
    cfg_mod = importlib.import_module(f"{pkg}.configuration_wavjepa")
    ModelCls = getattr(model_mod, "WavJEPAModel")
    ConfigCls = getattr(cfg_mod, "WavJEPAConfig")
    config = ConfigCls.from_pretrained(WAVJEPA_BASE_REPO, trust_remote_code=True)
    model = ModelCls(config)
    sf_path = os.path.join(snap_dir, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(sf_path)
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.float()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


from torchao.quantization import quantize_, Int8WeightOnlyConfig

print(f"[diag2] baseline: {tegra()}MiB", flush=True)

print("\n=== VERSION 1 (deprecated, suspected leak) ===", flush=True)
m1 = load_fresh_model()
gc.collect()
after_construct_v1 = tegra()
print(f"[diag2] after construct (fp32, fresh): {after_construct_v1}MiB", flush=True)
quantize_(m1, Int8WeightOnlyConfig(version=1))
gc.collect()
trim()
time.sleep(1)
after_quant_v1 = tegra()
print(f"[diag2] after quantize v1 + gc: {after_quant_v1}MiB  (delta from construct: {after_quant_v1-after_construct_v1:+d})", flush=True)
del m1
gc.collect()
trim()
time.sleep(1)
after_del_v1 = tegra()
print(f"[diag2] after del m1 + gc: {after_del_v1}MiB", flush=True)

print("\n=== VERSION 2 (recommended) ===", flush=True)
m2 = load_fresh_model()
gc.collect()
after_construct_v2 = tegra()
print(f"[diag2] after construct (fp32, fresh): {after_construct_v2}MiB", flush=True)
quantize_(m2, Int8WeightOnlyConfig(version=2))
gc.collect()
trim()
time.sleep(1)
after_quant_v2 = tegra()
print(f"[diag2] after quantize v2 + gc: {after_quant_v2}MiB  (delta from construct: {after_quant_v2-after_construct_v2:+d})", flush=True)
del m2
gc.collect()
trim()
time.sleep(1)
after_del_v2 = tegra()
print(f"[diag2] after del m2 + gc: {after_del_v2}MiB", flush=True)

print(f"\n[diag2] === SUMMARY (single process, same baseline, no cross-process noise) ===")
print(f"[diag2] v1 steady-state after quantize: {after_quant_v1}MiB (construct was {after_construct_v1}MiB)")
print(f"[diag2] v2 steady-state after quantize: {after_quant_v2}MiB (construct was {after_construct_v2}MiB)")
print(f"[diag2] v1 quantize delta: {after_quant_v1-after_construct_v1:+d}MiB   v2 quantize delta: {after_quant_v2-after_construct_v2:+d}MiB")
