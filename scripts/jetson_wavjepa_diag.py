"""scripts/jetson_wavjepa_diag.py — isolate WHERE WavJEPA-base's memory
footprint actually goes (measured +1593MiB on Jetson vs a theoretical
~352MiB post-int8 estimate). Fine-grained tegrastats snapshots at every
sub-step of construction / quantization / GPU move, run on the Jetson only
(unified-memory behavior does not reproduce on a discrete-GPU box).

Usage: python3 jetson_wavjepa_diag.py
"""
import gc
import re
import subprocess
import sys
import time

import torch
import torch.nn as nn


def tegra_used_mib():
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    return int(m.group(1)) if m else None


def snap(tag):
    torch.cuda.synchronize()
    t = tegra_used_mib()
    ta = torch.cuda.memory_allocated() / 1024**2
    print(f"[diag] {tag:45s} tegrastats={t}MiB  torch_alloc={ta:.1f}MiB", flush=True)
    return t, ta


device = torch.device("cuda")
_ = torch.zeros(1, device=device)
torch.cuda.synchronize()
snap("00_cuda_context")

from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, _find_snapshot, _import_snapshot_pkg
import importlib, os

snap_dir = _find_snapshot(WAVJEPA_BASE_REPO)
pkg = _import_snapshot_pkg(snap_dir)
model_mod = importlib.import_module(f"{pkg}.modeling_wavjepa")
cfg_mod = importlib.import_module(f"{pkg}.configuration_wavjepa")
ModelCls = getattr(model_mod, "WavJEPAModel")
ConfigCls = getattr(cfg_mod, "WavJEPAConfig")

config = ConfigCls.from_pretrained(WAVJEPA_BASE_REPO, trust_remote_code=True)
snap("01_config_loaded")

model = ModelCls(config)
snap("02_fresh_fp32_model_constructed_on_cpu")

sf_path = os.path.join(snap_dir, "model.safetensors")
from safetensors.torch import load_file
sd = load_file(sf_path)
snap("03_state_dict_loaded_from_disk_cpu")

model.load_state_dict(sd, strict=True)
snap("04_state_dict_applied")

del sd
gc.collect()
snap("05_after_del_sd_and_gc")

model = model.float()
snap("06_after_explicit_float_cast")

for p in model.parameters():
    p.requires_grad_(False)
model.eval()
snap("07_frozen_eval")

# ---- quantize on CPU ----
from torchao.quantization import quantize_, Int8WeightOnlyConfig
quantize_(model, Int8WeightOnlyConfig())
snap("08_int8_quantized_still_on_cpu")

gc.collect()
snap("09_after_gc_post_quant")

# ---- move to GPU ----
model = model.to(device)
torch.cuda.synchronize()
snap("10_moved_to_gpu")

gc.collect()
torch.cuda.empty_cache()
snap("11_after_gc_and_empty_cache_on_gpu")

# ---- param accounting ----
total_params = sum(p.numel() for p in model.parameters())
linear_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters())
print(f"\n[diag] total_params={total_params/1e6:.1f}M  linear_params={linear_params/1e6:.1f}M  "
      f"other_params={(total_params-linear_params)/1e6:.1f}M")

# actual per-parameter dtype/byte breakdown post-quantization
bytes_by_dtype = {}
for n, p in model.named_parameters():
    key = str(p.dtype)
    bytes_by_dtype[key] = bytes_by_dtype.get(key, 0) + p.numel() * p.element_size()
print("[diag] regular nn.Parameter bytes by dtype (does NOT include torchao's wrapped int8 tensors "
      "if they're not exposed as plain .dtype -- see next line):")
for k, v in bytes_by_dtype.items():
    print(f"    {k}: {v/1024**2:.1f} MiB")

# torchao-quantized tensors often show up as AffineQuantizedTensor wrapping
# int8 data + a separate fp scale/zero-point -- walk buffers too, and try
# to size the actual underlying storage directly
total_storage_mib = 0.0
for n, p in model.named_parameters():
    try:
        # AffineQuantizedTensor exposes .tensor_impl or similar; fall back
        # to naive numel*element_size if it's a plain tensor
        nbytes = p.numel() * p.element_size()
    except Exception as e:
        nbytes = -1
    total_storage_mib += max(0, nbytes) / 1024**2
print(f"[diag] naive numel*element_size total: {total_storage_mib:.1f} MiB")
