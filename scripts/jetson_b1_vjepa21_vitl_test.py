"""scripts/jetson_b1_vjepa21_vitl_test.py — B item 2b: real Jetson memory
+ latency test for V-JEPA 2.1 ViT-L at 384x384/64f/int8 weight-only,
mirroring scripts/jetson_phase0_memory.py's exact methodology (same
tegrastats-ground-truth snapshot(), same q_int8_cpu_then_move quantize-on-
CPU-then-move pattern, same reset_peak_memory_stats+isolated-forward
measurement that produced the 2.43-2.45s current-ViT-L baseline) so the
two numbers are directly comparable.

Run ON the Jetson, after tools/jetson_preflight.sh has PASSED.

Usage:
    python3 jetson_b1_vjepa21_vitl_test.py --ckpt ~/vjepa21_vitl.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time

import torch
import torch.nn as nn


def tegra_ram_mib() -> dict:
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if not m:
        return {"used_mib": None, "total_mib": None, "raw": line}
    return {"used_mib": int(m.group(1)), "total_mib": int(m.group(2)), "raw": line}


def torch_mem_mib() -> dict:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


_RESULTS_PATH = None


def _dump(log):
    if _RESULTS_PATH is None:
        return
    os.makedirs(os.path.dirname(_RESULTS_PATH) or ".", exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump({"log": log, "status": "IN_PROGRESS"}, f, indent=2)


def snapshot(tag: str, log: list) -> dict:
    torch.cuda.synchronize()
    entry = {"tag": tag, "t": time.time(), "tegrastats": tegra_ram_mib(), "torch": torch_mem_mib()}
    log.append(entry)
    print(f"[b2] {tag:45s} tegrastats_used={entry['tegrastats']['used_mib']}MiB  "
          f"torch_alloc={entry['torch']['allocated_mib']:.0f}MiB  "
          f"torch_max_alloc={entry['torch']['max_allocated_mib']:.0f}MiB", flush=True)
    _dump(log)
    return entry


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module: nn.Module, tag: str, device) -> nn.Module:
    module = module.to("cpu")
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
        print(f"[b2]   int8-quantized on CPU: {tag}", flush=True)
    except Exception as e:
        print(f"[b2]   INT8 QUANTIZATION FAILED for {tag}: {e!r} -- moving at original dtype", flush=True)
    gc.collect()
    _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize()
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    return module


def _clean_backbone_key(state_dict):
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.expanduser("~/vjepa21_vitl.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/vjepa21_vitl_jetson_results.json"))
    args = p.parse_args()

    global _RESULTS_PATH
    _RESULTS_PATH = args.out

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    log = []
    print(f"[b2] device={torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)
    snapshot("00_process_start_no_cuda_context", log)

    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize()
    snapshot("01_cuda_context_created", log)

    print("[b2] building V-JEPA 2.1 ViT-L architecture via torch.hub (pretrained=False)...", flush=True)
    encoder, _pred = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_large_384", pretrained=False)
    snapshot("02_arch_built_cpu", log)

    print(f"[b2] loading local checkpoint {args.ckpt} (ema_encoder)...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _clean_backbone_key(ckpt["ema_encoder"])
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"[b2] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del ckpt, sd
    gc.collect()
    _malloc_trim()
    snapshot("03_ckpt_loaded_cpu", log)

    encoder = q_int8_cpu_then_move(encoder, "vjepa21_vitl", device)
    snapshot("04_vjepa21_vitl_int8_on_gpu", log)

    # isolate: ONE 64f/384px forward, peak activation memory -- mirrors
    # jetson_phase0_memory.py's V-JEPA2 ViT-L isolated-forward measurement
    # exactly (same reset_peak_memory_stats + snapshot pattern) for a
    # directly comparable number.
    torch.cuda.reset_peak_memory_stats()
    B, C, T, H, W = 1, 3, 64, 384, 384
    dummy = torch.randn(B, C, T, H, W)
    t0 = time.time()
    with torch.no_grad():
        out = encoder(dummy.to(device))
    torch.cuda.synchronize()
    fwd_s = time.time() - t0
    peak = snapshot("05_VJEPA21_VITL_ISOLATED_FORWARD_64f_384px_int8", log)
    peak["forward_latency_s"] = fwd_s
    peak["output_shape"] = list(out.shape)
    print(f"[b2] *** V-JEPA 2.1 ViT-L isolated forward: {fwd_s:.2f}s, "
          f"peak_torch_alloc={peak['torch']['max_allocated_mib']:.0f}MiB, "
          f"tegrastats_used={peak['tegrastats']['used_mib']}MiB / "
          f"{peak['tegrastats']['total_mib']}MiB total ***", flush=True)

    with open(args.out, "w") as f:
        json.dump({"log": log, "status": "DONE"}, f, indent=2)
    print(f"[b2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
