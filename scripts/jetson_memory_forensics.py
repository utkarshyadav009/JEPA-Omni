"""scripts/jetson_memory_forensics.py — where does the memory ACTUALLY go?

TWO QUESTIONS, one measurement.

**Q1: SigLIP2 measures 519-1,170 MiB resident but its vision tower is 177 MiB of fp16
weights.** Where are the other ~400 MiB? Candidates, and they need very different fixes:
  * torch's CUDA caching allocator holding freed blocks (fix: config / empty_cache)
  * activation + workspace peaks from encoding 4 frames at once (fix: batch 1)
  * the CUDA context itself (fix: nothing, it is a fixed tax)
  * Python + torch import overhead attributed to the model by mistake (fix: measure properly)

The distinction is measurable: `torch.cuda.memory_allocated()` counts live tensors,
`memory_reserved()` counts the pool torch has taken from the driver, and `free -m` counts
what the OS lost. Their differences name the culprit.

**Q2: would a C++ runtime be better?** That is really "how much of the footprint is the
Python/torch stack rather than the models". The stack is already mostly C++ underneath --
llama.cpp for both LLMs, sherpa-onnx for STT, ONNX Runtime for the codec -- and PyTorch is
the one Python-heavy piece, carrying perception. If `import torch` alone costs more than
SigLIP2's weights, that reframes the whole question; if it is small, a C++ rewrite buys
little and costs a great deal.

Prints a ledger where every line is attributable, so the answer is not a guess.

Run on the Jetson:
    python3 scripts/jetson_memory_forensics.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

PROD = os.path.expanduser("~/bmo_production")
sys.path.insert(0, f"{PROD}/pipeline")


def sys_avail() -> float:
    """MiB available per the OS -- the only number that matters on unified memory."""
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    return float(out.splitlines()[1].split()[-1])


def rss() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


class Ledger:
    def __init__(self):
        self.rows = []
        self.a0 = sys_avail()
        self.r0 = rss()

    def mark(self, tag: str, torch_mod=None):
        a, r = sys_avail(), rss()
        row = {"stage": tag, "sys_used_MiB": round(self.a0 - a, 1), "rss_MiB": round(r, 1)}
        if torch_mod is not None and torch_mod.cuda.is_available():
            row["cuda_alloc_MiB"] = round(torch_mod.cuda.memory_allocated() / 2**20, 1)
            row["cuda_reserved_MiB"] = round(torch_mod.cuda.memory_reserved() / 2**20, 1)
        self.rows.append(row)
        extra = ""
        if "cuda_alloc_MiB" in row:
            extra = (f"  cuda_alloc={row['cuda_alloc_MiB']:7.1f}"
                     f"  cuda_reserved={row['cuda_reserved_MiB']:7.1f}")
        print(f"{tag:38s} sys_used={row['sys_used_MiB']:7.1f}  rss={row['rss_MiB']:7.1f}{extra}",
              flush=True)
        return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4, help="scene frames per encode")
    ap.add_argument("--out", default=os.path.expanduser("~/memory_forensics.json"))
    args = ap.parse_args()

    L = Ledger()
    L.mark("00_bare_python")

    import numpy as np                                    # noqa: F401
    L.mark("01_numpy")

    import torch
    L.mark("02_import_torch", torch)

    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
    L.mark("03_cuda_context", torch)

    import transformers                                    # noqa: F401
    L.mark("04_import_transformers", torch)

    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    L.mark("05_processor", torch)

    m = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16)
    L.mark("06_siglip_cpu_both_towers", torch)

    del m.text_model
    gc.collect()
    L.mark("07_text_tower_dropped", torch)

    m = m.to("cuda").eval()
    L.mark("08_vision_tower_on_gpu", torch)

    from PIL import Image
    imgs = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
            for _ in range(args.frames)]
    px = proc(images=imgs, return_tensors="pt").to("cuda")
    px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in px.items()}
    L.mark(f"09_pixels_{args.frames}frames", torch)

    with torch.no_grad():
        t0 = time.time()
        o = m.vision_model(**px)
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
    L.mark(f"10_after_forward ({ms:.0f} ms)", torch)

    del o
    gc.collect()
    L.mark("11_del_output", torch)

    torch.cuda.empty_cache()
    L.mark("12_after_empty_cache", torch)

    # one frame at a time -- does the peak follow batch size?
    px1 = {k: (v[:1] if hasattr(v, "shape") else v) for k, v in px.items()}
    with torch.no_grad():
        for _ in range(args.frames):
            o1 = m.vision_model(**px1)
        torch.cuda.synchronize()
    del o1
    gc.collect()
    L.mark("13_sequential_1frame_x4", torch)
    torch.cuda.empty_cache()
    L.mark("14_empty_cache_again", torch)

    total = L.rows[-1]["sys_used_MiB"]
    ctx = L.rows[2]["sys_used_MiB"] - L.rows[1]["sys_used_MiB"]
    torch_imp = L.rows[1]["sys_used_MiB"] - L.rows[0]["sys_used_MiB"]
    weights = L.rows[7]["sys_used_MiB"] - L.rows[6]["sys_used_MiB"]
    print("\n" + "=" * 74)
    print(f"{'python + numpy + torch import':38s} {torch_imp:7.1f} MiB   <- C++ would remove this")
    print(f"{'CUDA context (fixed tax)':38s} {ctx:7.1f} MiB   <- any runtime pays it")
    print(f"{'vision tower weights -> GPU':38s} {weights:7.1f} MiB   <- what int8 could halve")
    print(f"{'TOTAL at end':38s} {total:7.1f} MiB")
    print("=" * 74)
    print("cuda_reserved - cuda_alloc = caching allocator holding freed blocks.")
    print("If that gap is large, the fix is allocator config, not quantization.")

    with open(args.out, "w") as f:
        json.dump(L.rows, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
