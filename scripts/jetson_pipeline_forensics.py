"""scripts/jetson_pipeline_forensics.py — attribute EVERY MiB in the pipeline, not just SigLIP2.

The SigLIP2 audit found that `transformers.AutoProcessor` cost 327-500 MiB for a resize and a
normalise -- more than the model it fed, and 4-5x what perfect int8 of that model could save.
That was not a SigLIP2 problem. It was a pattern:

    models/vision_encoder.py:51   AutoVideoProcessor.from_pretrained(...)   <- V-JEPA2
    models/audio_encoder.py:182   AutoFeatureExtractor.from_pretrained(...) <- WavJEPA
    scripts/extract_siglip2_scene*.py   AutoProcessor.from_pretrained(...)  <- TRAINING, x64 shards

So this stages the whole on-device pipeline and, for each component, separates:

    MODEL WEIGHTS       -- irreducible without quantization
    FRAMEWORK OVERHEAD  -- processors, feature extractors, tokenizers: usually replaceable
    RUNTIME/WORKSPACE   -- CUDA context, cuBLAS/cuDNN scratch: paid in any language

and prints the split, because the fix is completely different for each.

TRAINING MATTERS TOO, and more than it looks. The extraction scripts run **64 shards in
parallel** (the measured-optimal config: 64 x 4 threads = 256 threads = core count). If each
shard pays ~500 MiB for a processor it barely uses, that is up to **32 GB of RAM** spent on
image resizing during a run that was already CPU-bound. `--training-estimate` reports that
number rather than leaving it as an intuition.

Run on the Jetson:
    python3 scripts/jetson_pipeline_forensics.py
    python3 scripts/jetson_pipeline_forensics.py --training-estimate
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


def avail() -> float:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    return float(out.splitlines()[1].split()[-1])


class Audit:
    def __init__(self):
        self.rows = []
        self.prev = avail()
        self.start = self.prev

    def step(self, tag: str, kind: str = "model"):
        now = avail()
        d = self.prev - now
        self.prev = now
        self.rows.append({"stage": tag, "kind": kind, "delta_MiB": round(d, 1),
                          "cum_MiB": round(self.start - now, 1)})
        print(f"  {tag:42s} {kind:9s} {d:+8.1f} MiB   (cum {self.start-now:7.1f})", flush=True)
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-estimate", action="store_true",
                    help="report the per-shard processor cost x the 64-shard extraction fan-out")
    ap.add_argument("--out", default=os.path.expanduser("~/pipeline_forensics.json"))
    args = ap.parse_args()

    A = Audit()
    print("\n=== BASELINE ===", flush=True)
    import numpy as np                                   # noqa: F401
    A.step("numpy", "framework")
    import torch
    A.step("import torch", "framework")
    torch.cuda.init(); _ = torch.zeros(1, device="cuda")
    A.step("CUDA context", "runtime")
    import transformers                                  # noqa: F401
    A.step("import transformers", "framework")

    print("\n=== PERCEPTION: V-JEPA2 ===", flush=True)
    from transformers import AutoVideoProcessor
    vp = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    proc_v = A.step("AutoVideoProcessor  <-- suspect", "framework")
    from transformers import AutoModel
    vm = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.bfloat16)
    A.step("V-JEPA2 ViT-L weights (CPU)", "model")
    vm = vm.to("cuda").eval()
    A.step("V-JEPA2 -> GPU", "model")

    print("\n=== PERCEPTION: WavJEPA ===", flush=True)
    try:
        from transformers import AutoFeatureExtractor
        _fe = AutoFeatureExtractor.from_pretrained("labhamlet/wavjepa-base",
                                                   trust_remote_code=True)
        proc_a = A.step("AutoFeatureExtractor  <-- suspect", "framework")
    except Exception as e:
        print(f"    (feature extractor unavailable: {type(e).__name__})", flush=True)
        proc_a = 0.0

    print("\n=== PERCEPTION: SigLIP2 (processor already removed) ===", flush=True)
    sm = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16)
    del sm.text_model
    gc.collect()
    A.step("SigLIP2 vision tower (CPU)", "model")
    sm = sm.to("cuda").eval()
    A.step("SigLIP2 -> GPU", "model")
    from models.m5_motion_crop import siglip2_preprocess     # the 6-line replacement
    A.step("siglip2_preprocess (our version)", "framework")

    print("\n=== FORWARD PASSES (workspace) ===", flush=True)
    from PIL import Image
    imgs = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8")) for _ in range(4)]
    px = siglip2_preprocess(imgs).to("cuda").to(torch.bfloat16)
    with torch.no_grad():
        sm.vision_model(pixel_values=px)
        torch.cuda.synchronize()
    A.step("SigLIP2 forward", "runtime")

    totals = {}
    for r in A.rows:
        totals[r["kind"]] = totals.get(r["kind"], 0.0) + r["delta_MiB"]
    print("\n" + "=" * 78)
    for k in ("model", "framework", "runtime"):
        v = totals.get(k, 0.0)
        note = {"model": "irreducible without quantization",
                "framework": "<- usually REPLACEABLE",
                "runtime": "paid in any language"}[k]
        print(f"  {k.upper():10s} {v:8.1f} MiB   {note}")
    print(f"  {'TOTAL':10s} {sum(totals.values()):8.1f} MiB")
    print("=" * 78)

    proc_total = proc_v + proc_a
    print(f"\n  processors/extractors alone: {proc_total:.1f} MiB on-device")
    if args.training_estimate:
        print("\n=== TRAINING-SIDE EXTRAPOLATION ===")
        print("  scripts/extract_siglip2_scene*.py load AutoProcessor PER SHARD, and the")
        print("  measured-optimal extraction config is 64 shards (64 x 4 threads = 256 cores).")
        for per in (327.0, 500.0):
            print(f"    {per:5.0f} MiB/shard x 64 shards = {per*64/1024:6.1f} GB of RAM")
        print("  ...during a run that was already CPU-bound at 249 clips/s.")

    with open(args.out, "w") as f:
        json.dump({"rows": A.rows, "totals": totals,
                   "processor_MiB": proc_total}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
