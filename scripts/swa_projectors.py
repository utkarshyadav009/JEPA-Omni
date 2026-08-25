"""Stochastic Weight Averaging of projector checkpoints -- average the weights
of several late checkpoints (e.g. 8k/10k/12k) to smooth the alignment space
without over-indexing on any single step's sequence-length bias."""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sds = [torch.load(c, map_location="cpu") for c in a.ckpts]
keys = sds[0].keys()
avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}
avg = {k: avg[k].to(sds[0][k].dtype) for k in keys}
torch.save(avg, a.out)
print(f"SWA of {len(sds)} checkpoints -> {a.out}", flush=True)
