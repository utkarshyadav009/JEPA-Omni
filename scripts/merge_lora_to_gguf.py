"""Merge a LoRA adapter into its base model and convert to GGUF for Jetson.
Used for the LFM2.5-350M fast-tier retrain (bmo_lfm25_350m_lora/best)."""
import argparse, subprocess, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/home/utkarsh/hf_models/LFM2.5-350M")
ap.add_argument("--adapter", default="checkpoints/bmo_lfm25_350m_lora/best")
ap.add_argument("--merged", default="checkpoints/bmo_lfm25_350m_merged")
ap.add_argument("--gguf", default="checkpoints/bmo_lfm25_350m_v1_Q8_0.gguf")
ap.add_argument("--convert", default="/home/utkarsh/repos/llama.cpp/convert_hf_to_gguf.py")
args = ap.parse_args()

print("loading base + adapter, merging ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.merged)
AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True).save_pretrained(args.merged)
print(f"merged -> {args.merged}", flush=True)

print("converting to GGUF Q8_0 ...", flush=True)
r = subprocess.run([sys.executable, args.convert, args.merged, "--outfile", args.gguf, "--outtype", "q8_0"],
                   capture_output=True, text=True)
print(r.stdout[-500:] if r.returncode == 0 else r.stderr[-1500:], flush=True)
print("GGUF:", args.gguf, os.path.getsize(args.gguf) if os.path.exists(args.gguf) else "MISSING", flush=True)
