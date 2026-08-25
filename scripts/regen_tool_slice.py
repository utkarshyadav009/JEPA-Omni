"""Regenerate ONLY the tool-use slice for the companion corpus, in SMALL
batches. The full run's tool slice failed: at n=72 GPT-OSS-120B's harmony
'analysis' preamble + the array blew past max_new_tokens, truncating the JSON
(unterminated ']'). Small batches (n<=10) keep each generation short enough to
finish cleanly. Appends the recovered tool lines to the existing draft."""
import sys, json, time
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model
from scripts.generate_bmo_text_corpus_gptoss import (
    BMO_CHARACTER, TOOL_USE_PROMPT_TEMPLATE, generate_with_retry)

MODEL = "/home/utkarsh/hf_models/gpt-oss-120b"
DRAFT = "/home/utkarsh/JEPA-Omni/data/bmo_companion_tools_v8_DRAFT.jsonl"
BATCHES, PER = 6, 10  # 60 tool lines total, small enough to avoid truncation

print("loading GPT-OSS-120B ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
n_gpus = torch.cuda.device_count()
mm = {i: "85GiB" for i in range(n_gpus)}; mm["cpu"] = "300GiB"
model = dispatch_model(model, device_map=infer_auto_device_map(model, max_memory=mm, no_split_module_classes=model._no_split_modules))
print(f"dispatched across {n_gpus} GPU(s)", flush=True)

got = []
for b in range(BATCHES):
    prompt = TOOL_USE_PROMPT_TEMPLATE.format(character=BMO_CHARACTER, n=PER)
    try:
        lines = generate_with_retry(model, tok, prompt, f"tool_batch{b}", max_new_tokens=2500)
        got.extend([l for l in lines if isinstance(l, str) and "<tool_call" in l])
        print(f"  batch {b}: +{len(lines)} (total {len(got)})", flush=True)
    except Exception as e:
        print(f"  batch {b} failed: {e}", flush=True)

with open(DRAFT, "a") as f:
    for text in got:
        f.write(json.dumps({"text": text, "category": "tool_use",
                            "state": {"energy": 0.6, "mood": "curious"}}) + "\n")
print(f"DONE: appended {len(got)} tool_use lines to {DRAFT}", flush=True)
