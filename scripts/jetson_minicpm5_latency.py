"""scripts/jetson_minicpm5_latency.py — real Jetson latency/memory measurement
for openbmb/MiniCPM5-1B (1.08B params, 24 layers, GQA 16Q/2KV, hidden_size=1536
-- matches Qwen2.5-1.5B's hidden_size exactly, so drop-in compatible with
M3Connector/UltravoxProjectorConfig's llm_hidden if it replaces Qwen later).

Directly comparable to the existing Qwen2.5-1.5B-int8 numbers already measured
on this Jetson: A1 (checkpoints/m5_jetson/PHASE_A1_LOCKED_CHECKPOINTS_MEMORY_RESULTS.json)
measured 12.2s for 60 tokens = ~203ms/token as part of the full stack. This
script isolates JUST the LLM (no ViT-L/WavJEPA/M2/M3 loaded) for a clean
apples-to-apples per-token latency number, same int8 + malloc_trim recipe
already proven on this Jetson (scripts/jetson_m5_live_demo.py's fix for the
NvMap OOM crash).

Usage:
    python3 scripts/jetson_minicpm5_latency.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import re
import time
import types
import importlib.machinery

_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import torch

PROMPTS = [
    "Describe what a busy city street looks like.",
    "Explain how a bicycle works in simple terms.",
    "What is the weather like on a sunny day?",
    "Tell me about your favorite kind of music.",
    "Summarize what happens during a thunderstorm.",
]


def tegra_ram_mib() -> dict:
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if not m:
        return {"used_mib": None, "total_mib": None}
    return {"used_mib": int(m.group(1)), "total_mib": int(m.group(2))}


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, device):
    import gc
    module = module.to("cpu")
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
        print("[minicpm5-latency]   int8-quantized", flush=True)
    except Exception as e:
        print(f"[minicpm5-latency]   int8 quant failed (running bf16 instead): {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openbmb/MiniCPM5-1B")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--out", default=os.path.expanduser("~/jetson_minicpm5_latency_results.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[minicpm5-latency] device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)

    ram_start = tegra_ram_mib()
    print(f"[minicpm5-latency] RAM before load: {ram_start}", flush=True)

    print(f"[minicpm5-latency] loading {args.model}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    load_s = time.perf_counter() - t0
    print(f"[minicpm5-latency]   HF load took {load_s:.1f}s", flush=True)

    llm = q_int8_cpu_then_move(llm, device)
    llm.eval()
    ram_after_load = tegra_ram_mib()
    print(f"[minicpm5-latency] RAM after load+quantize: {ram_after_load}", flush=True)

    print(f"[minicpm5-latency] hidden_size={llm.config.hidden_size} "
          f"num_hidden_layers={getattr(llm.config, 'num_hidden_layers', '?')} "
          f"vocab_size={llm.config.vocab_size}", flush=True)

    # warm-up (first call often pays extra one-time cost, same convention as jetson_tts_latency.py)
    def to_chat_ids(text: str):
        # MiniCPM5-1B has a real chat_template (confirmed via tokenizer.chat_template
        # is not None, 2026-08-01) -- feeding raw prompt text directly (as the first
        # version of this script did) produced quiz-completion-style garbage instead
        # of answering the prompt; this is a formatting mismatch, not a broken model.
        msgs = [{"role": "user", "content": text}]
        # enable_thinking=False (2026-08-01 finding): default thinking mode burns the
        # whole token budget inside <think>...</think> and never reaches an answer at
        # max_new_tokens=60 -- the docs explicitly flag this ("always use
        # enable_thinking=False... with thinking ON the model may not reach a
        # completed function call"). Real speed number needs actual answers, not
        # truncated reasoning.
        out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             enable_thinking=False, return_tensors="pt")
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return ids.to(device)

    warm_ids = to_chat_ids("Warm up.")
    with torch.no_grad():
        llm.generate(warm_ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    per_token_ms: list[float] = []
    total_latencies_s: list[float] = []
    generations = []
    for i, prompt in enumerate(PROMPTS[: args.n_prompts]):
        ids = to_chat_ids(prompt)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = llm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - ids.shape[1]
        ms_per_tok = (elapsed / max(1, n_new)) * 1000.0
        per_token_ms.append(ms_per_tok)
        total_latencies_s.append(elapsed)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        generations.append({"prompt": prompt, "n_new_tokens": n_new, "elapsed_s": elapsed,
                             "ms_per_token": ms_per_tok, "text": text})
        print(f"[minicpm5-latency] [{i+1}/{args.n_prompts}] {n_new} tokens in {elapsed:.2f}s "
              f"({ms_per_tok:.1f}ms/token) -> {text!r}", flush=True)

    ram_peak = tegra_ram_mib()
    results = {
        "model": args.model,
        "hidden_size": llm.config.hidden_size,
        "ram_before_load_mib": ram_start,
        "ram_after_load_mib": ram_after_load,
        "ram_peak_mib": ram_peak,
        "hf_load_s": load_s,
        "generations": generations,
        "ms_per_token_mean": statistics.mean(per_token_ms),
        "ms_per_token_stdev": statistics.stdev(per_token_ms) if len(per_token_ms) > 1 else 0.0,
        "total_latency_s_mean": statistics.mean(total_latencies_s),
        "comparison_qwen25_1_5b_ms_per_token": 203.0,  # A1's measured 12.2s/60 tokens
    }
    print("\n[minicpm5-latency] === RESULTS ===", flush=True)
    print(json.dumps({k: v for k, v in results.items() if k != "generations"}, indent=2), flush=True)
    speedup = 203.0 / results["ms_per_token_mean"]
    print(f"\n[minicpm5-latency] MiniCPM5-1B: {results['ms_per_token_mean']:.1f}ms/token "
          f"vs Qwen2.5-1.5B: 203ms/token measured in A1 -> "
          f"{'FASTER' if speedup > 1 else 'SLOWER'} by {speedup:.2f}x", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[minicpm5-latency] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
