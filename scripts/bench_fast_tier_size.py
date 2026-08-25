"""Compare fast-tier LLM candidates on the Jetson by raw llama.cpp latency +
memory -- answers 'do we need 700M, or is 350M enough + faster + lighter?'.
Uses the low-level llama.cpp path directly (no HF tokenizer, so a candidate
with a different vocab still gives valid LATENCY/MEMORY numbers; output text
quality needs the matching tokenizer + a retrain, out of scope for sizing)."""
import sys, time, argparse, gc

def rss_mb():
    # Jetson unified memory: used MiB from /proc/meminfo (MemTotal-MemAvailable)
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); d[k] = int(v.strip().split()[0])
    return (d["MemTotal"] - d["MemAvailable"]) / 1024.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--n-tokens", type=int, default=24)  # fast-tier budget
    ap.add_argument("--n-ctx", type=int, default=512)
    args = ap.parse_args()

    from llama_cpp import Llama
    m0 = rss_mb()
    t0 = time.perf_counter()
    llm = Llama(model_path=args.gguf, n_gpu_layers=-1, n_ctx=args.n_ctx, logits_all=False, verbose=False)
    load_ms = (time.perf_counter() - t0) * 1000.0
    m1 = rss_mb()

    prompt = b"user: How are you feeling today, Beemo?\nassistant:"
    toks = llm.tokenize(prompt, add_bos=True)
    # warmup
    llm.reset()
    for i, _ in enumerate(llm.generate(toks, temp=0.0)):
        if i >= 4: break

    # timed generate
    llm.reset()
    t0 = time.perf_counter(); n = 0
    for tid in llm.generate(toks, temp=0.0):
        n += 1
        if n >= args.n_tokens: break
    gen_ms = (time.perf_counter() - t0) * 1000.0
    m2 = rss_mb()

    print(f"[{args.label}] load={load_ms:.0f}ms  weights_mem=+{m1-m0:.0f}MB  "
          f"gen_{args.n_tokens}tok={gen_ms:.0f}ms  per_tok={gen_ms/max(1,n):.1f}ms  "
          f"peak_mem=+{m2-m0:.0f}MB", flush=True)
    del llm; gc.collect()

if __name__ == "__main__":
    main()
