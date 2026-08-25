"""scripts/jetson_kv_logits_probe.py — price the llama.cpp memory knobs, for real.

WHY. The memory plan assumed 100-250 MiB from KV-cache work. Two facts make that suspect:
  * `n_ctx` is already **512** (lowered from 2048 on 2026-08-07 after a real OOM), not 8192,
    so there is no big context reduction left to harvest -- KV at 512 is already small.
  * `GGUFFastTier` passes **`logits_all=True`**, which allocates a logits buffer for EVERY
    position: vocab x n_ctx x 4 bytes. At a ~65k vocab and n_ctx 512 that is ~134 MB PER
    MODEL, and it dwarfs the KV cache it was lumped in with.

`logits_all=True` is there to read per-token logprobs for confidence routing (escalate to the
thinker or not). But the generate loop only ever reads the LAST row --
`self.llm.scores[self.llm.n_tokens - 1, :]` -- which is exactly what llama.cpp keeps when
`logits_all=False`. If the signal survives, this is free memory.

MEASURES, per configuration: RSS delta across the load, the load time, and -- the part that
decides it -- whether `mean_neg_logprob` is still finite and close to the baseline. A memory
saving that silently breaks confidence routing is not a saving, it is a regression that would
show up later as bad escalation decisions.

Run on the Jetson (loads ONE model at a time, so it is safe under memory pressure):
    python3 scripts/jetson_kv_logits_probe.py
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

PROMPT = "You see: a person sitting. Say one short friendly line."


def avail_mib() -> float:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    return float(out.splitlines()[1].split()[-1])


def try_cfg(gguf: str, tok_dir: str, name: str, **kw) -> dict:
    """Load with the given kwargs, generate once, report memory + the confidence signal."""
    from transformers import AutoTokenizer
    from llama_cpp import Llama

    gc.collect()
    before = avail_mib()
    t0 = time.time()
    try:
        llm = Llama(model_path=gguf, n_gpu_layers=-1, verbose=False, **kw)
    except Exception as e:
        return {"cfg": name, "error": f"{type(e).__name__}: {e}"}
    load_s = time.time() - t0
    after_load = avail_mib()

    tok = AutoTokenizer.from_pretrained(tok_dir)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                     add_generation_prompt=True, tokenize=False)
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    text_ids, neg_lps, err = [], [], None
    try:
        for i, t in enumerate(llm.generate(tokens, temp=0.0, repeat_penalty=1.15)):
            if t == llm.token_eos() or i >= 24:
                break
            # the ONLY row the production code reads -- the whole question is whether this
            # is still valid without logits_all
            logits = llm.scores[llm.n_tokens - 1, :]
            lp = Llama.logits_to_logprobs(logits)[t]
            neg_lps.append(-float(lp))
            text_ids.append(t)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    after_gen = avail_mib()
    txt = llm.detokenize(text_ids).decode("utf-8", errors="replace") if text_ids else ""
    mnlp = (sum(neg_lps) / len(neg_lps)) if neg_lps else None
    del llm
    gc.collect()

    return {"cfg": name, "load_s": round(load_s, 1),
            "load_MiB": round(before - after_load, 1),
            "gen_MiB": round(after_load - after_gen, 1),
            "total_MiB": round(before - after_gen, 1),
            "mean_neg_logprob": (round(mnlp, 4) if mnlp is not None else None),
            "n_tokens": len(text_ids), "text": txt.strip()[:70],
            "logprob_error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf")
    ap.add_argument("--tok", default=f"{PROD}/tokenizers/lfm25_350m_tok")
    ap.add_argument("--out", default=os.path.expanduser("~/kv_logits_probe.json"))
    args = ap.parse_args()

    CFGS = [
        ("baseline (n_ctx=512, logits_all=True)", dict(n_ctx=512, logits_all=True)),
        ("logits_all=False", dict(n_ctx=512, logits_all=False)),
        ("logits_all=False + flash_attn", dict(n_ctx=512, logits_all=False, flash_attn=True)),
        ("logits_all=False + FA + q8_0 KV",
         dict(n_ctx=512, logits_all=False, flash_attn=True, type_k=8, type_v=8)),
        ("logits_all=False + FA + q4_0 KV",
         dict(n_ctx=512, logits_all=False, flash_attn=True, type_k=2, type_v=2)),
        ("n_ctx=256, logits_all=False + FA", dict(n_ctx=256, logits_all=False, flash_attn=True)),
    ]
    res = []
    for name, kw in CFGS:
        r = try_cfg(args.gguf, args.tok, name, **kw)
        res.append(r)
        if "error" in r:
            print(f"── {name:44s} FAILED: {r['error'][:80]}", flush=True)
        else:
            print(f"── {name:44s} load={r['load_MiB']:7.1f} MiB  total={r['total_MiB']:7.1f}  "
                  f"mnlp={r['mean_neg_logprob']}  n={r['n_tokens']}", flush=True)
            if r["logprob_error"]:
                print(f"     LOGPROB BROKE: {r['logprob_error'][:90]}", flush=True)
            print(f"     {r['text']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[probe] wrote {args.out}")
    print("A config only counts if mean_neg_logprob is finite AND close to baseline --")
    print("confidence routing decides when to escalate to the thinker.")


if __name__ == "__main__":
    main()
