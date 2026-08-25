"""scripts/jetson_logprob_recover.py — keep the 249 MiB, get the confidence signal back.

`logits_all=True` costs ~249 MiB on the fast tier (measured, jetson_kv_logits_probe.py) and
exists only so the generate loop can read a per-token logprob for confidence routing. With
`logits_all=False` the memory is freed but the production read
`llm.scores[llm.n_tokens - 1, :]` returns ZEROS -- mean_neg_logprob pins to 11.0904 in every
configuration, which is exactly ln(65536), the vocab size, i.e. a uniform distribution over
an all-zero buffer. That is a silently broken router, not a saving.

llama.cpp still computes the LAST position's logits with logits_all=False; only the earlier
rows are dropped. So the fix is to read the right row. This tries the candidate accessors and
scores each against the baseline value (0.7932) obtained with logits_all=True.
"""
from __future__ import annotations
import json, os, sys
PROD = os.path.expanduser("~/bmo_production"); sys.path.insert(0, f"{PROD}/pipeline")
PROMPT = "You see: a person sitting. Say one short friendly line."
BASELINE_MNLP = 0.7932

def run(logits_all: bool, reader: str):
    from transformers import AutoTokenizer
    from llama_cpp import Llama
    import numpy as np
    llm = Llama(model_path=f"{PROD}/models_gguf/bmo_lfm25_350m_v2_Q8_0.gguf",
                n_gpu_layers=-1, n_ctx=512, logits_all=logits_all, verbose=False)
    tok = AutoTokenizer.from_pretrained(f"{PROD}/tokenizers/lfm25_350m_tok")
    p = tok.apply_chat_template([{"role":"user","content":PROMPT}],
                                add_generation_prompt=True, tokenize=False)
    toks = llm.tokenize(p.encode(), add_bos=False, special=True)
    neg, ids, err = [], [], None
    try:
        for i, t in enumerate(llm.generate(toks, temp=0.0, repeat_penalty=1.15)):
            if t == llm.token_eos() or i >= 24: break
            if   reader == "scores[n_tokens-1]": lg = llm.scores[llm.n_tokens - 1, :]
            elif reader == "scores[0]":          lg = llm.scores[0, :]
            elif reader == "scores[-1]":         lg = llm.scores[-1, :]
            elif reader == "eval_logits[-1]":    lg = np.asarray(llm.eval_logits[-1])
            elif reader == "get_logits_ith(-1)": lg = np.asarray(llm._ctx.get_logits_ith(-1))[:llm.n_vocab()]
            else: raise ValueError(reader)
            neg.append(-float(Llama.logits_to_logprobs(lg)[t])); ids.append(t)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    txt = llm.detokenize(ids).decode("utf-8", "replace") if ids else ""
    m = (sum(neg)/len(neg)) if neg else None
    del llm
    return {"logits_all": logits_all, "reader": reader,
            "mean_neg_logprob": (round(m,4) if m is not None else None),
            "n": len(ids), "text": txt.strip()[:60], "error": err}

def main():
    out = []
    for la, rd in [(True, "scores[n_tokens-1]"),
                   (False, "scores[n_tokens-1]"),
                   (False, "scores[0]"),
                   (False, "scores[-1]"),
                   (False, "eval_logits[-1]"),
                   (False, "get_logits_ith(-1)")]:
        try: r = run(la, rd)
        except Exception as e: r = {"logits_all": la, "reader": rd, "error": f"{type(e).__name__}: {e}"}
        m = r.get("mean_neg_logprob")
        ok = (m is not None and abs(m - BASELINE_MNLP) < 0.15)
        r["matches_baseline"] = ok
        tag = "MATCH " if ok else ("BROKEN" if m and m > 5 else "      ")
        print(f"── logits_all={str(r['logits_all']):5s} {r['reader']:22s} mnlp={m} {tag}", flush=True)
        if r.get("error"): print(f"     err: {str(r['error'])[:90]}", flush=True)
        out.append(r)
    json.dump(out, open(os.path.expanduser("~/logprob_recover.json"), "w"), indent=2)
    print("\nbaseline (logits_all=True) mnlp = 0.7932; 11.0904 == ln(65536) == all-zero buffer")

if __name__ == "__main__": main()
