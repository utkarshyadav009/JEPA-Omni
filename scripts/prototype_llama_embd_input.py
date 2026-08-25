"""PROTOTYPE — prove llama.cpp (llama-cpp-python 0.3.34) accepts a custom
embedding prefix (the projector's output) instead of token IDs, so the Ultravox
no-text path can drive the deployed GGUF fast tier WITHOUT patching llama.cpp.

Test: take a text prompt, compute its input embeddings with the HF model, feed
THOSE FLOATS into the GGUF via a llama_batch with the `embd` buffer, greedily
continue, and compare to the normal token-fed HF continuation. If they match,
the embd-input mechanism works AND the HF-space embeddings the projector emits
are compatible with the GGUF. (f16 GGUF is used to isolate mechanism from
quantization; a Q8 pass can follow.)
"""
import sys, ctypes
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import numpy as np
import torch
import llama_cpp
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF = "checkpoints/lfm25_350m_base_f16.gguf"
HF = "/home/utkarsh/hf_models/LFM2.5-350M"
PROMPT = "The capital of France is"
N_GEN = 12


def _first(*names):
    for n in names:
        f = getattr(llama_cpp, n, None)
        if f is not None:
            return f
    raise AttributeError(names)


# ---- HF: embeddings for the prompt ----
tok = AutoTokenizer.from_pretrained(HF)
hf = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).eval()
ids = tok(PROMPT, return_tensors="pt")["input_ids"]
emb = hf.get_input_embeddings()(ids)[0].detach().numpy().astype(np.float32)  # (T,H)
T, H = emb.shape
print(f"prompt={PROMPT!r} tokens={ids[0].tolist()} T={T} H={H}", flush=True)

# ---- llama.cpp load ----
llama_cpp.llama_backend_init()
mp = llama_cpp.llama_model_default_params(); mp.n_gpu_layers = 0
load_model = _first("llama_model_load_from_file", "llama_load_model_from_file")
model = load_model(GGUF.encode(), mp)
cp = llama_cpp.llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512
new_ctx = _first("llama_init_from_model", "llama_new_context_with_model")
ctx = new_ctx(model, cp)
n_embd = _first("llama_model_n_embd", "llama_n_embd")(model)
try:
    vocab = llama_cpp.llama_model_get_vocab(model)
    n_vocab = _first("llama_vocab_n_tokens", "llama_n_vocab")(vocab)
except Exception:
    n_vocab = _first("llama_n_vocab")(model)
print(f"gguf n_embd={n_embd} n_vocab={n_vocab}", flush=True)
assert n_embd == H, f"embd dim mismatch {n_embd} vs {H}"


def decode_embd(emb_arr, pos0):
    T = emb_arr.shape[0]
    batch = llama_cpp.llama_batch_init(T, n_embd, 1)
    batch.n_tokens = T
    flat = np.ascontiguousarray(emb_arr.reshape(-1).astype(np.float32))
    ctypes.memmove(batch.embd, flat.ctypes.data, flat.nbytes)
    for i in range(T):
        batch.pos[i] = pos0 + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if i == T - 1 else 0
    rc = llama_cpp.llama_decode(ctx, batch)
    llama_cpp.llama_batch_free(batch)
    return rc


def decode_token(t, pos):
    b = llama_cpp.llama_batch_init(1, 0, 1)
    b.n_tokens = 1
    b.token[0] = t; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1
    rc = llama_cpp.llama_decode(ctx, b)
    llama_cpp.llama_batch_free(b)
    return rc


def last_logits():
    lp = llama_cpp.llama_get_logits_ith(ctx, -1)
    ptr = ctypes.cast(lp, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(n_vocab,))


rc = decode_embd(emb, 0)
print("llama_decode(embd prefix) rc =", rc, "(0 = OK)", flush=True)

out, pos = [], T
for _ in range(N_GEN):
    nt = int(np.argmax(last_logits()))
    out.append(nt)
    if nt == tok.eos_token_id:
        break
    decode_token(nt, pos); pos += 1

print("EMBD-path (GGUF) :", repr(tok.decode(out)), flush=True)
with torch.no_grad():
    g = hf.generate(ids, max_new_tokens=N_GEN, do_sample=False)
print("TOKEN-path (HF)  :", repr(tok.decode(g[0][ids.shape[1]:])), flush=True)
print("PROTOTYPE_DONE", flush=True)
