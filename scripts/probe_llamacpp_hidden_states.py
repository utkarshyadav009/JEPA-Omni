"""scripts/probe_llamacpp_hidden_states.py — can GLR actually RUN on the deployed thinker?

GLR's latent rollout is `ê_i = ê_{i-1} + g_φ(h_{i-1})`, so each step needs TWO things from the
runtime:

    (A) feed a raw embedding IN, bypassing the token vocabulary  -- ALREADY PROVEN.
        `llama_batch.embd` accepts a custom embedding prefix and produced byte-identical output
        to the token path (scripts/prototype_llama_embd_input.py, llama-cpp-python 0.3.34).
    (B) read the last hidden state h OUT, once per step           -- NOT PROVEN. This probe.

(B) is the real risk and it is worth an hour now rather than after a training run. If llama.cpp
cannot return per-step hidden states, GLR on-device needs a different host for the rollout (a
small torch copy of the 0.6B, which the Jetson's memory budget cannot currently afford) and the
whole track changes shape. **Do not schedule on-device GLR work until this prints PASS.**

Three things are checked, in increasing order of what they'd cost us to be wrong about:

  1. does `llama_get_embeddings`-style access return anything at all, and of the right width;
  2. is what comes back the LAST HIDDEN STATE (d_model) or a POOLED SENTENCE EMBEDDING? These
     have the same shape on many models and are trivially confused. Distinguished by feeding two
     prompts that share a prefix but differ in the final token: a last-token hidden state must
     differ; a mean-pooled sentence embedding will differ far less. Also checked against the
     known d_model of Qwen3-0.6B (1024).
  3. does it update PER DECODE STEP with a KV cache in play, which is what a rollout needs --
     a value that only refreshes on a full re-eval is useless here.

Usage (on the Jetson):
    python3 scripts/probe_llamacpp_hidden_states.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROD = os.path.expanduser("~/bmo_production")
GGUF = f"{PROD}/models_gguf/bmo_thinker_qwen3_v5_Q8_0.gguf"
D_MODEL_EXPECTED = 1024        # Qwen3-0.6B


def main() -> None:
    import llama_cpp
    from llama_cpp import Llama
    print(f"llama_cpp {llama_cpp.__version__}")

    verdicts = {}

    # embedding=True is what exposes the embedding buffer at all. It is NOT free -- it changes
    # how the graph is built -- so the probe also has to confirm generation still works.
    llm = Llama(model_path=GGUF, n_gpu_layers=-1, n_ctx=512, embedding=True,
                logits_all=False, verbose=False)

    # --- 1. is anything exposed, at the right width? ---------------------------------------
    try:
        e = llm.embed("The room is dim and someone is at a screen.")
        arr = np.asarray(e, dtype=np.float32)
        if arr.ndim == 2:                      # per-token embeddings: exactly what we want
            width, kind = arr.shape[-1], f"per-token {arr.shape}"
        else:
            width, kind = arr.shape[-1], f"pooled {arr.shape}"
        print(f"[1] embed() -> {kind}, width={width}")
        verdicts["exposed"] = width == D_MODEL_EXPECTED
        verdicts["per_token"] = arr.ndim == 2
    except Exception as ex:
        print(f"[1] embed() FAILED: {type(ex).__name__}: {ex}")
        verdicts["exposed"] = verdicts["per_token"] = False
        arr = None

    # --- 2. last hidden state, or pooled sentence embedding? -------------------------------
    # Shared prefix, different final token. A last-token hidden state must move a lot; a
    # mean-pooled sentence vector over ~10 tokens moves comparatively little.
    try:
        a = np.asarray(llm.embed("the person is sitting and looking at a bright"),
                       dtype=np.float32)
        b = np.asarray(llm.embed("the person is sitting and looking at a dark"),
                       dtype=np.float32)
        if a.ndim == 2:
            a, b = a[-1], b[-1]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"[2] cos(shared-prefix, differing final token) = {cos:.4f}")
        print("    (a LAST-TOKEN hidden state should be clearly < 0.99; a mean-pooled "
              "sentence embedding tends to sit very close to 1.0)")
        verdicts["looks_like_last_hidden"] = cos < 0.99
    except Exception as ex:
        print(f"[2] FAILED: {type(ex).__name__}: {ex}")
        verdicts["looks_like_last_hidden"] = False

    # --- 3. does it update per decode step, with KV cache in play? -------------------------
    # This is the one that actually decides feasibility: a rollout needs h after EACH single
    # -token decode, not only after a fresh full evaluation.
    # The obvious accessor (`_ctx.get_embeddings()`) raises
    #   ValueError: '&<f' is not a valid PEP 3118 buffer format string
    # which is NOT a "hidden states are unavailable" result -- it is the SAME broken ctypes
    # buffer binding already documented for `get_logits_ith` in m4_cognitive_core.py:227. The
    # C function returns a bare `float*`; the wrapper's attempt to expose it as a Python buffer
    # is what fails. Read the pointer ourselves and the data is right there.
    import ctypes

    def h_last(ll) -> np.ndarray:
        n = ll._model.n_embd() if hasattr(ll._model, "n_embd") else D_MODEL_EXPECTED
        p = llama_cpp.llama_get_embeddings_ith(ll._ctx.ctx, -1)
        if not p:
            raise RuntimeError("llama_get_embeddings_ith returned NULL")
        buf = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * n)).contents
        return np.frombuffer(buf, dtype=np.float32, count=n).copy()

    try:
        llm.reset()
        toks = llm.tokenize(b"The room is dim.", add_bos=True)
        llm.eval(toks)
        h1 = h_last(llm)
        nxt = int(np.argmax(llm.scores[-1])) if getattr(llm, "scores", None) is not None \
            else llm.sample()
        llm.eval([nxt])
        h2 = h_last(llm)
        moved = float(np.linalg.norm(h1 - h2))
        cos12 = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-9))
        print(f"[3] per-step h via raw pointer: dim={h1.shape[0]}  "
              f"||h1-h2||={moved:.4f}  cos={cos12:.4f}")
        verdicts["per_step_update"] = moved > 1e-3 and h1.shape[0] == D_MODEL_EXPECTED
    except Exception as ex:
        print(f"[3] FAILED: {type(ex).__name__}: {ex}")
        verdicts["per_step_update"] = False

    # --- 4. generation still works with embedding=True -------------------------------------
    try:
        llm.reset()
        out = llm("Say hello.", max_tokens=12)
        txt = out["choices"][0]["text"].strip()
        print(f"[4] generation with embedding=True: {txt[:60]!r}")
        verdicts["gen_still_works"] = bool(txt)
    except Exception as ex:
        print(f"[4] generation FAILED: {type(ex).__name__}: {ex}")
        verdicts["gen_still_works"] = False

    print("\n" + "=" * 60)
    for k, v in verdicts.items():
        print(f"  {k:26s} {'ok' if v else 'NO'}")
    ok = all(verdicts.get(k) for k in
             ("exposed", "looks_like_last_hidden", "per_step_update", "gen_still_works"))
    # NOTE: single-line f-string expression on purpose -- the Jetson runs Python 3.10, which
    # does not allow a newline inside an f-string replacement field (that is 3.12+).
    msg = ("PASS - GLR can run under llama.cpp on-device" if ok else
           "FAIL - GLR rollout needs a different host; see module docstring")
    print("\nVERDICT: " + msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
