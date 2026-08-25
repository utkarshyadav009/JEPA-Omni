"""models/quantized_text_encoder.py — int8 quantization for the query text encoder.

WHY THIS MODULE EXISTS: EmbeddingGemma is the largest single consumer in the Jetson
describe pipeline (~587 MiB measured), and the obvious `torchao` linear-only int8 pass only
recovers ~105 MiB. The reason is structural and worth stating, because it determines the
whole approach:

    total params      302.9M  (577.7 MiB @bf16)
    embedding table   201.3M  (384.0 MiB)  = 66.5%   <- (262144 x 768) vocab table
    linear layers     101.4M  (193.5 MiB)  = 33.5%

**Two thirds of the model is a vocabulary table that linear quantization never sees.** That
looks like the win to chase -- and it was chased, and it FAILED. See the boxed result below:
the 105 MiB from the linears is the only saving available here without breaking the encoder.

WHAT THIS DOES
  * linears -> torchao Int8WeightOnlyConfig. **MEASURED SAFE**: encoder-output cosine vs bf16
    = 0.9998, retrieval correct-field 0.939 -> 0.939. Saves ~105 MiB. Use this.
  * `Int8Embedding` exists and is **DELIBERATELY OFF BY DEFAULT** -- see the warning below.

  ┌─ MEASURED NEGATIVE RESULT: DO NOT int8-QUANTIZE THIS EMBEDDING TABLE ──────────────┐
  │ Quantizing the 262144x768 table looks like the obvious 384 MiB win, and every cheap │
  │ proxy says it is safe:                                                              │
  │     per-token cosine(bf16 row, int8 row) = 0.99991  (min 0.99964)                    │
  │     a random-nn.Embedding unit test      = 0.999975                                 │
  │ But the ENCODER OUTPUT collapses:                                                   │
  │     int8 LINEAR only      -> output cosine 0.9998   retrieval field 0.939            │
  │     int8 EMBEDDING only   -> output cosine 0.31-0.58                                 │
  │     int8 BOTH             -> output cosine 0.30-0.57  retrieval field 0.272          │
  │     (+ re-encoding the bank with it collapses correct-clip to 0.000)                 │
  │ Why: Gemma scales embeddings by sqrt(hidden)=27.71 and then runs a deep residual     │
  │ stack, so a per-token error that is invisible to cosine (cosine is dominated by the  │
  │ large components; the table's median |w| is 0.041 against a per-row scale of ~0.007) │
  │ compounds layer over layer.                                                          │
  │ LESSON: cosine on the QUANTIZED WEIGHTS is a misleading proxy. The only honest test  │
  │ is end-to-end encoder output, and then the downstream retrieval metric.              │
  └─────────────────────────────────────────────────────────────────────────────────────┘

ON "ACTIVATION-AWARE" (measured, not assumed): true AWQ was attempted first and is NOT
available here -- `torchao.prototype.awq` requires a base config implementing
`SupportsActivationPreScaling` (the int8 dynamic-activation config does not), and the int4
path additionally needs `mslk >= 1.0.0`, which is not installed. The activation-aware
alternative that IS available, `Int8DynamicActivationInt8WeightConfig`, was measured
**11x SLOWER** (163 ms vs 14.8 ms per query on Blackwell) for the same ~105 MiB saving.
So plain int8 weight-only is the choice here on evidence, not preference -- and it was
measured to cost no retrieval quality (correct-clip 0.456 -> 0.461, correct-field 0.939 ->
0.939 on a 6k-caption bank).

Harness for re-validating any future change: `scripts/quantize_embeddinggemma_eval.py`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Int8Embedding(nn.Module):
    """Drop-in replacement for nn.Embedding storing int8 weights + per-row fp16 scales.

    Memory: (V x D) bytes + (V x 2) bytes, versus (V x D x 2) for bf16 -- i.e. ~2x smaller,
    which on a 262144x768 table is 384 MiB -> ~192.5 MiB.

    Dequantization happens on the GATHERED rows only (batch x seq x D), never on the full
    table, so the runtime cost is proportional to the query length rather than the vocab."""

    def __init__(self, weight: torch.Tensor, padding_idx: Optional[int] = None):
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.to(torch.float16))
        self.padding_idx = padding_idx
        self.num_embeddings, self.embedding_dim = w.shape
        self._out_dtype = weight.dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.qweight[ids]                       # (..., D) int8, gathered
        s = self.scale[ids].to(torch.float32)          # (..., 1)
        out = rows.to(torch.float32) * s
        return out.to(self._out_dtype)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, int8+per-row-scale"


def quantize_text_encoder(base: nn.Module, do_linear: bool = True,
                          do_embedding: bool = False, verbose: bool = True) -> nn.Module:
    """int8-quantize a text encoder in place. Returns the same module for chaining.

    Order matters: embeddings are swapped FIRST (a plain module replacement), then torchao
    runs over the linears. Doing it the other way round makes torchao's tensor subclasses
    visible to the embedding walk, which needlessly complicates the traversal."""
    def _mib(m):
        t = 0
        for p in m.parameters():
            t += p.numel() * p.element_size()
        for b in m.buffers():
            t += b.numel() * b.element_size()
        return t / 2**20

    before = _mib(base)

    if do_embedding:
        print("[quant] WARNING: embedding-table int8 was MEASURED to break this encoder "
              "(output cosine 0.31-0.58, retrieval field 0.939->0.272). See module docstring.",
              flush=True)
        n = 0
        for name, mod in list(base.named_modules()):
            for child_name, child in list(mod.named_children()):
                if isinstance(child, nn.Embedding):
                    setattr(mod, child_name,
                            Int8Embedding(child.weight, getattr(child, "padding_idx", None)))
                    n += 1
        if verbose:
            print(f"[quant] replaced {n} embedding table(s) with int8+per-row-scale", flush=True)

    if do_linear:
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(base, Int8WeightOnlyConfig(),
                      filter_fn=lambda m, fqn: isinstance(m, nn.Linear))
            if verbose:
                print("[quant] linears -> int8 weight-only (torchao)", flush=True)
        except Exception as e:
            print(f"[quant] linear int8 SKIPPED ({e!r})", flush=True)

    if verbose:
        print(f"[quant] encoder {before:.1f} MiB -> {_mib(base):.1f} MiB", flush=True)
    return base


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = nn.Embedding(5000, 256)
    q = Int8Embedding(emb.weight)
    ids = torch.randint(0, 5000, (2, 11))
    ref, got = emb(ids), q(ids)
    err = (ref - got).abs().max().item()
    rel = ((ref - got).norm() / ref.norm()).item()
    fp_b = emb.weight.numel() * emb.weight.element_size()
    q_b = q.qweight.numel() + q.scale.numel() * 2
    print(f"shape={tuple(got.shape)} max_abs_err={err:.5f} rel_err={rel:.5f} "
          f"bytes {fp_b} -> {q_b} ({fp_b/q_b:.2f}x smaller)")
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    print(f"cosine(ref, int8) = {cos:.6f}  -> {'OK' if cos > 0.999 else 'DEGRADED'}")
