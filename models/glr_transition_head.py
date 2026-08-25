"""models/glr_transition_head.py — Geometric Latent Reasoning for the BMO thinker.

Implements arXiv:2606.02248 (*Geometric Latent Reasoning Induces Shorter Generations in LLMs*)
against our deployed thinker, Qwen3-0.6B — the paper's own primary model, which is why this is
worth building rather than adapting.

WHY THIS EXISTS, with the number that justifies it. Until 2026-08-16 the thinker's native CoT
was silently disabled (`enable_thinking` inherited as `False`; see CLAUDE.md). Turning it on
moved the thinker from **650 ms to 1,749–3,509 ms** and made it the single dominant leg of the
whole BMO pipeline, ahead of perception. Nearly all of that is autoregressive CoT tokens. GLR's
claim is precisely that most of those discrete transitions are redundant: on SVAMP it reaches a
correct answer in ~100 tokens where CoT-SFT needs 500–700, **with no explicit length objective**.

NOT the reason to do this: KV-cache growth. At `n_ctx=512` on a 0.6B model the KV is negligible
(KV quantization was already measured worthless at this context size). The win is wall-clock.

## The method, exactly

The head is a **single linear layer** `g_φ: R^d -> R^d` — ~1M params on 0.6B — that predicts a
*displacement* in embedding space rather than an absolute embedding:

    Δê_i = g_φ(h_{i-1})            ê_i = ê_{i-1} + Δê_i

Targets are the differences between consecutive **input embeddings** of the real CoT tokens:

    Δe_i = E_in(t_i) - E_in(t_{i-1})

Training is two forward passes per batch:

  1. teacher pass over the true discrete sequence `q <think> t_{1:m} </think> a_{1:l}`,
     collecting `h` at every thought position;
  2. student pass with the thought-token embeddings **replaced** by the rolled-out latents.

    L = L_CE(answer tokens only) + λ · L_Δ
    L_Δ = (1/m) Σ γ^{i-1} || g_φ(h_{i-1}) - Δe_i ||²         γ = 0.999

Two constraints in the paper are load-bearing and easy to get wrong:

  * **Cross-entropy is NOT applied to the latent replacement positions.** Supervising them pulls
    every latent state toward an immediate vocabulary prediction, which is exactly the discrete
    behaviour the method is trying to relax. `ce_on_latent=False` here, and the loss masks
    assert it.
  * **The input embedding matrix is FROZEN.** The targets Δe are built from it, so training it
    makes the regression targets non-stationary — the head would chase a moving objective.

The position discount γ^{i-1} is what makes this *geometric relaxation* rather than plain
distillation: later latent states are permitted to drift off the token-induced path, which is
where the token savings come from.

## Inference

    ê_1 = E_in(<think>)
    for i in 1..K:  ê_i = ê_{i-1} + g_φ(h_{i-1})     # fed straight back, no vocab projection
    then resume ordinary autoregressive decoding

K is a budget knob, not a constant: the paper sweeps {5, 10, 20, 50} on 0.6B and reports that
accuracy degrades at large K from accumulated geometric drift. Expect to tune it per task.

## Deployment note (read before trusting a latency estimate)

The Jetson runs the thinker as a **GGUF under llama.cpp**, and the rollout needs the model's
last hidden state `h` at every latent step. Feeding embeddings in is solved and proven —
`llama_batch.embd` accepts a custom embedding prefix and produced byte-identical output to the
token path (`scripts/prototype_llama_embd_input.py`). Getting hidden states back out per step is
NOT yet proven and is the real deployment risk. `scripts/probe_llamacpp_hidden_states.py` tests
it in isolation; do not schedule the on-device work until that probe passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GLRConfig:
    d_model: int
    gamma: float = 0.999        # position discount on the transition loss (paper value)
    lambda_delta: float = 1.0   # weight of L_Delta against L_CE
    k_latent: int = 10          # inference rollout length; paper sweeps {5,10,20,50} on 0.6B
    ce_on_latent: bool = False  # MUST stay False -- see module docstring
    bias: bool = True


class TransitionHead(nn.Module):
    """`g_φ: R^d -> R^d`. One linear layer, per the paper — deliberately not an MLP.

    The point of GLR is that a *local directional correction* along a trajectory the base model
    already induces is enough; a deeper head would start doing the reasoning itself and would no
    longer be a cheap add-on to a frozen backbone."""

    def __init__(self, cfg: GLRConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        # ZERO INIT, and it is not a stylistic choice -- `std=0.02` was tried and produced a
        # measured failure. This head is a RESIDUAL update on a trajectory, so at init it must
        # emit Delta = 0 and leave the latent path exactly at E_in(<think>), then earn every
        # deviation.
        #
        # What std=0.02 actually does here: `h` is Qwen3's post-RMSNorm final hidden state, so
        # ||h|| ~ sqrt(1024) ~ 32, and a N(0, 0.02^2) matrix maps that to ||out|| ~ 0.02*32*32
        # ~ 20 -- against a TARGET ||Delta_e|| of **1.25** (measured over the real embedding
        # matrix: mean ||e|| 0.925, mean ||Delta_e|| 1.245, mean ||Delta_e||^2 1.567). The head
        # therefore starts ~16x too large, and at inference those errors accumulate over K
        # steps instead of cancelling: the K=10 rollout reached ||latent|| = **423**, generated
        # 330 tokens against the K=0 baseline's 83, and its EOS rate collapsed to 0.38.
        # A rollout that diverges is not a shorter chain of thought, it is a hang.
        nn.init.zeros_(self.proj.weight)
        if cfg.bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d) hidden states -> (B, T, d) predicted displacements."""
        return self.proj(h)


def transition_loss(pred_delta: torch.Tensor,
                    target_delta: torch.Tensor,
                    mask: torch.Tensor,
                    gamma: float = 0.999,
                    normalize: bool = True) -> torch.Tensor:
    """Position-discounted MSE, `L_Δ = (1/m) Σ γ^{i-1} ||Δê_i - Δe_i||²`.

    `mask` (B, T) marks real thought positions. The discount index is the position WITHIN the
    thought span, not the absolute sequence position — otherwise the discount would depend on
    how long the question happened to be, and two identical CoTs after different prompts would
    be weighted differently.

    `normalize` divides by the mean squared target norm, which is what makes λ interpretable.
    The paper writes `L = L_CE + λ·L_Δ` and never states λ, and BOTH naive choices fail on this
    model because the two terms are ~3 orders of magnitude apart in raw units:
      * λ=1.0  -> L_Δ (a squared L2 SUMMED over d=1024, landing in the thousands) contributes
                  ~99.9% of the gradient and CE is ignored, so nothing keeps the answer correct;
      * λ=1e-3 -> the opposite, measured: val_delta only fell 2380->1505 over 5 epochs, whereas
                  a λ=1.0 smoke run reached 1251 within 180 steps. The head never learned to
                  shrink its output and the rollout diverged.
    Normalising makes L_Δ ≈ 1.0 for a head that predicts nothing (`mean ||Δe||² = 1.567` on this
    embedding matrix), so it sits alongside CE ≈ 2 and λ≈1 means what it looks like it means.
    """
    B, T, _ = pred_delta.shape
    se = ((pred_delta - target_delta) ** 2).sum(-1)                     # (B, T)
    # index within the masked span, per row
    idx = (mask.cumsum(dim=1) - 1).clamp(min=0).to(pred_delta.dtype)    # 0-based, (B, T)
    disc = torch.pow(torch.as_tensor(gamma, dtype=pred_delta.dtype,
                                     device=pred_delta.device), idx)
    m = mask.to(pred_delta.dtype)
    num = (se * disc * m).sum()
    den = m.sum().clamp(min=1.0)
    loss = num / den
    if normalize:
        scale = ((target_delta ** 2).sum(-1) * m).sum() / den
        loss = loss / scale.clamp(min=1e-6)
    return loss


@torch.no_grad()
def rollout_latents(model,
                    head: TransitionHead,
                    prefix_ids: torch.Tensor,
                    k: int,
                    embed_fn=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference-time latent rollout. Returns (latents (1,K,d), hidden of last step).

    Starts from the input embedding of the final prefix token — at inference that token is
    `<think>`, matching the paper's `ê_1 = E_in(<think>)`. Each step feeds the updated latent
    straight back in **without a vocabulary projection**, which is both the point of the method
    and the reason each latent step is cheaper than a decoded token.
    """
    embed = embed_fn or model.get_input_embeddings()
    e = embed(prefix_ids)                                   # (1, T, d)
    out = model(inputs_embeds=e, use_cache=True, output_hidden_states=True)
    past, h = out.past_key_values, out.hidden_states[-1][:, -1:]        # (1,1,d)

    cur = e[:, -1:]                                          # ê_0 = E_in(<think>)
    lat = []
    for _ in range(k):
        cur = cur + head(h)                                  # ê_i = ê_{i-1} + g_φ(h_{i-1})
        lat.append(cur)
        out = model(inputs_embeds=cur, past_key_values=past,
                    use_cache=True, output_hidden_states=True)
        past, h = out.past_key_values, out.hidden_states[-1][:, -1:]
    return (torch.cat(lat, dim=1) if lat else e[:, :0]), h


def build_delta_targets(embed_weight: torch.Tensor,
                        thought_ids: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """`Δe_i = E_in(t_i) - E_in(t_{i-1})` over the thought span.

    Position 0 of each span gets a zero target: there is no preceding thought token, and the
    rollout is initialised from `E_in(<think>)` rather than predicted, so supervising a
    displacement there would be regressing against something the model never produces.
    """
    e = torch.nn.functional.embedding(thought_ids, embed_weight)        # (B, T, d)
    d = torch.zeros_like(e)
    d[:, 1:] = e[:, 1:] - e[:, :-1]
    return d * mask.unsqueeze(-1).to(e.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GLRConfig(d_model=64)
    head = TransitionHead(cfg)
    B, T, d = 2, 7, 64
    h = torch.randn(B, T, d)
    mask = torch.zeros(B, T); mask[0, 1:5] = 1; mask[1, 2:7] = 1
    ids = torch.randint(0, 100, (B, T))
    W = torch.randn(100, d)
    tgt = build_delta_targets(W, ids, mask)
    loss = transition_loss(head(h), tgt, mask, cfg.gamma)
    n = sum(p.numel() for p in head.parameters())
    print(f"[glr] head params {n} (d={d}) | L_delta {loss.item():.4f}")
    assert torch.isfinite(loss)
    # the discount must depend on position WITHIN the span, not absolute position
    assert (tgt[0, 0] == 0).all() and (tgt[1, 0] == 0).all(), "span position 0 must be zero-target"
    assert (tgt[1, 1] == 0).all(), "outside-mask positions must be zeroed"
    print("[glr] ok")
