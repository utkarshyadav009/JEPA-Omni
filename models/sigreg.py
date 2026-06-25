"""
sigreg.py — Sketched Isotropic Gaussian Regularization (SIGReg)

Faithful transcription of Algorithm 1 (Epps-Pulley variant) from
LeJEPA: Balestriero & LeCun, arXiv:2511.08544v3, p.10.

What it does (paper Sec. 4): pushes the *batch* distribution of embeddings
toward an isotropic Gaussian N(0, I) by
  (1) projecting embeddings onto M random unit directions (sketching),
  (2) comparing each 1-D projected empirical characteristic function (ECF)
      against the N(0,1) target CF via the Epps-Pulley weighted-L2 statistic,
  (3) averaging the per-direction statistics (Def. 2 uses the AVERAGE, not the
      max, to avoid sparse gradients across directions).
No EMA, no stop-gradient, no teacher-student, no negatives. O(N) time & memory.

Role in JEPA-Omni
  - M1 (now): geometry probe + implementation validation. Use as
        loss = info_nce(...) + lambda * sigreg(pooled_video_emb, global_step)
    to study its effect on embedding isotropy/uniformity and retrieval.
    NOTE on what this validates: it validates that (a) the estimator is correct
    (bounded, no NaNs, DDP-correct) and (b) SIGReg shapes geometry to isotropic
    *without destabilising a working pipeline*. It does NOT validate "SIGReg as
    anti-collapse for cross-modal masked prediction" — M1 has a frozen text
    anchor + InfoNCE that already prevent collapse and has no masked prediction.
  - M2 (later): geometry-shaping term on the World-State vector. With FROZEN
    prediction targets, collapse is prevented by construction, so SIGReg's job
    there is to keep the World-State isotropic/full-rank for the M3 connector,
    NOT to be the load-bearing anti-collapse mechanism.

VERIFY before trusting exact magnitudes: diff against the official repo
(rbalestr-lab/lejepa) for the cached/optimised implementation and any
post-print fixes. Two spots transcribed exactly from Algorithm 1 but worth a
second pair of eyes: the Gaussian-window weighting `.mul(target_cf)` and the
`* N` test-statistic scaling (folds into lambda).
"""

import torch
import torch.distributed as dist


def _ddp_avg(t: torch.Tensor) -> torch.Tensor:
    """all_reduce(AVG) with complex + single-GPU support."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    ws = dist.get_world_size()
    if torch.is_complex(t):
        tr = torch.view_as_real(t).contiguous()
        dist.all_reduce(tr, op=dist.ReduceOp.SUM)
        tr /= ws
        return torch.view_as_complex(tr)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / ws
    return t


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Args
        x            : (N, K) embeddings to regularise (pooled World-State, or in
                       M1 the pooled video embedding). Do NOT pre-standardise:
                       SIGReg enforces unit variance per projection by design.
        global_step  : seeds the projection sampler so directions are identical
                       across DDP ranks and resampled every step (resampling
                       beats fixed directions — paper Fig. 7).
        num_slices   : M = |A|, number of random projection directions
                       (paper default = 256).
        reduce       : 'mean' -> scalar loss (Def. 2 average over directions);
                       'none' -> per-direction statistic, shape (num_slices,).

    Returns
        Scalar SIGReg loss, or (num_slices,) tensor if reduce='none'.
    """
    assert x.dim() == 2, f"expected (N, K), got {tuple(x.shape)}"
    dev = dict(device=x.device)

    # --- slice sampling: synced across devices via global_step seed ---
    g = torch.Generator(**dev)
    g.manual_seed(int(global_step))
    A = torch.randn((x.size(1), num_slices), generator=g, **dev)  # (K, M)
    A = A / A.norm(p=2, dim=0)                                    # unit-norm columns

    # --- Epps-Pulley statistic toward N(0,1) ---
    t = torch.linspace(-5.0, 5.0, 17, **dev)        # quadrature grid (Algorithm 1)
    target_cf = torch.exp(-0.5 * t**2)              # CF of N(0,1) (real) + Gauss. window

    x_t = (x.to(A.dtype) @ A).unsqueeze(2) * t      # (N, M, T)
    ecf = (1j * x_t).exp().mean(dim=0)              # (M, T) empirical CF, batch-mean
    ecf = _ddp_avg(ecf)                             # global ECF across ranks

    # weighted-L2 between empirical and target CF, weighted by the Gaussian window
    err = (ecf - target_cf).abs().square().mul(target_cf)   # (M, T)
    N = x.size(0) * _world_size()
    per_dir = torch.trapz(err, t, dim=1) * N               # (M,)

    if reduce == "mean":
        return per_dir.mean()
    if reduce == "none":
        return per_dir
    raise ValueError(f"reduce must be 'mean' or 'none', got {reduce!r}")


if __name__ == "__main__":
    # smoke test: isotropic Gaussian should score ~0; collapsed/anisotropic high.
    torch.manual_seed(0)
    iso = torch.randn(256, 128)                     # ~N(0, I)  -> low
    collapsed = torch.randn(256, 128) * 1e-3        # near-constant -> high
    aniso = torch.randn(256, 128); aniso[:, 1:] *= 1e-2   # 1 active dim -> high
    for name, z in [("isotropic", iso), ("collapsed", collapsed), ("anisotropic", aniso)]:
        print(f"{name:12s} SIGReg = {sigreg(z, global_step=0).item():.4f}")
