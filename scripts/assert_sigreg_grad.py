"""scripts/assert_sigreg_grad.py — STOP-0 grad-flow assertion.

Verifies that sigreg(world_state(feats, tbins)) backpropagates
gradient into pool_query AND at least one transformer block parameter.

Usage:
    conda run -n jepa-omni python scripts/assert_sigreg_grad.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.sigreg import sigreg

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg   = AVJepaConfig()
model = AVJepaPredictor(cfg).to(device)
model.train()
for p in model.parameters():
    p.requires_grad_(True)

B, Tv, Ta = 8, 32, 50
feats = {
    "vision":  torch.randn(B, Tv, 1024, device=device),
    "ambient": torch.randn(B, Ta, 768,  device=device),
}
tbins = {
    "vision":  torch.linspace(0, 99, Tv).long().unsqueeze(0).expand(B, -1).to(device),
    "ambient": torch.linspace(0, 99, Ta).long().unsqueeze(0).expand(B, -1).to(device),
}

lam = 0.1
ws      = model.world_state(feats, tbins)   # grad-enabled path
sr_loss = sigreg(ws.float(), global_step=0, num_slices=256)
loss    = lam * sr_loss
loss.backward()

# ── check pool_query grad ────────────────────────────────────────────────────
pq_norm = model.pool_query.grad.norm().item() if model.pool_query.grad is not None else 0.0
print(f"pool_query grad norm    : {pq_norm:.6e}")

# ── check transformer block params ──────────────────────────────────────────
block_norms = []
for name, p in model.named_parameters():
    if "blocks" in name and p.grad is not None:
        block_norms.append((name, p.grad.norm().item()))

if block_norms:
    # report the first one with nonzero grad
    blk_name, blk_norm = max(block_norms, key=lambda x: x[1])
    print(f"max block param grad    : {blk_norm:.6e}  ({blk_name})")
else:
    blk_norm = 0.0
    print("max block param grad    : 0.0  (NO block grads found)")

# ── assertions ───────────────────────────────────────────────────────────────
THRESHOLD = 1e-8
assert pq_norm  > THRESHOLD, f"FAIL: pool_query grad {pq_norm:.2e} <= {THRESHOLD}"
assert blk_norm > THRESHOLD, f"FAIL: block grad {blk_norm:.2e} <= {THRESHOLD}"
n_nonzero_blocks = sum(1 for _, n in block_norms if n > THRESHOLD)
print(f"block params with grad>threshold: {n_nonzero_blocks}/{len(block_norms)}")
print()
print("STOP-0 GRAD ASSERTION: PASSED")
