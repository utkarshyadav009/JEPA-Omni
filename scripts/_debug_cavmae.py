import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/home/utkarsh/JEPA-Omni')
from scripts.cavmae_retrieval import CAVMAEEncoder, AUDIO_TIME_FRAMES, AUDIO_N_PATCHES

model = CAVMAEEncoder.from_checkpoint('/home/utkarsh/models/cav-mae/audio_model.25.pth', device='cpu')

spec_a = torch.randn(1, 1, 128, AUDIO_TIME_FRAMES) * 10
spec_b = torch.randn(1, 1, 128, AUDIO_TIME_FRAMES) * 0.01

def trace_audio(spec, label):
    print(f"=== {label} ===")
    x = model.patch_embed_a(spec)
    print(f"  after patch_embed: mean={x.mean().item():.4f} std={x.std().item():.4f}")
    x = x + model.pos_embed_a + model.modality_a
    print(f"  after pos+mod:     mean={x.mean().item():.4f} std={x.std().item():.4f}")
    cls = model.cls_token_a.expand(1, -1, -1)
    x = torch.cat([cls, x], dim=1)
    for i, blk in enumerate(model.blocks_a):
        x = blk(x)
        if i == 0 or i == 10:
            print(f"  after block {i}: cls_norm={x[0,0].norm().item():.4f} mean={x.mean().item():.4f}")
    for blk in model.blocks_u:
        x = blk(x)
    x = model.norm_a(x)
    emb = F.normalize(x[:, 0], dim=-1)
    print(f"  final emb norm: {emb.norm().item():.4f}")
    return emb

emb_a = trace_audio(spec_a, "Input A (large std)")
emb_b = trace_audio(spec_b, "Input B (tiny std)")
sim = (emb_a @ emb_b.T).item()
print(f"\nSimilarity A vs B: {sim:.6f}")

# Also test same vs different with real-ish values
spec_c = torch.zeros(1, 1, 128, AUDIO_TIME_FRAMES)
spec_d = torch.zeros(1, 1, 128, AUDIO_TIME_FRAMES)
spec_d[0, 0, 64, 208] = 100.0  # big spike in center
emb_c = model.encode_audio(spec_c)
emb_d = model.encode_audio(spec_d)
sim_cd = (emb_c @ emb_d.T).item()
print(f"Silence vs spike: {sim_cd:.6f}")
