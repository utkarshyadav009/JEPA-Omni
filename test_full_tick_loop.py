"""Real end-to-end test of the full tonight's-work integration: homeostatic
state -> AppraisalVector -> face selection, FastTier (fine-tuned MiniCPM5-1B)
for both user exchanges and unprompted speech, across a realistic multi-
stage scenario."""
import sys, time
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from models.m4_cognitive_core import FastTier, CognitiveCoreRouter, AsyncThinker
from models.homeostatic_state import load_face_table
from models.bmo_duplex_tick import BmoDuplexTick

device = torch.device("cuda:0")
base = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(base, "checkpoints/bmo_minicpm5_lora/best").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("checkpoints/bmo_minicpm5_lora/best", trust_remote_code=True)

fast_tier = FastTier(model, tokenizer, device, max_new_tokens=20)
router = CognitiveCoreRouter(fast_tier, nll_escalate_threshold=3.0)


class _NoOpThinker:
    """Stub: no real DuplexLoop/vision/audio context in this isolated test."""
    def poll(self):
        return None
    def is_running(self):
        return False


tick = BmoDuplexTick(router=router, thinker=_NoOpThinker(),
                      face_table=None)  # face table needs the Jetson path; test appraisal only here

print("=== Stage 1: 3 minutes of silence, no user, static scene ===", flush=True)
for t in range(0, 180, 20):
    r = tick.tick(dt_s=20.0, transcript=None, user_speaking=False, user_present=False,
                  scene_embedding_drift=0.0, input_arousal_signal=0.0)
    print(f"  t={t+20:3d}s  social_need={r.homeostatic['social_need']:.2f}  "
          f"speak={r.speak_reason}" + (f"  -> {r.text_to_speak!r}" if r.text_to_speak else ""), flush=True)

print("\n=== Stage 2: user greets BMO ===", flush=True)
r = tick.tick(dt_s=1.0, transcript="hey BMO, sorry I was busy", user_speaking=True, user_present=True,
              scene_embedding_drift=0.3, input_arousal_signal=0.0)
print(f"  social_need={r.homeostatic['social_need']:.2f}  speak={r.speak_reason} -> {r.text_to_speak!r}", flush=True)

print("\n=== Stage 3: sudden loud noise (stress event) ===", flush=True)
for t in range(0, 15, 3):
    r = tick.tick(dt_s=3.0, transcript=None, user_speaking=False, user_present=True,
                  scene_embedding_drift=0.1, input_arousal_signal=0.9)
    print(f"  t={t+3:2d}s  stress={r.homeostatic['stress']:.2f}  speak={r.speak_reason}" +
          (f"  -> {r.text_to_speak!r}" if r.text_to_speak else ""), flush=True)

print("\n=== Stage 4: calm again, user asks a real question ===", flush=True)
r = tick.tick(dt_s=5.0, transcript="what should we do today?", user_speaking=True, user_present=True,
              scene_embedding_drift=0.0, input_arousal_signal=0.0)
print(f"  speak={r.speak_reason} -> {r.text_to_speak!r}", flush=True)
print(f"  final appraisal (V,A,C,N,O) = {tuple(round(x,3) for x in r.appraisal)}", flush=True)

print("\nDONE", flush=True)
