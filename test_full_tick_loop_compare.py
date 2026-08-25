"""Parametrized version of test_full_tick_loop.py -- same real scenario
(silence->lonely, greeting, stress event, calm question), but takes
--base-model / --checkpoint so the exact same held-out test can be run
against any of the three real checkpoints now available (original v1 on the
41-line corpus, v2 on the expanded 150-line corpus, LFM2-700M v2) for an
honest side-by-side comparison rather than eyeballing separate runs.
"""
import argparse
import sys
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from models.m4_cognitive_core import FastTier, CognitiveCoreRouter
from models.bmo_duplex_tick import BmoDuplexTick

ap = argparse.ArgumentParser()
ap.add_argument("--base-model", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--label", required=True)
args = ap.parse_args()

device = torch.device("cuda:0")
base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(base, args.checkpoint).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

fast_tier = FastTier(model, tokenizer, device, max_new_tokens=20)
router = CognitiveCoreRouter(fast_tier, nll_escalate_threshold=3.0)


class _NoOpThinker:
    def poll(self):
        return None
    def is_running(self):
        return False


tick = BmoDuplexTick(router=router, thinker=_NoOpThinker(), face_table=None)

print(f"\n########## {args.label} ##########", flush=True)

for t in range(0, 180, 20):
    r = tick.tick(dt_s=20.0, transcript=None, user_speaking=False, user_present=False,
                  scene_embedding_drift=0.0, input_arousal_signal=0.0)
    if r.text_to_speak:
        print(f"[RESULT:{args.label}] silence t={t+20}s speak={r.speak_reason} -> {r.text_to_speak!r}", flush=True)

r = tick.tick(dt_s=1.0, transcript="hey BMO, sorry I was busy", user_speaking=True, user_present=True,
              scene_embedding_drift=0.3, input_arousal_signal=0.0)
print(f"[RESULT:{args.label}] greeting -> {r.text_to_speak!r}", flush=True)

for t in range(0, 15, 3):
    r = tick.tick(dt_s=3.0, transcript=None, user_speaking=False, user_present=True,
                  scene_embedding_drift=0.1, input_arousal_signal=0.9)
    if r.text_to_speak:
        print(f"[RESULT:{args.label}] stress t={t+3}s -> {r.text_to_speak!r}", flush=True)

for t in range(0, 90, 30):
    tick.tick(dt_s=30.0, transcript=None, user_speaking=False, user_present=True,
              scene_embedding_drift=0.0, input_arousal_signal=0.0)
# Real bug found and fixed here (2026-08-07): stress_decay_per_s=1/60 in
# models/homeostatic_state.py means stress takes ~60s to meaningfully unwind
# -- the original 5s gap here left stress still ~92% of its peak, so this
# stage was never actually "calm" despite the label. The model's
# stress-flavored answers to "what should we do today" in that state were
# CORRECT continuous-state behavior, not a training/corpus bug -- this was
# a mislabeled test scenario, not a model quality gap. Added a real ~90s
# decay period (matching the documented time constant) before asking.
r = tick.tick(dt_s=5.0, transcript="what should we do today?", user_speaking=True, user_present=True,
              scene_embedding_drift=0.0, input_arousal_signal=0.0)
print(f"[RESULT:{args.label}] calm_question -> {r.text_to_speak!r}", flush=True)

print(f"DONE:{args.label}", flush=True)
