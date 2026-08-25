"""scripts/siglip2_room_test.py — is the perception ENCODER the bottleneck?

Established (PERCEPTION_GENERALIZATION_PLAN.md): on a real frame from BMO's camera — a dim
bedroom with chairs, posters, a door and a desk — BOTH output heads failed, because they
consume the same M2/V-JEPA2 world-state and that representation is out of distribution:
V-JEPA2 + M2 were trained on VGGSound/Ego4D *action and sound events in video*, not rooms.

That is a claim about the REPRESENTATION, so the test holds everything else fixed and swaps
only the encoder:

    same room frame, same 121,104-caption bank, different image encoder
      A) V-JEPA2 -> M2 -> QueryPredictor -> retrieve   (current stack)
      B) SigLIP2 (zero-shot, its own joint image-text space) -> retrieve

SigLIP2 needs NO training here — it is natively an image-text model, so it can score the
bank captions directly. That makes this a clean, cheap, same-day answer rather than another
training run. This repo already has precedent for taking it seriously: the M1 gate used a
zero-shot SigLIP2 baseline that scored **R@1 32.5 vs the V-JEPA2 spine's 22.5**.

Also probes a small hand-written set of room-vs-event captions, because the 121k bank is
itself VGGSound/Action100M-flavoured — if the bank contains no good sentence for a bedroom,
even a perfect encoder cannot retrieve one, and separating "bad encoder" from "bad bank"
matters for deciding what to fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# A deliberately mixed probe set: room/scene sentences that the VGGSound-derived bank does
# NOT contain, plus event sentences it does. A working encoder should prefer the room ones
# for a photo of a room.
PROBE = [
    "a bedroom with a desk, an office chair and posters on the wall",
    "an empty dim room with furniture and pictures on the wall",
    "an indoor room with a door, shelves and a chair",
    "a home office with a desk and a computer chair",
    "a person opening a microwave oven in a kitchen",
    "a vacuum cleaner being operated on a carpet",
    "a car door being opened in a parking lot",
    "soldiers marching in a military parade",
    "a dog barking loudly in a yard",
    "a person playing an accordion indoors",
]


def load_frame(path: str, rotate: int, gain: float, res: int = 256) -> Image.Image:
    im = Image.open(path)
    if rotate:
        im = im.rotate(rotate, expand=True)
    if gain != 1.0:
        im = ImageEnhance.Brightness(im).enhance(gain)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.convert("RGB").resize((res, res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="jetson_artifacts/benchmarks/home/cam_gst.jpg")
    ap.add_argument("--rotate", type=int, default=90)
    ap.add_argument("--gain", type=float, default=1.8)
    ap.add_argument("--siglip", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--bank", default="checkpoints/perception_bank_max_fp16.pt")
    ap.add_argument("--bank-limit", type=int, default=40000,
                    help="cap bank captions re-encoded with SigLIP2 (its text tower is slower)")
    ap.add_argument("--out", default="checkpoints/SIGLIP2_ROOM_TEST.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    img = load_frame(args.frame, args.rotate, args.gain)
    print(f"[sig] frame {img.size} from {args.frame}", flush=True)

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(args.siglip, dtype=torch.bfloat16).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.siglip)
    print(f"[sig] {args.siglip} loaded", flush=True)

    with torch.no_grad():
        px = proc(images=[img], return_tensors="pt").to(device)
        px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in px.items()}
        _o = model.get_image_features(**px)
        _o = _o.pooler_output if hasattr(_o, "pooler_output") else _o
        z_img = F.normalize(_o.float(), dim=-1)

    def encode_texts(texts, bs=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                t = proc(text=texts[i:i + bs], padding="max_length", truncation=True,
                         max_length=64, return_tensors="pt").to(device)
                _t = model.get_text_features(**t)
                _t = _t.pooler_output if hasattr(_t, "pooler_output") else _t
                out.append(F.normalize(_t.float(), dim=-1).cpu())
        return torch.cat(out, 0)

    # ---- probe set: does SigLIP2 prefer ROOM sentences for a room photo? ----
    zp = encode_texts(PROBE).to(device)
    sims = (z_img @ zp.T)[0]
    order = sims.argsort(descending=True)
    print("\n== SigLIP2 zero-shot on the probe set (room sentences vs event sentences) ==")
    for r, i in enumerate(order.tolist()):
        tag = "ROOM " if i < 4 else "event"
        print(f"  {r+1:2d}. [{tag}] {sims[i]:+.4f}  {PROBE[i]}")
    top4_room = sum(1 for i in order[:4].tolist() if i < 4)
    print(f"  -> room sentences in top-4: {top4_room}/4")

    # ---- the real bank, same one the current stack retrieves from ----
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    texts = bank["text"][: args.bank_limit]
    print(f"\n[sig] re-encoding {len(texts)} bank captions with SigLIP2 text tower ...", flush=True)
    zb = encode_texts(texts).to(device)
    s = (z_img @ zb.T)[0]
    top = s.topk(5)
    print("\n== SigLIP2 retrieval from the SAME 121k-caption bank ==")
    for sc, ix in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  [{sc:+.4f}] {texts[ix][:150]}")

    res = {"probe": [{"text": PROBE[i], "score": float(sims[i]), "is_room": i < 4}
                     for i in order.tolist()],
           "room_in_top4": top4_room,
           "bank_top5": [{"score": float(v), "text": texts[i]}
                         for v, i in zip(top.values.tolist(), top.indices.tolist())],
           "bank_captions_scored": len(texts)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[sig] wrote {args.out}")


if __name__ == "__main__":
    main()
