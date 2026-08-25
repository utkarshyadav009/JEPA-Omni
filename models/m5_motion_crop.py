"""models/m5_motion_crop.py — one camera stream, three consumers.

WHY THIS EXISTS (measured on the device, 2026-08-15):

**The CSI sensor is EXCLUSIVE.** With `face_engine/motion_tracker` holding sensor-id=0, a
second `nvarguscamerasrc` client fails with
`Failed to create CaptureSession` -- and it fails in the silent way: OpenCV reports
`isOpened() == True` and then every `read()` returns False. So the deployed choice was
really either/or:

    motion_tracker running  ->  the eyes track you, and perception is blind
    perception running      ->  perception works, and the eyes are dead

Both is not an option with two processes. But it *is* an option with one: the perception
capture thread already pulls frames continuously, and the motion centroid is ~15 lines of
arithmetic on a downscaled copy. So this module folds motion tracking INTO the perception
capture path, and gets face crops out of the same signal for free:

    camera ──► capture thread ─┬─► perception window (V-JEPA2 / SigLIP2 / WavJEPA)
                               ├─► motion centroid ──► /dev/shm/bmo_motion.txt (face engine)
                               └─► face crop around the centroid ──► identity head

Three things fall out:
  * the eyes can track WHILE perception runs, which was previously impossible;
  * the identity head gets a localised crop instead of a whole 1280x720 frame;
  * `motion_tracker`'s **133 MiB** RSS is no longer spent (measured -- not the "tens of MB"
    its own source comment estimates).

THE HONEST LIMIT. A motion centroid is not a face box. It is the centre of mass of whatever
MOVED, which for a seated person sits around the chest/shoulders, and for a still person does
not exist at all. `HEAD_BIAS` shifts the crop upward to compensate, which is a heuristic, not
a detector. The identity head was trained on VoxCeleb2 FACE crops, so this is closer to its
training distribution than a full frame but still not in it. A real detector remains the
correct fix; this is the free 80% that needs no new model and no second camera client.

Output format matches `face_engine/motion_tracker.cpp` byte-for-byte so the face engine's
poller needs no change: `"1 <cx> <cy>\\n"` when moving, `"0 0 0\\n"` when not, cx/cy
normalised to -1..1 with (0,0) at frame centre.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MOTION_PATH = "/dev/shm/bmo_motion.txt"

# Matched to motion_tracker.cpp so behaviour is identical for the face engine.
GRID_W, GRID_H = 80, 60
DIFF_THRESHOLD = 25       # per-pixel brightness delta counted as "moved"
MIN_MOTION_PIXELS = 40    # below this it is sensor noise, not a person

# A motion centroid sits at the centre of mass of movement -- torso height for a seated
# person. Shift the crop up by this fraction of the crop size to land nearer the head.
HEAD_BIAS = 0.22
CROP_FRAC = 0.55          # crop side length as a fraction of the smaller frame dimension


@dataclass
class Motion:
    moving: bool
    cx: float             # -1..1, 0 = centre
    cy: float
    pixels: int


class MotionCentroid:
    """Frame differencing on a downscaled grayscale copy. Stateless apart from one previous
    frame, so memory stays flat -- same design rationale as the C++ tracker it replaces."""

    def __init__(self, write_path: Optional[str] = MOTION_PATH):
        self._prev: Optional[np.ndarray] = None
        self.write_path = write_path
        self.last = Motion(False, 0.0, 0.0, 0)

    def push(self, frame_hwc_u8: np.ndarray) -> Motion:
        """frame_hwc_u8: (H, W, 3) RGB or BGR uint8. Returns the current Motion and, if
        `write_path` is set, updates the file the face engine polls."""
        g = frame_hwc_u8
        if g.ndim == 3:
            g = g.mean(axis=2)
        # nearest-neighbour downsample to the tracker's grid -- cheap and sufficient
        ys = (np.linspace(0, g.shape[0] - 1, GRID_H)).astype(np.int32)
        xs = (np.linspace(0, g.shape[1] - 1, GRID_W)).astype(np.int32)
        small = g[ys][:, xs].astype(np.int16)

        m = Motion(False, 0.0, 0.0, 0)
        if self._prev is not None:
            d = np.abs(small - self._prev)
            mask = d > DIFF_THRESHOLD
            n = int(mask.sum())
            if n > MIN_MOTION_PIXELS:
                yy, xx = np.nonzero(mask)
                cx = float(xx.mean()) / GRID_W * 2.0 - 1.0
                cy = float(yy.mean()) / GRID_H * 2.0 - 1.0
                m = Motion(True, cx, cy, n)
            else:
                m = Motion(False, 0.0, 0.0, n)
        self._prev = small
        self.last = m

        if self.write_path:
            try:
                with open(self.write_path, "w") as f:
                    f.write(f"1 {m.cx:.4f} {m.cy:.4f}\n" if m.moving else "0 0 0\n")
            except OSError:
                pass       # the face engine can miss a frame; perception must not die for it
        return m


def crop_around(frame_chw: "np.ndarray | object", cx: float, cy: float,
                out_size: int = 224, crop_frac: float = CROP_FRAC,
                head_bias: float = HEAD_BIAS):
    """Crop a square around a normalised (-1..1) centroid, biased upward toward the head.

    Accepts and returns a torch (C,H,W) uint8 tensor. Clamps to the frame, so a centroid at
    the edge yields a valid crop rather than an exception."""
    import torch
    import torch.nn.functional as F

    C, H, W = frame_chw.shape
    side = int(min(H, W) * crop_frac)
    # centroid -> pixel space, then lift toward the head
    px = int((cx + 1.0) * 0.5 * W)
    py = int((cy + 1.0) * 0.5 * H) - int(side * head_bias)
    x0 = max(0, min(W - side, px - side // 2))
    y0 = max(0, min(H - side, py - side // 2))
    patch = frame_chw[:, y0:y0 + side, x0:x0 + side]
    return F.interpolate(patch.unsqueeze(0).float(), size=(out_size, out_size),
                         mode="bilinear", align_corners=False)[0].to(frame_chw.dtype)


if __name__ == "__main__":
    # Mechanism smoke test: a bright block that moves should be found, a static frame not.
    import torch
    mc = MotionCentroid(write_path=None)
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    mc.push(base)
    print("static frame ->", mc.push(base))

    moved = base.copy()
    moved[40:90, 200:260] = 255          # upper-right block appears
    m = mc.push(moved)
    print("moved block  ->", m)
    assert m.moving and m.cx > 0 and m.cy < 0, "centroid should be upper-right"

    f = torch.randint(0, 255, (3, 240, 320), dtype=torch.uint8)
    c = crop_around(f, m.cx, m.cy, out_size=224)
    print("crop:", tuple(c.shape), c.dtype)
    assert c.shape == (3, 224, 224)
    print("m5_motion_crop OK")


# ── SigLIP2 preprocessing without transformers.AutoProcessor ────────────────────
# MEASURED 2026-08-15 on the Jetson: `AutoProcessor.from_pretrained(...)` costs
# **327-500 MiB** -- more than SigLIP2's own vision tower (177 MiB fp16) and 4-5x more than
# perfect int8 quantization of that tower could ever save (89 MiB). For a resize and a
# normalise.
#
# The config it loads is trivial: size 224x224, rescale 1/255, mean 0.5, std 0.5,
# resample=2 (PIL BILINEAR -- NOT bicubic; using bicubic drops pixel cosine to 0.992).
# Reimplementing it exactly:
#     pixel cosine    vs AutoProcessor : 0.99994
#     EMBEDDING cosine vs AutoProcessor : 0.99993 - 0.99996
#     cosine between two DIFFERENT images: 0.9941   <- the discrimination scale
# The error is an order of magnitude below the scale the model actually distinguishes at,
# so this is equivalent in every way that matters -- and it matters that it be equivalent,
# because 587,303 cached scene features were extracted with AutoProcessor on mercury.
#
# This is the single largest memory saving found on this device, and it is not a model
# change at all. Look at what the FRAMEWORK is doing before compressing the model.
SIGLIP2_SIZE = 224
SIGLIP2_MEAN = 0.5
SIGLIP2_STD = 0.5


def siglip2_preprocess(pils, size: int = SIGLIP2_SIZE):
    """PIL images -> (N, 3, size, size) float32, matching SigLIP2's processor.

    Pass the result through `.to(device).to(torch.bfloat16)` exactly as the AutoProcessor
    output was. Deliberately takes PIL images rather than arrays so the resize filter is
    PIL's BILINEAR, which is what `resample=2` in the model's preprocessor_config means."""
    import torch
    from PIL import Image
    out = []
    for im in pils:
        if not isinstance(im, Image.Image):
            im = Image.fromarray(np.asarray(im).astype(np.uint8))
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).float().div_(255.0)
        out.append(((x - SIGLIP2_MEAN) / SIGLIP2_STD).permute(2, 0, 1))
    return torch.stack(out)
