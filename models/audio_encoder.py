"""
models/audio_encoder.py — M2 frozen WavJEPA wrapper.

Loads BOTH labhamlet/wavjepa-base and labhamlet/wavjepa-nat-base from HF.
WavJEPA ships custom modeling code (trust_remote_code=True), but the
transformers 5.x AutoModel.from_pretrained path has a known compat bug
('all_tied_weights_keys' AttributeError).  We work around it by loading the
model class directly from the HF snapshot and loading safetensors manually.

GROUNDED FACTS (confirmed by probe on a 10s clip at 16 kHz):
  wavjepa-base:
      Input:   (B, 1, n_samples)   — 1 channel, raw PCM at 16 kHz
      Output:  (B, 996, 768)       — 996 tokens, 768-d, for a 10s clip
      Token rate: 99.6 Hz

  wavjepa-nat-base:
      Input:   (B, 2, n_samples)   — 2 channels (binaural), raw PCM at 16 kHz
      Output:  (B, 2, 996, 768)    — retains channel dim; we mean-pool → (B, 996, 768)
      Token rate: 99.6 Hz (per channel; same after pooling)

Both models: 768-d output, MIT license, ~196-200M params.

This module is FROZEN: eval(), requires_grad_(False), bf16, no_grad forward.
It is the only place the WavJEPA HF API is touched — everything downstream
sees a clean (B, T, 768) bfloat16 tensor.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Repo constants (authoritative)
# ─────────────────────────────────────────────────────────────────────────────
WAVJEPA_BASE_REPO = "labhamlet/wavjepa-base"
WAVJEPA_NAT_REPO  = "labhamlet/wavjepa-nat-base"

WAVJEPA_AUDIO_SAMPLE_RATE: int   = 16_000   # Hz (confirmed by extractor config)
WAVJEPA_OUT_DIM:            int   = 768      # base hidden size for both variants

# Measured token rates (probe_wavjepa.py on 10s clip at 16 kHz):
#   base    → 996 tokens / 10s = 99.6 Hz
#   nat     → (after ch-pool) 996 tokens / 10s = 99.6 Hz
WAVJEPA_BASE_TOKEN_RATE_HZ: float = 99.6
WAVJEPA_NAT_TOKEN_RATE_HZ:  float = 99.6


def _hf_cache_root() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface/hub"),
    )


def _find_snapshot(repo_id: str) -> str:
    """Return path to the local HF snapshot for `repo_id` (fail clearly if absent)."""
    repo_slug = "models--" + repo_id.replace("/", "--")
    snap_root = os.path.join(_hf_cache_root(), repo_slug, "snapshots")
    if not os.path.isdir(snap_root):
        raise FileNotFoundError(
            f"No local snapshot for {repo_id}.\n"
            f"Run:  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{repo_id}')\""
        )
    snaps = sorted(os.listdir(snap_root))
    return os.path.join(snap_root, snaps[-1])   # latest snapshot


def _import_snapshot_pkg(snap_dir: str) -> str:
    """Register `snap_dir` as an importable package; return pkg name."""
    init = os.path.join(snap_dir, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    parent = os.path.dirname(snap_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = os.path.basename(snap_dir)
    if pkg not in sys.modules:
        importlib.import_module(pkg)
    return pkg


def _extract_tensor(out) -> Tensor:
    """Normalise heterogeneous WavJEPA outputs to (B, T, D)."""
    if isinstance(out, Tensor):
        feat = out
    elif isinstance(out, (tuple, list)):
        feat = out[0]
    else:
        raise ValueError(f"Cannot extract tensor from output type {type(out)}")
    # nat-base returns (B, 2, T, D); base returns (B, T, D)
    if feat.dim() == 4:
        feat = feat.mean(dim=1)   # mean-pool channel dim → (B, T, D)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# AudioEncoder
# ─────────────────────────────────────────────────────────────────────────────
class AudioEncoder(nn.Module):
    """Frozen wrapper around ONE WavJEPA variant (base or nat-base).

    Loads directly from the local HF snapshot (bypassing the broken
    AutoModel.from_pretrained path in transformers 5.x).

    Parameters
    ----------
    repo : str
        HF repo id — WAVJEPA_BASE_REPO or WAVJEPA_NAT_REPO.
    n_channels : int
        1 for base (mono), 2 for nat-base (binaural).
    dtype : torch.dtype
        bfloat16 for extraction (matches vision encoder).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        repo: str,
        n_channels: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.repo      = repo
        self.n_channels = n_channels
        self.dtype     = dtype
        self.device_str = device

        snap_dir = _find_snapshot(repo)
        pkg = _import_snapshot_pkg(snap_dir)

        # ── load model class from snapshot ───────────────────────────────
        if "nat" in repo:
            model_mod_name = "modeling_wavjepa_nat"
            cls_name = "WavJEPANatModel"
            cfg_mod_name = "configuration_wavjepa_nat"
            cfg_cls_name = "WavJEPANatConfig"
        else:
            model_mod_name = "modeling_wavjepa"
            cls_name = "WavJEPAModel"
            cfg_mod_name = "configuration_wavjepa"
            cfg_cls_name = "WavJEPAConfig"

        model_mod = importlib.import_module(f"{pkg}.{model_mod_name}")
        cfg_mod   = importlib.import_module(f"{pkg}.{cfg_mod_name}")
        ModelCls  = getattr(model_mod, cls_name)
        ConfigCls = getattr(cfg_mod,   cfg_cls_name)

        config = ConfigCls.from_pretrained(repo, trust_remote_code=True)
        model  = ModelCls(config)

        # ── load weights from safetensors ────────────────────────────────
        sf_path = os.path.join(snap_dir, "model.safetensors")
        from safetensors.torch import load_file
        sd = load_file(sf_path)
        model.load_state_dict(sd, strict=True)

        # ── freeze ────────────────────────────────────────────────────────
        # NOTE: WavJEPA uses torch.masked.MaskedTensor internally, which does
        # NOT support bfloat16.  Load the model in float32; we cast the OUTPUT
        # tensor to bf16 after inference.  The dtype arg is kept for the output.
        model = model.float().to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        # ── feature extractor (for numpy→input_values conversion) ────────
        try:
            from transformers import AutoFeatureExtractor
            self.extractor = AutoFeatureExtractor.from_pretrained(
                repo, trust_remote_code=True
            )
        except Exception:
            self.extractor = None

        # ── measure actual token rate via 2s probe ────────────────────────
        self._measured_token_rate_hz = self._probe_token_rate(probe_duration_s=2.0)

    # ──────────────────────────────────────────────────────────────────────
    def _probe_token_rate(self, probe_duration_s: float = 2.0) -> float:
        """Run a silent clip to measure tokens/second."""
        n = int(probe_duration_s * WAVJEPA_AUDIO_SAMPLE_RATE)
        # Model must run in float32 (MaskedTensor constraint)
        x = torch.zeros(1, self.n_channels, n, dtype=torch.float32, device=self.device_str)
        with torch.no_grad():
            out  = self.model(x)
            feat = _extract_tensor(out)     # (1, T, D)
        return feat.shape[1] / probe_duration_s

    @property
    def token_rate_hz(self) -> float:
        return self._measured_token_rate_hz

    @property
    def out_dim(self) -> int:
        return WAVJEPA_OUT_DIM

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, waveform: Tensor) -> Tensor:
        """Encode a batch of waveforms.

        Parameters
        ----------
        waveform : Tensor
            (B, n_channels, n_samples) float32 raw PCM at WAVJEPA_AUDIO_SAMPLE_RATE.
            For base: n_channels=1.  For nat-base: n_channels=2.

        Returns
        -------
        Tensor
            (B, T, 768) bfloat16 frozen features.
        """
        # WavJEPA requires float32 internally (MaskedTensor bf16 restriction).
        inp  = waveform.float().to(self.device_str)
        out  = self.model(inp)
        feat = _extract_tensor(out)     # (B, T, 768) float32
        return feat.to(torch.bfloat16)  # cast output to bf16 for storage

    def forward(self, waveform: Tensor) -> Tensor:
        return self.encode(waveform)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load both variants at once
# ─────────────────────────────────────────────────────────────────────────────
def load_both_encoders(
    device: str = "cuda",
) -> "tuple[AudioEncoder, AudioEncoder]":
    """Return (base_enc, nat_enc) — both frozen on `device`."""
    base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=device)
    nat  = AudioEncoder(WAVJEPA_NAT_REPO,  n_channels=2, device=device)
    return base, nat


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CLIP_S = 10.0
    SR     = WAVJEPA_AUDIO_SAMPLE_RATE
    n      = int(CLIP_S * SR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"AudioEncoder smoke test — 10s clip at {SR} Hz on {device}")
    print("=" * 60)

    for repo, n_ch in [(WAVJEPA_BASE_REPO, 1), (WAVJEPA_NAT_REPO, 2)]:
        print(f"\n--- {repo} ---")
        t0 = time.time()
        enc = AudioEncoder(repo, n_channels=n_ch, device=device)
        load_t = time.time() - t0

        wav = torch.zeros(1, n_ch, n)   # (B=1, n_ch, n_samples)
        t1 = time.time()
        feats = enc.encode(wav)
        enc_t = time.time() - t1

        T, D = feats.shape[-2], feats.shape[-1]
        rate = T / CLIP_S

        print(f"  load time:    {load_t:.1f}s")
        print(f"  encode time:  {enc_t:.2f}s")
        print(f"  output shape: {tuple(feats.shape)}")
        print(f"  T (tokens):   {T}")
        print(f"  D (dim):      {D}")
        print(f"  token rate:   {rate:.2f} Hz  ← MEASURED (not assumed)")
        print(f"  probe rate:   {enc.token_rate_hz:.2f} Hz  (from 2s probe at load)")
        print(f"  dtype:        {feats.dtype}")

    print("\n[audio_encoder] Done. Use these rates in manifest (NOT 100 Hz).")
