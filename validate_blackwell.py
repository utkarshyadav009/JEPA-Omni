#!/usr/bin/env python3
"""
validate_blackwell.py — run this FIRST on the Blackwell box, before any M2 work.

Blackwell RTX PRO 6000 is compute capability sm_120. PyTorch wheels built against
CUDA < 12.8 have no sm_120 kernels and fail with "no kernel image is available for
execution on the device" — but only when a kernel actually launches, so a bare
`torch.cuda.is_available()` (which returns True) is NOT a sufficient check. This
script forces real kernels (matmul, conv, bf16, SDPA attention) on every visible
GPU so a broken build fails loudly here, in seconds, instead of mid-run.

PASS = exit 0 and "ALL CHECKS PASSED". Any FAIL → reinstall torch for cu128+:
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
(or the cu129/cu130 nightly if 12.8 still lacks your card). Driver 595.x / CUDA 13.2
is forward-compatible; the wheel's bundled CUDA is what must carry sm_120 kernels.
"""
import sys


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
        return cond

    try:
        import torch
    except Exception as e:
        print(f"  [FAIL] import torch — {e}")
        return 1

    print(f"torch {torch.__version__} | bundled CUDA {torch.version.cuda}")
    if not check("CUDA available", torch.cuda.is_available()):
        return 1

    n = torch.cuda.device_count()
    print(f"visible GPUs: {n}")
    # cu128 is the first toolkit shipping sm_120; warn if the wheel predates it.
    cuda_ok = torch.version.cuda is not None and tuple(
        int(x) for x in torch.version.cuda.split(".")[:2]
    ) >= (12, 8)
    check("wheel CUDA >= 12.8 (sm_120 kernels)", cuda_ok, torch.version.cuda)

    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        print(f"\nGPU {i}: {name}  sm_{cap[0]}{cap[1]}")
        check(f"GPU{i} is Blackwell sm_120", cap == (12, 0), f"got sm_{cap[0]}{cap[1]}")
        dev = torch.device(f"cuda:{i}")
        try:
            # real fp32 matmul — the canonical "no kernel image" tripwire
            a = torch.randn(2048, 2048, device=dev)
            (a @ a).sum().item()
            check(f"GPU{i} fp32 matmul launches", True)
        except Exception as e:
            check(f"GPU{i} fp32 matmul launches", False, str(e)[:120])
            continue
        try:
            b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
            (b @ b).float().sum().item()
            check(f"GPU{i} bf16 matmul", True)
        except Exception as e:
            check(f"GPU{i} bf16 matmul", False, str(e)[:120])
        try:
            # conv kernels are a separate codepath from GEMM — check both
            x = torch.randn(8, 16, 64, 64, device=dev)
            w = torch.randn(32, 16, 3, 3, device=dev)
            torch.nn.functional.conv2d(x, w).sum().item()
            check(f"GPU{i} conv2d", True)
        except Exception as e:
            check(f"GPU{i} conv2d", False, str(e)[:120])
        try:
            # SDPA = the attention path V-JEPA2 / the predictor actually use
            q = torch.randn(2, 8, 256, 64, device=dev, dtype=torch.bfloat16)
            torch.nn.functional.scaled_dot_product_attention(q, q, q).float().sum().item()
            check(f"GPU{i} scaled_dot_product_attention (bf16)", True)
        except Exception as e:
            check(f"GPU{i} SDPA (bf16)", False, str(e)[:120])

    # optional: flash-attn (WavJEPA's HF card uses it; sdpa is a safe fallback)
    try:
        import flash_attn  # noqa: F401
        print(f"\nflash-attn {flash_attn.__version__} importable (optional)")
    except Exception:
        print("\nflash-attn not installed — fine, SDPA is the safe fallback on Blackwell")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
