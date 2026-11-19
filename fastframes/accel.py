#
# ROCm port of FastFrames: platform-specific kernel selection.
#
"""Kernel selection for CUDA (upstream behaviour) and ROCm (AMD GPUs, e.g. Strix Halo gfx1151).

Upstream FastFrames hard-codes PyTorch's cuDNN SDPA backend, which does not exist on
ROCm. Every attention call in the package goes through :func:`sdpa` instead, which
dispatches to:

* ``cudnn``      - upstream behaviour (NVIDIA only);
* ``sdpa``       - PyTorch's own dispatch (on ROCm: AOTriton flash / memory-efficient);
* ``sdpa-flash`` - force PyTorch's flash backend (AOTriton on ROCm);
* ``sdpa-mem``   - force PyTorch's memory-efficient backend (AOTriton on ROCm);
* ``fa``         - the ``flash_attn`` package (CK or Triton-AMD build), ``[B, L, H, D]`` native;
* ``triton``     - FastFrames' own Triton flash attention (``fastframes.kernels.attention``), autotuned
                   for the per-frame shapes; reads the ``[B, L, H, D]`` KV cache in place;
* ``math``       - reference math kernel (slow; for debugging).

``FASTFRAMES_ATTN`` selects the backend; ``auto`` (default) picks ``cudnn`` on NVIDIA and the
fastest measured backend on ROCm (see ``rocm/bench_attention.py``).
"""
from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["IS_ROCM", "attention_backend", "sdpa", "sdpa_bshd", "set_attention_backend", "gpu_arch"]

IS_ROCM = torch.version.hip is not None

_BACKENDS = ("cudnn", "sdpa", "sdpa-flash", "sdpa-mem", "fa", "triton", "math")
_override = None


def gpu_arch():
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").split(":")[0] if IS_ROCM else f"sm{props.major}{props.minor}"


@functools.cache
def _have_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _rocm_default():
    # Strix Halo (gfx1151, ROCm 7.13, PyTorch 2.11): the Triton kernel runs at 24-27 TFLOP/s on the
    # per-frame shapes vs 13-17 for AOTriton through SDPA (rocm/README.md, bench_attention.py).
    return "triton"


def attention_backend():
    name = _override or os.environ.get("FASTFRAMES_ATTN", "auto")
    if name == "auto":
        name = _rocm_default() if IS_ROCM else "cudnn"
    if name not in _BACKENDS:
        raise ValueError(f"FASTFRAMES_ATTN={name!r}; expected one of {('auto',) + _BACKENDS}")
    if name == "cudnn" and IS_ROCM:
        name = _rocm_default()
    if name == "fa" and not _have_flash_attn():
        raise RuntimeError("FASTFRAMES_ATTN=fa but flash_attn is not installed")
    return name


def set_attention_backend(name):
    """Process-wide override (``None`` restores ``FASTFRAMES_ATTN`` / auto)."""
    global _override
    if name is not None and name not in _BACKENDS + ("auto",):
        raise ValueError(name)
    _override = name


_SDPA_KERNELS = {
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "sdpa-flash": [SDPBackend.FLASH_ATTENTION],
    "sdpa-mem": [SDPBackend.EFFICIENT_ATTENTION],
    "math": [SDPBackend.MATH],
}


def sdpa(q, k, v, *, is_causal=False, dropout_p=0.0, scale=None, backend=None):
    """``[B, H, L, D]`` scaled dot-product attention through the selected backend."""
    name = backend or attention_backend()
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p, scale=scale)
    if name == "triton" and not is_causal and dropout_p == 0.0 and q.shape[-1] in (64, 128):
        from .kernels.attention import flash_attention
        return flash_attention(q, k, v, scale=scale)
    if name == "triton":
        name = "sdpa"
    if name == "fa":
        from flash_attn import flash_attn_func
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                              dropout_p=dropout_p, softmax_scale=scale, causal=is_causal)
        return out.transpose(1, 2)
    if name == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p, scale=scale)
    with sdpa_kernel(_SDPA_KERNELS[name]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p, scale=scale)


def sdpa_bshd(q, k, v, *, is_causal=False, dropout_p=0.0, scale=None, backend=None):
    """``[B, L, H, D]`` in and out (avoids two transposes for ``flash_attn``)."""
    name = backend or attention_backend()
    if name == "fa" and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=is_causal)
    out = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
               is_causal=is_causal, dropout_p=dropout_p, scale=scale, backend=name)
    return out.transpose(1, 2)
