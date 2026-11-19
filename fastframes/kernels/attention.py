#
# ROCm port of FastFrames: forward-only flash attention in Triton, tuned for RDNA 3.5 (gfx1151).
#
"""Non-causal, unmasked BF16 flash attention (forward only) for FastFrames' per-frame shapes.

FastFrames attends one latent frame (1560 queries at 480p) to a KV cache of up to 21 frames
(32760 keys), head dim 128, no mask. That is a single, regular shape family, so a small
hand-written kernel can be autotuned for it instead of relying on a generic library kernel.

``flash_attention(q, k, v)`` takes ``[B, H, L, D]`` tensors with arbitrary strides (so the
``[B, L, H, D]`` KV cache is read in place, without a transpose copy).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["flash_attention"]


def _configs():
    """Search space for RDNA 3.5; ``FASTFRAMES_ATTN_TUNE=full`` widens it (slower first call)."""
    import os
    full = os.environ.get("FASTFRAMES_ATTN_TUNE") == "full"
    out = []
    for bm in (64, 128) + ((256,) if full else ()):
        for bn in (32, 64, 128):
            for warps in (4, 8):
                for stages in (1, 2):
                    # waves_per_eu=1 won every shape in the wide sweep on gfx1151 (+4..28%).
                    for wpe in ((0, 1, 2, 4) if full else (0, 1)):
                        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "waves_per_eu": wpe},
                                                 num_warps=warps, num_stages=stages))
    return out


@triton.autotune(configs=_configs(), key=["H", "LQ", "LK_BUCKET", "D"])
@triton.jit
def _fwd(Q, K, V, O, sm_scale,
         sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sob, soh, som, sod,
         H, LQ, LK, LK_BUCKET, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q_ptrs = Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
    qk_scale = sm_scale * 1.4426950408889634  # log2(e): use exp2
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    k_base = K + b * skb + h * skh
    v_base = V + b * svb + h * svh
    # Full key blocks (no bounds checks), then one masked tail block.
    full = (LK // BLOCK_N) * BLOCK_N
    for start in range(0, full, BLOCK_N):
        kt = tl.load(k_base + (start + offs_n)[None, :] * skn + offs_d[:, None] * skd)
        s = tl.dot(q, kt) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + (start + offs_n)[:, None] * svn + offs_d[None, :] * svd)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    if full < LK:
        kv_mask = (full + offs_n) < LK
        kt = tl.load(k_base + (full + offs_n)[None, :] * skn + offs_d[:, None] * skd, mask=kv_mask[None, :],
                     other=0.0)
        s = tl.dot(q, kt) * qk_scale
        s = tl.where(kv_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + (full + offs_n)[:, None] * svn + offs_d[None, :] * svd, mask=kv_mask[:, None],
                    other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < LQ)


def _launch(q, k, v, scale):
    b, h, lq, d = q.shape
    lk = k.shape[2]
    # Output laid out as [B, Lq, H, D] so callers' transpose back to [B, L, H, D] is free.
    out = torch.empty(b, lq, h, d, device=q.device, dtype=q.dtype).transpose(1, 2)

    def grid(meta):
        return triton.cdiv(lq, meta["BLOCK_M"]), b * h

    _fwd[grid](q, k, v, out, scale, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
               h, lq, lk, triton.next_power_of_2(lk), D=d)
    return out


# Opaque to torch.compile: Inductor calls the autotuned kernel instead of re-tracing it.
@torch.library.custom_op("fastframes::flash_attention", mutates_args=())
def _flash_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    return _launch(q, k, v, scale)


@_flash_attention_op.register_fake
def _(q, k, v, scale):
    b, h, lq, d = q.shape
    return q.new_empty(b, lq, h, d).transpose(1, 2)


def flash_attention(q, k, v, scale=None):
    """``[B, H, Lq, D]`` x ``[B, H, Lk, D]`` -> ``[B, H, Lq, D]`` (no mask, no dropout; BF16/FP16)."""
    d = q.shape[-1]
    if d not in (64, 128):
        raise ValueError(f"head dim {d} not supported")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"dtype mismatch {q.dtype} {k.dtype} {v.dtype}")
    return _flash_attention_op(q, k, v, d ** -0.5 if scale is None else float(scale))
