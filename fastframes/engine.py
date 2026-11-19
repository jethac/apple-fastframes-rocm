#
# ROCm port of FastFrames: compiled per-frame inference engine.
#
"""Fast per-frame inference for :class:`fastframes.pipeline.FastFrames`.

The reference rollout (``FastFrames.rollout``) runs the DiT eagerly. ``FastEngine``
re-implements the per-frame DiT call with the Sec. 3.4 optimisations of the paper's
latency profiler (``evaluation/latency/dit.py``), adapted to a cache that grows by one
latent frame per step:

* one fused QKV GEMM per layer, FP32 real-valued RoPE tables, GEMM patch embedding;
* text K/V projected once per prompt, timestep modulation precomputed per timestep;
* every transformer block split into two statically-shaped ``torch.compile`` regions
  (``_pre``: adaLN + QKV + RoPE; ``_post``: out-proj + text cross-attention + FFN).
  The self-attention over the growing KV cache sits between them and calls the
  platform's attention kernel directly, so the compiled code never recompiles when the
  cache length changes (a single compile serves all 21 frames and every layer);
* the KV-refinement stem uses the GEMM patch embedding instead of Conv3d.

Numerics match the reference path to BF16 rounding (fused GEMMs and RoPE tables change
accumulation order); ``rocm/check_engine.py`` reports the agreement on real weights.
"""
from __future__ import annotations

import os

import torch
import torch._dynamo
import torch._inductor.config as inductor
import torch.nn.functional as F

from . import accel
from .wan.causal_model import FastOps, rope_real

__all__ = ["FastEngine"]


def _pre(norm1, modulation, norm_q, norm_k, x, e0, qkv_w, qkv_b, rope, heads):
    b, tokens, dim = x.shape
    shape = (b, tokens, heads, dim // heads)
    mod = (modulation.unsqueeze(1) + e0).chunk(6, dim=2)
    h = (norm1(x).unsqueeze(1) * (1 + mod[1]) + mod[0]).flatten(1, 2)
    q, k, v = F.linear(h, qkv_w, qkv_b).chunk(3, dim=-1)
    q = rope_real(norm_q(q).view(shape), rope)
    k = rope_real(norm_k(k).view(shape), rope)
    return q, k, v.reshape(shape)


def _rope_exact(x, table):
    """The reference complex128 RoPE (``causal_rope_apply``) as FP64 real rotations."""
    pairs = x.double().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = table[..., 0], table[..., 1]
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).reshape_as(x).to(x.dtype)


def _pre_exact(norm1, modulation, sa, x, e0, rope, heads):
    """Reference-recipe variant (``fast_ops=False``): separate Q/K/V GEMMs, FP64 RoPE."""
    b, tokens, dim = x.shape
    mod = (modulation.unsqueeze(1) + e0).chunk(6, dim=2)
    h = (norm1(x).unsqueeze(1) * (1 + mod[1]) + mod[0]).flatten(1, 2)
    q, k, v = sa.qkv(h)
    return _rope_exact(q, rope).type_as(v), _rope_exact(k, rope).type_as(v), v


def _patch_exact(conv, latent):
    return conv(latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)


def _post(block, x, y, e0, text_k, text_v):
    b, tokens, dim = x.shape
    sa, ca = block.self_attn, block.cross_attn
    mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
    y = sa.o(y.reshape(b, tokens, dim))
    x = x + (y.unsqueeze(1) * mod[2]).flatten(1, 2)
    cq = ca.norm_q(ca.q(block.norm3(x))).view(b, tokens, ca.num_heads, ca.head_dim)
    y = accel.sdpa(cq.transpose(1, 2), text_k, text_v).transpose(1, 2)
    x = x + ca.o(y.reshape(b, tokens, dim))
    y = block.ffn((block.norm2(x).unsqueeze(1) * (1 + mod[4]) + mod[3]).flatten(1, 2))
    return x + (y.unsqueeze(1) * mod[5]).flatten(1, 2)


def _head(head, x, e_head, h, w):
    tokens = head(x, e_head)  # [1, 1, L, 64]
    return tokens.reshape(1, 1, h, w, 2, 2, 16).permute(0, 1, 6, 2, 4, 3, 5).reshape(1, 1, 16, 2 * h, 2 * w)


class BlockStreamer:
    """Keep the last ``count`` transformer blocks in pinned host memory and stream them in.

    On an APU (Strix Halo) host and device memory are the same LPDDR5X, so a block (0.7 GB for
    the 14B model) copies in a few ms on a side stream while the previous block computes. Two
    device slots are double-buffered; events order copy-after-use and use-after-copy. Values are
    untouched, so results are bit-identical to keeping every block resident.

    Offloaded blocks' parameters are re-pointed at the slots, so only the engine may run them.
    """

    def __init__(self, dit, fast, count):
        self.dit, self.fast = dit, fast
        self.first = dit.num_layers - count
        self.host, slot_template = {}, None
        for i in range(self.first, dit.num_layers):
            tensors = self._tensors(i)
            self.host[i] = {k: t.detach().to("cpu").pin_memory() for k, t in tensors.items()}
            slot_template = tensors
        device = next(iter(slot_template.values())).device
        self.slots = [{k: torch.empty_like(t, device=device) for k, t in slot_template.items()} for _ in range(2)]
        self.bytes_per_block = sum(t.numel() * t.element_size() for t in slot_template.values())
        for i in range(self.first, dit.num_layers):  # drop the device copies
            self._bind(i, self.slots[0])
        torch.cuda.empty_cache()
        self.stream = torch.cuda.Stream()
        self.loaded = [torch.cuda.Event() for _ in range(2)]
        self.free = [torch.cuda.Event() for _ in range(2)]
        for e in self.free:
            e.record()

    def _tensors(self, i):
        block = self.dit.blocks[i]
        fused = self.fast is not None
        skip = {f"self_attn.{n}.{a}" for n in "qkv" for a in ("weight", "bias")} if fused else set()
        out = {k: p for k, p in block.named_parameters() if k not in skip}
        if fused:
            out["__qkv_w"], out["__qkv_b"] = self.fast.qkv[i]
        return out

    def _bind(self, i, slot):
        block = self.dit.blocks[i]
        for k, p in block.named_parameters():
            if k in slot:
                p.data = slot[k]
        if self.fast is not None:
            w, b = slot["__qkv_w"], slot["__qkv_b"]
            sa, dim = block.self_attn, self.dit.dim
            for j, lin in enumerate((sa.q, sa.k, sa.v)):
                lin.weight.data = w[j * dim:(j + 1) * dim]
                lin.bias.data = b[j * dim:(j + 1) * dim]
            self.fast.qkv[i] = (w, b)

    def prefetch(self, i):
        s = (i - self.first) % 2
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.free[s])
            for k, dst in self.slots[s].items():
                dst.copy_(self.host[i][k], non_blocking=True)
            self.loaded[s].record(self.stream)

    def acquire(self, i):
        """Bind block ``i`` to its slot once loaded and start loading block ``i + 1``."""
        s = (i - self.first) % 2
        torch.cuda.current_stream().wait_event(self.loaded[s])
        self._bind(i, self.slots[s])
        if i + 1 < self.dit.num_layers:
            self.prefetch(i + 1)

    def load_sync(self, i):
        """Load and bind block ``i`` outside the streamed forward (per-prompt work)."""
        s = (i - self.first) % 2
        torch.cuda.current_stream().wait_event(self.free[s])
        for k, dst in self.slots[s].items():
            dst.copy_(self.host[i][k])
        self._bind(i, self.slots[s])

    def release(self, i):
        self.free[(i - self.first) % 2].record()


def plan_offload(pipeline, margin_gib=3.0):
    """Blocks to stream so that weights + a full KV cache + ``margin`` fit in free device memory."""
    dit, cfg = pipeline.dit, pipeline.config
    kv = dit.num_layers * 2 * cfg.num_frames * pipeline.frame_tokens * dit.dim * 2
    free, _ = torch.cuda.mem_get_info()
    need = kv + margin_gib * 2 ** 30 - free
    if need <= 0:
        return 0
    block = sum(p.numel() * p.element_size() for p in dit.blocks[0].parameters())
    count = int(-(-need // block)) + 2  # two device slots stay resident
    if count > dit.num_layers:
        raise RuntimeError(f"model does not fit: needs {need / 2**30:.1f} GiB more than streaming can free")
    return count


class FastEngine:
    """Compiled replacement for ``CausalWanModel.forward_frame`` and the refiner stem.

    ``compile``: ``None`` (eager), ``"default"`` or ``"max-autotune-no-cudagraphs"``.
    ``exact``: follow the reference ops of a ``fast_ops=False`` recipe (FF-S): separate Q/K/V
    GEMMs, FP64 RoPE and the Conv3d patch embedding. Default: ``not config.fast_ops``.
    """

    def __init__(self, pipeline, compile="max-autotune-no-cudagraphs", exact=None, offload="auto"):
        self.pipe = pipeline
        dit = pipeline.dit
        cfg = pipeline.config
        self.dit = dit
        self.height, self.width = cfg.latent_size
        self.grid = (self.height // 2, self.width // 2)
        self.heads = dit.num_heads
        dit.freqs = dit.freqs.to(pipeline.device)
        self.exact = (not cfg.fast_ops) if exact is None else exact
        self.fast = None if self.exact else (pipeline.fast or FastOps(dit, self.height, self.width, cfg.num_frames))
        if self.exact:
            # complex128 tables of the reference RoPE, stored as FP64 (cos, sin) pairs.
            h, w = self.grid
            half = dit.dim // dit.num_heads // 2
            t, y, xx = dit.freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
            self.rope64 = [torch.view_as_real(torch.cat((
                t[f:f + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                y[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                xx[:w].view(1, 1, w, -1).expand(1, h, w, -1)), dim=-1).reshape(h * w, 1, half)).contiguous()
                for f in range(cfg.num_frames)]
        mode = None if compile in (None, "none") else compile
        if mode is not None:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            inductor.fx_graph_cache = True
            # Round fused intermediates to BF16 exactly where eager PyTorch does, so compiled code
            # follows the numerics the distilled models were evaluated with (rocm/README.md).
            inductor.emulate_precision_casts = os.environ.get("FASTFRAMES_EMULATE_CASTS", "1") == "1"
            kw = dict(dynamic=False, fullgraph=False)
            if mode != "default":
                kw["mode"] = mode
            self._pre = torch.compile(_pre_exact if self.exact else _pre, **kw)
            self._post = torch.compile(_post, **kw)
            self._head = torch.compile(_head, **kw)
            patch = (lambda latent: _patch_exact(dit.patch_embedding, latent)) if self.exact else self.fast.patch_embed
            self._patch = torch.compile(patch, **kw)
        else:
            self._pre, self._post, self._head = (_pre_exact if self.exact else _pre), _post, _head
            self._patch = ((lambda latent: _patch_exact(dit.patch_embedding, latent)) if self.exact
                           else self.fast.patch_embed)
        self._text = None
        count = plan_offload(pipeline) if offload == "auto" else int(offload or 0)
        self.streamer = BlockStreamer(dit, self.fast, count) if count else None
        self.offloaded_blocks = count

    # ------------------------------------------------------------------ per prompt
    def text_cache(self, prompt_embeds):
        """``CausalWanModel.text_cache`` that also works with streamed blocks."""
        if self.streamer is None:
            return self.dit.text_cache(prompt_embeds)
        context = self.dit.embed_text(prompt_embeds)
        caches = []
        for i, block in enumerate(self.dit.blocks):
            if i >= self.streamer.first:
                self.streamer.load_sync(i)
            k, v = block.cross_attn.text_kv(context)
            caches.append({"k": k, "v": v})
            if i >= self.streamer.first:
                self.streamer.release(i)
        return caches

    def set_text(self, text_cache):
        """Per-layer text K/V in ``[B, H, 512, D]`` (contiguous for the attention kernel)."""
        self._text = [(c["k"].transpose(1, 2).contiguous(), c["v"].transpose(1, 2).contiguous())
                      for c in text_cache]

    # ------------------------------------------------------------------ DiT
    def forward_frame(self, latent, e0, e_head, kv_cache, frame):
        """Same contract as ``CausalWanModel.forward_frame`` (text cache set by ``set_text``)."""
        dit, fast = self.dit, self.fast
        x = self._patch(latent)
        tokens = x.shape[1]
        start, end = frame * tokens, (frame + 1) * tokens
        rope = self.rope64[frame] if self.exact else fast.rope[frame]
        streamer = self.streamer
        if streamer is not None:
            streamer.prefetch(streamer.first)
        for i, block in enumerate(dit.blocks):
            offloaded = streamer is not None and i >= streamer.first
            if offloaded:
                streamer.acquire(i)
            sa = block.self_attn
            if self.exact:
                q, k, v = self._pre(block.norm1, block.modulation, sa, x, e0, rope, self.heads)
            else:
                q, k, v = self._pre(block.norm1, block.modulation, sa.norm_q, sa.norm_k, x, e0, *fast.qkv[i], rope,
                                    self.heads)
            cache = kv_cache[i]
            cache["k"][:, start:end].copy_(k)
            cache["v"][:, start:end].copy_(v)
            y = accel.sdpa(q.transpose(1, 2), cache["k"][:, :end].transpose(1, 2),
                           cache["v"][:, :end].transpose(1, 2))
            x = self._post(block, x, y.transpose(1, 2), e0, *self._text[i])
            if offloaded:
                streamer.release(i)
        return self._head(dit.head, x, e_head, *self.grid), x

    def stem(self, x0):
        """Refiner stem tokens of a clean latent ``[1, 1, 16, H, W]`` (GEMM patch embedding)."""
        return self._patch(x0)


def default_compile_mode():
    return os.environ.get("FASTFRAMES_COMPILE", "max-autotune-no-cudagraphs")
