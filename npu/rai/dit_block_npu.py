"""One FastFrames-S (Causal Forcing++ Wan2.1-1.3B) DiT block, one latent frame, on the XDNA2 NPU via
Ryzen AI 1.9 (VAIML BF16).

The graph is ``CausalWanAttentionBlock.forward_frame`` for frame ``f`` with static shapes:
inputs x [1560,1536], e6 [6,1536] (block modulation + time embedding, tiny, host-side),
kprev/vprev [f*1560,1536] (the KV cache of earlier frames), text K/V [512,1536] (per-prompt cache);
outputs x_out, and this frame's RoPE'd k and v (to append to the cache). RoPE is rewritten as
``q*C + (q @ R)*S`` with a constant 128x128 pair-rotation R so that it is elementwise + GEMM.

References (CPU, torch, upstream code): FP32 everything; and the release precision (BF16 weights,
BF16 activations, upstream BF16 attention) to show the precision floor.

Usage: python dit_block_npu.py --ckpt .../framewise-2step.pt --frame 0 --out DIR [--layer 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from npu_common import compare, run_cpu, run_vitisai  # noqa: E402

TOK, DIM, NH, HD, LT = 1560, 1536, 12, 128, 512
LAT_H, LAT_W = 30, 52


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def attn(q, k, v):  # [Lq,DIM],[Lk,DIM],[Lk,DIM] -> [Lq,DIM]
    q = q.reshape(-1, NH, HD).transpose(0, 1)
    k = k.reshape(-1, NH, HD).permute(1, 2, 0)
    v = v.reshape(-1, NH, HD).transpose(0, 1)
    p = torch.softmax((q @ k) * HD ** -0.5, dim=-1)
    return (p @ v).transpose(0, 1).reshape(-1, DIM)


class BlockGraph(nn.Module):
    def __init__(self, blk, cos, sin, has_prev):
        super().__init__()
        self.b, self.has_prev = blk, has_prev
        self.register_buffer("C", cos)  # [TOK, DIM]
        self.register_buffer("S", sin)
        R = torch.zeros(HD, HD)
        for i in range(HD // 2):  # (r, i) -> (-i, r)
            R[2 * i + 1, 2 * i] = -1.0
            R[2 * i, 2 * i + 1] = 1.0
        self.register_buffer("R", R)

    def rope(self, t):
        rot = (t.reshape(TOK * NH, HD) @ self.R).reshape(TOK, DIM)
        return t * self.C + rot * self.S

    def forward(self, x, e6, tk, tv, kprev=None, vprev=None):
        b = self.b
        sa, ca = b.self_attn, b.cross_attn
        e = [e6[i:i + 1] for i in range(6)]
        h = F.layer_norm(x, (DIM,), eps=b.norm1.eps) * (1 + e[1]) + e[0]
        q = self.rope(rms(sa.q(h), sa.norm_q.weight, sa.norm_q.eps))
        k = self.rope(rms(sa.k(h), sa.norm_k.weight, sa.norm_k.eps))
        v = sa.v(h)
        keys = torch.cat([kprev, k], 0) if self.has_prev else k
        vals = torch.cat([vprev, v], 0) if self.has_prev else v
        x = x + sa.o(attn(q, keys, vals)) * e[2]
        h3 = F.layer_norm(x, (DIM,), b.norm3.weight, b.norm3.bias, b.norm3.eps)
        q2 = rms(ca.q(h3), ca.norm_q.weight, ca.norm_q.eps)
        x = x + ca.o(attn(q2, tk, tv))
        h2 = F.layer_norm(x, (DIM,), eps=b.norm2.eps) * (1 + e[4]) + e[3]
        x = x + b.ffn(h2) * e[5]
        return x, k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", default=str(Path.home() / "fastframes-npu/src"))
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    import fastframes.wan.causal_model as cm
    from fastframes.wan.model import WAN_CONFIGS, sinusoidal_embedding_1d
    from safetensors.torch import load_file, save_file

    args.out.mkdir(parents=True, exist_ok=True)
    wfile = args.out / f"dit_layer{args.layer}_weights.safetensors"
    if not wfile.exists():  # pull one block + time embedding out of the 5.6 GB pickle once
        sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)["generator_ema"]
        sd = {k.removeprefix("model.").removeprefix("_fsdp_wrapped_module."): v for k, v in sd.items()}
        keep = {k: v.float().contiguous() for k, v in sd.items()
                if k.startswith(f"blocks.{args.layer}.") or k.startswith(("time_embedding.", "time_projection."))}
        save_file(keep, str(wfile))
    sd = load_file(str(wfile))
    cfg = WAN_CONFIGS["Wan2.1-T2V-1.3B"]
    blk = cm.CausalWanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.num_heads, cfg.qk_norm, cfg.cross_attn_norm, cfg.eps)
    pre = f"blocks.{args.layer}."
    blk.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
    blk.eval()
    te = nn.Sequential(nn.Linear(cfg.freq_dim, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
    te.load_state_dict({k.removeprefix("time_embedding."): v for k, v in sd.items() if k.startswith("time_embedding.")})
    tp = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 6 * DIM))
    tp.load_state_dict({k.removeprefix("time_projection."): v for k, v in sd.items() if k.startswith("time_projection.")})
    from fastframes.wan.model import rope_params
    freqs = torch.cat([rope_params(1024, HD - 4 * (HD // 6)), rope_params(1024, 2 * (HD // 6)),
                       rope_params(1024, 2 * (HD // 6))], dim=1)  # == WanBackbone.rope_table()

    f, Lp = args.frame, args.frame * TOK
    g = torch.Generator().manual_seed(77 + f)
    grid = torch.tensor([[1, LAT_H, LAT_W]])
    report = dict(component="dit_block", layer=args.layer, frame=f, kv_len=Lp + TOK, tp_size=args.tp_size)

    def run_ref(dtype, attn_fp32):
        """Upstream forward_frame for frames 0..f; returns (x_in, e6, text kv, cache, outputs of frame f)."""
        b = blk.to(dtype)
        orig = cm.attention
        cm.attention = ((lambda q, k, v: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
            if attn_fp32 else (lambda q, k, v: orig(q, k, v, backend="auto")))
        import fastframes.wan.model as wm
        orig_x = wm.cross_attention
        wm.cross_attention = cm.attention
        gg = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            t = torch.full((1,), 1000.0)
            emb = te(sinusoidal_embedding_1d(cfg.freq_dim, t).float())
            e0 = tp(emb).unflatten(1, (6, DIM)).unsqueeze(1).to(dtype)            # [1,1,6,DIM]
            ctx = torch.randn(1, LT, DIM, generator=gg).to(dtype)
            tk, tv = b.cross_attn.text_kv(ctx)
            cache = {"k": torch.zeros(1, (f + 1) * TOK, NH, HD, dtype=dtype),
                     "v": torch.zeros(1, (f + 1) * TOK, NH, HD, dtype=dtype)}
            for fr in range(f + 1):
                xin = torch.randn(1, TOK, DIM, generator=gg).to(dtype)
                out = b.forward_frame(xin, e0, grid, freqs, {"k": tk, "v": tv}, cache, fr)
        cm.attention, wm.cross_attention = orig, orig_x
        e6 = (b.modulation.unsqueeze(1) + e0)[0, 0]
        return xin[0], e6, tk[0].reshape(LT, DIM), tv[0].reshape(LT, DIM), cache, out[0]

    xin, e6, tk, tv, cache, ref_x = run_ref(torch.float32, attn_fp32=True)
    ref_k = cache["k"][0, Lp:].reshape(TOK, DIM)
    ref_v = cache["v"][0, Lp:].reshape(TOK, DIM)
    kprev = cache["k"][0, :Lp].reshape(Lp, DIM)
    vprev = cache["v"][0, :Lp].reshape(Lp, DIM)
    _, _, _, _, cache_b, bf_x = run_ref(torch.bfloat16, attn_fp32=False)
    blk.float()
    report["torch_bf16_release_vs_fp32"] = dict(x=compare(bf_x.float().numpy(), ref_x.numpy()),
                                                k=compare(cache_b["k"][0, Lp:].reshape(TOK, DIM).float().numpy(),
                                                          ref_k.numpy()))
    # Rope tables for frame f in the rotation form (cos/sin repeated over each (r,i) pair, all heads).
    c = HD // 2
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    ang = torch.cat([fs[0][f:f + 1].view(1, 1, 1, -1).expand(1, LAT_H, LAT_W, -1),
                     fs[1][:LAT_H].view(1, LAT_H, 1, -1).expand(1, LAT_H, LAT_W, -1),
                     fs[2][:LAT_W].view(1, 1, LAT_W, -1).expand(1, LAT_H, LAT_W, -1)], dim=-1).reshape(TOK, 1, c)
    cos = ang.real.float().repeat_interleave(2, -1).expand(TOK, NH, HD).reshape(TOK, DIM).contiguous()
    sin = ang.imag.float().repeat_interleave(2, -1).expand(TOK, NH, HD).reshape(TOK, DIM).contiguous()
    graph = BlockGraph(blk, cos, sin, Lp > 0).eval()
    ins = dict(x=xin, e6=e6, tk=tk, tv=tv)
    if Lp:
        ins.update(kprev=kprev, vprev=vprev)
    with torch.no_grad():
        gx, gk, gv = graph(*ins.values())
    report["graph_vs_upstream_fp32"] = dict(x=compare(gx.numpy(), ref_x.numpy()), k=compare(gk.numpy(), ref_k.numpy()),
                                            v=compare(gv.numpy(), ref_v.numpy()))
    onnx_path = args.out / f"dit_L{args.layer}_f{f}.onnx"
    if not onnx_path.exists():
        torch.onnx.export(graph, tuple(ins.values()), str(onnx_path), input_names=list(ins),
                          output_names=["x_out", "k_new", "v_new"], opset_version=17, dynamo=False)
    feeds = {k: v.detach().numpy().astype(np.float32) for k, v in ins.items()}
    _, cpu_t = run_cpu(onnx_path, feeds, iters=3)
    report["onnx_cpu_fp32_ms"] = cpu_t
    res = run_vitisai(onnx_path, feeds, args.out / f"vai_{onnx_path.stem}_tp{args.tp_size}_{int(time.time())}",
                      tp_size=args.tp_size or None, iters=args.iters, optimize_level=args.opt_level)
    report["npu"] = res["report"]
    o = res["outputs"]
    report["npu_vs_upstream_fp32"] = dict(x=compare(o["x_out"], ref_x.numpy()), k=compare(o["k_new"], ref_k.numpy()),
                                          v=compare(o["v_new"], ref_v.numpy()))
    L = Lp + TOK
    flops = 2 * TOK * DIM * DIM * 4 + 2 * 2 * TOK * L * DIM + 2 * TOK * DIM * DIM * 2 + 2 * 2 * TOK * LT * DIM \
        + 2 * 2 * TOK * DIM * cfg.ffn_dim
    report["gflop_per_call"] = flops / 1e9
    report["npu_effective_tflops"] = flops / (report["npu"]["wall_ms_min"] * 1e-3) / 1e12
    (args.out / f"report_{onnx_path.stem}_tp{args.tp_size}_o{args.opt_level}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "npu"}, indent=1))
    print(json.dumps({k: v for k, v in report["npu"].items() if k not in ("wall_ms_all", "loaded_libs")}, indent=1))


if __name__ == "__main__":
    main()
