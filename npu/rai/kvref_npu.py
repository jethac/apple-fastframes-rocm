"""FastFrames-S KV refinement network on the XDNA2 NPU via Ryzen AI 1.9 (VAIML BF16).

Two graphs, both with the fixed refine-timestep embedding folded in (it is a constant per model):
  * stem:  stem tokens [1560,1536] -> phi [1560,768]   (stem_in + 4 adaLN transformer blocks + LN)
  * head:  (nK, nV [1560,1536], phi [1560,768]) -> (K', V') for ONE DiT layer (30 such per frame)

Reference: the unmodified upstream modules in FP32 on CPU (torch). The release runs the refiner in
BF16, so a torch-CPU BF16 run is also reported to show the precision floor the NPU should be judged by.

Usage: python kvref_npu.py --weights .../kv_refiner.safetensors --part stem|head [--layer 0] --out DIR
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

TOK, DIM, DA = 1560, 1536, 768
REFINE_T = 833.3333129882812


class StemGraph(nn.Module):
    """compute_phi with the modulation of each block precomputed from the constant embedding."""

    def __init__(self, net, e):
        super().__init__()
        self.net = net
        self.mods = [[m.detach() for m in blk.mod(e)] for blk in net.stem_blocks]  # each [1,1,DA]

    def forward(self, stem_tokens):
        n = self.net
        x = n.stem_in(stem_tokens)                                     # [TOK, DA]
        for blk, (sh1, sc1, g1, sh2, sc2, g2) in zip(n.stem_blocks, self.mods):
            sh1, sc1, g1, sh2, sc2, g2 = (t.reshape(1, DA) for t in (sh1, sc1, g1, sh2, sc2, g2))
            h = F.layer_norm(x, (DA,)) * (1 + sc1) + sh1
            qkv = blk.qkv(h).reshape(TOK, 3, blk.n_heads, blk.head_dim).permute(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]                           # [heads, TOK, hd]
            a = torch.softmax(q @ k.transpose(1, 2) * blk.head_dim ** -0.5, dim=-1) @ v
            x = x + g1 * blk.proj(a.transpose(0, 1).reshape(TOK, DA))
            h = F.layer_norm(x, (DA,)) * (1 + sc2) + sh2
            x = x + g2 * blk.mlp(h)
        return n.stem_norm(x)


class HeadGraph(nn.Module):
    """GroupGEMM.forward for one layer, modulation precomputed, phi_proj included."""

    def __init__(self, net, e, layer):
        super().__init__()
        g = net.heads_grouped.gemms[0]
        self.g, self.L = g, layer
        self.phi_proj = net.heads_grouped.phi_proj[str(g.d_v)]
        e1 = F.silu(e).reshape(1, DA)
        mod = (e1 @ g.mod_w[layer] + g.mod_b[layer]).detach()
        self.shift, self.scale = mod.chunk(2, dim=-1)
        self.vscale = torch.exp(g.v_logscale[layer]).detach().reshape(1, DIM)

    def forward(self, nK, nV, phi):
        g, L = self.g, self.L
        pk = nK @ g.proj_k_w[L] + g.proj_k_b[L][0]
        pv = nV @ g.proj_v_w[L] + g.proj_v_b[L][0]
        h = torch.cat([pk, pv, self.phi_proj(phi)], dim=-1) @ g.fuse_w[L] + g.fuse_b[L][0]
        h = F.layer_norm(h, (g.d_v,)) * (1 + self.scale) + self.shift
        h = h @ g.m0_w[L] + g.m0_b[L][0]
        h = F.gelu(h, approximate="tanh") @ g.m2_w[L] + g.m2_b[L][0]
        k = nK + (h @ g.dK_w[L] + g.dK_b[L][0])
        kh, rh = k.reshape(TOK, g.num_heads, g.head_dim), nK.reshape(TOK, g.num_heads, g.head_dim)
        kh = kh * (rh.norm(dim=-1, keepdim=True) / (kh.norm(dim=-1, keepdim=True) + 1e-6))
        v = (nV + (h @ g.dV_w[L] + g.dV_b[L][0])) * self.vscale
        return kh.reshape(TOK, DIM), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default=str(Path.home() / "fastframes-npu/src"))
    ap.add_argument("--part", choices=["stem", "head"], required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    from fastframes.models.kv_refinement import KVRefinementNetwork
    from safetensors.torch import load_file

    args.out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.float() for k, v in load_file(args.weights).items()}
    net = KVRefinementNetwork(dim=DIM, num_heads=12, head_dim=128, num_blocks=30, d_a=sd["stem_in.weight"].shape[0])
    net.load_state_dict(sd, strict=True)
    net.eval()
    g = torch.Generator().manual_seed(4321)
    report = dict(component=f"kv_refiner_{args.part}", layer=args.layer if args.part == "head" else None,
                  tp_size=args.tp_size)
    with torch.no_grad():
        e = net.embed_timestep(REFINE_T, 1, "cpu", torch.float32)       # [1, DA]
        net_bf = KVRefinementNetwork(dim=DIM, num_heads=12, head_dim=128, num_blocks=30, d_a=DA)
        net_bf.load_state_dict(sd, strict=True)
        net_bf = net_bf.to(torch.bfloat16).eval()
        e_bf = net_bf.embed_timestep(REFINE_T, 1, "cpu", torch.bfloat16)
        if args.part == "stem":
            stem = torch.randn(1, TOK, DIM, generator=g)
            ref = net.compute_phi(stem, e)[0]                           # upstream, FP32
            ref_bf = net_bf.compute_phi(stem.bfloat16(), e_bf)[0].float()
            model = StemGraph(net, e).eval()
            mine = model(stem[0])
            inputs = dict(stem_tokens=stem[0])
            outs = ["phi"]
            refs = dict(phi=ref)
            refs_bf = dict(phi=ref_bf)
            flops = 2 * TOK * DIM * DA + 4 * (2 * TOK * DA * 3 * DA + 2 * 2 * TOK * TOK * DA + 2 * TOK * DA * DA
                                              + 2 * 2 * TOK * DA * 4 * DA)
        else:
            L = args.layer
            # K post-RoPE/RMSNorm: unit-ish per element; random stand-ins with the same scale.
            nK = torch.randn(TOK, DIM, generator=g)
            nV = torch.randn(TOK, DIM, generator=g)
            phi = torch.randn(TOK, DA, generator=g)
            gm = net.heads_grouped.gemms[0]
            phi_v = net.heads_grouped.phi_proj[str(gm.d_v)](phi).unsqueeze(0)
            rk, rv = gm(nK[None], nV[None], phi_v, e, layers=slice(L, L + 1))
            gb = net_bf.heads_grouped.gemms[0]
            phi_vb = net_bf.heads_grouped.phi_proj[str(gb.d_v)](phi.bfloat16()).unsqueeze(0)
            bk, bv = gb(nK[None].bfloat16(), nV[None].bfloat16(), phi_vb, e_bf, layers=slice(L, L + 1))
            model = HeadGraph(net, e, L).eval()
            mine = model(nK, nV, phi)
            inputs = dict(nK=nK, nV=nV, phi=phi)
            outs = ["k_ref", "v_ref"]
            refs = dict(k_ref=rk[0], v_ref=rv[0])
            refs_bf = dict(k_ref=bk[0].float(), v_ref=bv[0].float())
            flops = 2 * TOK * (2 * DIM * DA + DA * DA + 3 * DA * DA + 2 * DA * 2 * DA + 2 * DA * DIM)
        mine = mine if isinstance(mine, tuple) else (mine,)
        report["graph_vs_upstream_fp32"] = {o: compare(m.numpy(), refs[o].numpy()) for o, m in zip(outs, mine)}
        report["torch_bf16_vs_fp32"] = {o: compare(refs_bf[o].numpy(), refs[o].numpy()) for o in outs}

    onnx_path = args.out / (f"kvref_stem.onnx" if args.part == "stem" else f"kvref_head_L{args.layer}.onnx")
    if not onnx_path.exists():
        torch.onnx.export(model, tuple(inputs.values()), str(onnx_path), input_names=list(inputs),
                          output_names=outs, opset_version=17, dynamo=False)
    feeds = {k: v.numpy().astype(np.float32) for k, v in inputs.items()}
    cpu_out, cpu_t = run_cpu(onnx_path, feeds, iters=3)
    report["onnx_cpu_fp32_ms"] = cpu_t
    res = run_vitisai(onnx_path, feeds, args.out / f"vai_{onnx_path.stem}_tp{args.tp_size}_{int(time.time())}",
                      tp_size=args.tp_size or None, iters=args.iters, optimize_level=args.opt_level)
    report["npu"] = res["report"]
    report["npu_vs_upstream_fp32"] = {o: compare(res["outputs"][o], refs[o].numpy()) for o in outs}
    report["gemm_attn_gflop_per_call"] = flops / 1e9
    report["npu_effective_tflops"] = flops / (report["npu"]["wall_ms_min"] * 1e-3) / 1e12
    for o in outs:
        np.save(args.out / f"npu_{onnx_path.stem}_{o}_tp{args.tp_size}.npy", res["outputs"][o])
    (args.out / f"report_{onnx_path.stem}_tp{args.tp_size}_o{args.opt_level}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "npu"}, indent=1))
    print(json.dumps({k: v for k, v in report["npu"].items() if k not in ("wall_ms_all", "loaded_libs")}, indent=1))


if __name__ == "__main__":
    main()
