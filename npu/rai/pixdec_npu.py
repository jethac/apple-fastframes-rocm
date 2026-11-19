"""FastFrames-S pixel decoder, one latent frame per call, on the XDNA2 NPU via Ryzen AI 1.9 (VAIML BF16).

The release ``PixelDecoder`` is re-expressed as a per-frame streaming graph with only 2D/Gemm ops:
each causal 3x3x3 depthwise Conv3d becomes three 3x3 depthwise Conv2d (one per time tap: the two
history frames and the current frame) summed; 1x1x1 Conv3d become MatMuls over the 1560 tokens.
History (each block's input of the previous two frames) is passed in and out as graph I/O.

Reference: the unmodified upstream clip-level ``PixelDecoder.forward`` in FP32 on CPU (torch).

Usage: python pixdec_npu.py --weights .../pixel_decoder.safetensors --out DIR [--tp-size 1] [--blocks 3]
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

H, W, DIM = 30, 52, 1536
TOK = H * W


class PixelDecoderFrame(nn.Module):
    def __init__(self, core, n_blocks=3, head=True):
        super().__init__()
        self.core = core
        self.blocks = (list(core.conv_low) + list(core.conv_high))[:n_blocks]
        self.head = head

    def forward(self, feat, shift, scale, *hist):
        # feat [TOK, DIM]; shift/scale [1, DIM]; hist: 2 per block, each [1, DIM, H, W]
        c = self.core
        x = F.layer_norm(feat, (DIM,), eps=c.norm.eps) * (1 + scale) + shift
        x = F.layer_norm(x, (DIM,), c.entry_norm.weight, c.entry_norm.bias, c.entry_norm.eps)
        x = x @ c.entry_pw.weight.view(DIM, DIM).t() + c.entry_pw.bias          # tokens [TOK, DIM]
        new_hist = []
        for i, b in enumerate(self.blocks):
            xs = x.t().reshape(1, DIM, H, W)                                     # NCHW
            new_hist.append(xs)
            w = b.dw.weight                                                      # [DIM,1,3,3,3]
            y = (F.conv2d(hist[2 * i], w[:, :, 0], None, padding=1, groups=DIM)
                 + F.conv2d(hist[2 * i + 1], w[:, :, 1], None, padding=1, groups=DIM)
                 + F.conv2d(xs, w[:, :, 2], b.dw.bias, padding=1, groups=DIM))
            y = y.reshape(DIM, TOK).t()
            y = F.layer_norm(y, (DIM,), b.norm.weight, b.norm.bias, b.norm.eps)
            y = y @ b.pw1.weight.flatten(1).t() + b.pw1.bias
            y = F.gelu(y)
            y = y @ b.pw2.weight.flatten(1).t() + b.pw2.bias
            x = x + y
        out = x @ c.fc_out.weight.t() + c.fc_out.bias if self.head else x
        return (out, *new_hist)


def unpatchify(tokens):  # [TOK, 3072] -> [4, 3, 480, 832] (StreamingPixelDecoder layout)
    t = tokens.reshape(H, W, 4, 16, 16, 3)
    return t.transpose(2, 5, 0, 3, 1, 4).reshape(4, 3, H * 16, W * 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default=str(Path.home() / "fastframes-npu/src"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--opt-level", type=int, default=1)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    from fastframes.models.pixel_decoder import PixelDecoder
    from safetensors.torch import load_file

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dec = PixelDecoder(DIM)
    sd = {k.removeprefix("pixel_head."): v.float() for k, v in load_file(args.weights).items()}
    dec.load_state_dict(sd, strict=True)
    dec.eval()
    head = not args.no_head
    frame = PixelDecoderFrame(dec, args.blocks, head).eval()

    g = torch.Generator().manual_seed(1234)
    Fn = args.frames
    feats = torch.randn(1, Fn * TOK, DIM, generator=g)
    mods = 0.2 * torch.randn(1, Fn, 1, DIM, generator=g)

    # Torch per-frame (FP32) and upstream clip-level reference (only meaningful for full model).
    with torch.no_grad():
        hist = [torch.zeros(1, DIM, H, W) for _ in range(2 * args.blocks)]
        feeds_all, torch_frames = [], []
        for f in range(Fn):
            shift, scale = (dec.modulation.unsqueeze(1) + mods[:, f:f + 1]).chunk(2, dim=2)
            fd = dict(feat=feats[0, f * TOK:(f + 1) * TOK], shift=shift.reshape(1, DIM), scale=scale.reshape(1, DIM))
            out = frame(fd["feat"], fd["shift"], fd["scale"], *hist)
            feeds_all.append((fd, [h.clone() for h in hist]))
            torch_frames.append(out[0].numpy())
            nh = []
            for i in range(args.blocks):
                nh += [hist[2 * i + 1], out[1 + i]]
            hist = nh
        report = dict(component="pixel_decoder", blocks=args.blocks, head=head, frames=Fn,
                      tp_size=args.tp_size, inputs="randn features, 0.2*randn modulation (latency-script convention)")
        if args.blocks == 3 and head:
            grid = torch.tensor([[Fn, H, W]])
            t0 = time.perf_counter()
            clip = dec(feats, mods, grid)[0].numpy()  # [4F, 3, 480, 832]
            report["torch_cpu_clip_ms_per_frame"] = (time.perf_counter() - t0) * 1e3 / Fn
            mine = np.concatenate([unpatchify(t) for t in torch_frames])
            report["torch_frame_graph_vs_upstream_clip"] = compare(mine, clip)
            np.save(args.out / "ref_clip_pixels.npy", clip)

    # ONNX export of the per-frame graph.
    onnx_path = args.out / "pixdec_frame.onnx"
    names_in = ["feat", "shift", "scale"] + [f"hist{i}_{j}" for i in range(args.blocks) for j in "ab"]
    names_out = ["tokens"] + [f"x{i}" for i in range(args.blocks)]
    fd0, h0 = feeds_all[0]
    if not onnx_path.exists():
        torch.onnx.export(frame, (fd0["feat"], fd0["shift"], fd0["scale"], *h0), str(onnx_path),
                          input_names=names_in, output_names=names_out, opset_version=17, dynamo=False)

    def np_feeds(fd, h):
        d = {k: v.numpy().astype(np.float32) for k, v in fd.items()}
        d.update({names_in[3 + i]: t.numpy().astype(np.float32) for i, t in enumerate(h)})
        return d

    cpu_out, cpu_t = run_cpu(onnx_path, np_feeds(fd0, h0), iters=3)
    report["onnx_cpu_fp32_ms"] = cpu_t
    report["onnx_cpu_vs_torch_frame0"] = compare(cpu_out["tokens"], torch_frames[0])

    res = run_vitisai(onnx_path, np_feeds(fd0, h0), args.out / f"vai_tp{args.tp_size}_{int(time.time())}",
                      tp_size=args.tp_size or None, iters=args.iters, optimize_level=args.opt_level)
    report["npu"] = res["report"]
    sess = res["session"]
    npu_frames = [res["outputs"]["tokens"]]
    # Frames >= 1 fed with the torch reference history (isolates per-frame error; no drift).
    for f in range(1, Fn):
        fd, h = feeds_all[f]
        npu_frames.append(sess.run(["tokens"], np_feeds(fd, h))[0])
    report["npu_vs_torch_tokens"] = [compare(n, t) for n, t in zip(npu_frames, torch_frames)]
    if head:
        report["npu_vs_torch_pixels"] = [compare(unpatchify(n), unpatchify(t)) for n, t in zip(npu_frames, torch_frames)]
        # uint8 frame error as the user would see it
        u = lambda a: np.round((np.clip(a, -1, 1) + 1) * 127.5)
        report["npu_pixel_uint8_max_abs_diff"] = [float(np.abs(u(unpatchify(n)) - u(unpatchify(t))).max())
                                                  for n, t in zip(npu_frames, torch_frames)]
        report["npu_pixel_uint8_mean_abs_diff"] = [float(np.abs(u(unpatchify(n)) - u(unpatchify(t))).mean())
                                                   for n, t in zip(npu_frames, torch_frames)]
    np.save(args.out / f"npu_tokens_tp{args.tp_size}.npy", np.stack(npu_frames))
    np.save(args.out / "torch_tokens.npy", np.stack(torch_frames))
    flops = 2 * TOK * DIM * DIM + args.blocks * 2 * (2 * TOK * DIM * 4 * DIM) + (2 * TOK * DIM * 3072 if head else 0)
    report["gemm_gflop_per_call"] = flops / 1e9
    report["npu_effective_tflops"] = flops / (report["npu"]["wall_ms_min"] * 1e-3) / 1e12
    out = args.out / f"report_tp{args.tp_size}_o{args.opt_level}_b{args.blocks}{'' if head else '_nohead'}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "npu"}, indent=1))
    print(json.dumps({k: v for k, v in report["npu"].items() if k not in ("wall_ms_all", "loaded_libs")}, indent=1))


if __name__ == "__main__":
    main()
