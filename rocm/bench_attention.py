#!/usr/bin/env python3
"""Attention microbenchmark on FastFrames shapes (one 480p latent frame against the KV cache).

  python rocm/bench_attention.py --model 1.3b      # 12 heads x 128
  python rocm/bench_attention.py --model 14b       # 40 heads x 128

Times every available backend of ``fastframes.accel`` for cache lengths of 1, 6, 11, 16
and 21 latent frames, checks each against the math kernel, and prints effective TFLOP/s.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes import accel  # noqa: E402

HEADS = {"1.3b": 12, "14b": 40}
TOKENS = 1560


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=HEADS, default="1.3b")
    p.add_argument("--frames", default="1,6,11,16,21")
    p.add_argument("--backends", default="sdpa,triton")
    p.add_argument("--cross", action="store_true", help="also time text cross-attention (512 keys)")
    p.add_argument("--bshd", action="store_true", help="K/V as transposed views of a [B, L, H, D] cache")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    heads, d = HEADS[args.model], 128
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    lens = [int(f) * TOKENS for f in args.frames.split(",")] + ([512] if args.cross else [])
    for lk in lens:
        q = torch.randn(1, heads, TOKENS, d, device="cuda", dtype=torch.bfloat16, generator=g)
        kv_shape = (1, lk, heads, d) if args.bshd else (1, heads, lk, d)
        k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, generator=g)
        if args.bshd:
            k, v = k.transpose(1, 2), v.transpose(1, 2)
        ref = accel.sdpa(q.float(), k.float(), v.float(), backend="math")
        flops = 4 * heads * TOKENS * lk * d
        for name in args.backends.split(","):
            row = dict(model=args.model, kv_len=lk, backend=name)
            try:
                out = accel.sdpa(q, k, v, backend=name)
                row["max_err"] = (out.float() - ref).abs().max().item()
                ms = bench(lambda: accel.sdpa(q, k, v, backend=name))
                row.update(ms=round(ms, 3), tflops=round(flops / ms / 1e9, 2))
            except Exception as e:  # backend unavailable on this build / arch
                row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            rows.append(row)
            print(json.dumps(row), flush=True)
    try:  # report the configurations the autotuner picked
        from fastframes.kernels.attention import _fwd
        for key, cfg in _fwd.cache.items():
            print("best", key, cfg, flush=True)
    except Exception:
        pass
    if args.json:
        args.json.write_text(json.dumps(dict(arch=accel.gpu_arch(), torch=torch.__version__,
                                             hip=torch.version.hip, rows=rows), indent=1))


if __name__ == "__main__":
    main()
