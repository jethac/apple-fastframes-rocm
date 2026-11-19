#!/usr/bin/env python3
"""BF16 GEMM microbenchmark on the FastFrames per-frame shapes (M = 1560 tokens).

  python rocm/bench_gemm.py                       # default BLAS (hipBLASLt on gfx1151)
  TORCH_BLAS_PREFER_HIPBLASLT=0 python rocm/bench_gemm.py   # rocBLAS
  PYTORCH_TUNABLEOP_ENABLED=1 python rocm/bench_gemm.py     # TunableOp search
"""
import json
import time

import torch

M = 1560
SHAPES = {  # name: (K, N)
    "1.3b qkv": (1536, 4608), "1.3b o": (1536, 1536), "1.3b ffn1": (1536, 8960), "1.3b ffn2": (8960, 1536),
    "14b qkv": (5120, 15360), "14b o": (5120, 5120), "14b ffn1": (5120, 13824), "14b ffn2": (13824, 5120),
}


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    torch.manual_seed(0)
    for name, (k, n) in SHAPES.items():
        x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
        b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        ms = bench(lambda: torch.nn.functional.linear(x, w, b))
        print(json.dumps(dict(gemm=name, m=M, k=k, n=n, ms=round(ms, 3), tflops=round(2 * M * k * n / ms / 1e9, 2))),
              flush=True)


if __name__ == "__main__":
    main()
