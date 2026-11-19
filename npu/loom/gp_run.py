#!/usr/bin/env python3
"""Run the documented XDNA gated-projection example (ffn_gate_up_quadratic_bf16,
loom/docs/examples/targets/xdna-pipelines/gated-projection.cxx) on the NPU via
`loom-test --device=xdna` and check against a numpy reference.

loom-run only invokes kernel.def roots, so we append a check.case that reads
npy inputs, launches the pipeline, and writes the output npy.

Usage: python gp_run.py <imported gated-projection.loom> <workdir> [K]
"""
import subprocess, sys, os, numpy as np

def bf16_round(x):
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    b = x.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000  # RNE
    return b.astype(np.uint32).view(np.float32)

def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> 16).astype(np.uint16).view(np.int16)

src, wd = sys.argv[1], os.path.abspath(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 512
os.makedirs(wd, exist_ok=True)
rng = np.random.default_rng(0)
x = bf16_round(rng.uniform(-1, 1, K) * 0.25)
w = bf16_round(rng.uniform(-1, 1, (2, K)) * 0.25)
np.save(f"{wd}/x.npy", bf16_bits(x)); np.save(f"{wd}/w.npy", bf16_bits(w).reshape(-1))

gate = bf16_round(np.float32(np.dot(x.astype(np.float64), w[0])))[0]
up = bf16_round(np.float32(np.dot(x.astype(np.float64), w[1])))[0]
lin = bf16_round(gate * np.float32(0.25))[0]
scale = bf16_round(np.float32(0.5) + lin)[0]
act = bf16_round(gate * scale)[0]
ref = float(bf16_round(act * up)[0])

case = f"""
check.case public @gp_npy {{
  %x = check.file.read.npy path("{wd}/x.npy") : tensor<{K}xi16>
  %w = check.file.read.npy path("{wd}/w.npy") : tensor<{2*K}xi16>
  %o = check.generate.fill value(-12345.0) : tensor<1xf32>
  kernel.launch @ffn_gate_up_quadratic_bf16(%x, %w, %o) : (tensor<{K}xi16>, tensor<{2*K}xi16>, tensor<1xf32>)
  check.file.write.npy value(%o) path("out.npy") mode(always) : tensor<1xf32>
  check.return
}}
"""
full = f"{wd}/gp_check.loom"
open(full, "w").write(open(src).read() + case)
cmd = ["loom-test", full, "--device=xdna", f"--config=ffn_gate_up.input_size={K}",
       f"--file-output-root={wd}"]
print(" ".join(cmd), flush=True)
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-3000:]); print(p.stderr[-3000:], file=sys.stderr)
got = float(np.load(f"{wd}/out.npy")[0]) if os.path.exists(f"{wd}/out.npy") else None
print(f"reference gate={float(gate)} up={float(up)} result={ref!r}")
print(f"npu result={got!r}  exact_match={got == ref}")
