"""Single Gemm/MatMul at FastFrames shapes on the NPU: exactness, zero-column check, throughput.

Inputs are small positive integers and weights are a +-1/0 pattern, so BF16 results are exact
(all partial sums are integers < 2^8 * K... kept small) and any dropped column is unambiguous.
Usage: python gemm_probe.py --m 1560 --k 1536 --n 6144 --out DIR [--tp-size 1|0]  (0 = compiler default)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from npu_common import compare, run_vitisai  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=1560)
ap.add_argument("--k", type=int, default=1536)
ap.add_argument("--n", type=int, default=6144)
ap.add_argument("--tp-size", type=int, default=1)
ap.add_argument("--opt-level", type=int, default=1)
ap.add_argument("--random", action="store_true", help="randn data instead of exact integer data")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
M, K, N = args.m, args.k, args.n
rng = np.random.default_rng(0)
if args.random:
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
else:
    x = (np.arange(M * K).reshape(M, K) % 7 - 3).astype(np.float32)            # -3..3
    w = rng.integers(-1, 2, size=(K, N)).astype(np.float32)                      # -1,0,1
b = np.zeros(N, np.float32)
model = helper.make_model(helper.make_graph(
    [helper.make_node("MatMul", ["x", "w"], ["xw"]), helper.make_node("Add", ["xw", "b"], ["y"])], "gemm",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")]),
    opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
tag = f"{M}x{K}x{N}_tp{args.tp_size}_o{args.opt_level}{'_rand' if args.random else ''}"
path = args.out / f"gemm_{tag}.onnx"
onnx.save(model, path)
ref = x.astype(np.float64) @ w.astype(np.float64)
tp = None if args.tp_size == 0 else args.tp_size
res = run_vitisai(path, {"x": x}, args.out / f"vai_{tag}_{int(time.time())}", tp_size=tp, iters=20,
                  optimize_level=args.opt_level)
rep = dict(shape=[M, K, N], tp_size=args.tp_size, exact_data=not args.random,
           cmp=compare(res["outputs"]["y"], ref), npu=res["report"])
rep["exact"] = bool(np.array_equal(res["outputs"]["y"], ref.astype(np.float32))) if not args.random else None
rep["tflops"] = 2 * M * K * N / (rep["npu"]["wall_ms_min"] * 1e-3) / 1e12
(args.out / f"report_{tag}.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(dict(tag=tag, exact=rep["exact"], cmp=rep["cmp"], ms=rep["npu"]["wall_ms_min"],
                      tflops=rep["tflops"], npu_executed=rep["npu"]["npu_executed"],
                      placement=rep["npu"]["placement"]), indent=1))
