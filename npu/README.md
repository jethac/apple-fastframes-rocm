# FastFrames-S on the Strix Halo XDNA2 NPU

Experiment: can Apple's FastFrames-S (Causal Forcing++ Wan2.1-1.3B causal DiT + KV refiner + pixel
decoder) run on the Ryzen AI MAX+ 395 NPU, and how does it compare with the Radeon 8060S iGPU?

**Short answer: it works, but it is slow.**
- **Ryzen AI 1.9 (ONNX / VitisAI EP, VAIML BF16):**
  - Every component compiles and runs 100% on the NPU, and matches the reference once the
    stamp-bug workaround is applied.
  - It is roughly 5–15x slower than the GPU, and comparable to the 16-core CPU.
  - The only sensible use is a hybrid: the pixel decoder runs on the NPU, concurrently with the GPU DiT.
- **HRX Loom:**
  - Bit-exact BF16 GEMMs on the NPU, with no stamp bug.
  - No op library and no model import. GEMM throughput is about 0.29 TFLOP/s, so it is not viable
    for FastFrames today.
  - Details in [loom/NOTES.md](loom/NOTES.md).

Host: Arch/Omarchy, kernel 7.2.8, amdxdna in-tree driver, XRT 2.27 (/opt/xilinx/xrt), Ryzen AI 1.9.0
runtime extracted privately (not dpkg-installed), onnxruntime-vitisai 1.30.0, Python 3.12.16.
All host work is in `~/fastframes-npu/` (Ryzen AI route in `rai/`, Loom route in `loom/`).

## The stamp bug at FastFrames shapes, and a second workaround

Ryzen AI 1.9 BF16 Gemms can silently zero half their output columns
([repro](https://github.com/jethac/ryzen-ai-bf16-stamp-repro)). That repro was re-run in this
environment: the baseline still fails and `tp_size=1` still fixes it (`results/repro-001`).
It hits FastFrames directly. With the defaults (`tp_size` 2, `optimize_level` 1):

| Graph | Result with the defaults |
|---|---|
| Gemm 1560x1536x6144 | columns 3072–6143 are zero |
| Pixel decoder | token columns 1536–3071 are zero; cosine 0.024; uint8 frames off by ~54 on average |
| DiT block | V columns 768–1535 are zero; cosine of x is 0.48 |

**New finding: `optimize_level=2` with the default `tp_size` is also correct.**
- It holds for the 128x768 identity-style repro shape, the 1560x1536x6144 GEMM and every component.
- It keeps both compute stamps, so it is the fastest correct configuration.
- The compiler's plan is unchanged: 2 stamps split on N, and only `mini_stamp_0` appears in
  `multi_layer_overlay.h`. So the opt-2 code path avoids the bug by some route not visible in those
  files. Treat it as an observation, not a proof, and keep checking outputs.
- `optimize_level=3` is unusable: the compiler loops forever printing
  `[AIE ERROR] _XAie2p_GetTTypefromLoc(): Invalid column: 5` (a 28 GB log in 10 minutes; killed).

Every compile here uses `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`: BF16 matmuls are emulated
with block FP16. That is why NPU error is about 3x the error of a real BF16 run (table below).

## Results (Ryzen AI route, best correct config: optimize_level 2, default tp_size)

Inputs are random, with the same shapes and seeds as the references. Reference is the upstream
PyTorch module in FP32 on CPU. "BF16 floor" is the same upstream code run in BF16 on CPU, which is the
release precision. NPU latency is the minimum wall time of `session.run` with FP32 I/O.

| Component (one latent frame) | NPU ms | rel_l2 NPU / BF16 floor | cosine | CPU ORT fp32 ms | GPU bf16 eager ms |
|---|---|---|---|---|---|
| Pixel decoder (3 ConvNeXt blocks + head) | 76 frame 0 (zero history), ~125 steady state | 0.029 / 0.010 | 0.9996 | 164 | 16.9 |
| KV refiner stem (4 blocks, 1560 tokens) | 112.5 | 0.107 / 0.045 | 0.994 | 137 | 11.4 |
| KV refiner head, one layer (30 per frame) | 20.8 | K 0.034 / 0.008, V 0.019 / 0.005 | 0.9994 | 33 | 1.46 (all 30, batched: 45.6) |
| DiT block, frame 0 (KV 1560) | 85.9 | x 0.016 / 0.006 | 0.9999 | 157 | 8.7 |
| DiT block, frame 20 (KV 32760) | 531 | x 0.021 / 0.0065 | 0.9998 | 328 | 52.1 |
| Gemm 1560x1536x6144 | 8.2 (3.6 TFLOP/s) | 1e-4 | 1.0 | – | 0.95 (30.8 TFLOP/s) |

Pixel decoder as uint8 frames: the NPU differs from FP32 by a mean of 0.9 and a maximum of 11
levels. BF16 differs by a mean of 0.32 and a maximum of 4.

`tp_size=1` (opt 2) is also correct but slower: pixel decoder 176 ms, stem 170 ms, head 24.8 ms,
DiT f0 125 ms, DiT f20 836 ms. At `optimize_level=1` everything is a further 2–3x slower.

**Evidence that it ran on the NPU** (every `results/**/report_*.json` has an `npu` section):
- `final-vaiml-pass-summary.txt` for every graph reports "operators offloaded by VAIML: 100%",
  "partitions offloaded to NPU: 1" and "executed on CPU: 0", with Target Device stx, overlay
  rai_2x4x4 and data type bfloat16.
- ORT profiling shows kernel events only for `VitisAIExecutionProvider`. Each report records the
  loaded libxrt and libvaiml2 libraries.
- Host CPU time per call is small (1–35 ms) against NPU wall time of 20–800 ms, so the time is spent
  on the device.
- The node counts in `summary.json` (e.g. 61/99) count ONNX `Constant` nodes in the total. VAIML
  absorbs those, and its summary reports 100% of the compute operators.

**Whole-model projection** (NPU, best config, per latent frame, 21 frames per clip):
- DiT: 2 evals x 30 blocks = 5–32 s.
- KV refiner: 112 + 30 x 21 ≈ 0.74 s.
- Pixel decoder: about 0.13 s.

Running the whole model on the NPU is therefore not worth doing. Static shapes would also need one
compiled graph per (layer, KV length): 30 x 21 graphs at about 35–55 s each to compile.

**Hybrid, measured** (`rai/concurrency_result.txt`): the NPU pixel decoder loop ran concurrently with
a GPU DiT block loop (frame 20, eager).
- The GPU block went from 51.6 to 52.1 ms median, so it was effectively unaffected.
- The NPU decoder went from ~129 to ~199 ms median.
- So the decoder can move off the GPU for free in wall-clock terms. It runs concurrently with the
  next frame's DiT, which takes far longer than 200 ms per frame.
- The gain is the ~17 ms per latent frame of GPU time the decoder would otherwise use: a few percent
  of FastFrames-S frame time. The KV refiner should stay on the GPU (0.74 s on the NPU vs 57 ms on the
  GPU per frame, and it sits on the critical path).

## Ryzen AI vs Loom

| | Ryzen AI 1.9 VAIML | HRX Loom 0.2.0 |
|---|---|---|
| Model import | ONNX, unchanged graph | none: hand-written Loom IR or C++ |
| Ops available | all FastFrames ops (Conv, LN, GELU, softmax attention, MatMul) | none; one hand-written GEMM |
| Stamp bug | present by default; avoided with tp_size=1 or optimize_level=2 | not present (identity probe exact) |
| BF16 numerics | BFP16-emulated matmul, ~3x the BF16 error | fp32 accumulate, near-exact |
| Gemm 1560x1536x6144 | 8.2 ms | 100 ms (16 cores, 0.29 TFLOP/s) |
| Effort for a component | done here | estimated 1–2 engineer-weeks for a slow decoder |

## Files
- `rai/npu_common.py`: VitisAI session harness, placement and profile evidence, numerics (including
  the zero-column check).
- `rai/pixdec_npu.py`, `rai/kvref_npu.py`, `rai/dit_block_npu.py`: per-component export, run and
  compare. They re-express the upstream code with 2D/Gemm ops; the graph-vs-upstream FP32 check is
  in each report.
- `rai/gemm_probe.py`: stamp-bug and throughput probe at any M/K/N.
- `rai/pixdec_bf16_floor.py`: BF16 floor for the decoder.
- `rai/gpu_ref.py`, `rai/gpu_loop.py`, `rai/npu_loop.py`: GPU reference and the concurrency test
  (ROCm torch container).
- `rai/summarize.py`: collects all reports into `results/summary.json`.
- `rai/results/`: every report JSON, VAIML pass summaries, the repro pair results and `gpu_ref.txt`.
  No AMD binaries, no ONNX models or weights.
- Run on the host with `~/fastframes-npu/rai/run.sh SCRIPT ARGS`. It wraps `run-sdk.sh` / `npu-run.sh`
  from the bf16-stamp-repro repo, with SDK=`rai/runtime/opt/amd/ryzenai/1.9.0` and venv=`rai/venv`.
  Queue scripts are in `results/`.
- `loom/`: the Loom route (see its NOTES.md).
