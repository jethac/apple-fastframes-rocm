# Loom (hrx-loom) on the Strix Halo XDNA2 NPU: evaluation notes

Date: 2026-11-19. Host: `nucbox-strixhalo` (Ryzen AI MAX+ 395, kernel 7.2.8-9-omarchy,
amdxdna in-tree driver, NPU firmware 1.1.2.65, XRT 2.27.0). Host work dir: `~/fastframes-npu/loom/`.

## 1. Install

- Official path: the PyPI wheel. `uv venv --python 3.12 ~/fastframes-npu/loom/.venv` then
  `uv pip install "hrx-loom[hf]" numpy`. This installed **hrx-loom 0.2.0** (manylinux_2_39 wheel;
  the host has glibc 2.44). No source build was needed.
- Tools in the venv: loom-run, loom-test, loom-benchmark, loom-compile, loom-import-cxx,
  loom-format, loom-serve, amdf-info, and others. `loom-compile --version` is not a recognised flag,
  so the version comes from `pip show hrx-loom`.

## 2. Device discovery: works

- `loom-run --list_drivers` lists `xdna: AMD XDNA execution using libamdf` (logs/list_drivers.log).
- `loom-run --list_devices=xdna` prints `xdna://id/0000010500000000ea94362d296f5eca`
  (logs/list_devices_xdna.log).
- `amdf-info --device=xdna --activate` reports "Activation: OK", target `amd.xdna.strix_halo.17f0_11 (AIE2P)`,
  and "Live array 8 columns from 0, 6 rows" (results/logs/evidence/amdf-info-*.txt).

## 3. Documented XDNA example (gated projection): works, results are bit-exact

- Ran `loom-import-cxx`, then `loom-format --check`, then `loom-compile ... --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 --format=xdna`.
  The compile succeeded and produced a 6416 B `projection.xdna` plus `projection.report.json`.
- `loom-run` can only invoke `kernel.def` roots. With the pipeline root it fails with
  `module_function.c:576: NOT_FOUND; module function symbol was not found`. With the `.xdna` it fails with
  `no linked input provider accepts`. Pipelines run through `loom-test --device=xdna` with a `check.case`.
  `gp_run.py` appends a case that reads npy inputs, launches the pipeline and writes an npy output.
- Results:
  - K=512: NPU = -0.054931640625, numpy reference (with the same bf16 rounding steps) = -0.054931640625. Exact.
  - K=4096: NPU = -0.32421875, reference = -0.32421875. Exact.
  - Logs: logs/gp_run_K512.log and logs/gp_run_K4096.log.
- Upstream XDNA hardware tests copied from the repo (hal_execution, packed_matrix_groups, hardware_loop,
  finite_ingress) also pass on `--device=xdna` (logs/cts_*.log).

## 4. BF16 GEMM on XDNA (hand-written, because no GEMM library exists for aie2p)

Nothing ready-made exists. I searched hrx-system (loom docs/examples, libamdf/cts/xdna,
loom/target/amd/xdna tests), hrx-loom-library and loom-skills:
- No GEMM, matmul or linear provider for aie2p.
- The only BF16 matrix instruction Loom exposes on aie2p is `vector.mma` with
  vector<8xbf16> x vector<8xbf16> -> vector<64xf32>. It lowers to the m8n8k1 outer product
  `mma.bf16bf16.m8n8k1`, plus a broadcast and two shuffles per call.
- A block-FP instruction also exists (`mma.bfp16ebs8.m8n8k8`). It is not bf16.

What I wrote:
- `gemm_gen.py` (`kernel.def`, one core). Correct at small sizes. A `kernel.def` stages *all*
  bindings into the 64 KiB of tile memory, so 128x768x768 is rejected:
  `XDNA/051 ... requires enough tile-local memory for staged kernel bindings`.
- `gemm_pipe_gen.py` (resident `pipeline.def`). Each worker core DMAs 8-wide-packed A/B k-chunks from DDR
  into channel records in tile memory, accumulates 8x8 f32 tiles with `vector.mma`, and DMAs the f32 C block
  back to DDR with a 3-D strided copy.
  - Use `--fused --wide`, where each core's own strand issues its DMAs.
  - The variant with separate ingress strands fails once there are more than 64 distinct record offsets.
    The DMA-only strand can't be realised and needs its own core:
    `XDNA/051 ... requires an independently assigned compute tile for each instruction-executing strand`.
  - The maximum is 2 worker rows per column, so 16 cores. With 3 or 4 rows the compile fails:
    `XDNA/051 ... requires available DMA queues for simultaneous construction lifetimes`.
  - Multi-column images need the `--device=xdna://0?columns=8` URI. With plain `--device=xdna` they fail:
    `validation.c:362: FAILED_PRECONDITION; XDNA image does not match the admitted native context`.
- Host packing: Ap[M/8][K][8] and Bp[N/8][K][8]. Output: f32 row-major C.

Numerics. The reference is numpy fp64 computed on the same bf16-rounded inputs.
"Zero blocks" counts all-zero column blocks of width 8, 64 or 384.

| shape (M x K x N) | mode | max abs err | cosine | zero cols / zero blocks |
|---|---|---|---|---|
| 128x768x768 | identity, A=(arange%16)+1 | **0.0 (exact)** | 1.0 | 0 / none |
| 128x768x768 | random U(-1,1) | 2.2e-5 (max \|ref\| 42) | 0.99999999999998 | 0 / none |
| 1560x1536x6144 | random | 6.8e-5 (max \|ref\| 70) | 0.99999999999996 | 0 / none |
| 1560x6144x1536 | random | 3.2e-4 (max \|ref\| 134) | 0.9999999999998 | 0 / none |
| 1560x1536x768 | random | 5.3e-5 (max \|ref\| 64) | 0.99999999999996 | 0 / none |

**No sign of the stamp bug.** Columns 384-767 of the identity probe are exact. Every sentinel was
overwritten, so every output element was written.

Timing comes from `loom-benchmark --measure=dispatch_complete`, which measures host submission through
queue completion. The figures are min / p50 across 2-3 repeated runs of 10-150 dispatches each.
FLOPs = 2*M*N*K.

| shape | workers | min | p50 | TFLOP/s (min) |
|---|---|---|---|---|
| 128x768x768 | 1 (non-wide) | 9.47 ms | 9.62 ms | 0.016 |
| 128x768x768 | 1 (wide) | 8.26 ms | 8.34 ms | 0.018 |
| 128x768x768 | 8 | 1.32 ms | 1.33 ms | 0.11 |
| 128x768x768 | 16 (wide) | 0.655 ms | 0.666 ms | 0.23 |
| 1560x1536x6144 | 8 (wide) | 200.7 ms | 201.5 ms | 0.147 |
| 1560x1536x6144 | 16 (wide) | 100.4 ms | 100.5-101.1 ms | **0.293** |
| 1560x6144x1536 | 16 (wide) | 103.0 ms | 103.2-103.5 ms | 0.286 |
| 1560x1536x768 | 16 (wide) | 13.15 ms | 13.2-13.6 ms | 0.280 |

- Scaling is linear in cores, and each core is compute-bound at about 18 GFLOP/s, roughly 7 ns per
  m8n8k1 `vector.mma`. That is an order of magnitude or more below what AIE2P BF16 hardware should reach.
- The bottleneck is Loom's current bf16 matrix lowering (outer product plus shuffles) and the lack of a
  tuned schedule. My kernel adds no software pipelining of DMA and compute.
- The non-wide 1560x1536x6144 run (121-144 ms) was slower and noisier, possibly from concurrent NPU use.
  During the evidence capture xrt-smi showed only our hardware context.

## 5. Evidence the work ran on the NPU

Files are in results/logs/evidence/:
- `proc_during_run.txt`: the loom-benchmark PID holds fd 4 open on `/dev/accel/accel0`.
- `xrt-smi_aie-partitions_during_*.txt`: run with `sudo xrt-smi examine -r aie-partitions`. As the normal
  user it fails with an mmap EAGAIN (memlock limit 8 MiB). The output shows partition columns [0..7], one
  hardware context whose PID matches the loom-benchmark PID, status Active, and submissions and
  completions climbing (10/9, 6/5, ...).
- CPU fallback is impossible. `loom-test ... --device=task` on the same module fails with
  `target family 'x86' cannot specialize roots authored for target family 'amd.xdna.aie2p'`.
- The compile report lists `backend=xdna`, `bundle=aie2p-core`, `realization=resident-configuration`,
  `workers=16`, and a 72.5 KB XDNA ELF.

## 6. Effort to run a whole FastFrames component (pixel decoder) through Loom on the NPU

- **No model import path.** There is no ONNX or PyTorch importer. Input is Loom IR or a C++ subset
  (`loom-import-cxx`). Models in hrx-loom-library are hand-authored Loom programs.
- **No op library for aie2p.**
  - hrx-loom-library's tensor motifs (linear, convolution, channel_norm, attention, the BF16 activations
    used by its DiT model Krea-2-Turbo) target AMD GPUs only: `amdgpu.target<gfx11-generic>` with subgroup ops.
  - The only NPU model, Qwen3-Embedding-0.6B, is a fixed 32-core array composition of about 7k hand-written
    lines (array.cxx, workers_*.loom, tile_math.cxx with inline AIE2P assembly). Its projections use
    BFP16EBS8 block-FP, not true BF16.
  - loom-skills' xdna2 knowledge base says "planned", with no device rows.
- **No optimized GEMM for aie2p.** The GEMM above is the best available and reaches about 0.29 TFLOP/s.
- What would have to be hand-authored for the pixel decoder:
  - LayerNorm: row reductions across tiles.
  - 1x1 convs as 1536<->6144 GEMMs: possible today, but slow.
  - GELU: the lookup-table and transcendental paths exist as vector ops; numerics need checking.
  - Depthwise 3x3 conv: halo or strided DMA layouts.
  - The final 1536->3072 linear.
  - Inter-op data movement: each op is a separate invocation that round-trips DDR, unless all ops are
    fused into one resident pipeline (the Qwen style).
- Constraints to design around:
  - 16 KiB instructions per tile.
  - 64 KiB of tile memory.
  - DMA queue limits, which allow 2 DMA-hungry workers per column.
  - At most 64 distinct DMA record offsets per DMA-only strand.
  - No tile-state retention between submissions.
- Estimate:
  - A correct but slow decoder: about 1-2 engineer-weeks.
  - Competitive performance needs a real aie2p BF16 GEMM, which Loom doesn't currently lower well
    (m8n8k1 only), or a switch to BFP16. That is weeks to months and depends on AMD compiler work.
- At 0.29 TFLOP/s the decoder's 1x1-conv GEMMs alone (2 x 29.4 GFLOP) take about 200 ms per pass on the NPU.
- Possibly more useful: Loom's tuned primary targets are RDNA 3.5 GPUs. The Radeon 8060S on this box is
  gfx1151, and hrx-loom-library already runs a BF16 DiT (Krea-2-Turbo) there. That was not tested here.

## Files

Local (`B:\fastframes-rocm\npu\loom\`):
- `gp_run.py`: gated-projection example runner and numeric check.
- `gemm_gen.py`: single-core `kernel.def` GEMM, for small shapes only.
- `gemm_pipe_gen.py`: multi-core resident-pipeline BF16 GEMM generator.
- `gemm_run.py`: packs inputs, generates the kernel, runs `loom-test --device=...`, checks numerics.
- `bench.sh`: repeated `loom-benchmark` dispatch_complete timing.
- `evidence.sh`: proc fd and xrt-smi capture during a run.
- `results/`: copies of the host result.json, bench_*.json and logs.

Host: `~/fastframes-npu/loom/{.venv,src,runs,logs,gp,ex}`.

Reproduce the main runs:
```
source ~/fastframes-npu/loom/.venv/bin/activate; cd ~/fastframes-npu/loom/runs
python ../src/gemm_run.py 128 768 768 probe --mode identity --fused --wide --cols 8 --rows 2 --device "xdna://0?columns=8"
python ../src/gemm_run.py 1560 1536 6144 ff --fused --wide --cols 8 --rows 2 --mb 40 --nb 128 --kb 32 --device "xdna://0?columns=8"
../src/bench.sh ff 3 "xdna://0?columns=8"
```
