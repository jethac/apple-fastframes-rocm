# FastFrames on ROCm / AMD Strix Halo

This port runs Apple's [FastFrames-S](https://huggingface.co/apple/FastFrames-S) (1.3B) and
[FastFrames-L](https://huggingface.co/apple/FastFrames-L) (14B) text-to-video models on AMD GPUs
through ROCm. It is tuned for **Strix Halo** (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151,
128 GB unified memory).

## Results

All numbers are for one 81-frame, 832x480 clip (5.06 s at 16 fps), measured after warm-up with
`rocm/profile_generate.py` ([JSON](results/)). The setup was PyTorch 2.11 + ROCm 7.13 (AMD
gfx1151 wheels), with the GPU otherwise idle and memory configured as a shared pool (see
[Memory setup](#memory-setup-on-strix-halo)).

| Model | Text encode | Rollout (21 latent frames) | Pixel decode + H.264 | **Total** | vs. real time | Peak GPU memory |
|---|---|---|---|---|---|---|
| FastFrames-S | 0.35 s | 14.1 s | 1.1 s | **15.6 s** | 3.1x | 21 GiB |
| FastFrames-L | 0.58 s | 87.8 s | 1.2 s | **89.6 s** | 17.7x | 60 GiB |

Time per DiT evaluation grows with the KV cache:

| | Frame 0 | Frame 10 | Frame 20 |
|---|---|---|---|
| FF-S | 153 ms | 298 ms | 443 ms |
| FF-L | 1.15 s | 1.85 s | 2.55 s |

Earlier runs used the default 64 GB VRAM carve-out with another process holding 7 GB of it. In
that setup FF-L had to stream 15 of its 40 blocks from host memory and took 97.2 s; FF-S took
17.2 s ([JSON](results/profile_ff-l_final.json)).

### Memory setup on Strix Halo

By default the BIOS reserves a fixed VRAM carve-out (often 64 GB), and the GPU can only borrow
half of the remaining RAM on top of it. Shrinking the carve-out and raising the kernel's GTT
limit turns the 128 GB into one pool shared by the CPU and GPU:

1. In the BIOS (**Advanced → AMD CBS → NBIO Common Options → GFX Configuration** on AMI
   firmware), set **UMA Frame buffer Size** to its minimum (2 GB on the GMKtec EVO-X2).
2. Add `ttm.pages_limit=28835840` to the kernel command line. That is 110 GiB in 4 KiB pages;
   amdgpu and ROCm read it when the driver loads.

After a reboot `torch.cuda.mem_get_info()` reports 110 GiB free, and FF-L runs with its full
KV cache and no block streaming. `BlockStreamer` still handles smaller carve-outs automatically.

For reference, the paper's numbers on one B200 are 36 ms (FF-S) and 158 ms (FF-L) per latent
frame.

### Speedup over the upstream path on ROCm

Upstream's eager path, once its cuDNN-only attention calls are rerouted, takes **41 s** for the
FF-S rollout. Its first decode also costs 144 s while MIOpen tunes its conv kernels. Upstream
FF-L does not fit in a 64 GB carve-out at 21 frames.

## Running it

```bash
docker build -f rocm/Dockerfile -t fastframes-rocm .            # ubuntu:24.04 + AMD gfx1151 wheels
# checkpoints: see docs/installation.md#checkpoints (ff-l: add --include 'google/*' etc. per pattern)
FASTFRAMES_CHECKPOINTS=/path/to/checkpoints rocm/run.sh \
    python scripts/generate.py --model ff-s --prompt "A corgi surfing a wave at sunset" --output outputs/corgi.mp4
FASTFRAMES_CHECKPOINTS=/path/to/checkpoints rocm/run.sh \
    python scripts/generate.py --model ff-l --prompt-file data/prompts/demos.txt --output-dir outputs/ff-l
```

- **Other GPUs.** `--build-arg GFX=gfx1150` (Strix Point) or `GFX=gfx110X-all` (RDNA3 discrete)
  selects another of AMD's wheel indexes. Only gfx1151 has been tested.
- **Caches.** Compiler, autotuning, MIOpen and TunableOp caches live in the `fastframes-cache`
  volume. The first generation takes 2–3 minutes longer while they warm up.
- **`--engine eager`** gives bit-identical output to upstream, about 4x faster than upstream but
  slower than the compiled default. `--engine none` runs the untouched upstream path.

## What the port changes, and why

1. **Attention without cuDNN.**
   - Upstream forces PyTorch's cuDNN SDPA backend, which does not exist on ROCm. All attention now
     goes through `fastframes/accel.py` (`FASTFRAMES_ATTN=triton|sdpa|sdpa-flash|sdpa-mem|fa|math`).
     On NVIDIA it still defaults to cuDNN.
   - The ROCm default is `fastframes/kernels/attention.py`, a forward-only Triton flash-attention
     kernel autotuned for FastFrames' shapes: 1560 queries against up to 32760 keys, head dim 128,
     no mask. It reads the `[B, L, H, D]` KV cache in place and is registered as an opaque
     `torch.library` op so Inductor does not re-trace it.

   | Shape (bf16) | AOTriton via SDPA | FastFrames Triton kernel |
   |---|---|---|
   | 1.3B, full cache (32760 keys) | 13.0 TFLOP/s | 26.2–27.8 TFLOP/s |
   | 14B, full cache | 13.8 TFLOP/s | 24.7 TFLOP/s |
   | 14B, one frame | 15.9 TFLOP/s | 24.5 TFLOP/s |

   Maximum error against the FP32 math kernel is identical for both.
2. **AOTriton has to be enabled.** On gfx1151 PyTorch's flash and memory-efficient kernels stay
   disabled unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Without it, SDPA silently falls
   back to the math kernel at 1.5 TFLOP/s. The image sets this.
3. **Compiled per-frame engine** (`fastframes/engine.py`, on by default in `scripts/generate.py`).
   - Text K/V are projected once per prompt and RoPE tables are precomputed.
   - Each transformer block is split into two fixed-shape `torch.compile` regions, with the
     self-attention over the growing cache between them. One compile therefore serves all 21
     cache lengths and all layers.
   - The engine follows each recipe's ops. FF-S was released with `fast_ops=False`, so for FF-S
     it uses separate Q/K/V GEMMs, FP64 RoPE and the Conv3d patch embed. FF-L uses the fused fast
     ops.
   - `inductor.emulate_precision_casts` is on. Without it, fused FP32 intermediates move the
     2-step distilled models noticeably further from the reference (FF-S frame 0, latent cosine:
     0.978 without, 0.998 with).
4. **Fitting FF-L in 64 GB.**
   - Upstream's `FastOps` kept both the fused QKV weights and the originals, a 6.3 GB duplicate on
     14B. Q/K/V now alias the fused copy.
   - Safetensors load straight into GPU memory.
   - umT5-XXL lives in pinned host memory and runs on the GPU only while encoding, when the KV
     cache is not allocated.
   - When the weights plus a full KV cache still do not fit, `BlockStreamer` keeps the last *N*
     transformer blocks in pinned host memory and double-buffers them in on a side stream. On an
     APU the copy runs at about 73 GB/s.
   - With a 64 GB carve-out and 7 GB taken by other processes, *N* = 15 of 40 blocks. With the
     shared-pool setup above, *N* = 0. Streaming is bit-exact (the
     eager engine with streaming gives bit-identical latents) and costs about 11% of FF-L's
     frame-0 GPU time.
5. **Tuning and caches in the image.**
   - TunableOp is on (`PYTORCH_TUNABLEOP_ENABLED=1`). It gains 7–20% per GEMM in microbenchmarks
     (`rocm/bench_gemm.py`, peaking at 40 TFLOP/s), but little end to end, because Inductor's
     max-autotune already picks good kernels.
   - The MIOpen find-db is persisted.
   - `expandable_segments` is set in the allocator config.
6. **`rocm/run.sh`** forwards only environment variables that are actually set. A bare `-e VAR`
   would unset the image's defaults; that is how bug 2 first appeared.

### Where the time goes

`rocm/profile_kernels.py` gives a kernel-level view.
- **FF-L, frame 0, with 15 blocks streamed:** GEMMs are 72% (hipBLASLt at about 33 TFLOP/s,
  roughly 55% of the 8060S's BF16 peak), host-to-device streaming 12%, attention 10%, fused
  elementwise about 5%. Without streaming, frame 0 drops from 1.34 s to 1.15 s.
- **FF-S at late frames:** attention dominates (about 350 of 500 ms at frame 20).

Both models are throughput-bound on these kernels. The next big wins would need faster GEMM or
attention kernels for RDNA 3.5.

## Numerical agreement

Autoregressive sampling amplifies per-call BF16 noise, so the reference point here is how far two
equally valid **upstream** rollouts drift when only the attention kernel changes
(`rocm/check_divergence.py`).

| Model, frames | Reference vs. reference (AOTriton ↔ Triton) | Compiled engine vs. reference |
|---|---|---|
| FF-S, 21 | latent cos 0.969, pixel PSNR 21.0 dB | latent cos 0.959, 19.9 dB |
| FF-L, 6 | latent cos 0.982, 23.2 dB | latent cos 0.969, 21.0 dB |

The eager engine is bit-identical to upstream for FF-S (all 6 latent frames checked), including
with block streaming. For FF-L its latents are bit-identical (3 frames checked); decoded pixels
differ by 55 dB PSNR because MIOpen picks its conv algorithm differently on a first call.
`rocm/check_frame.py` checks single calls: frame 3 has 1.5% relative L2 against a 1.4% noise
floor. Sample videos: `data/prompts/demos.txt` prompts 0, 36, 72 and 108 render coherently with
both models.

## NPU (XDNA2) experiment: FastFrames-S

See [`npu/README.md`](../npu/README.md).
- **Ryzen AI 1.9 SDK.** Every FF-S component compiles and runs entirely on the NPU, and matches
  the reference once the
  [BF16 missing-stamp bug](https://github.com/jethac/ryzen-ai-bf16-stamp-repro) is avoided. With
  default settings the bug hits FastFrames directly: half the GEMM output columns come back as
  zeros. `tp_size=1` fixes it; so does `optimize_level=2`, which is also faster. The second fix is
  observed, not explained.
- **Speed.** The NPU is 5–15x slower than the iGPU, so the only useful split is the pixel decoder
  on the NPU concurrently with the GPU DiT, worth a few percent.
- **HRX Loom.** Exact BF16 GEMMs on the NPU with no stamp bug, but no op library or model import
  yet, and only 0.29 TFLOP/s for hand-written GEMMs.

## Scripts

| Script | Purpose |
|---|---|
| `bench_gemm.py`, `bench_attention.py` | Kernel microbenchmarks on FastFrames shapes |
| `check_engine.py` | Engine vs. upstream rollout: latent / feature / pixel agreement and timing |
| `check_frame.py` | Single DiT call agreement, with an FP32-attention noise floor |
| `check_divergence.py` | Reference-vs-reference rollout sensitivity (the yardstick above) |
| `profile_generate.py` | Stage and per-frame timing of a whole generation |
| `profile_kernels.py` | `torch.profiler` kernel table for one DiT evaluation |
