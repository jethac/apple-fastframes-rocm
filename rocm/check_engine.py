#!/usr/bin/env python3
"""Compare the compiled ROCm engine against the upstream reference rollout on real weights.

  python rocm/check_engine.py --model ff-s --frames 6
  python rocm/check_engine.py --model ff-l --frames 4 --text-encoder-device cpu

Runs the reference (``engine=none``) and the engine rollout for the same prompt and seed,
then reports latent / feature agreement and decoded-pixel PSNR, plus per-rollout timing.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes import accel  # noqa: E402
from fastframes.pipeline import MODELS, FastFrames, configure_backends, to_uint8  # noqa: E402


def stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return dict(max_abs=(a - b).abs().max().item(), rel_l2=((a - b).norm() / b.norm()).item(),
                cos=torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="ff-s")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--frames", type=int, default=21, help="latent frames to roll out (21 = full clip)")
    p.add_argument("--prompt", default="A corgi surfing a wave at sunset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", default="max-autotune-no-cudagraphs")
    p.add_argument("--text-encoder-device", default=None)
    p.add_argument("--offload", default="auto", help="blocks streamed from host memory (int or 'auto')")
    p.add_argument("--json", type=Path)
    args = p.parse_args()

    spec = MODELS[args.model]
    config = dataclasses.replace(spec.config, num_frames=args.frames, num_pixel_frames=4 * args.frames - 3)
    configure_backends(config)
    root = args.checkpoint_dir
    model = FastFrames.from_pretrained(config, root / spec.wan_dir, root / spec.dit, root / spec.refiner,
                                       root / spec.decoder, text_encoder_device=args.text_encoder_device)
    emb = model.encode_prompt(args.prompt)

    def run(label):
        torch.cuda.synchronize()
        t = time.perf_counter()
        lat, feat, mod = model.rollout(emb, args.seed)
        torch.cuda.synchronize()
        t_roll = time.perf_counter() - t
        t = time.perf_counter()
        pix = to_uint8(model.decode(feat, mod)[0])
        torch.cuda.synchronize()
        print(f"{label}: rollout {t_roll:.2f}s decode {time.perf_counter() - t:.2f}s", flush=True)
        return lat, feat, pix, t_roll

    ref = run("reference")
    model.enable_engine(compile=None if args.compile == "none" else args.compile, offload=args.offload)
    print(f"engine: offloaded blocks {model.engine.offloaded_blocks}", flush=True)
    run("engine (warm-up / compile)")
    eng = run("engine")
    per_frame = [round(torch.nn.functional.cosine_similarity(eng[0][0, f].float().flatten(),
                                                              ref[0][0, f].float().flatten(), dim=0).item(), 4)
                 for f in range(args.frames)]
    out = dict(model=args.model, per_frame_latent_cos=per_frame, frames=args.frames, attention=accel.attention_backend(), compile=args.compile,
               latents=stats(eng[0], ref[0]), features=stats(eng[1], ref[1]), pixel_psnr_db=psnr(eng[2], ref[2]),
               reference_rollout_s=ref[3], engine_rollout_s=eng[3],
               offloaded_blocks=model.engine.offloaded_blocks,
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
