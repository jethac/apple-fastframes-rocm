#!/usr/bin/env python3
"""Kernel-level breakdown of one engine DiT evaluation (torch.profiler).

  python rocm/profile_kernels.py --model ff-l --frame 10
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes.pipeline import MODELS, FastFrames, configure_backends  # noqa: E402
from fastframes.wan.causal_model import allocate_kv_cache  # noqa: E402


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ff-l")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--compile", default="max-autotune-no-cudagraphs")
    p.add_argument("--offload", default="auto")
    p.add_argument("--rows", type=int, default=25)
    args = p.parse_args()
    spec = MODELS[args.model]
    cfg = spec.config
    configure_backends(cfg)
    root = Path("checkpoints")
    m = FastFrames.from_pretrained(cfg, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder,
                                   text_encoder_device="cpu")
    m.enable_engine(compile=None if args.compile == "none" else args.compile, offload=args.offload)
    emb = m.encode_prompt("A corgi surfing a wave at sunset")
    kv = allocate_kv_cache(m.dit, cfg.num_frames, m.frame_tokens, device=m.device)
    lat = torch.randn(1, 1, 16, *cfg.latent_size, device=m.device, dtype=torch.bfloat16)
    with m.autocast():
        m.engine.set_text(m.engine.text_cache(emb))
        _, e0, eh = m._time_conditioning(833.0)
        for _ in range(3):
            m.engine.forward_frame(lat, e0, eh, kv, args.frame)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            m.engine.forward_frame(lat, e0, eh, kv, args.frame)
            torch.cuda.synchronize()
    print(f"offloaded blocks: {m.engine.offloaded_blocks}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=70))


if __name__ == "__main__":
    main()
