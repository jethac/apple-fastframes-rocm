#!/usr/bin/env python3
"""Single DiT call agreement: reference ``forward_frame`` vs the engine (eager and compiled).

Isolates per-call numerics from rollout divergence. Runs frame 0 (empty cache) and frame
``--frame`` against a cache filled by the reference path, for the same latent / timestep.
Also reports an FP32 reference so BF16 noise can be told apart from bugs.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes import accel  # noqa: E402
from fastframes.engine import FastEngine  # noqa: E402
from fastframes.pipeline import MODELS, FastFrames, configure_backends  # noqa: E402
from fastframes.wan.causal_model import allocate_kv_cache  # noqa: E402


def cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return dict(rel_l2=round(((a - b).norm() / b.norm()).item(), 5),
                cos=round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 6))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="ff-s")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--frame", type=int, default=3)
    p.add_argument("--text-encoder-device", default=None)
    args = p.parse_args()
    spec = MODELS[args.model]
    cfg = spec.config
    configure_backends(cfg)
    root = args.checkpoint_dir
    m = FastFrames.from_pretrained(cfg, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder,
                                   text_encoder_device=args.text_encoder_device)
    emb = m.encode_prompt("A corgi surfing a wave at sunset")
    g = torch.Generator(device=m.device).manual_seed(0)
    lat = torch.randn(1, args.frame + 1, 16, *cfg.latent_size, generator=g, device=m.device, dtype=torch.bfloat16)
    text = m.dit.text_cache(emb)
    out = {}
    engines = {"engine_eager": FastEngine(m, compile=None), "engine_compiled": FastEngine(m)}
    for e in engines.values():
        e.set_text(text)

    def run(label, fn, frame, cache):
        with m.autocast():
            t, e0, eh = m._time_conditioning(833.0)
            return fn(lat[:, frame:frame + 1], e0, eh, cache, frame)

    # Reference cache for frames < frame (clean pass at t=0 like the rollout's first frame).
    caches = {k: allocate_kv_cache(m.dit, cfg.num_frames, m.frame_tokens, device=m.device)
              for k in ["reference", *engines]}
    with m.autocast():
        _, e0c, ehc = m._time_conditioning(0.0)
        for f in range(args.frame):
            m.dit.forward_frame(lat[:, f:f + 1], e0c, ehc, text, caches["reference"], f, fast=m.fast)
    for k in engines:
        for a, b in zip(caches[k], caches["reference"]):
            a["k"].copy_(b["k"])
            a["v"].copy_(b["v"])
    for frame in (0, args.frame):
        ref = run("reference", lambda *a: m.dit.forward_frame(*a[:3], text, *a[3:], fast=m.fast), frame,
                  caches["reference"])
        for k, e in engines.items():
            got = run(k, e.forward_frame, frame, caches[k])
            out[f"frame{frame}/{k}"] = dict(flow=cmp(got[0], ref[0]), features=cmp(got[1], ref[1]))
        # FP32-attention reference to show the BF16 noise floor of the reference itself.
        accel.set_attention_backend("math")
        ref32 = run("reference_math", lambda *a: m.dit.forward_frame(*a[:3], text, *a[3:], fast=m.fast), frame,
                    caches["reference"])
        accel.set_attention_backend(None)
        out[f"frame{frame}/reference_math_attention"] = dict(flow=cmp(ref32[0], ref[0]))
    print(json.dumps(dict(model=args.model, fast_ops=cfg.fast_ops, autocast=cfg.autocast, **out), indent=1))


if __name__ == "__main__":
    main()
