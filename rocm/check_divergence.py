#!/usr/bin/env python3
"""Rollout sensitivity: how far do two *reference* rollouts drift when only the attention kernel
changes (AOTriton SDPA vs FastFrames' Triton kernel)? Gives the yardstick for engine-vs-reference
rollout differences (autoregressive sampling amplifies per-call BF16 noise)."""
import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes import accel  # noqa: E402
from fastframes.pipeline import MODELS, FastFrames, configure_backends, to_uint8  # noqa: E402

name, frames = (sys.argv[1] if len(sys.argv) > 1 else "ff-s"), int(sys.argv[2]) if len(sys.argv) > 2 else 6
spec = MODELS[name]
cfg = dataclasses.replace(spec.config, num_frames=frames, num_pixel_frames=4 * frames - 3)
configure_backends(cfg)
root = Path("checkpoints")
m = FastFrames.from_pretrained(cfg, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder,
                               text_encoder_device="cpu" if name == "ff-l" else None)
emb = m.encode_prompt("A corgi surfing a wave at sunset")
res = {}
for backend in ("triton", "sdpa"):
    accel.set_attention_backend(backend)
    lat, feat, mod = m.rollout(emb, 0)
    res[backend] = (lat, to_uint8(m.decode(feat, mod)[0]))
a, b = res["triton"], res["sdpa"]
cos = torch.nn.functional.cosine_similarity(a[0].float().flatten(), b[0].float().flatten(), dim=0).item()
per_frame = [round(torch.nn.functional.cosine_similarity(a[0][0, f].float().flatten(), b[0][0, f].float().flatten(),
                                                          dim=0).item(), 4) for f in range(frames)]
mse = ((a[1].float() - b[1].float()) ** 2).mean().item()
print(json.dumps(dict(model=name, frames=frames, latent_cos=cos, per_frame_cos=per_frame,
                      pixel_psnr_db=10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item())))
