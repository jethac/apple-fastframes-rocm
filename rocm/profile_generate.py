#!/usr/bin/env python3
"""Stage-by-stage timing of one text-to-video generation (after a warm-up generation).

  python rocm/profile_generate.py --model ff-s
  python rocm/profile_generate.py --model ff-l --json outputs/profile_ff-l.json

Stages: text encoding, rollout (DiT + KV refinement for all 21 latent frames), pixel
decoding, uint8 conversion + host copy, H.264 encoding. Also reports per-latent-frame
rollout time and the number of transformer blocks streamed from host memory.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import load_model  # noqa: E402

from fastframes import accel  # noqa: E402
from fastframes.pipeline import to_uint8  # noqa: E402
from fastframes.video import save_video  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ff-s")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--engine", default="max-autotune-no-cudagraphs")
    p.add_argument("--prompt", default="A corgi surfing a wave at sunset")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    model = load_model(args.model, args.checkpoint_dir, engine=args.engine)
    frame_times = []
    if model.engine is not None:  # per-latent-frame timing through the engine
        inner = model.engine.forward_frame

        def wrapped(latent, e0, e_head, kv_cache, frame):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = inner(latent, e0, e_head, kv_cache, frame)
            torch.cuda.synchronize()
            frame_times.append((frame, time.perf_counter() - t))
            return out
        model.engine.forward_frame = wrapped
    for run in range(args.runs):
        frame_times.clear()
        emb, t_text = timed(lambda: model.encode_prompt(args.prompt))
        (lat, feat, mod), t_roll = timed(lambda: model.rollout(emb, 0))
        pix, t_dec = timed(lambda: model.decode(feat, mod))
        frames, t_u8 = timed(lambda: to_uint8(pix[0]).cpu())
        with tempfile.TemporaryDirectory() as d:
            _, t_enc = timed(lambda: save_video(Path(d) / "x.mp4", frames))
        total = t_text + t_roll + t_dec + t_u8 + t_enc
        print(f"run {run}: text {t_text:.2f}s rollout {t_roll:.2f}s decode {t_dec:.2f}s "
              f"to_uint8 {t_u8:.2f}s h264 {t_enc:.2f}s total {total:.2f}s", flush=True)
    dit_by_frame = {}
    for f, t in frame_times:
        dit_by_frame.setdefault(f, []).append(round(t * 1e3, 1))
    out = dict(model=args.model, engine=args.engine, attention=accel.attention_backend(), arch=accel.gpu_arch(),
               torch=torch.__version__, hip=torch.version.hip,
               offloaded_blocks=getattr(model.engine, "offloaded_blocks", None),
               text_encoder_device=str(model.text_encoder.device),
               seconds=dict(text=t_text, rollout=t_roll, decode=t_dec, to_uint8=t_u8, h264=t_enc, total=total),
               video_seconds=81 / 16, realtime_factor=total / (81 / 16),
               dit_ms_by_frame=dit_by_frame, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
