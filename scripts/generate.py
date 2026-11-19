#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Text-to-video generation with FastFrames-S / FastFrames-L.

  python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
      --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4

  # one prompt per line, sharded over GPUs (run one process per GPU)
  CUDA_VISIBLE_DEVICES=3 python scripts/generate.py --model ff-l --checkpoint-dir checkpoints \
      --prompt-file prompts.txt --output-dir outputs/ff-l --shard 3 --num-shards 8
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes.pipeline import MODELS, FastFrames, configure_backends  # noqa: E402
from fastframes.video import save_video  # noqa: E402


def load_model(name, checkpoint_dir, engine="max-autotune-no-cudagraphs", text_encoder_device=None):
    """Build a released model from ``<checkpoint_dir>`` (see docs/installation.md#checkpoints).

    ROCm port: ``engine`` selects the compiled per-frame engine (``fastframes.engine``; ``"none"``
    keeps the upstream eager path) and ``text_encoder_device`` can place umT5-XXL on the CPU.
    """
    spec = MODELS[name]
    root = Path(checkpoint_dir)
    configure_backends(spec.config)
    if text_encoder_device is None and name == "ff-l":
        text_encoder_device = "cpu"  # 11 GB umT5-XXL on the host keeps the 14B DiT + KV cache on the GPU
    model = FastFrames.from_pretrained(
        spec.config, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder,
        text_encoder_device=text_encoder_device)
    if engine != "none":
        model.enable_engine(compile=None if engine == "eager" else engine)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="seed of the first prompt (prompt i uses seed + i)")
    parser.add_argument("--output", type=Path, help="output path for --prompt")
    parser.add_argument("--output-dir", type=Path, help="output directory for --prompt-file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--engine", default="max-autotune-no-cudagraphs",
                        choices=("max-autotune-no-cudagraphs", "default", "eager", "none"),
                        help="ROCm port: compiled per-frame engine mode; 'eager' = engine without "
                             "torch.compile; 'none' = upstream reference path")
    parser.add_argument("--text-encoder-device", default=None,
                        help="device of umT5-XXL (default: cpu for ff-l, the GPU for ff-s)")
    args = parser.parse_args()

    if args.prompt is not None:
        jobs = [(args.prompt, args.seed, args.output or Path(f"{args.model}_seed{args.seed}.mp4"))]
    else:
        prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
        out = args.output_dir or Path("outputs") / args.model
        jobs = [(p, args.seed + i, out / f"{i:04d}.mp4") for i, p in enumerate(prompts)][args.shard::args.num_shards]

    model = load_model(args.model, args.checkpoint_dir, engine=args.engine,
                       text_encoder_device=args.text_encoder_device)
    for prompt, seed, path in jobs:
        start = time.perf_counter()
        frames = model.generate(prompt, seed)
        save_video(path, frames)
        print(f"{path}  seed={seed}  {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
