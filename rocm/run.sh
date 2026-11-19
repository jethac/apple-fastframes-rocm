#!/usr/bin/env bash
# Run a command in the fastframes-rocm image with the GPU, checkpoints, outputs and caches mounted.
#
#   rocm/run.sh python scripts/generate.py --model ff-s --prompt "..." --output outputs/x.mp4
#
# FASTFRAMES_CHECKPOINTS  checkpoint directory (default: ./checkpoints)
# FASTFRAMES_IMAGE        image tag (default: fastframes-rocm)
# FASTFRAMES_DEV=1        mount this checkout over /workspace (edit code without rebuilding)
# DOCKER_FLAGS            extra docker run flags
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT=${FASTFRAMES_CHECKPOINTS:-$ROOT/checkpoints}
mkdir -p "$ROOT/outputs"
DEV=()
if [[ -n "${FASTFRAMES_DEV:-}" ]]; then DEV=(-v "$ROOT":/workspace); fi
# Forward only variables that are set: a bare `-e VAR` would unset the image's own defaults.
ENVS=()
for v in FASTFRAMES_ATTN FASTFRAMES_COMPILE PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_FILENAME \
         TORCH_BLAS_PREFER_HIPBLASLT TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL; do
  if [[ -n "${!v+x}" ]]; then ENVS+=(-e "$v"); fi
done
exec docker run --rm ${DOCKER_FLAGS:-} --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  "${DEV[@]}" -v "$CKPT":/workspace/checkpoints:ro -v "$ROOT/outputs":/workspace/outputs \
  -v fastframes-cache:/cache "${ENVS[@]}" \
  "${FASTFRAMES_IMAGE:-fastframes-rocm}" "$@"
