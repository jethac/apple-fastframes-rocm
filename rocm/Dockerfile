# FastFrames on ROCm: AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151) and other ROCm GPUs.
#
#   docker build -f rocm/Dockerfile -t fastframes-rocm .
#   docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \
#       --security-opt seccomp=unconfined --ipc=host \
#       -v $PWD/checkpoints:/workspace/checkpoints -v $PWD/outputs:/workspace/outputs \
#       -v fastframes-cache:/cache fastframes-rocm \
#       python scripts/generate.py --model ff-s --prompt "A corgi surfing a wave at sunset" --output outputs/corgi.mp4
# Base: plain Ubuntu + AMD's per-architecture PyTorch wheels (ROCm runtime ships inside the wheels).
# GFX selects the wheel index: gfx1151 = Strix Halo; gfx1150 = Strix Point; gfx110X-all = RDNA3 dGPUs.
FROM ubuntu:24.04
ARG GFX=gfx1151
ARG TORCH_INDEX=https://repo.amd.com/rocm/whl/${GFX}/

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # Persistent compiler / autotuning caches (mount a volume at /cache to keep them across runs).
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    XDG_CACHE_HOME=/cache/xdg \
    HF_HOME=/cache/hf \
    MIOPEN_USER_DB_PATH=/cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
    # Strix Halo tuning (see rocm/README.md for the measurements behind each setting).
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    TORCH_BLAS_PREFER_HIPBLASLT=1 \
    PYTORCH_TUNABLEOP_ENABLED=1 \
    PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop_gfx1151.csv \
    HIP_FORCE_DEV_KERNARG=1 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev \
        build-essential ca-certificates curl git libnuma1 && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /workspace
RUN pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX} "torch==2.11.0+rocm7.13.0" "torchvision==0.26.0+rocm7.13.0"
COPY rocm/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && python -c "import torch, triton; assert torch.version.hip, torch.__version__; print(torch.__version__, triton.__version__)"

COPY pyproject.toml README.md ./
COPY fastframes ./fastframes
COPY scripts ./scripts
COPY evaluation/latency ./evaluation/latency
COPY rocm ./rocm
COPY data ./data
RUN pip install --no-deps -e . \
    # Bake Cisco's OpenH264 2.7 binary (downloaded on first use otherwise) out of the volume path.
    && XDG_CACHE_HOME=/opt/openh264-cache python -m fastframes.openh264 \
    && ln -s /opt/openh264-cache/fastframes/openh264/libopenh264-2.7.0-linux64.8.so /usr/local/lib/libopenh264.so
ENV OPENH264_LIBRARY=/usr/local/lib/libopenh264.so

CMD ["bash"]
