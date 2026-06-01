FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Cloud Run L4 GPU nodes ship driver R535 (CUDA 12.2). torch 2.5.1 + CUDA 12.1
# is the latest pairing that runs there. The base image preinstalls
# torch 2.5.1+cu121; `.[web_api-gpu]` resolves with `torch>=2.5`, which pip
# treats as already satisfied and leaves untouched. PIP_EXTRA_INDEX_URL keeps
# any transitively-pulled torch/torchvision resolve on the matching cu121
# wheel index so it cannot pull a CUDA 13 wheel and re-break driver compat.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York \
    PYTHONPATH=/opt/http \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /opt/http

# ---- System packages -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy metadata ---------------------------------------------------
COPY pyproject.toml /opt/http/

# cutie declares cchardet>=2.1.7, which does not build on the base image's
# Python. Install an empty stub to satisfy the requirement (so the resolution
# install below does not try to build the real one) plus faust-cchardet to
# provide the actual top-level `cchardet` module.
RUN mkdir -p /tmp/cc_stub \
    && printf 'from setuptools import setup\nsetup(name="cchardet", version="2.1.7", py_modules=[])\n' > /tmp/cc_stub/setup.py \
    && pip install --no-deps /tmp/cc_stub \
    && rm -rf /tmp/cc_stub
RUN --mount=type=cache,target=/root/.cache/pip pip install faust-cchardet

# ---- Install project + web_api-gpu extra -----------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[web_api-gpu]'

# ---- Install cue CLI (needed for builder.build with spec JSON files) --
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://github.com/cue-lang/cue/releases/download/v0.11.1/cue_v0.11.1_linux_amd64.tar.gz \
       | tar xz -C /usr/local/bin cue \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---- Copy full source ------------------------------------------------
COPY . /opt/http

# ---- Run API ---------------------------------------------------------
WORKDIR /opt/http/web_api
CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:80"]
