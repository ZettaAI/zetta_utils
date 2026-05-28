# syntax=docker/dockerfile:1.4
# GPU Dockerfile tuned for layer + BuildKit cache reuse on repeated builds.
# Semantically equivalent to ./gpu.Dockerfile; the difference is layer ordering
# and extra cache mounts so unrelated source edits don't invalidate the pip
# layers.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Cloud Run L4 GPU nodes ship driver R535 (CUDA 12.2). torch 2.5.1 + CUDA 12.1
# is the latest pairing that runs there. Force pip to resolve torch/torchvision
# from the matching cu121 wheel index so a transitive resolve cannot pull a
# CUDA 13 wheel from default PyPI and re-break the driver compatibility.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York \
    PYTHONPATH=/opt/http \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /opt/http

# Keep apt's downloaded .debs and package lists in BuildKit caches across
# builds (Debian's docker-clean hook deletes them by default).
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# ---- System packages -------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        unixodbc-dev \
        libboost-dev \
        libboost-all-dev \
        build-essential \
        g++ \
        python3-dev \
        curl \
        ca-certificates

# Copy only the dependency manifest first; pyproject.toml is held back so
# editing it does NOT invalidate the heavy pip layers below.
COPY web_api/requirements.txt /opt/http/

# ---- 1. NumPy 1.26.4 (for abiss/lsd) ---------------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install --force-reinstall "numpy==1.26.4"

# ---- 2. typing_extensions (for NotRequired) --------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install "typing_extensions>=4.6.0"

# ---- 3. Cython -------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install "cython>=3.0"

# ---- 4. lsd -----------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/ZettaAI/lsd.git@cebe976

# ---- 5. requirements.txt ---------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# ---- 6. nanobind (build-time, for witty via waterz on Python 3.12) ---
RUN --mount=type=cache,target=/root/.cache/pip pip install nanobind

# ---- cue CLI – curl already installed above, single small layer ------
RUN curl -fsSL https://github.com/cue-lang/cue/releases/download/v0.11.1/cue_v0.11.1_linux_amd64.tar.gz \
       | tar xz -C /usr/local/bin cue

# ---- 7. Project metadata + modules extras (last among pip steps) -----
COPY pyproject.toml /opt/http/
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[modules]'

# ---- 8. Copy full source ---------------------------------------------
COPY . /opt/http

# ---- Run API ---------------------------------------------------------
WORKDIR /opt/http/web_api
CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:80"]
