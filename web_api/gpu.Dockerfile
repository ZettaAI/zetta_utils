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
    
# ---- System packages -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# ---- Copy metadata ---------------------------------------------------
COPY pyproject.toml web_api/requirements.txt /opt/http/

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

# ---- 6. Install project + modules ------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip pip install nanobind # Required for witty via waterz, only for built-time compatibility with Python 3.12. Can be removed later
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[modules]'

# ---- Install cue CLI (needed for builder.build with spec JSON files) --
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://github.com/cue-lang/cue/releases/download/v0.11.1/cue_v0.11.1_linux_amd64.tar.gz \
       | tar xz -C /usr/local/bin cue \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---- 7. Copy full source ---------------------------------------------
COPY . /opt/http

# ---- Run API ---------------------------------------------------------
WORKDIR /opt/http/web_api
CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:80"]