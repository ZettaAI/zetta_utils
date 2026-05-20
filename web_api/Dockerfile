# syntax=docker/dockerfile:1.4
# CPU-only Dockerfile
# For GPU support, use gpu.Dockerfile instead
FROM python:3.12

ENV PYTHONPATH=/opt/http
WORKDIR /opt/http

COPY pyproject.toml requirements.modules.txt web_api/requirements.txt /opt/http/

RUN apt update && apt install -y git ffmpeg libsm6 libxext6 unixodbc-dev libboost-dev

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade 'pip>=25.2'

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=20 \
    PIP_RESUME_RETRIES=20

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip install -r requirements.modules.txt \
    && pip install --no-deps .

# cchardet==2.1.7 does not build on Py3.12+ (API removals). Cutie declares
# `cchardet >= 2.1.7`, so install an empty stub to satisfy the requirement
# and faust-cchardet to provide the actual top-level `cchardet` module.
RUN mkdir -p /tmp/cc_stub \
    && printf 'from setuptools import setup\nsetup(name="cchardet", version="2.1.7", py_modules=[])\n' > /tmp/cc_stub/setup.py \
    && pip install --no-deps /tmp/cc_stub \
    && rm -rf /tmp/cc_stub
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install faust-cchardet

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "cutie @ git+https://github.com/hkchengrex/Cutie.git"

# ---- Install cue CLI (needed for builder.build with spec JSON files) --
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://github.com/cue-lang/cue/releases/download/v0.11.1/cue_v0.11.1_linux_amd64.tar.gz \
       | tar xz -C /usr/local/bin cue \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . /opt/http
RUN zetta --help

WORKDIR /opt/http/web_api
CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:80"]
