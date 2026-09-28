# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/molai-venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/opt/molai-venv/bin:${PATH}"

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libglib2.0-0 \
        libgl1 \
        libxext6 \
        libxrender1 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependency layers remain cached when only source files change.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY LICENSE THIRD_PARTY_NOTICES.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["bash"]
