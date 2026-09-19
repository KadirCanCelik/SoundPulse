FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev; \
    else \
        uv venv && uv pip install -r pyproject.toml; \
    fi

COPY . .

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" appuser && \
    useradd -l -u "${UID}" -g "${GID}" -d /app appuser && \
    mkdir -p exports data/uploads data/output && \
    chown -R appuser:appuser /app && \
    chmod -R 750 /app/exports /app/data

USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]