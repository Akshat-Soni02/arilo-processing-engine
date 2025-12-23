FROM python:3.14-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Sync deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd --create-home --shell /bin/bash app-user

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    PORT=8080

COPY --from=builder --chown=app-user:app-user /app/.venv /app/.venv
COPY --chown=app-user:app-user . .

RUN mkdir -p /app/logs && chown -R app-user:app-user /app/logs

USER app-user

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8080

CMD ["fastapi", "run", "src/main.py", "--port", "8080"]
