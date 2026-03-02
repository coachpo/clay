# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Prime dependency layer first for cache efficiency.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Install project into the managed venv without dev dependencies.
COPY src ./src
COPY start_proxy.py ./start_proxy.py
RUN uv sync --locked --no-dev

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/start_proxy.py /app/start_proxy.py
COPY pyproject.toml /app/pyproject.toml

USER app

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8082/health', timeout=4)" || exit 1

CMD ["clay"]
