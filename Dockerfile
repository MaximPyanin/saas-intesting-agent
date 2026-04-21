# syntax=docker/dockerfile:1.7

# ============================================================================
# ORACLE — multi-stage Dockerfile (Step 18)
# ============================================================================
# Stage 1: builder — uses uv to resolve + install deps into /app/.venv
# Stage 2: runtime — slim python:3.13-slim, copies .venv + src/, runs as
#                    non-root user, persists state to /app/data volume
#
# Build:   docker build -t oracle:latest .
# Run:     docker run --rm --env-file .env -v oracle-data:/app/data oracle:latest
# Compose: docker compose up -d
#
# Image target: ~350-450 MB (python:3.13-slim ~120 MB + dependencies)
# Entry point: python -m oracle.bot.main (starts bot + scheduler + alerts)
# ============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.13-slim AS builder

# Grab uv binary from the official image — zero extra layers to maintain
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Build tools for any C extensions (pandas, numpy, aiosqlite, etc. usually
# ship wheels but keep these as safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# UV_COMPILE_BYTECODE=1 precompiles .pyc for faster cold start in runtime
# UV_LINK_MODE=copy avoids hardlink errors across volume mounts
# UV_PYTHON_DOWNLOADS=never forces use of the system Python (already 3.13)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.13

# Copy lock-affecting files first so this layer can be cached across builds
# that don't change dependencies
COPY pyproject.toml uv.lock ./

# Install deps without the project itself — maximizes layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the source tree and install the oracle package itself
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:3.13-slim AS runtime

# Runtime-only packages. curl for healthcheck, tzdata for pytz/zoneinfo
# (Europe/Warsaw for the scheduler), ca-certificates for HTTPS feeds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r oracle -g 1000 \
    && useradd -r -u 1000 -g oracle -m -d /home/oracle oracle

# Copy the fully prepared venv and source from builder
# Both directories owned by the oracle user (non-root)
COPY --from=builder --chown=oracle:oracle /app/.venv /app/.venv
COPY --from=builder --chown=oracle:oracle /app/src   /app/src

# Persistent data dir for SQLite DBs + Telethon session file
# Mount a volume here: -v oracle-data:/app/data
RUN mkdir -p /app/data && chown -R oracle:oracle /app /app/data

# Make the venv's Python the default + ensure UTF-8 + Europe/Warsaw timezone
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Europe/Warsaw \
    ORACLE_CONTAINER=1

# Run as non-root, working out of the data volume so oracle.db / oracle_data.db
# / oracle_reader.session all land inside the mounted volume (persistent
# across container restarts).
USER oracle
WORKDIR /app/data

# Healthcheck: process is alive AND domain DB is readable.
# --start-period=60s gives uv.lock + init_db + scheduler setup time on boot.
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('oracle_data.db').execute('SELECT 1').fetchone()" || exit 1

# Entry point: start the Telegram bot (which hosts the scheduler + alerts
# poll as in-process cron jobs via APScheduler).
# The --dry-run flag lets CI smoke-test without a real Telegram token:
#   docker run --rm oracle:latest --dry-run
ENTRYPOINT ["python", "-m", "oracle.bot.main"]
CMD []
