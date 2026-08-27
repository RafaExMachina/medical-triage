# -------------------------------------------------------------------
# Stage 1: Builder
# -------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Copy uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Copy dependency files first to improve Docker layer caching.
COPY pyproject.toml uv.lock ./

# Install only production dependencies.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project


# -------------------------------------------------------------------
# Stage 2: Runtime
# -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------------------------
# Locale required by ONNX Runtime StringNormalizer.
# The converted TF-IDF pipeline requires en_US.UTF-8.
# -------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Add the virtual environment binaries to PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Allow Python to locate the src-layout package.
ENV PYTHONPATH="/app/src"

# Copy only the virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy application source code.
COPY src ./src

# Copy production model artifacts.
# .dockerignore restricts which files from models/ enter the image.
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "medical_triage.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]