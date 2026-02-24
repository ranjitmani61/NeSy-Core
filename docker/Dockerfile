# ── NeSy-Core Production Image ────────────────────────────────────────────
# Multi-stage build: build wheel → slim runtime.
# Usage:
#   docker build -f docker/Dockerfile -t nesy-core .
#   docker run -p 8000:8000 nesy-core
# ──────────────────────────────────────────────────────────────────────────

# Stage 1 – builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml setup.cfg README.md ./
COPY nesy/ nesy/
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

# Stage 2 – runtime
FROM python:3.11-slim AS runtime
LABEL maintainer="NeSy-Core Contributors"

# Non-root user
RUN groupadd -r nesy && useradd -r -g nesy nesy

WORKDIR /app

# Install the wheel + server extras
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" \
    && rm -rf /tmp/*.whl

# Copy server entry-point
COPY nesy/deployment/server/ /app/server/
COPY scripts/serve.py /app/serve.py

# Switch to non-root
USER nesy

EXPOSE 8000
ENV NESY_LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "nesy.deployment.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
