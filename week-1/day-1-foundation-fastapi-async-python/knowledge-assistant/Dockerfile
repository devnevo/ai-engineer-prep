# ─── Build stage ──────────────────────────────────────────────────────────────
# Multi-stage build: this stage installs dependencies, the next stage copies
# only what's needed → final image is 10x smaller

FROM python:3.11-slim AS builder

WORKDIR /build

# Copy ONLY dependency files first (layer caching optimization)
# If deps don't change, Docker reuses this layer even when app code changes
COPY pyproject.toml .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# ─── Runtime stage ────────────────────────────────────────────────────────────
# Slim image: no build tools, no pip cache, just Python + your deps

FROM python:3.11-slim AS runtime

# Non-root user for security
# Running as root in a container = if attacker gets in, they own everything
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/

# Create upload directory
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check — K8s can also use this, but Docker Compose relies on it
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
