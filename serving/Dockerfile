# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies into an isolated prefix so we copy only what we need
COPY serving/requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (keep directory structure so imports work)
COPY serving/src/ ./serving/src/
COPY serving/__init__.py ./serving/__init__.py
COPY serving/models/ ./serving/models/

# Runtime configuration (overridable via docker run -e / docker-compose env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI=sqlite:///mlruns.db \
    MODEL_NAME=ecommerce-churn-predictor \
    CHAMPION_PCT=80

# Switch to non-root
USER appuser

EXPOSE 8000

# Healthcheck — calls the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
