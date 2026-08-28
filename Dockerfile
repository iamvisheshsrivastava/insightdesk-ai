FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Build frontend
COPY frontend/ frontend/
WORKDIR /app/frontend
RUN npm install --no-audit --no-fund && npm run build

# Back to app directory
WORKDIR /app

# Copy production requirements and install Python dependencies.
# No torch/transformers - embeddings use fastembed (ONNX, ~100MB) to keep
# the image and runtime memory small enough for free-tier hosting.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ src/
COPY models/ models/

# Create necessary directories and hand them (plus the rest of /app) to the
# non-root user below - without this, appuser can't write feedback_data/,
# data/, plots/, or logs/ at runtime (mkdir raises PermissionError, caught
# silently, and endpoints like /feedback/summary return 503).
RUN mkdir -p data plots logs feedback_data && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Expose port (Heroku overrides this via the $PORT env var)
EXPOSE 8000

# Run as non-root for better container security
USER appuser

# Shell form so $PORT is expanded at runtime (Heroku injects its own port)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
