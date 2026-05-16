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

# Install CPU-only PyTorch first to avoid pulling in 2GB+ of CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy production requirements and install remaining Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ src/
COPY models/ models/

# Create necessary directories
RUN mkdir -p data plots logs && useradd --create-home --shell /bin/bash appuser

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Expose port (Heroku overrides this via the $PORT env var)
EXPOSE 8000

# Run as non-root for better container security
USER appuser

# Shell form so $PORT is expanded at runtime (Heroku injects its own port)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
