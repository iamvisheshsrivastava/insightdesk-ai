# Makefile for InsightDesk AI Development

.PHONY: help install lint format test test-unit test-integration test-performance clean build run docker-build docker-run docker-test security docs

# Default target
help:
	@echo "InsightDesk AI Development Commands"
	@echo "=================================="
	@echo "Setup:"
	@echo "  install          Install development dependencies"
	@echo "  install-prod     Install production dependencies only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security scans"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-test      Test Docker container"
	@echo ""
	@echo "Development:"
	@echo "  run              Start development server"
	@echo "  build            Full build (lint, test, docker)"
	@echo "  clean            Clean temporary files"
	@echo "  docs             Generate documentation"

# Installation
install:
	@echo "📦 Installing development dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black isort flake8 bandit safety pytest pytest-cov
	@echo "✅ Dependencies installed"

install-prod:
	@echo "📦 Installing production dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Production dependencies installed"

# Code formatting and linting
format:
	@echo "🎨 Formatting code..."
	black .
	isort .
	@echo "✅ Code formatted"

lint:
	@echo "🔍 Running linting checks..."
	black --check --diff .
	isort --check-only --diff .
	flake8 .
	@echo "✅ Linting passed"

security:
	@echo "🔒 Running security scans..."
	bandit -r src/ -f text
	safety check
	@echo "✅ Security scans completed"

# Testing
test:
	@echo "🧪 Running all tests with coverage..."
	pytest --cov=src --cov-report=term-missing --cov-report=html tests/
	@echo "✅ Tests completed"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest -m unit tests/ -v
	@echo "✅ Unit tests completed"

test-integration:
	@echo "🧪 Running integration tests..."
	pytest -m integration tests/ -v
	@echo "✅ Integration tests completed"

test-performance:
	@echo "⚡ Running performance benchmarks..."
	python scripts/test_benchmark_quick.py
	@echo "✅ Performance tests completed"

# Docker operations
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t insightdesk-ai:latest .
	@echo "✅ Docker image built"

docker-run:
	@echo "🚀 Running Docker container..."
	docker run --rm -p 8000:8000 insightdesk-ai:latest

docker-test:
	@echo "🧪 Testing Docker container..."
	docker build --target test -t insightdesk-ai:test .
	docker run --rm insightdesk-ai:test
	@echo "✅ Docker tests passed"

# Development server
run:
	@echo "🚀 Starting development server..."
	cd src && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Full build pipeline (mimics CI/CD)
build: format lint security test docker-build
	@echo "🎉 Full build completed successfully!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "API docs available at: http://localhost:8000/docs (when server running)"
	@echo "Coverage report available at: htmlcov/index.html"
	@echo "✅ Documentation generated"

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup completed"

# Development workflow shortcuts
dev-setup: install format
	@echo "🔧 Development environment ready!"

pre-commit: format lint security test-unit
	@echo "✅ Pre-commit checks passed!"

ci-check: lint test security
	@echo "✅ CI checks passed!"

# Quick demos
demo-benchmark:
	@echo "📊 Running benchmarking demo..."
	python scripts/demo_benchmarking.py

demo-api:
	@echo "🔥 Testing API endpoints..."
	@echo "Starting server in background..."
	cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
	@sleep 5
	@echo "Testing health endpoint..."
	curl -f http://localhost:8000/health || echo "❌ Health check failed"
	@echo "Testing models status..."
	curl -f http://localhost:8000/models/status || echo "❌ Models status failed"
	@echo "Stopping server..."
	@pkill -f "uvicorn api.main:app" || true

# MLflow operations
mlflow-ui:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --backend-store-uri file://./mlruns --port 5000

# Training shortcuts
train-models:
	@echo "🤖 Training models..."
	python scripts/train_xgboost.py
	python scripts/train_tensorflow.py
	@echo "✅ Models trained"

# Data operations
prepare-data:
	@echo "📊 Preparing data..."
	python scripts/unzip_and_load.py
	python scripts/build_features.py
	@echo "✅ Data prepared"

# Full development cycle
dev-cycle: clean dev-setup prepare-data train-models test demo-benchmark
	@echo "🎉 Full development cycle completed!"

# Production readiness check
prod-check: clean install-prod lint security test docker-build docker-test
	@echo "🚀 Production readiness check completed!"

# Streamlit dashboard commands
dashboard:
	@echo "🎨 Starting Streamlit dashboard..."
	streamlit run app.py --server.port 8501

dashboard-demo:
	@echo "🧪 Starting Streamlit demo..."
	streamlit run demo_dashboard.py --server.port 8502

full-stack:
	@echo "🚀 Launching full stack (FastAPI + Streamlit)..."
	python scripts/launch_full_stack.py

# Dashboard development
dashboard-dev: install
	@echo "🛠️  Setting up dashboard development environment..."
	pip install streamlit requests plotly
	@echo "✅ Dashboard development ready!"