# Multi-stage Dockerfile for ML Pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data /app/feast_repo

# Set permissions
RUN chmod +x /app

# Expose ports
EXPOSE 8000 8501 5000

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    ipykernel

# Create jupyter config
RUN jupyter notebook --generate-config

# API stage
FROM base as api

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Dashboard stage
FROM base as dashboard

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# MLflow stage
FROM base as mlflow

# Create MLflow directories
RUN mkdir -p /app/mlruns /app/mlflow-artifacts

# Health check for MLflow
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlflow-artifacts"]

# Prefect worker stage
FROM base as prefect-worker

# Run Prefect worker
CMD ["prefect", "worker", "start", "--pool", "default-agent-pool"]

# Production stage (default)
FROM base as production

# Copy startup script
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/start.sh"]