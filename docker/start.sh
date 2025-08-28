#!/bin/bash

# Start script for production deployment
set -e

echo "Starting ML Pipeline Application..."

# Wait for dependencies
echo "Waiting for database..."
while ! nc -z postgres 5432; do
  sleep 0.1
done
echo "Database is ready!"

echo "Waiting for MLflow..."
while ! nc -z mlflow 5000; do
  sleep 0.1
done
echo "MLflow is ready!"

# Initialize Feast repository if it doesn't exist
if [ ! -f "/app/feast_repo/feature_store.yaml" ]; then
    echo "Initializing Feast repository..."
    cd /app
    python -c "
from config.feast_config import feast_store
feast_store.create_feature_repo()
feast_store.create_sample_data()
print('Feast repository initialized')
"
fi

# Run database migrations or setup if needed
echo "Setting up database tables..."
python -c "
from config.supabase_client import supabase_client
import asyncio
asyncio.run(supabase_client.create_tables())
print('Database setup complete')
"

# Start the FastAPI application
echo "Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4