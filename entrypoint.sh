#!/bin/sh
set -e

# Wait for the PostgreSQL server to be available
while ! nc -z host.docker.internal 5432; do
  echo "Waiting for the PostgreSQL server to start..."
  sleep 1
done


echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting Uvicorn server..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1