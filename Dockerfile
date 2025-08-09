# Dockerfile

FROM python:3.12-slim-bullseye

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and netcat (for DB wait)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy project files
COPY . .

# Configure poetry to create virtualenv inside the project and install dependencies
RUN poetry config virtualenvs.in-project true && poetry install --no-root

# Add the virtualenv binaries to PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]