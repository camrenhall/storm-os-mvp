# Dockerfile for Generator Container - Optimized for Render deployment
FROM python:3.11-slim

# Install minimal system dependencies for eccodes only
RUN apt-get update && apt-get install -y \
    libeccodes-tools \
    libeccodes-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (fixed paths - no pipeline/ subdirectory)
COPY *.py ./
COPY generator/ ./generator/
COPY schema.sql .

# Create cache directory
RUN mkdir -p /tmp/mrms_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN groupadd -r generator && useradd -r -g generator generator
RUN chown -R generator:generator /app /tmp/mrms_cache
USER generator

# Entry point
CMD ["python", "-m", "generator.run"]