# Dockerfile for Generator Container - Optimized for Render deployment
FROM python:3.11-slim

# Install system dependencies for eccodes AND geopandas
RUN apt-get update && apt-get install -y \
    libeccodes-tools \
    libeccodes-dev \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install core Python dependencies first
RUN pip install --no-cache-dir -r requirements.txt

# Install geopandas with proper system library detection
RUN pip install --no-cache-dir geopandas shapely fiona rasterio

# Copy source code matching your structure
COPY pipeline/ ./pipeline/
COPY generator/ ./generator/
COPY db.py .
COPY schema.sql .

# Replace the existing mkdir commands with:
RUN mkdir -p /app/cache/flash && \
    mkdir -p /app/cache/qpe && \
    mkdir -p /app/cache/ffw && \
    chmod -R 755 /app/cache

# Create non-root user for security
RUN groupadd -r generator && useradd -r -g generator generator

# Fix permissions for cache directories and app
RUN chown -R generator:generator /app

USER generator

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "-m", "generator.run"]