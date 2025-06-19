FROM python:3.13.4-slim-bookworm

# Install system dependencies for geospatial libraries and eccodes
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libeccodes-dev \
    libeccodes-tools \
    libproj-dev \
    libgeos-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pipeline/ ./pipeline/
COPY *.py ./

# Create necessary directories
RUN mkdir -p /app/flash_cache /app/qpe_cache /app/ffw_cache /app/validation_output

# Set Python path
ENV PYTHONPATH=/app:/app/pipeline

# Default command (can be overridden)
CMD ["python", "-m", "pipeline.flood_classifier"]