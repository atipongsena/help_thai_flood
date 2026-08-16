# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root (assuming build context is root)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
# We copy the entire api and utils folders
COPY api/ ./api/
COPY utils/ ./utils/

# Copy models
# NOTE: If models are very large, consider using Google Cloud Storage and downloading them at runtime
COPY models/ ./models/

# Set PYTHONPATH to include /app so imports work
# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root (assuming build context is root)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
# We copy the entire api and utils folders
COPY api/ ./api/
COPY utils/ ./utils/

# Copy models
# NOTE: If models are very large, consider using Google Cloud Storage and downloading them at runtime
COPY models/ ./models/

# Set PYTHONPATH to include /app so imports work
ENV PYTHONPATH=/app

# Expose port 8000
EXPOSE 8000

# Run the application
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}
