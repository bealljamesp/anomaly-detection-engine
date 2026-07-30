# Use an official lightweight Python 3.12 base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and source code
COPY pyproject.toml .
COPY src/ src/
COPY main.py .

# Upgrade pip and install the package with cloud & dev dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[cloud,dev]

# Container execution entrypoint
CMD ["python", "main.py"]
