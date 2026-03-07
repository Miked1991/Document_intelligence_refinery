
### `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# Create necessary directories
RUN mkdir -p .refinery/profiles .refinery/pageindex

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY}

# Run
CMD ["python", "-m", "src.main"]