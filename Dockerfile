FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .[dev]

COPY src/ ./src/
COPY rubric/ ./rubric/
COPY tests/ ./tests/

ENTRYPOINT ["python", "-m", "src.main"]