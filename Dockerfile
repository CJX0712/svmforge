# SVMForge — reproducible runtime image (author 晨星)
FROM python:3.12-slim

WORKDIR /app

# Runtime deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dev/lint/test tooling.
RUN pip install --no-cache-dir ruff pytest

# Project source (single-layer copy is fine for a small repo).
COPY . .

# Run the deterministic 2-run demo by default (writes benchmark.json).
CMD ["python", "examples/run_demo.py"]
