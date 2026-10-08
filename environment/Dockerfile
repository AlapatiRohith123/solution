FROM python:3.11-slim@sha256:7d8a6b68a3f81e05e5d3ab72bd9d0a649d2146f254b1f63a232f384a5113d8d6

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
