FROM python:3.11-slim@sha256:7d8a6b68a3f81e05e5d3ab72bd9d0a649d2146f254b1f63a232f384a5113d8d6

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.39.2-1.1 \
    curl=7.88.1-10+deb12u5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt

ENV PYTHONPATH=/app
