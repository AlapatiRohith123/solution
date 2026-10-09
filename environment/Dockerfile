# CPU-only default base image. For GPU tasks: set gpus = 1 in task.toml,
# use a CUDA base image instead, and pin the CUDA build of your ML
# framework in requirements.txt. Examples (pick one, delete the FROM below):
#   FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# Tasks are validated in single-container, single-GPU sandboxes — do not
# assume more than one GPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates procps ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# EDIT: Change the path and filename to match your task's input data.
COPY task_inputs/ /app/data/

# EDIT: Create the directories your task expects and verify input data exists.
RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && test -f /app/data/input_data.npz

# EDIT: Set env var pointing to your task's input data.
ENV OTTER_DATA_PATH=/app/data/input_data.npz
