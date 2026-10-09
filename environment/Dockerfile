# CPU-only default base image. For GPU tasks: set gpus = 1 in task.toml,
# use a CUDA base image instead, and pin the CUDA build of your ML
# framework in requirements.txt. Examples (pick one, delete the FROM below):
#   FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# Tasks are validated in single-container, single-GPU sandboxes — do not
# assume more than one GPU.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
WORKDIR /app


COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt

# EDIT: Change the path and filename to match your task's input data.
COPY task_inputs/ /app/data/

# EDIT: Create the directories your task expects and verify input data exists.
RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && test -f /app/data/input_data.npz

# EDIT: Set env var pointing to your task's input data.
ENV OTTER_DATA_PATH=/app/data/input_data.npz
