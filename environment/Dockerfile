FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

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
