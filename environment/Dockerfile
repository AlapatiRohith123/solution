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
    && test -f /app/data/metadata.csv

# EDIT: Set env var pointing to your task's input data.
ENV OTTER_DATA_PATH=/app/data/metadata.csv
