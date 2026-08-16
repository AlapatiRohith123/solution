FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

LABEL org.opencontainers.image.revision="ppgn-completion-budget-20260808-r6"

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip

COPY task_inputs/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && test -s /app/data/train.npz \
    && test -s /app/data/valid.npz \
    && test -s /app/data/test.npz \
    && test -s /app/data/dataset_manifest.json \
    && test -s /app/data/checksums.json \
    && test -s /app/data/LICENSE_DATASET.md

ENV OTTER_DATA_DIR=/app/data
