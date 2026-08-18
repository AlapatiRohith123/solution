# This pins the tested Torch 2.3.1 and CUDA 12.1 stack for L4 training.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime@sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        asciinema=2.1.0-1 \
        tmux=3.2a-4ubuntu0.2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes \
        -r /tmp/requirements.txt \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.3.1'"

COPY task_inputs/shakespeare.npz /app/data/shakespeare.npz
COPY task_inputs/train.py task_inputs/convert.py /app/
COPY shakespeare_data.md training_contract.md data_feeder.py run_training.py /app/

RUN mkdir -p /app/artifacts \
    && chmod 0700 /app/data \
    && chmod 0755 /app/run_training.py \
    && echo "0efb587858b59c7d385a6340ed03de9c8aa283440d65da952a8a904bc9720c91  /app/data/shakespeare.npz" | sha256sum -c -

ENV OTTER_DATA_PATH=/app/data/shakespeare.npz
