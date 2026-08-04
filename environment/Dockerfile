FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates procps ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock /tmp/
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock

COPY task_inputs/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports \
    && test -f /app/data/dataset_manifest.json \
    && test -f /app/data/encoder_protocol.md \
    && test -f /app/data/kg/entities.tsv \
    && test -f /app/data/kg/relations.tsv \
    && test -f /app/data/kg/train.tsv \
    && test -f /app/data/kg/valid.tsv \
    && test -f /app/data/lp/eval_queries.tsv \
    && test -f /app/data/lp/val_answers.tsv \
    && python -c "import json; m=json.load(open('/app/data/dataset_manifest.json')); assert m['n_entities']==17050 and m['n_relations']==51 and m['n_eval_queries']==20621"

ENV OTTER_DATA_DIR=/app/data
