#!/usr/bin/env bash
set -e

# Resolve script and repository locations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Working directory selection ────────────────────────────────────────────────
DEFAULT_WORK="/app"
FALLBACK_WORK="${ROOT_DIR}/workdir"

if [[ -d "${DEFAULT_WORK}" && -w "${DEFAULT_WORK}" ]]; then
    RUN_DIR="${DEFAULT_WORK}"
else
    RUN_DIR="${FALLBACK_WORK}"
fi
RUN_DIR="${OTTER_APP_DIR:-${RUN_DIR}}"

# ── Python interpreter ────────────────────────────────────────────────────────
PY="${PYTHON:-python3}"

# ── Create required output directories ────────────────────────────────────────
for sub in artifacts reports artifacts/embeddings artifacts/predictions encoder; do
    mkdir -p "${RUN_DIR}/${sub}"
done

# ── Deploy solution modules into the run directory ────────────────────────────
MODULES=(
    kg_io.py
    embedding_models.py
    optimisation.py
    ranking.py
    narrative.py
    pipeline.py
    run.py
)

for mod in "${MODULES[@]}"; do
    cp "${SCRIPT_DIR}/${mod}" "${RUN_DIR}/${mod}"
done

# ── Locate input data ─────────────────────────────────────────────────────────
INPUT_DIR="${OTTER_DATA_DIR:-/app/data}"
if [[ ! -f "${INPUT_DIR}/dataset_manifest.json" ]]; then
    INPUT_DIR="${ROOT_DIR}/environment/task_inputs"
fi

# ── Run the pipeline ──────────────────────────────────────────────────────────
OTTER_APP_DIR="${RUN_DIR}" \
OTTER_DATA_DIR="${INPUT_DIR}" \
PYTHONPATH="${RUN_DIR}" \
    "${PY}" "${RUN_DIR}/run.py"
