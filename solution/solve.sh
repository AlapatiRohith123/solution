#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_APP_DIR="/app"
if [[ ! -d "${DEFAULT_APP_DIR}" || ! -w "${DEFAULT_APP_DIR}" ]]; then
  DEFAULT_APP_DIR="${TASK_DIR}/workdir"
fi
APP_DIR="${OTTER_APP_DIR:-${DEFAULT_APP_DIR}}"
DATA_DIR="${OTTER_DATA_DIR:-/app/data}"
if [[ ! -f "${DATA_DIR}/dataset_manifest.json" ]]; then
  DATA_DIR="${TASK_DIR}/environment/task_inputs"
fi

mkdir -p "${APP_DIR}/src"
cp "${SCRIPT_DIR}/src/"*.py "${APP_DIR}/src/"
cp "${SCRIPT_DIR}/run.py" "${APP_DIR}/run.py"

OTTER_APP_DIR="${APP_DIR}" \
OTTER_DATA_DIR="${DATA_DIR}" \
PYTHONPATH="${APP_DIR}" \
  "${PYTHON:-python3}" "${APP_DIR}/run.py"
