#!/usr/bin/env bash
set -euo pipefail

app_dir="${APP_DIR:-/app}"
solution_dir="${SOLUTION_DIR:-/solution}"

cp "${solution_dir}/train.py" "${app_dir}/train.py"
cp "${solution_dir}/convert.py" "${app_dir}/convert.py"

output="${app_dir}/artifacts/reference-full"
mkdir -p "${output}"
python3 "${app_dir}/run_training.py" \
    "${app_dir}/data/shakespeare.npz" \
    "${output}" \
    --train-steps 100000
cp "${output}/model.onnx" "${app_dir}/model.onnx"
