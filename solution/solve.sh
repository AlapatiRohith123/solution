#!/usr/bin/env bash
set -euo pipefail

app_dir="${APP_DIR:-/app}"
solution_dir="${SOLUTION_DIR:-solution}"

mkdir -p "${app_dir}/data"
mkdir -p "${app_dir}/artifacts"

cp environment/task_inputs/shakespeare.npz "${app_dir}/data/shakespeare.npz"
cp environment/shakespeare_data.md environment/training_contract.md environment/data_feeder.py environment/run_training.py "${app_dir}/"

cp "${solution_dir}/train.py" "${app_dir}/train.py"
cp "${solution_dir}/convert.py" "${app_dir}/convert.py"

output="${app_dir}/artifacts/reference-full"
mkdir -p "${output}"
python3 "${app_dir}/run_training.py" \
    "${app_dir}/data/shakespeare.npz" \
    "${output}" \
    --train-steps 10000 > /kaggle/working/logs/training.log 2>&1
cp "${output}/model.onnx" "${app_dir}/model.onnx"
