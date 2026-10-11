import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def evaluate(context=None):
    # Enforce deterministic evaluation
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    results = {}
    app_dir = os.environ.get("APP_DIR", "/app")
    metrics = {"mcc": 0.0, "auc_pr": 0.0, "worst_channel_f1": 0.0}

    # 1. Check model artifact existence
    artifacts_dir = f"{app_dir}/artifacts"
    model_artifact_found = False
    if os.path.isdir(artifacts_dir):
        valid_exts = ('.pth', '.pt', '.bin', '.onnx', '.pkl', '.pickle', '.joblib', '.h5', '.keras', '.ckpt', '.weights')
        for fname in os.listdir(artifacts_dir):
            if fname in ('inference.py', 'metrics.json', 'hidden_preds.csv'):
                continue
            fpath = os.path.join(artifacts_dir, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                if any(fname.endswith(ext) for ext in valid_exts) or fname.startswith('model'):
                    model_artifact_found = True
                    break
    results['model_artifact_exists'] = model_artifact_found

    # 2. Check inference script existence
    inference_path = f"{app_dir}/artifacts/inference.py"
    inference_file_ok = os.path.isfile(inference_path) and os.path.getsize(inference_path) > 0
    results['inference_exists'] = bool(inference_file_ok and model_artifact_found)

    if not (inference_file_ok and model_artifact_found):
        return _pack_results(results, metrics)

    # 3. Check metrics.json existence and content
    metrics_path = f"{app_dir}/artifacts/metrics.json"
    metrics_json_ok = False
    if os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0:
        try:
            with open(metrics_path, 'r') as f:
                reported = json.load(f)
            required_keys = {"mcc", "auc_pr", "worst_channel_f1"}
            if isinstance(reported, dict) and required_keys.issubset(reported.keys()):
                valid_values = True
                for k in required_keys:
                    val = reported[k]
                    if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                        valid_values = False
                        break
                if valid_values:
                    metrics_json_ok = True
        except Exception:
            metrics_json_ok = False
    results['metrics_json_exists'] = metrics_json_ok

    # 4. Run inference on hidden split
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    inference_cmd = ['python3', f"{app_dir}/artifacts/inference.py", '--input', f"{app_dir}/tests/data/hidden_test.csv", '--output', f"{app_dir}/artifacts/hidden_preds.csv"]
    try:
        subprocess.run(inference_cmd, check=True, env=env)
        results['inference_runs'] = True
    except Exception:
        try:
            inference_cmd[0] = 'python'
            subprocess.run(inference_cmd, check=True, env=env)
            results['inference_runs'] = True
        except Exception:
            results['inference_runs'] = False
            return _pack_results(results, metrics)

    # 5. Evaluate predictions
    try:
        preds_df = pd.read_csv(f"{app_dir}/artifacts/hidden_preds.csv")
        gt_df = pd.read_csv(f"{app_dir}/tests/data/hidden_test.csv")

        # Merge by segment
        merged = preds_df.merge(gt_df, on='segment')
        y_true = merged['anomaly']
        y_pred_prob = merged['prediction']
        y_pred = (y_pred_prob > 0.5).astype(int)

        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
        metrics['auc_pr'] = float(average_precision_score(y_true, y_pred_prob))

        # Worst channel F1
        channel_f1s = []
        for channel in merged['channel'].unique():
            ch_data = merged[merged['channel'] == channel]
            if len(ch_data) > 0 and len(ch_data['anomaly'].unique()) > 1:
                channel_f1s.append(f1_score(ch_data['anomaly'], (ch_data['prediction'] > 0.5).astype(int)))

        metrics['worst_channel_f1'] = float(min(channel_f1s)) if channel_f1s else 0.0
        results['performance_measured'] = True
    except Exception:
        results['performance_measured'] = False

    return _pack_results(results, metrics)


def _pack_results(results, metrics):
    criteria_list = [
        {"id": "model_artifact_exists", "milestone_id": "model_artifacts", "passed": results.get("model_artifact_exists", False)},
        {"id": "inference_exists", "milestone_id": "model_artifacts", "passed": results.get("inference_exists", False)},
        {"id": "metrics_json_exists", "milestone_id": "model_artifacts", "passed": results.get("metrics_json_exists", False)},
        {"id": "inference_runs", "milestone_id": "inference_evaluation", "passed": results.get("inference_runs", False)},
        {"id": "performance_measured", "milestone_id": "inference_evaluation", "passed": results.get("performance_measured", False)}
    ]
    return {'criteria': criteria_list, 'metrics': metrics}

