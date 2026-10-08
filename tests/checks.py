import os
import subprocess

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def evaluate(context=None):
    results = {}
    app_dir = os.environ.get("APP_DIR", "/app")

    metrics = {"mcc": 0.0, "auc_pr": 0.0, "worst_channel_f1": 0.0}
    
    # Check if artifacts exist
    if not os.path.exists(f"{app_dir}/artifacts/inference.py"):
        results['inference_exists'] = False
        return _pack_results(results, metrics)
        
    results['inference_exists'] = True
    
    if not os.path.exists(f"{app_dir}/artifacts/metrics.json"):
        results['metrics_json_exists'] = False
    else:
        results['metrics_json_exists'] = True
        
    # Run inference on hidden split
    try:
        subprocess.run(['python', f"{app_dir}/artifacts/inference.py", '--input', f"{app_dir}/tests/data/hidden_test.csv", '--output', f"{app_dir}/artifacts/hidden_preds.csv"], check=True)
        results['inference_runs'] = True
    except Exception:  # noqa: BLE001
        results['inference_runs'] = False
        return _pack_results(results, metrics)
        
    # Evaluate predictions
    try:
        preds_df = pd.read_csv(f"{app_dir}/artifacts/hidden_preds.csv")
        gt_df = pd.read_csv(f"{app_dir}/tests/data/hidden_test.csv")
        
        # Merge by segment
        merged = preds_df.merge(gt_df, on='segment')
        y_true = merged['anomaly']
        y_pred_prob = merged['prediction']
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_prob)
        
        # Worst channel F1
        channel_f1s = []
        for channel in merged['channel'].unique():
            ch_data = merged[merged['channel'] == channel]
            if len(ch_data) > 0 and len(ch_data['anomaly'].unique()) > 1:
                channel_f1s.append(f1_score(ch_data['anomaly'], (ch_data['prediction'] > 0.5).astype(int)))
        
        metrics['worst_channel_f1'] = min(channel_f1s) if channel_f1s else 0.0
        results['performance_measured'] = True
        
    except Exception:  # noqa: BLE001
        results['performance_measured'] = False
        
    return _pack_results(results, metrics)


def _pack_results(results, metrics):
    criteria_list = [
        {"id": "inference_exists", "passed": results.get("inference_exists", False)},
        {"id": "metrics_json_exists", "passed": results.get("metrics_json_exists", False)},
        {"id": "inference_runs", "passed": results.get("inference_runs", False)},
        {"id": "performance_measured", "passed": results.get("performance_measured", False)}
    ]
    return {'criteria': criteria_list, 'metrics': metrics}
