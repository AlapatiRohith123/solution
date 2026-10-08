import json
import os
import subprocess
import pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score

def evaluate(context=None):
    results = {}
    metrics = {}
    
    # Check if artifacts exist
    if not os.path.exists('/app/artifacts/inference.py'):
        results['inference_exists'] = False
        return {'criteria': results, 'metrics': metrics}
    results['inference_exists'] = True
    
    if not os.path.exists('/app/artifacts/metrics.json'):
        results['metrics_json_exists'] = False
    else:
        results['metrics_json_exists'] = True
        
    # Run inference on hidden split
    try:
        subprocess.run(['python', '/app/artifacts/inference.py', '--input', '/app/tests/data/hidden_test.csv', '--output', '/app/artifacts/hidden_preds.csv'], check=True)
        results['inference_runs'] = True
    except Exception:
        results['inference_runs'] = False
        return {'criteria': results, 'metrics': metrics}
        
    # Evaluate predictions
    try:
        preds_df = pd.read_csv('/app/artifacts/hidden_preds.csv')
        gt_df = pd.read_csv('/app/tests/data/hidden_test.csv')
        
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
        
    except Exception as e:
        results['performance_measured'] = False
        
    return {'criteria': results, 'metrics': metrics}
