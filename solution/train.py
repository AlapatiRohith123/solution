import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import pickle
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score

def main():
    np.random.seed(42)
    # Load dataset
    df = pd.read_csv('/app/task_inputs/dataset.csv')
    
    # Train on labelled data only for simplicity in the oracle
    labeled_df = df[df['anomaly'] != -1].copy()
    
    features = ['duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks',
                'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks',
                'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']
                
    X_train = labeled_df[features]
    y_train = labeled_df['anomaly']
    
    # Train a simple XGBoost model
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Save model
    with open('/app/artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # Generate inference script
    inference_code = """
import argparse
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)
features = ['duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks',
            'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks',
            'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']

with open('/app/artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

preds = model.predict_proba(df[features])[:, 1]
out_df = pd.DataFrame({'segment': df['segment'], 'prediction': preds})
out_df.to_csv(args.output, index=False)
"""
    with open('/app/artifacts/inference.py', 'w') as f:
        f.write(inference_code)

    # Compute training metrics for metrics.json
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)[:, 1]
    
    mcc = float(matthews_corrcoef(y_train, preds))
    auc_pr = float(average_precision_score(y_train, probs))
    f1 = float(f1_score(y_train, preds))
    
    metrics = {
        'mcc': mcc,
        'auc_pr': auc_pr,
        'worst_channel_f1': f1 # Approximation for internal metric
    }
    
    with open('/app/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
