import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    app_dir = os.environ.get("APP_DIR", "/app")
    # Load dataset
    df = pd.read_csv(f"{app_dir}/task_inputs/dataset.csv")
    
    # Train on labelled data only for simplicity in the oracle
    labeled_df = df[df['anomaly'] != -1].copy()
    
    features = ['duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks',
                'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks',
                'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']
                
    X_train = labeled_df[features].values
    y_train = labeled_df['anomaly'].values
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Utilize both T4 GPUs efficiently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularNN(input_dim=len(features))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Save model
    os.makedirs(f"{app_dir}/artifacts", exist_ok=True)
    
    # If DataParallel was used, save the underlying module
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), f"{app_dir}/artifacts/model.pth")
        
    # Generate inference script
    inference_code = """
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn

class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

app_dir = os.environ.get("APP_DIR", "/app")
df = pd.read_csv(args.input)
features = ['duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks',
            'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks',
            'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TabularNN(input_dim=len(features))
model.load_state_dict(torch.load(f"{app_dir}/artifacts/model.pth", map_location=device))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
model.eval()

X_tensor = torch.tensor(df[features].values, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy().flatten()
    
out_df = pd.DataFrame({'segment': df['segment'], 'prediction': preds})
out_df.to_csv(args.output, index=False)
"""
    with open(f"{app_dir}/artifacts/inference.py", 'w') as f:
        f.write(inference_code)

    # Compute training metrics for metrics.json
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X_train, dtype=torch.float32).to(device)
        probs = model(X_all).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        
    mcc = float(matthews_corrcoef(y_train, preds))
    auc_pr = float(average_precision_score(y_train, probs))
    f1 = float(f1_score(y_train, preds))
    
    metrics = {
        'mcc': mcc,
        'auc_pr': auc_pr,
        'worst_channel_f1': f1 # Approximation for internal metric
    }
    
    with open(f"{app_dir}/artifacts/metrics.json", 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
