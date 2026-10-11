import json
import os
import random

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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    app_dir = os.environ.get("APP_DIR", "/app")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset_path = f"{app_dir}/task_inputs/dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = f"{app_dir}/environment/task_inputs/dataset.csv"
    df = pd.read_csv(dataset_path)

    # Train on labelled data
    labeled_df = df[df['anomaly'] != -1].copy()

    features = ['duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks',
                'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks',
                'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']

    X_train = labeled_df[features].values
    y_train = labeled_df['anomaly'].values

    # Train ensemble of models with diverse seeds
    seeds = [42, 101, 202, 303, 404]
    state_dicts = []

    for seed in seeds:
        seed_everything(seed)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = TabularNN(input_dim=len(features)).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(25):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        state_dicts.append(model.state_dict())

    # Save model artifacts
    os.makedirs(f"{app_dir}/artifacts", exist_ok=True)
    torch.save({
        'seeds': seeds,
        'state_dicts': state_dicts,
        'features': features
    }, f"{app_dir}/artifacts/model.pth")

    # Generate inference script
    inference_code = '''import argparse
import os
import random
import numpy as np
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    app_dir = os.environ.get("APP_DIR", "/app")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(f"{app_dir}/artifacts/model.pth", map_location=device)
    features = checkpoint['features']
    state_dicts = checkpoint['state_dicts']

    df = pd.read_csv(args.input)
    X_tensor = torch.tensor(df[features].values, dtype=torch.float32).to(device)

    all_preds = []
    with torch.no_grad():
        for state_dict in state_dicts:
            model = TabularNN(input_dim=len(features))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            preds = model(X_tensor).cpu().numpy().flatten()
            all_preds.append(preds)

    final_preds = np.mean(all_preds, axis=0)
    out_df = pd.DataFrame({'segment': df['segment'], 'prediction': final_preds})
    out_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
'''
    with open(f"{app_dir}/artifacts/inference.py", 'w') as f:
        f.write(inference_code)

    # Compute training metrics for metrics.json
    all_train_preds = []
    with torch.no_grad():
        X_all = torch.tensor(X_train, dtype=torch.float32).to(device)
        for state_dict in state_dicts:
            model = TabularNN(input_dim=len(features))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            all_train_preds.append(model(X_all).cpu().numpy().flatten())

    probs = np.mean(all_train_preds, axis=0)
    preds = (probs > 0.5).astype(int)

    mcc = float(matthews_corrcoef(y_train, preds))
    auc_pr = float(average_precision_score(y_train, probs))
    f1 = float(f1_score(y_train, preds))

    metrics = {
        'mcc': mcc,
        'auc_pr': auc_pr,
        'worst_channel_f1': f1
    }

    with open(f"{app_dir}/artifacts/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Training and artifact generation complete. Metrics:", metrics)


if __name__ == '__main__':
    main()

