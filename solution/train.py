import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, k=5, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, padding=(k // 2) * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TelemetryTransformerResNet(nn.Module):
    def __init__(self, seq_len=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.SiLU(),
        )
        self.res1 = ConvBlock1D(48, 64, k=5, dilation=1)
        self.res2 = ConvBlock1D(64, 96, k=5, dilation=2)

        self.proj = nn.Linear(96, 96)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(96 * 2, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.res2(h)

        t_seq = h.permute(0, 2, 1)
        t_seq = self.transformer(t_seq)
        h = t_seq.permute(0, 2, 1)

        avg_p = self.pool_avg(h).squeeze(-1)
        max_p = self.pool_max(h).squeeze(-1)
        feat = torch.cat([avg_p, max_p], dim=1)
        return self.head(feat).squeeze(-1)


def main():
    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training requires GPU acceleration with CUDA as specified in task.toml."
        )
    device = torch.device('cuda')
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0)})")

    app_dir = os.environ.get("APP_DIR", "/app")

    dataset_path = os.path.join(app_dir, "task_inputs", "dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(app_dir, "environment", "task_inputs", "dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "environment", "task_inputs", "dataset.csv"
        )
    df = pd.read_csv(dataset_path)

    seq_cols = [f"seq_{i:03d}" for i in range(128)]

    # Filter labeled data
    labeled_df = df[df['anomaly'] != -1].copy().reset_index(drop=True)
    X_lab = labeled_df[seq_cols].values[:, np.newaxis, :]
    y_lab = labeled_df['anomaly'].values.astype(np.float32)

    # 3-fold cross validation with GPU acceleration
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_logits = np.zeros(len(labeled_df))
    saved_weights = []

    pos_weight = torch.tensor([(len(y_lab) - sum(y_lab)) / sum(y_lab)]).to(device)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lab, y_lab)):
        model = TelemetryTransformerResNet(128).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-3)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)

        tr_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_lab[tr_idx], dtype=torch.float32),
                torch.tensor(y_lab[tr_idx], dtype=torch.float32),
            ),
            batch_size=32,
            shuffle=True,
        )

        best_loss = 1e9
        best_w = None

        for ep in range(30):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                jitter = torch.randn_like(bx) * 0.02
                opt.zero_grad()
                loss = crit(model(bx + jitter), by)
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vx = torch.tensor(X_lab[va_idx], dtype=torch.float32).to(device)
                vy = torch.tensor(y_lab[va_idx], dtype=torch.float32).to(device)
                vl = crit(model(vx), vy).item()
                if vl < best_loss:
                    best_loss = vl
                    best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
        saved_weights.append(best_w)

        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_lab[va_idx], dtype=torch.float32).to(device)
            oof_logits[va_idx] = model(vx).cpu().numpy()

    # Discover optimal threshold calibration on out-of-fold logits
    best_logit_th = 0.0
    best_f1 = -1.0
    for candidate_th in np.linspace(-2.0, 2.0, 81):
        f = f1_score(y_lab, (oof_logits > candidate_th).astype(int))
        if f > best_f1:
            best_f1 = f
            best_logit_th = float(candidate_th)

    calibrated_val_probs = 1.0 / (1.0 + np.exp(-(oof_logits - best_logit_th)))
    calibrated_val_probs = np.clip(calibrated_val_probs, 0.0, 1.0)
    val_preds = (calibrated_val_probs > 0.5).astype(int)

    val_mcc = float(matthews_corrcoef(y_lab, val_preds))
    val_auc_pr = float(average_precision_score(y_lab, calibrated_val_probs))

    # Evaluate channel-wise F1 on validation folds
    val_channel_f1s = []
    for channel in labeled_df['channel'].unique():
        ch_mask = (labeled_df['channel'] == channel).values
        ch_data = labeled_df[labeled_df['channel'] == channel]
        pos_count = int((ch_data['anomaly'] == 1).sum())
        if len(ch_data) >= 20 and pos_count >= 5 and len(ch_data['anomaly'].unique()) > 1:
            ch_f1 = f1_score(ch_data['anomaly'], val_preds[ch_mask])
            val_channel_f1s.append(float(ch_f1))

    worst_channel_f1 = float(min(val_channel_f1s)) if val_channel_f1s else float(f1_score(y_lab, val_preds))

    metrics = {
        'mcc': val_mcc,
        'auc_pr': val_auc_pr,
        'worst_channel_f1': worst_channel_f1,
    }

    # Save PyTorch model artifacts
    artifacts_dir = f"{app_dir}/artifacts"
    if not os.path.exists(app_dir):
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pth")
    torch.save({
        'models': saved_weights,
        'seq_cols': seq_cols,
        'best_logit_th': float(best_logit_th),
        'seq_len': 128,
    }, model_path)

    # Save internal validation metrics
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create inference script
    inference_code = '''import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, k=5, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, padding=(k // 2) * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TelemetryTransformerResNet(nn.Module):
    def __init__(self, seq_len=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.SiLU(),
        )
        self.res1 = ConvBlock1D(48, 64, k=5, dilation=1)
        self.res2 = ConvBlock1D(64, 96, k=5, dilation=2)

        self.proj = nn.Linear(96, 96)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(96 * 2, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.res2(h)

        t_seq = h.permute(0, 2, 1)
        t_seq = self.transformer(t_seq)
        h = t_seq.permute(0, 2, 1)

        avg_p = self.pool_avg(h).squeeze(-1)
        max_p = self.pool_max(h).squeeze(-1)
        feat = torch.cat([avg_p, max_p], dim=1)
        return self.head(feat).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    app_dir = os.environ.get("APP_DIR", "/app")
    model_path = os.path.join(app_dir, "artifacts", "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    seq_cols = checkpoint['seq_cols']
    best_th = float(checkpoint['best_logit_th'])
    saved_weights = checkpoint['models']
    seq_len = checkpoint.get('seq_len', 128)

    df = pd.read_csv(args.input)
    X = df[seq_cols].values[:, np.newaxis, :]

    total_logits = np.zeros(len(df))
    for weights in saved_weights:
        model = TelemetryTransformerResNet(seq_len=seq_len).to(device)
        model.load_state_dict({k: v.to(device) for k, v in weights.items()})
        model.eval()
        with torch.no_grad():
            batch_size = 512
            for i in range(0, len(X), batch_size):
                batch_tensor = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                logits = model(batch_tensor).cpu().numpy()
                total_logits[i:i+batch_size] += logits / len(saved_weights)

    probs = 1.0 / (1.0 + np.exp(-(total_logits - best_th)))
    probs = np.clip(probs, 0.0, 1.0)
    out_df = pd.DataFrame({'segment': df['segment'], 'prediction': probs})
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
'''
    inference_script_path = os.path.join(artifacts_dir, "inference.py")
    with open(inference_script_path, 'w') as f:
        f.write(inference_code)

    print("Training and artifact generation complete. Internal validation metrics:", metrics)


if __name__ == '__main__':
    main()
