import os

import numpy as np
import pandas as pd

print("Downloading dataset...")
url = "https://zenodo.org/api/records/12588359/files/dataset.csv/content"
df = pd.read_csv(url)

# Remove the 'train' column as it's not needed (or we'll create our own splits)
if "train" in df.columns:
    df = df.drop(columns=["train"])

# Shuffle
np.random.seed(42)
df = df.sample(frac=1.0).reset_index(drop=True)

# Opaque IDs
df["segment"] = ["seq_" + str(i) for i in range(len(df))]

# Amplitudes rescaled (standardize numerical columns)
num_cols = [
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len",
]

for col in num_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

channels = df["channel"].unique()
print("Channels:", channels)

# Pick 3 held-out channels
hidden_channels = channels[-3:]
visible_channels = channels[:-3]

print("Visible:", visible_channels)
print("Hidden:", hidden_channels)

visible_df = df[df["channel"].isin(visible_channels)].copy()
hidden_df = df[df["channel"].isin(hidden_channels)].copy()

# For visible_df, we provide 10% labels.
# We'll split visible_df into train (10%) and test (90%).
visible_df = visible_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_train = int(len(visible_df) * 0.1)

train_df = visible_df.iloc[:n_train].copy()
unlabeled_df = visible_df.iloc[n_train:].copy()
in_domain_test_df = visible_df.iloc[n_train:].copy()  # The labels for the remaining 90%

# Remove labels for unlabeled
unlabeled_df["anomaly"] = -1

# Combine train and unlabeled for the agent
agent_df = (
    pd.concat([train_df, unlabeled_df])
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

# Save agent_df to task_inputs
os.makedirs("environment/task_inputs", exist_ok=True)
agent_df.to_csv("environment/task_inputs/dataset.csv", index=False)

# Save hidden evaluation to tests/
os.makedirs("tests/data", exist_ok=True)
hidden_df.to_csv("tests/data/hidden_test.csv", index=False)
in_domain_test_df.to_csv("tests/data/in_domain_test.csv", index=False)

print("Data preparation complete.")
