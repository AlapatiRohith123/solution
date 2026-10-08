Build an end-to-end tabular ML system for CubeSat Telemetry Fragment Anomaly Detection on the OPSSAT-AD dataset. 

You are provided with tabular telemetry fragments in `/app/task_inputs/dataset.csv`.
- The dataset contains features derived from univariate fragments.
- It includes data from 6 channels.
- Only ~10% of the fragments are labelled (where `anomaly` is 0 or 1). The rest are unlabelled (`anomaly` is -1).
- The goal is to detect anomalous fragments (binary classification).

You must:
1. Train a model capable of detecting anomalies, robust to distribution shifts (new channels).
2. Save your trained model artifact(s) to `/app/artifacts/`.
3. Create an inference script `/app/artifacts/inference.py` that takes an input CSV path and an output CSV path as arguments (`python /app/artifacts/inference.py --input <input_csv> --output <output_csv>`). It should load your model and write predictions (with columns `segment` and `prediction`, where prediction is probability or binary label) to the output CSV.
4. Generate evaluation metrics on your internal validation set and save them to `/app/artifacts/metrics.json`. Your JSON should have keys: `mcc`, `auc_pr`, and `worst_channel_f1`.

We will evaluate your inference script on a hidden split containing 3 held-out channels and the remaining 90% of the visible channels. Pass requires beating the human baseline mean on the hidden evaluation.

The pass criteria require the model to achieve an mcc >= 0.60, an auc_pr >= 0.75, and a worst_channel_f1 >= 0.45.
