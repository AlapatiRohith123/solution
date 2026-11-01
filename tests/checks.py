import glob
import hashlib
import json
import os
import random
import subprocess
import sys

# Grader isolation: Strip /app, current working directory, and PYTHONPATH entries from sys.path
# to prevent any agent files from hijacking imports or verifier scoring.
if "PYTHONPATH" in os.environ:
    del os.environ["PYTHONPATH"]
cwd = os.path.abspath(os.getcwd())
sys.path = [
    p for p in sys.path
    if p and os.path.abspath(p) not in ("/app", cwd, os.path.join(cwd, "artifacts"))
]

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

EXPECTED_DATASET_SHA256 = "880698298f0cf4bda4cd818d78b8ea72eac1f9eac6ca75418774f6f0bedd6e79"


def _run_inference_cmd(cmd, sub_kwargs):
    """Run inference command strictly as unprivileged user (nobody).

    No root fallback is permitted to ensure security and isolation.
    """
    seen = set()
    interpreters = []
    for interp in [cmd[0], sys.executable, "python3", "python"]:
        if interp and interp not in seen:
            seen.add(interp)
            interpreters.append(interp)

    for interp in interpreters:
        try:
            c = list(cmd)
            c[0] = interp
            subprocess.run(c, check=True, capture_output=True, **sub_kwargs)
            return True
        except Exception:  # noqa: BLE001, S110
            pass

    return False


def _has_tensor_leaves(obj, depth=0):
    """Recursively check for floating-point PyTorch tensors and count parameters."""
    if depth > 5:
        return 0
    param_count = 0
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return obj.numel()
    if isinstance(obj, dict):
        for v in obj.values():
            param_count += _has_tensor_leaves(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            param_count += _has_tensor_leaves(v, depth + 1)
    return param_count


def evaluate(context=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    results = {}
    metrics = {"mcc": 0.0, "auc_pr": 0.0, "worst_channel_f1": 0.0}

    # Resolve app_dir
    app_dir = os.environ.get("APP_DIR")
    if not app_dir:
        if context and "workdir" in context:
            app_dir = str(context["workdir"])
        elif os.path.exists("/app"):
            app_dir = "/app"
        else:
            app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    artifacts_dir = os.path.join(app_dir, "artifacts")

    # 1. Check model artifact existence (strictly PyTorch deep neural network model, >= 1KB)
    model_artifact_found = False
    valid_exts = ('.pth', '.pt', '.bin')
    if os.path.isdir(artifacts_dir):
        for fname in sorted(os.listdir(artifacts_dir)):
            if fname in ('inference.py', 'metrics.json', 'hidden_preds.csv'):
                continue
            fpath = os.path.join(artifacts_dir, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) >= 1024 and any(fname.endswith(ext) for ext in valid_exts):
                try:
                    loaded = torch.load(fpath, map_location='cpu', weights_only=False)
                    total_params = 0
                    if isinstance(loaded, torch.nn.Module):
                        total_params = sum(p.numel() for p in loaded.parameters() if p.is_floating_point())
                    elif isinstance(loaded, dict):
                        total_params = _has_tensor_leaves(loaded)
                    if total_params >= 1000:
                        model_artifact_found = True
                        break
                except Exception:  # noqa: BLE001, S110
                    pass
    results['model_artifact_exists'] = model_artifact_found

    # 2. Check inference script existence, GPU/CUDA specification, and data integrity
    inference_path = os.path.join(artifacts_dir, "inference.py")
    inference_file_ok = os.path.isfile(inference_path) and os.path.getsize(inference_path) > 0

    torch_imported = False
    cuda_specified = False
    if inference_file_ok:
        try:
            with open(inference_path, "r", encoding="utf-8", errors="ignore") as f:
                inf_src = f.read()
            torch_imported = ("import torch" in inf_src) or ("from torch" in inf_src)
            cuda_specified = ("cuda" in inf_src) or ("torch.cuda" in inf_src)
        except Exception:  # noqa: BLE001, S110
            pass

    # Integrity verification: dataset.csv must match original SHA256
    dataset_intact = False
    dataset_candidates = [
        os.path.join(app_dir, "task_inputs", "dataset.csv"),
        os.path.join(app_dir, "environment", "task_inputs", "dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "environment", "task_inputs", "dataset.csv"),
    ]
    for dpath in dataset_candidates:
        if os.path.isfile(dpath):
            try:
                with open(dpath, "rb") as df_in:
                    fhash = hashlib.sha256(df_in.read()).hexdigest()
                if fhash == EXPECTED_DATASET_SHA256:
                    dataset_intact = True
                    break
            except Exception:  # noqa: BLE001, S110
                pass

    # Filesystem scan: disallow forbidden external data/label files in /app/artifacts
    clean_artifacts = True
    if os.path.isdir(artifacts_dir):
        for fname in os.listdir(artifacts_dir):
            if fname in ('inference.py', 'metrics.json', 'hidden_preds.csv', 'probe_preds.csv'):
                continue
            if not any(fname.endswith(ext) for ext in valid_exts):
                clean_artifacts = False
                break

    results['inference_exists'] = bool(
        inference_file_ok
        and model_artifact_found
        and torch_imported
        and cuda_specified
        and dataset_intact
        and clean_artifacts
    )

    if not (inference_file_ok and model_artifact_found):
        return _pack_results(results, metrics)

    # 3. Check metrics.json existence and content (hardened: valid bounds, realistic non-fabricated values)
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    metrics_json_ok = False
    if os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0:
        try:
            with open(metrics_path, 'r', encoding="utf-8") as f:
                reported = json.load(f)
            required_keys = {"mcc", "auc_pr", "worst_channel_f1"}
            if isinstance(reported, dict) and required_keys.issubset(reported.keys()):
                valid_values = True
                for k in required_keys:
                    val = reported[k]
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                        valid_values = False
                        break
                    if k == 'mcc' and not (0.10 <= val <= 0.98):
                        valid_values = False
                        break
                    if k == 'auc_pr' and not (0.10 <= val <= 0.995):
                        valid_values = False
                        break
                    if k == 'worst_channel_f1' and not (0.10 <= val <= 0.98):
                        valid_values = False
                        break
                # Minimum channel F1 cannot exceed overall AUC-PR + 0.15 or MCC + 0.25
                if valid_values:
                    if float(reported['worst_channel_f1']) > float(reported['auc_pr']) + 0.15:
                        valid_values = False
                    if float(reported['worst_channel_f1']) > float(reported['mcc']) + 0.25:
                        valid_values = False
                metrics_json_ok = valid_values
        except Exception:  # noqa: BLE001
            metrics_json_ok = False
    results['metrics_json_exists'] = metrics_json_ok

    # 4. Locate test data strictly from real tests directory
    if context and "tests_dir" in context:
        tests_dir = str(context["tests_dir"])
    else:
        tests_dir = os.environ.get("OTTER_TESTS_DIR", "/tests")
        if not os.path.exists(tests_dir):
            tests_dir = os.path.dirname(__file__)

    hidden_test_path = os.path.join(tests_dir, "holdout_data", "hidden_test.csv")
    in_domain_test_path = os.path.join(tests_dir, "holdout_data", "in_domain_test.csv")

    h_df = pd.read_csv(hidden_test_path)
    if os.path.exists(in_domain_test_path):
        id_df = pd.read_csv(in_domain_test_path)
        gt_df = pd.concat([h_df, id_df], ignore_index=True)
    else:
        gt_df = h_df

    # Strictly lock test data directory and ground-truth files
    tests_data_dir = os.path.join(tests_dir, "holdout_data")
    if os.path.exists(tests_data_dir):
        try:
            os.chmod(tests_data_dir, 0o700)
        except OSError:
            pass

    for gt_file in (hidden_test_path, in_domain_test_path):
        if os.path.exists(gt_file):
            try:
                os.chmod(gt_file, 0o600)
            except OSError:
                pass

    # Secure GPU device access in Linux for unprivileged inference
    if os.name != "nt":
        for dev_pattern in ("/dev/nvidia*", "/dev/dri/*", "/dev/nvhost*"):
            for p in glob.glob(dev_pattern):
                try:
                    os.chmod(p, 0o666)
                except OSError:
                    pass

    # Secure artifacts directory permissions
    try:
        os.chmod(artifacts_dir, 0o755)
        for fname in os.listdir(artifacts_dir):
            fpath = os.path.join(artifacts_dir, fname)
            if os.path.isfile(fpath):
                os.chmod(fpath, 0o755 if fname == 'inference.py' else 0o644)
    except OSError:
        pass

    # Dedicated evaluation directory (resolved from harness context or OUTPUT_DIR)
    if context and "output_dir" in context:
        eval_dir = str(context["output_dir"])
    else:
        eval_dir = os.environ.get("OUTPUT_DIR", os.path.join(app_dir, "outputs"))
    os.makedirs(eval_dir, exist_ok=True)
    try:
        os.chmod(eval_dir, 0o777)
    except OSError:
        pass

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["HOME"] = os.environ.get("HOME", "/tmp")
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = ""

    # Network block: isolate network access during inference
    env["http_proxy"] = "http://127.0.0.1:0"
    env["https_proxy"] = "http://127.0.0.1:0"
    env["all_proxy"] = "socks5://127.0.0.1:0"
    env["no_proxy"] = ""
    env["REQUESTS_CA_BUNDLE"] = "/dev/null"
    env["SSL_CERT_FILE"] = "/dev/null"

    sub_kwargs = {
        "env": env,
    }
    if os.name != "nt":
        import pwd
        try:
            nobody_pw = pwd.getpwnam("nobody")
            sub_kwargs["user"] = nobody_pw.pw_uid
            sub_kwargs["group"] = nobody_pw.pw_gid
            import grp
            extra_gids = []
            for gname in ("video", "render"):
                try:
                    grp_info = grp.getgrnam(gname)
                    extra_gids.append(grp_info.gr_gid)
                except KeyError:
                    pass
            sub_kwargs["extra_groups"] = extra_gids
        except (KeyError, AttributeError):
            pass

    # Physical hiding of holdout data during inference:
    # Temporarily move holdout_data to /root/.secure_holdout so it cannot be read
    # by unprivileged processes even if path probing is attempted.
    secure_holdout_dir = "/root/.secure_holdout"
    holdout_moved = False
    if os.name != "nt" and os.path.exists(tests_data_dir):
        try:
            os.makedirs(secure_holdout_dir, mode=0o700, exist_ok=True)
            target_holdout_dst = os.path.join(secure_holdout_dir, "holdout_data")
            if os.path.exists(target_holdout_dst):
                import shutil
                shutil.rmtree(target_holdout_dst, ignore_errors=True)
            os.rename(tests_data_dir, target_holdout_dst)
            holdout_moved = True
        except OSError:
            pass

    try:
        # 4a. Run synthetic input sensitivity probe (anti-lookup / anti-memorization guard)
        seq_cols = [f"seq_{i:03d}" for i in range(128)]
        probe_input_path = os.path.join(eval_dir, "probe_features_input.csv")
        probe_output_path = os.path.join(eval_dir, "probe_preds.csv")
        if os.path.exists(probe_output_path):
            try:
                os.remove(probe_output_path)
            except OSError:
                pass

        probe_df = pd.DataFrame([
            {'segment': 'probe_norm_01', 'channel': 'channel_probe', 'subsystem': 'EPS', **{c: 0.0 for c in seq_cols}},
            {'segment': 'probe_anom_02', 'channel': 'channel_probe', 'subsystem': 'EPS', **{c: (4.0 if i % 4 == 0 else -2.0) for i, c in enumerate(seq_cols)}}
        ])
        probe_df.to_csv(probe_input_path, index=False)
        try:
            os.chmod(probe_input_path, 0o644)
        except OSError:
            pass

        probe_cmd = [sys.executable, f"{app_dir}/artifacts/inference.py", '--input', probe_input_path, '--output', probe_output_path]
        probe_ok = _run_inference_cmd(probe_cmd, sub_kwargs)
        probe_verified = False
        if probe_ok and os.path.exists(probe_output_path):
            try:
                probe_out_df = pd.read_csv(probe_output_path)
                if (
                    'segment' in probe_out_df.columns
                    and 'prediction' in probe_out_df.columns
                    and len(probe_out_df) == 2
                    and set(probe_out_df['segment']) == {'probe_norm_01', 'probe_anom_02'}
                ):
                    p_norm = float(probe_out_df.loc[probe_out_df['segment'] == 'probe_norm_01', 'prediction'].iloc[0])
                    p_anom = float(probe_out_df.loc[probe_out_df['segment'] == 'probe_anom_02', 'prediction'].iloc[0])
                    if 0.0 <= p_norm <= 1.0 and 0.0 <= p_anom <= 1.0 and abs(p_anom - p_norm) >= 1e-6:
                        probe_verified = True
            except Exception:  # noqa: BLE001
                probe_verified = False

        # 4b. Run inference on full evaluation dataset
        eval_input_path = os.path.join(eval_dir, "eval_features_input.csv")
        eval_features_df = gt_df.drop(columns=['anomaly']) if 'anomaly' in gt_df.columns else gt_df.copy()
        eval_features_df.to_csv(eval_input_path, index=False)
        try:
            os.chmod(eval_input_path, 0o644)
        except OSError:
            pass

        preds_output_path = os.path.join(eval_dir, "hidden_preds.csv")
        if os.path.exists(preds_output_path):
            try:
                os.remove(preds_output_path)
            except OSError:
                pass

        inference_cmd = [sys.executable, f"{app_dir}/artifacts/inference.py", '--input', eval_input_path, '--output', preds_output_path]
        eval_run_ok = _run_inference_cmd(inference_cmd, sub_kwargs)

        if not (eval_run_ok and probe_verified and os.path.exists(preds_output_path)):
            results['inference_runs'] = False
            return _pack_results(results, metrics)

        results['inference_runs'] = True

        # 5. Evaluate predictions against ground truth
        try:
            preds_df = pd.read_csv(preds_output_path)
            if 'segment' not in preds_df.columns or 'prediction' not in preds_df.columns:
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            if len(preds_df) != len(gt_df):
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            # Reject duplicate segment IDs in predictions
            if preds_df['segment'].duplicated().any():
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            # Reject unknown or missing segment IDs
            if set(preds_df['segment']) != set(gt_df['segment']):
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            # Reject null or non-numeric predictions
            if preds_df['prediction'].isnull().any() or not pd.api.types.is_numeric_dtype(preds_df['prediction']):
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            # Strictly validate that continuous probability predictions are bounded in [0.0, 1.0]
            if (preds_df['prediction'] < 0.0).any() or (preds_df['prediction'] > 1.0).any():
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            # Merge 1-to-1 by segment
            merged = pd.merge(gt_df, preds_df, on='segment', how='inner')
            if len(merged) != len(gt_df):
                results['performance_measured'] = False
                return _pack_results(results, metrics)

            y_true = merged['anomaly']
            y_pred_prob = merged['prediction']
            y_pred = (y_pred_prob > 0.5).astype(int)

            metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
            metrics['auc_pr'] = float(average_precision_score(y_true, y_pred_prob))

            # Worst channel F1 (filtered to channels with >= 20 segments and >= 5 anomalies)
            channel_f1s = []
            for channel in merged['channel'].unique():
                ch_data = merged[merged['channel'] == channel]
                pos_count = int((ch_data['anomaly'] == 1).sum())
                if len(ch_data) >= 20 and pos_count >= 5 and len(ch_data['anomaly'].unique()) > 1:
                    channel_f1s.append(f1_score(ch_data['anomaly'], (ch_data['prediction'] > 0.5).astype(int)))

            metrics['worst_channel_f1'] = float(min(channel_f1s)) if channel_f1s else 0.0

            results['performance_measured'] = True
        except Exception:  # noqa: BLE001
            results['performance_measured'] = False

        return _pack_results(results, metrics)

    finally:
        # Restore holdout data safely to original location
        if holdout_moved:
            try:
                target_holdout_src = os.path.join(secure_holdout_dir, "holdout_data")
                if os.path.exists(target_holdout_src):
                    os.rename(target_holdout_src, tests_data_dir)
            except OSError:
                pass


def _pack_results(results, metrics):
    criteria_list = [
        {"id": "model_artifact_exists", "milestone_id": "model_artifacts", "passed": bool(results.get("model_artifact_exists", False))},
        {"id": "inference_exists", "milestone_id": "model_artifacts", "passed": bool(results.get("inference_exists", False))},
        {"id": "metrics_json_exists", "milestone_id": "metrics_artifacts", "passed": bool(results.get("metrics_json_exists", False))},
        {"id": "inference_runs", "milestone_id": "inference_evaluation", "passed": bool(results.get("inference_runs", False))},
        {"id": "performance_measured", "milestone_id": "inference_evaluation", "passed": bool(results.get("performance_measured", False))}
    ]
    return {'criteria': criteria_list, 'metrics': metrics}
