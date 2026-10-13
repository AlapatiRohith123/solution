"""Task-specific verification checks for Mars Evolved-Gas Spectra Detection.

This file implements deterministic verification criteria and metric evaluations
for the Otter benchmark harness. rubric_score.py loads this file and calls
evaluate(context) exactly once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Globals & Constants
# ---------------------------------------------------------------------------

TARGET_CLASSES = [
    "basalt",
    "carbonate",
    "chloride",
    "iron_oxide",
    "oxalate",
    "oxychlorine",
    "phyllosilicate",
    "silicate",
    "sulfate",
    "sulfide",
]

_INFERENCE_CACHE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))


def app_path(*parts: str) -> Path:
    """Resolve a path relative to the agent's working directory."""
    return APP_DIR.joinpath(*parts)


def load_json(path: Path) -> dict:
    """Load and validate a JSON file."""
    if not path.exists():
        raise AssertionError(f"Missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Invalid JSON in {path}: line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc


def require_keys(obj: dict | set, keys: set[str], label: str) -> None:
    """Assert that a dict or set contains all required keys."""
    if isinstance(obj, dict):
        missing = sorted(keys - set(obj))
    elif isinstance(obj, set):
        missing = sorted(keys - obj)
    else:
        raise TypeError(f"{label} must be a dict or set; got {type(obj).__name__}")
    if missing:
        raise AssertionError(
            f"{label} missing required keys {missing}; found: {sorted(obj) if isinstance(obj, set) else sorted(obj.keys())}"
        )


# ---------------------------------------------------------------------------
# Inference & Metric Evaluation
# ---------------------------------------------------------------------------


def run_inference_and_evaluate() -> dict[str, float]:
    """Execute predict.py on hidden test data and compute primary metrics."""
    global _INFERENCE_CACHE
    if _INFERENCE_CACHE is not None:
        return _INFERENCE_CACHE

    predict_script = app_path("predict.py")
    if not predict_script.exists():
        predict_script = Path(__file__).resolve().parent.parent / "predict.py"
    assert predict_script.exists(), f"predict.py not found at {predict_script}"

    tests_dir = Path(__file__).resolve().parent
    hidden_features = Path(
        os.environ.get(
            "OTTER_TEST_FEATURES", str(tests_dir / "data" / "hidden_test_features.zip")
        )
    )
    hidden_labels_path = Path(
        os.environ.get(
            "OTTER_TEST_LABELS", str(tests_dir / "data" / "hidden_test_labels.csv")
        )
    )

    assert hidden_features.exists(), f"Hidden features zip missing: {hidden_features}"
    assert hidden_labels_path.exists(), (
        f"Hidden labels CSV missing: {hidden_labels_path}"
    )

    artifacts_dir = app_path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preds_output = artifacts_dir / "predictions.csv"

    cmd = [
        sys.executable,
        str(predict_script),
        "--test_features",
        str(hidden_features),
        "--predictions_output",
        str(preds_output),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("predict.py timed out after 120 seconds") from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"predict.py failed with exit code {exc.returncode}.\nStderr: {exc.stderr}\nStdout: {exc.stdout}"
        ) from exc

    assert preds_output.exists(), "predict.py did not produce predictions.csv"
    assert preds_output.stat().st_size > 0, "predictions.csv is empty"

    preds = pd.read_csv(preds_output)
    true_labels = pd.read_csv(hidden_labels_path)

    assert "sample_id" in preds.columns, (
        "predictions.csv must have a 'sample_id' column"
    )

    preds = preds.set_index("sample_id").sort_index()
    true_labels = true_labels.set_index("sample_id").sort_index()

    # Align sample IDs: predictions must cover all hidden test samples
    missing_samples = set(true_labels.index) - set(preds.index)
    assert len(missing_samples) == 0, (
        f"predictions.csv is missing hidden test samples: {missing_samples}"
    )

    # Align columns: if target classes are present by name, align them; else require 10 numeric columns
    pred_class_cols = [c for c in preds.columns if c in TARGET_CLASSES]
    if len(pred_class_cols) == len(TARGET_CLASSES):
        preds = preds[TARGET_CLASSES]
    else:
        assert preds.shape[1] == len(TARGET_CLASSES), (
            f"Expected {len(TARGET_CLASSES)} prediction columns, got {preds.shape[1]}"
        )

    pred_values = preds.values.astype(np.float64)
    assert not np.isnan(pred_values).any(), "predictions.csv contains NaN values"
    assert (pred_values >= 0.0).all() and (pred_values <= 1.0).all(), (
        "Predictions must be valid probabilities in [0.0, 1.0]"
    )

    sam_idx = true_labels["instrument_type"] == "sam_testbed"
    com_idx = true_labels["instrument_type"] == "commercial"

    pred_sam = preds.loc[sam_idx].values
    true_sam = (
        true_labels.loc[sam_idx].drop("instrument_type", axis=1)[TARGET_CLASSES].values
    )
    pred_com = preds.loc[com_idx].values
    true_com = (
        true_labels.loc[com_idx].drop("instrument_type", axis=1)[TARGET_CLASSES].values
    )

    pred_sam = np.clip(pred_sam, 1e-15, 1 - 1e-15)
    pred_com = np.clip(pred_com, 1e-15, 1 - 1e-15)

    def compute_bce(y_t, y_p):
        if len(y_t) == 0:
            return 0.5
        y_p = np.clip(y_p.flatten().astype(np.float64), 1e-15, 1.0 - 1e-15)
        y_true_flat = y_t.flatten().astype(np.float64)
        return float(
            -np.mean(
                y_true_flat * np.log(y_p) + (1.0 - y_true_flat) * np.log(1.0 - y_p)
            )
        )

    sam_loss = compute_bce(true_sam, pred_sam)
    com_loss = compute_bce(true_com, pred_com)

    y_true = true_labels.drop("instrument_type", axis=1)[TARGET_CLASSES].values
    y_pred = preds.values
    aps = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    macro_ap = np.mean(aps) if aps else 0.0

    _INFERENCE_CACHE = {
        "sam_testbed_logloss": float(sam_loss),
        "commercial_logloss": float(com_loss),
        "macro_ap": float(macro_ap),
    }
    return _INFERENCE_CACHE


# ---------------------------------------------------------------------------
# Milestone 1: Data Preparation
# ---------------------------------------------------------------------------


def check_data_deliverables() -> str:
    """Check that required data-stage files exist and are non-empty."""
    split_path = app_path("artifacts/proxy_val_split.csv")
    summary_path = app_path("artifacts/data_summary.json")

    for path in [split_path, summary_path]:
        assert path.exists(), f"Missing required data artifact: {path}"
        assert path.stat().st_size > 0, f"Data artifact is empty: {path}"

    return "data deliverables exist"


def check_data_schema() -> str:
    """Validate structure and distribution-shift handling of data artifacts."""
    split_path = app_path("artifacts/proxy_val_split.csv")
    df = pd.read_csv(split_path)

    # Check required columns: sample_id, split, instrument_type
    require_keys(
        set(df.columns),
        {"sample_id", "split", "instrument_type"},
        "proxy_val_split.csv",
    )
    assert len(df) >= 20, f"Proxy validation split has too few samples ({len(df)} < 20)"

    # Check valid split values
    splits = set(df["split"].unique())
    assert {"train", "val"}.issubset(splits), (
        f"proxy_val_split.csv must contain both 'train' and 'val' splits; found: {splits}"
    )

    # Group-disjoint requirement: no sample in both train and val
    train_samples = set(df[df["split"] == "train"]["sample_id"])
    val_samples = set(df[df["split"] == "val"]["sample_id"])
    overlap = train_samples.intersection(val_samples)
    assert len(overlap) == 0, (
        f"Train and val splits must be strictly disjoint; found overlap: {overlap}"
    )

    # Check samples belong to known metadata (no synthetic or hallucinated IDs)
    meta_path = Path(os.environ.get("OTTER_DATA_DIR", "/app/data")) / "metadata.csv"
    if not meta_path.exists():
        meta_path = (
            Path(__file__).resolve().parent.parent
            / "environment"
            / "task_inputs"
            / "metadata.csv"
        )

    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        known_ids = set(meta_df["sample_id"])
        unknown = set(df["sample_id"]) - known_ids
        assert len(unknown) == 0, (
            f"proxy_val_split.csv contains unknown sample IDs not in metadata: {unknown}"
        )

        # Verify instrument assignments match known metadata
        meta_inst = dict(zip(meta_df["sample_id"], meta_df["instrument_type"]))
        for _, row in df.iterrows():
            expected_inst = meta_inst.get(row["sample_id"])
            if expected_inst is not None:
                assert row["instrument_type"] == expected_inst, (
                    f"Sample {row['sample_id']} instrument_type mismatch: expected {expected_inst}, got {row['instrument_type']}"
                )

    # Instrument distribution shift requirement:
    # Both commercial and sam_testbed must be covered in both train and val
    instruments_present = set(df["instrument_type"].unique())
    assert {"commercial", "sam_testbed"}.issubset(instruments_present), (
        f"proxy_val_split.csv must cover both 'commercial' and 'sam_testbed'; found: {instruments_present}"
    )

    train_instruments = set(df[df["split"] == "train"]["instrument_type"].unique())
    val_instruments = set(df[df["split"] == "val"]["instrument_type"].unique())
    assert {"commercial", "sam_testbed"}.issubset(train_instruments), (
        f"Train split must cover both instruments to support domain adaptation; found: {train_instruments}"
    )
    assert {"commercial", "sam_testbed"}.issubset(val_instruments), (
        f"Val split must cover both instruments to faithfully evaluate distribution shift; found: {val_instruments}"
    )

    # Check data_summary.json schema and exact values
    summary = load_json(app_path("artifacts/data_summary.json"))
    require_keys(
        summary,
        {"sample_count", "feature_dim", "instruments_covered"},
        "data_summary.json",
    )
    if meta_path.exists():
        assert summary["sample_count"] == len(meta_df), (
            f"data_summary.json sample_count ({summary['sample_count']}) must match metadata sample count ({len(meta_df)})"
        )
    else:
        assert summary["sample_count"] >= 20, "sample_count must be at least 20"

    assert 10 <= summary["feature_dim"] <= 2000, (
        f"feature_dim ({summary['feature_dim']}) must be in plausible range [10, 2000]"
    )
    assert isinstance(summary["instruments_covered"], list), (
        "instruments_covered must be a list"
    )
    assert set(summary["instruments_covered"]) == {"commercial", "sam_testbed"}, (
        f"instruments_covered must contain both 'commercial' and 'sam_testbed'; found: {summary['instruments_covered']}"
    )

    return "data schema and group-disjoint proxy validation split valid"


# ---------------------------------------------------------------------------
# Milestone 2: Model Training and Evaluation
# ---------------------------------------------------------------------------


def check_model_deliverables() -> str:
    """Check that model deliverables exist and on-disk model artifact is a valid, trained model."""
    predict_script = app_path("predict.py")
    metrics_file = app_path("artifacts/metrics.json")
    for path in [predict_script, metrics_file]:
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"File is empty: {path}"

    # Verify model artifact exists on disk
    model_candidates = [
        app_path("models/best.pt"),
        app_path("models/model.pt"),
        app_path("models/model.pth"),
        app_path("artifacts/best.pt"),
        app_path("artifacts/model.pt"),
        app_path("artifacts/model.pth"),
        app_path("models/model.pkl"),
        app_path("artifacts/model.pkl"),
        app_path("models/model.joblib"),
        app_path("artifacts/model.joblib"),
    ]

    models_dir = app_path("models")
    if models_dir.exists():
        model_candidates.extend(list(models_dir.glob("*.*")))
    artifacts_dir = app_path("artifacts")
    if artifacts_dir.exists():
        model_candidates.extend(list(artifacts_dir.glob("*.pt")))
        model_candidates.extend(list(artifacts_dir.glob("*.pkl")))
        model_candidates.extend(list(artifacts_dir.glob("*.joblib")))

    found_models = [
        p for p in set(model_candidates) if p.exists() and p.stat().st_size >= 1024
    ]
    assert len(found_models) > 0, (
        "No valid model artifact found (expected >= 1KB file in /app/models/ or /app/artifacts/)"
    )

    # Resilient loader: try PyTorch, joblib, and pickle for any valid checkpoint candidate
    loaded_obj = None
    loaded_type = None

    for m in found_models:
        # Try torch.load
        try:
            import torch

            obj = torch.load(m, map_location="cpu", weights_only=False)
            loaded_obj = obj
            loaded_type = "torch"
            break
        except Exception:  # noqa: BLE001, S110
            pass

        # Try joblib
        try:
            import joblib

            obj = joblib.load(m)
            loaded_obj = obj
            loaded_type = "joblib"
            break
        except Exception:  # noqa: BLE001, S110
            pass

        # Try pickle
        try:
            import pickle

            with open(m, "rb") as f:
                obj = pickle.load(f)
            loaded_obj = obj
            loaded_type = "pickle"
            break
        except Exception:  # noqa: BLE001, S110
            pass

    assert loaded_obj is not None, (
        f"Found model artifact(s) {[str(m) for m in found_models]} but none could be loaded via PyTorch, joblib, or pickle"
    )

    # Substantive model structure verification: ensure it is a real model trained for 10 classes
    if loaded_type == "torch":
        import torch

        if isinstance(loaded_obj, torch.nn.Module):
            total_params = sum(p.numel() for p in loaded_obj.parameters())
            has_nonzero = any(torch.any(p != 0) for p in loaded_obj.parameters())
            has_10_output = any(
                p.ndim > 0
                and (
                    p.shape[0] == len(TARGET_CLASSES)
                    or p.shape[-1] == len(TARGET_CLASSES)
                )
                for p in loaded_obj.parameters()
            )
        elif isinstance(loaded_obj, dict):
            # state_dict
            total_params = sum(
                v.numel() for v in loaded_obj.values() if isinstance(v, torch.Tensor)
            )
            has_nonzero = any(
                torch.any(v != 0)
                for v in loaded_obj.values()
                if isinstance(v, torch.Tensor)
            )
            has_10_output = any(
                isinstance(v, torch.Tensor)
                and v.ndim > 0
                and (
                    v.shape[0] == len(TARGET_CLASSES)
                    or v.shape[-1] == len(TARGET_CLASSES)
                )
                for v in loaded_obj.values()
            )
        else:
            total_params = 0
            has_nonzero = False
            has_10_output = False

        assert total_params >= 1000, (
            f"Model has insufficient parameters ({total_params} < 1000); trivial dummy rejected"
        )
        assert has_nonzero, "Model weights must not be all zeros"
        assert has_10_output, (
            f"Model output dimension must match {len(TARGET_CLASSES)} target classes"
        )

    elif loaded_type in ["joblib", "pickle"]:
        # Verify scikit-learn / fitted pipeline
        is_fitted = False
        if hasattr(loaded_obj, "predict") or hasattr(loaded_obj, "predict_proba"):
            is_fitted = True
        elif isinstance(loaded_obj, dict) and len(loaded_obj) >= len(TARGET_CLASSES):
            # Dictionary of per-class estimators
            is_fitted = all(
                hasattr(est, "predict") or hasattr(est, "predict_proba")
                for est in loaded_obj.values()
            )

        assert is_fitted, (
            "Loaded artifact is not a valid fitted estimator/pipeline with inference capability"
        )

    return "model deliverables exist and model artifact is a substantive, trained model"


def check_metrics_schema() -> str:
    """Validate metrics structure and value ranges."""
    metrics = load_json(app_path("artifacts/metrics.json"))
    require_keys(
        metrics,
        {"sam_testbed_logloss", "commercial_logloss", "macro_ap"},
        "metrics.json",
    )
    for k in ["sam_testbed_logloss", "commercial_logloss", "macro_ap"]:
        val = metrics[k]
        assert isinstance(val, (int, float)), (
            f"Metric {k} must be numeric, got {type(val)}"
        )
        assert not np.isnan(val) and not np.isinf(val), f"Metric {k} must be finite"

    # Plausible non-trivial metric values
    assert 0.0 < metrics["macro_ap"] <= 1.0, (
        f"macro_ap must be in (0, 1], got {metrics['macro_ap']}"
    )
    assert 0.0 < metrics["sam_testbed_logloss"] < 2.0, (
        f"sam_testbed_logloss must be plausible (0, 2), got {metrics['sam_testbed_logloss']}"
    )
    assert 0.0 < metrics["commercial_logloss"] < 2.0, (
        f"commercial_logloss must be plausible (0, 2), got {metrics['commercial_logloss']}"
    )
    return "metrics schema valid"


def check_metrics_thresholds() -> str:
    """Validate model inference runs successfully and meets quality thresholds."""
    try:
        metrics = run_inference_and_evaluate()
    except Exception as e:
        raise AssertionError(f"Inference execution failed: {e}") from e

    preds_output = app_path("artifacts/predictions.csv")
    preds = pd.read_csv(preds_output).set_index("sample_id")
    numeric_preds = preds.select_dtypes(include=[np.number])

    # Reject trivial / constant baselines: standard deviation across samples must exceed 1e-4
    assert (numeric_preds.std(axis=0) > 1e-4).all(), (
        "Predictions must not be constant across samples; non-trivial predictions required for all classes"
    )

    # Real quality thresholds on hidden distribution-shifted testbed
    assert metrics["sam_testbed_logloss"] <= 0.64, (
        f"sam_testbed_logloss ({metrics['sam_testbed_logloss']:.4f}) must be <= 0.64"
    )
    assert metrics["commercial_logloss"] <= 0.63, (
        f"commercial_logloss ({metrics['commercial_logloss']:.4f}) must be <= 0.63"
    )
    assert metrics["macro_ap"] >= 0.50, (
        f"macro_ap ({metrics['macro_ap']:.4f}) must be >= 0.50"
    )

    return "inference successful and metrics meet quality thresholds"


# ---------------------------------------------------------------------------
# Milestone 3: Deliverables and Report
# ---------------------------------------------------------------------------


def check_final_deliverables() -> str:
    """Check that final output files exist and claims.json is structurally valid and traceable."""
    for path in [
        app_path("artifacts/claims.json"),
        app_path("artifacts/report.md"),
    ]:
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"File is empty: {path}"

    claims = load_json(app_path("artifacts/claims.json"))
    metrics_path = app_path("artifacts/metrics.json")
    assert metrics_path.exists(), "metrics.json missing for claims verification"
    metrics = load_json(metrics_path)

    if "metrics" in claims:
        claim_metrics = claims["metrics"]
    else:
        claim_metrics = claims

    require_keys(
        claim_metrics,
        {"sam_testbed_logloss", "commercial_logloss", "macro_ap"},
        "claims.json metrics",
    )

    for metric_key in ["sam_testbed_logloss", "commercial_logloss", "macro_ap"]:
        claim_val = float(claim_metrics[metric_key])
        reported_val = float(metrics[metric_key])
        assert abs(claim_val - reported_val) < 1e-3, (
            f"Claims mismatch for {metric_key}: claims.json has {claim_val}, "
            f"but metrics.json has {reported_val}"
        )

    require_keys(claims, {"traceability"}, "claims.json")
    traceability = claims["traceability"]
    assert isinstance(traceability, dict), "claims.json traceability must be an object"
    require_keys(
        traceability,
        {"model_checkpoint", "validation_split"},
        "claims.json traceability",
    )

    model_ref = (
        app_path(traceability["model_checkpoint"])
        if not Path(traceability["model_checkpoint"]).is_absolute()
        else Path(traceability["model_checkpoint"])
    )
    assert model_ref.exists(), (
        f"Traceability checkpoint {traceability['model_checkpoint']} not found on disk"
    )

    split_ref = (
        app_path(traceability["validation_split"])
        if not Path(traceability["validation_split"]).is_absolute()
        else Path(traceability["validation_split"])
    )
    assert split_ref.exists(), (
        f"Traceability split {traceability['validation_split']} not found on disk"
    )

    return "final deliverables and traceable claims valid"


def check_report_content() -> str:
    """Verify report discusses required topics with substantive depth and structure."""
    raw_report = app_path("artifacts/report.md").read_text(encoding="utf-8")
    report = raw_report.lower()
    words = report.split()
    assert len(words) >= 250, f"Report must be at least 250 words; found {len(words)}"

    unique_words = set(words)
    assert len(unique_words) >= 100, (
        f"Report has insufficient vocabulary diversity ({len(unique_words)} unique words < 100)"
    )

    # Must contain at least 3 markdown section headings
    heading_count = (
        raw_report.count("\n# ")
        + raw_report.count("\n## ")
        + raw_report.count("\n### ")
    )
    if raw_report.startswith("#"):
        heading_count += 1
    assert heading_count >= 3, (
        f"Report must contain at least 3 markdown section headings; found {heading_count}"
    )

    # 1. Approach & Model Architecture (require at least 2 distinct keywords)
    model_keywords = [
        "model",
        "architecture",
        "mlp",
        "neural",
        "classifier",
        "network",
        "layer",
        "linear",
        "logistic",
        "optimizer",
        "loss",
        "parameters",
        "regularization",
        "dropout",
    ]
    model_matches = [k for k in model_keywords if k in report]
    assert len(model_matches) >= 2, (
        f"Report must discuss modeling approach with sufficient depth (matched keywords: {model_matches})"
    )

    # 2. Preprocessing & Feature Extraction (require at least 2 distinct keywords)
    feature_keywords = [
        "feature",
        "preprocess",
        "m/z",
        "mz",
        "abundance",
        "spectrum",
        "spectra",
        "binning",
        "temperature",
        "normalization",
        "scaling",
        "ramp",
    ]
    feature_matches = [k for k in feature_keywords if k in report]
    assert len(feature_matches) >= 2, (
        f"Report must discuss preprocessing and mass spectrometry features in depth (matched: {feature_matches})"
    )

    # 3. Instrument Shift & Proxy Validation (require at least 2 distinct keywords)
    shift_keywords = [
        "shift",
        "instrument",
        "sam",
        "commercial",
        "testbed",
        "domain",
        "proxy",
        "distribution",
        "disjoint",
        "stratified",
    ]
    shift_matches = [k for k in shift_keywords if k in report]
    assert len(shift_matches) >= 2, (
        f"Report must discuss instrument distribution shift and proxy validation (matched: {shift_matches})"
    )

    # 4. Evaluation & Metrics (require at least 2 distinct keywords)
    metrics_keywords = [
        "loss",
        "logloss",
        "metric",
        "ap",
        "precision",
        "macro",
        "validation",
        "evaluation",
        "results",
        "score",
        "threshold",
    ]
    metrics_matches = [k for k in metrics_keywords if k in report]
    assert len(metrics_matches) >= 2, (
        f"Report must discuss evaluation results and metrics in depth (matched: {metrics_matches})"
    )

    return "report content valid, substantive, and covers all required topics"


# ---------------------------------------------------------------------------
# Evaluator Dispatch
# ---------------------------------------------------------------------------


def _criterion(id: str, fn, milestone_id: str = "final") -> dict[str, Any]:
    try:
        detail = fn()
        return {
            "id": id,
            "passed": True,
            "detail": detail or "passed",
            "milestone_id": milestone_id,
        }
    except Exception:  # noqa: BLE001
        return {
            "id": id,
            "passed": False,
            "detail": traceback.format_exc(limit=6),
            "milestone_id": milestone_id,
        }


def evaluate(context: dict | None = None) -> dict:
    try:
        metrics = run_inference_and_evaluate()
    except Exception:  # noqa: BLE001
        metrics = {
            "sam_testbed_logloss": 1.0,
            "commercial_logloss": 1.0,
            "macro_ap": 0.0,
        }

    criteria = [
        _criterion(
            "data_deliverables",
            check_data_deliverables,
            milestone_id="data-preparation",
        ),
        _criterion("data_schema", check_data_schema, milestone_id="data-preparation"),
        _criterion(
            "model_deliverables",
            check_model_deliverables,
            milestone_id="model-evaluation",
        ),
        _criterion(
            "metrics_schema", check_metrics_schema, milestone_id="model-evaluation"
        ),
        _criterion(
            "metrics_thresholds",
            check_metrics_thresholds,
            milestone_id="model-evaluation",
        ),
        _criterion(
            "final_deliverables",
            check_final_deliverables,
            milestone_id="deliverables-and-report",
        ),
        _criterion(
            "report_content",
            check_report_content,
            milestone_id="deliverables-and-report",
        ),
    ]

    return {"criteria": criteria, "metrics": metrics}
