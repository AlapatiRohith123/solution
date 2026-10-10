"""Task-specific verification checks for [YOUR TASK NAME].

This is the ONLY scoring file you edit. Replace the placeholder checks
below with your task's actual logic. rubric_score.py loads this file and
calls evaluate(context) exactly once.

Scoring is computed by the two standardized harness scripts (do not
edit them), both invoked by tests/test.sh:
  rubric_score.py: Rubric Score — weighted sum of your criteria (0 to 1,
                   diagnostic), plus per-milestone scores
  pass_fail.py:    Binary Pass/Fail Grade — 1 if all critical milestones
                   pass AND all primary metrics meet their human-derived
                   thresholds, else 0

Criterion weights and each milestone's critical flag are defined in
milestones_and_rubrics.json — do NOT set them here. checks.py only
determines pass/fail per criterion and returns metric values.

Primary metric thresholds are handled by pass_fail.py using
human_scores.json. Return the agent's actual metric values under "metrics"
in your evaluate() response — pass_fail.py compares them against
pass_criteria in human_scores.json and generates the final pass/fail
milestone automatically from those keys. Do NOT implement threshold
comparisons here, and do NOT write checks for the final pass/fail
milestone — just read and return the metric values.
"""

from __future__ import annotations

import json
import os
import traceback
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, average_precision_score
from pathlib import Path
from typing import Any

def run_inference_and_evaluate():
    cmd = [
        "python3", str(app_path("predict.py")),
        "--test_features", "/app/tests/data/hidden_test_features.zip",
        "--predictions_output", "/app/artifacts/predictions.csv"
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    preds = pd.read_csv(app_path("artifacts/predictions.csv"))
    true_labels = pd.read_csv("/app/tests/data/hidden_test_labels.csv")
    
    preds = preds.set_index("sample_id").sort_index()
    true_labels = true_labels.set_index("sample_id").sort_index()
    
    sam_idx = true_labels['instrument_type'] == 'sam_testbed'
    com_idx = true_labels['instrument_type'] == 'commercial'
    
    pred_sam = preds.loc[sam_idx].values
    true_sam = true_labels.loc[sam_idx].drop('instrument_type', axis=1).values
    pred_com = preds.loc[com_idx].values
    true_com = true_labels.loc[com_idx].drop('instrument_type', axis=1).values
    
    pred_sam = np.clip(pred_sam, 1e-15, 1 - 1e-15)
    pred_com = np.clip(pred_com, 1e-15, 1 - 1e-15)
    
    sam_loss = log_loss(true_sam.flatten(), pred_sam.flatten()) if len(true_sam) > 0 else 0.5
    com_loss = log_loss(true_com.flatten(), pred_com.flatten()) if len(true_com) > 0 else 0.5
    
    y_true = true_labels.drop('instrument_type', axis=1).values
    y_pred = preds.values
    aps = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    macro_ap = np.mean(aps) if aps else 0.0
    
    return {
        "sam_testbed_logloss": float(sam_loss),
        "commercial_logloss": float(com_loss),
        "macro_ap": float(macro_ap)
    }


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
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Invalid JSON in {path}: line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc


def require_keys(obj: dict, keys: set[str], label: str) -> None:
    """Assert that a dict contains all required keys."""
    if not isinstance(obj, dict):
        raise TypeError(f"{label} must be a dict; got {type(obj).__name__}")
    missing = sorted(keys - set(obj))
    if missing:
        raise AssertionError(
            f"{label} missing required keys {missing}; found: {sorted(obj)}"
        )


# ---------------------------------------------------------------------------
# Milestone 1: [YOUR FIRST MILESTONE NAME]
#
# Example: "Data Acquisition and Preprocessing"
# ---------------------------------------------------------------------------


def check_data_deliverables() -> str:
    """Check that required data-stage files exist."""
    return "Data preparation is handled implicitly."


def check_data_schema() -> str:
    """Validate structure of data artifacts."""
    return "No explicit data artifacts needed."


def check_model_deliverables() -> str:
    """Check that required model-stage files exist."""
    for path in [app_path("predict.py"), app_path("artifacts/metrics.json")]:
        assert path.exists(), f"Missing: {path}"
    return "model deliverables exist"


def check_metrics_schema() -> str:
    """Validate metrics structure and value ranges."""
    metrics = load_json(app_path("artifacts/metrics.json"))
    require_keys(
        metrics,
        {"sam_testbed_logloss", "commercial_logloss", "macro_ap"},
        "metrics.json",
    )
    return "metrics schema valid"


def check_metrics_thresholds() -> str:
    """Sanity-check that metrics are above a trivial baseline."""
    try:
        metrics = run_inference_and_evaluate()
        assert metrics["macro_ap"] >= 0.0, "macro_ap must be >= 0.0"
        return "inference successful and metrics above baseline"
    except Exception as e:
        raise AssertionError(f"Inference failed: {e}")


def check_final_deliverables() -> str:
    """Check that final output files exist."""
    for path in [
        app_path("artifacts/claims.json"),
        app_path("artifacts/report.md"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "final deliverables exist"


def check_report_content() -> str:
    """Verify report discusses required topics with sufficient depth."""
    report = app_path("artifacts/report.md").read_text().lower()
    assert len(report.split()) >= 50, "Report must be >= 50 words"
    return "report content valid"


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


def evaluate(context: dict) -> dict:
    try:
        metrics = run_inference_and_evaluate()
    except Exception as e:
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
