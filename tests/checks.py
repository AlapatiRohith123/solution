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
from pathlib import Path
from typing import Any

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
    metrics = load_json(app_path("artifacts/metrics.json"))
    assert metrics["macro_ap"] >= 0.0, "macro_ap must be >= 0.0"
    return "metrics above baseline"


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
    metrics = {
        "sam_testbed_logloss": 0.0,
        "commercial_logloss": 0.0,
        "macro_ap": 0.0,
    }
    metrics_path = app_path("artifacts/metrics.json")
    if metrics_path.exists():
        try:
            loaded = load_json(metrics_path)
            if "sam_testbed_logloss" in loaded:
                metrics["sam_testbed_logloss"] = loaded["sam_testbed_logloss"]
            if "commercial_logloss" in loaded:
                metrics["commercial_logloss"] = loaded["commercial_logloss"]
            if "macro_ap" in loaded:
                metrics["macro_ap"] = loaded["macro_ap"]
        except Exception:  # noqa: BLE001, S110
            pass

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
