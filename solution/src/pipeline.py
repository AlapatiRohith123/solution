"""Reference pipeline for [YOUR TASK NAME].

This module contains the oracle solution. solve.sh copies it into /app/src/
and run.py calls run_full() to produce all required artifacts.

The agent is expected to produce code that generates the same artifacts
(as defined in instruction.md). checks.py validates those artifacts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# ... add your imports

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
ARTIFACTS = APP_DIR / "artifacts"
REPORTS = APP_DIR / "reports"
MODELS = APP_DIR / "models"


def _resolve_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = Path(
        os.environ.get("OTTER_DATA_PATH", APP_DIR / "data" / "input_data.npz")
    )
    if env_path.exists():
        return env_path
    raise FileNotFoundError(f"Missing input data: {env_path}")


# -------------------------------------------------------------------------
# Milestone 1: Data loading and preprocessing
# -------------------------------------------------------------------------

def load_and_preprocess(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load raw data, preprocess, and write artifacts.

    Args:
        data_path: Path to input data. Falls back to OTTER_DATA_PATH env var.
        output_dir: Where to write artifacts. Falls back to APP_DIR/artifacts.
        seed: Random seed for reproducibility.

    Returns:
        Dict summarizing what was produced, e.g.:
        {"n_samples": 500, "splits": {"train": 350, "valid": 75, "test": 75}}
    """
    # data_path = _resolve_data_path(data_path)
    # out = Path(output_dir) if output_dir else ARTIFACTS
    # out.mkdir(parents=True, exist_ok=True)
    #
    # # Load, preprocess, split...
    # # Write artifacts:
    # #   out / "dataset_manifest.json"
    # #   out / "processed_data.npz"
    #
    # return {"n_samples": ..., "splits": {...}}
    raise NotImplementedError("Replace with your data loading logic")


# -------------------------------------------------------------------------
# Milestone 2: Model training and evaluation
# -------------------------------------------------------------------------

def train_and_evaluate(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train model, evaluate, and write artifacts.

    Args:
        data_path: Path to input data.
        output_dir: Where to write artifacts and model checkpoints.
        seed: Random seed for reproducibility.

    Returns:
        Dict with metrics, e.g.:
        {"auroc": 0.85, "accuracy": 0.82, "f1": 0.80}
    """
    # data_path = _resolve_data_path(data_path)
    # out = Path(output_dir) if output_dir else ARTIFACTS
    # out.mkdir(parents=True, exist_ok=True)
    #
    # # Load preprocessed data, train model, evaluate...
    # # Write artifacts:
    # #   out / "metrics.json"
    # #   out / "predictions.csv"
    # #   MODELS / "trained_model.pkl"
    #
    # return {"auroc": ..., "accuracy": ..., "f1": ...}
    raise NotImplementedError("Replace with your training logic")


# -------------------------------------------------------------------------
# Full end-to-end pipeline
# -------------------------------------------------------------------------

def run_full(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the complete pipeline: load -> preprocess -> train -> evaluate -> report.

    When called with no arguments (from run.py), uses env vars and defaults.

    Returns:
        Dict with final results and metrics.
    """
    data_path = _resolve_data_path(data_path)
    out = Path(output_dir) if output_dir else ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Data
    # data_result = load_and_preprocess(data_path, out, seed)

    # Step 2: Train + evaluate
    # model_result = train_and_evaluate(data_path, out, seed)

    # Step 3: Write metrics, final results, and report.
    # The agent must write metrics.json — checks.py reads it and passes
    # the values to evaluate.py for comparison against human thresholds.
    #
    # metrics = {"auroc": model_result["auroc"], ...}
    # (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    #
    # final = {**data_result, **model_result}
    # (out / "final_results.json").write_text(json.dumps(final, indent=2))
    #
    # REPORTS.mkdir(parents=True, exist_ok=True)
    # (REPORTS / "report.md").write_text("# Results\n...")
    #
    # return final
    raise NotImplementedError("Replace with your full pipeline logic")
