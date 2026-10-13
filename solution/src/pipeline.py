"""Reference pipeline for Mars Evolved-Gas Spectra Detection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add parent directory of src (solution/) to sys.path to access run.py
SOLUTION_DIR = Path(__file__).resolve().parent.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

import run


def run_full(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end training and artifact generation pipeline."""
    run.main()
    return {"status": "success"}


if __name__ == "__main__":
    run_full()
