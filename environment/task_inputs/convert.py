"""Editable converter for the current byte classifier."""

from __future__ import annotations

from pathlib import Path

import torch


def convert(trainer, output_path: Path) -> None:
    """Export the live trainer with a dynamic raw-byte batch."""
    trainer.model.eval()
    torch.onnx.export(
        trainer.model,
        torch.zeros(1, 72, dtype=torch.uint8),
        output_path,
        input_names=["sample"],
        output_names=["label_scores"],
        dynamic_axes={"sample": {0: "batch"}, "label_scores": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
