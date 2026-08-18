"""ONNX exporter for GRU-based text classifier."""

import torch
from pathlib import Path

def convert(trainer, output_path: Path) -> None:
    """Export the GRU model to ONNX format."""
    trainer.model.eval()
    
    # Create dummy input of shape (1, 72) with raw bytes
    device = next(trainer.model.parameters()).device
    dummy_input = torch.zeros((1, 72), dtype=torch.uint8, device=device)
    
    torch.onnx.export(
        trainer.model,
        dummy_input,
        output_path,
        input_names=["text_bytes"],
        output_names=["class_logits"],
        dynamic_axes={
            "text_bytes": {0: "batch_size"},
            "class_logits": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    # Inline any external data created by newer PyTorch exporters
    import onnx
    import os
    if os.path.exists(str(output_path)):
        model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(model, str(output_path), save_as_external_data=False)
        # Remove the external data file if it exists to keep directory clean
        ext_data_path = str(output_path) + ".data"
        if os.path.exists(ext_data_path):
            os.remove(ext_data_path)
