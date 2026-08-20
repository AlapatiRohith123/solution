import os
from pathlib import Path

import onnx
import torch


def convert(trainer, output_path: Path) -> None:
    trainer.model.eval()
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
    
    if os.path.exists(str(output_path)):
        model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(model, str(output_path), save_as_external_data=False)
        ext_data_path = str(output_path) + ".data"
        if os.path.exists(ext_data_path):
            os.remove(ext_data_path)
