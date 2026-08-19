import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "solution"))
from train import Trainer
from convert import convert
from pathlib import Path

trainer = Trainer(num_labels=5, train_steps=100)
output_path = Path("test_model.onnx")
try:
    convert(trainer, output_path)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
