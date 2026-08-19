import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "solution"))
from train import Trainer, SequenceCNN

trainer = Trainer(num_labels=28, train_steps=100)
samples = torch.randint(0, 256, (256, 72), dtype=torch.uint8)
labels = torch.randint(0, 28, (256,), dtype=torch.long)

try:
    loss = trainer.step(samples, labels)
    print("Training step success! Loss:", loss)
    
    pred = trainer.predict(samples)
    print("Predict success! Shape:", pred.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
