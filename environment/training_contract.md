# Training contract

The fixed runner has this command line.

```text
python3 /app/run_training.py DATASET OUTPUT_DIR --train-steps STEPS
```

`STEPS` can range from 4 through 100,000. The output directory must start empty. The runner validates its NPZ data before importing submitted code.

The runner supplies unsigned-byte samples with shape `N x 72` and integer label indexes. It may vary the samples, candidate classes, or label indexes between fresh runs.

`/app/train.py` must define `Trainer`. Its constructor receives `num_labels` and `train_steps`. It must expose a PyTorch module as `model`. The constructor does not receive a corpus path.

`step(samples, label_indices)` receives one raw training batch. It returns one finite loss after performing an update.

`predict(samples)` receives raw samples in the same format. It returns finite scores with shape `N x num_labels`. It must not change model state.

`/app/convert.py` must define `convert(trainer, output_path)`. It serializes the current model without changing it. The exported graph accepts raw unsigned bytes. It returns one score per current label.

The graph must support variable batch sizes. It uses one input and one output. Only standard `ai.onnx` operators are allowed. External tensor files are not allowed.

The runner writes `model.onnx` and `trace.json`. It also writes five ONNX checkpoints. These are runner-owned observations.
