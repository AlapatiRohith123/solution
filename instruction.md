Deliver `/app/train.py`, `/app/convert.py`, and `/app/model.onnx` as a general text classification trainer.

`/app/run_training.py` supplies raw samples and label indexes. You can refer to `/app/training_contract.md` for interface details. Submitted code owns the network and training logic. Do not modify `/app/run_training.py` or `/app/data_feeder.py`. Do not modify the corpus archive.

The shipped model will be tested on labeled 72-byte dialogue lines from the plays of William Shakespeare. Each label identifies the source play. Supplied training batches contain a fixed training split. Separate dialogue lines are held out for scoring. Each sample contains one spoken line. Act and scene headers are absent. Speaker labels and stage directions are also absent. Lines are clipped to 72 ASCII bytes and shorter lines are zero-padded.

These are example lines before padding.

```text
In delivering my son from me, I bury a second husband.
And I in going, madam, weep o'er my father's death anew; but I must
attend his majesty's command, to whom I am now in ward, evermore in
```

A training run may remap label indexes. It may also replace the corpus with dialogue lines from another playwright. Each run starts from scratch and must remain effective under those changes.

A 10,000-step Shakespeare run must reach at least 4.5 times chance macro average precision. A label-reversed run must reach at least 4.5 times chance. A simple supplied problem must reach at least 10 times chance within 10,000 steps. A 100,000-step retrain must reach at least 5 times chance. A 10,000-step run on another playwright's corpus must reach at least 4.5 times chance. The shipped model must reach at least 5 times chance on held-out Shakespeare dialogue. You must ship a model trained for 100,000 steps.

Training must improve the current network. A separate full retrain must reproduce the submitted model's normalized predictions within 0.10 RMSE.

`/app/convert.py` exports the current state as one self-contained `ai.onnx` graph under 64 KiB. The graph accepts unsigned-byte batches shaped `N x 72`. It uses standard operators without external tensors.
