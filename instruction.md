# GPU PPGN Regression and Expressivity on Revised QM9

Build a reproducible one-GPU pipeline for molecular property regression with
a Provably Powerful Graph Network (PPGN), and demonstrate what the model's
quadratic order-2 operation contributes.

## Data

The frozen input directory is `/app/data`.

- `train.npz` contains 30,000 labeled molecules.
- `valid.npz` contains 3,000 labeled molecules.
- `test.npz` contains 3,000 molecules without labels.
- `dataset_manifest.json` documents the target order, units, training-set
  normalization statistics, atomic species, and deterministic bond rule.
- `checksums.json` records the frozen input hashes.

Each split contains task-local `molecule_id` values, padded atomic numbers,
Cartesian coordinates in angstrom, and an atom mask. The labeled splits also
contain four targets in this order: atomization energy, HOMO, LUMO, and dipole
moment.

The NPZ tensor contract is:

- `molecule_id`: one string per molecule;
- `atomic_numbers`: `uint8 [batch, 29]`;
- `coordinates`: `float32 [batch, 29, 3]`;
- `atom_mask`: `bool [batch, 29]`; and
- `targets`: `float32 [batch, 4]` on the labeled splits only.

The data is a deterministic subset of Revised QM9, distributed under
CC BY 4.0. Preserve the provided splits and do not attempt to recover hidden
labels from an external copy.

## Required work

1. Provide `/app/run.py`. A single invocation must remove and regenerate every
   file under `/app/models`, `/app/artifacts`, and `/app/reports`.
2. Train the primary molecular regressor on exactly one CUDA GPU. CPU fallback
   is not allowed. The model must be a PPGN-style order-2 network that:
   - operates on atom-by-atom feature tensors;
   - applies learned transformations along the feature dimension;
   - includes a learned feature-channel matrix product inside its equivariant
     blocks; and
   - uses a permutation-invariant graph readout.
3. Predict all four properties. Training may standardize targets using only the
   statistics in `dataset_manifest.json`, but exported predictions must be in
   the original units and order.
4. Train both the primary regressor and a comparable ablation for exactly 30
   epochs with batch size 128 (7,050 optimizer steps each). The ablation
   removes the quadratic matrix-product path but otherwise uses the same
   optimization budget. Evaluate both models on the frozen validation split.
5. Run a permutation-augmented expressivity experiment that distinguishes a
   six-cycle from two disjoint triangles. Compare the primary PPGN with the
   quadratic-free ablation and export both classifiers. Class `0` denotes the
   six-cycle and class `1` denotes two disjoint triangles.
6. Use training seed `1912019` and deterministic CUDA settings. Test
   predictions must contain exactly one row for every supplied test molecule.

The complete `/app/run.py` pipeline must finish within 45 minutes on the
provided L4 GPU.

## Required artifacts

Create these paths:

- `/app/models/training_checkpoint.pt`
  - a loadable PyTorch dictionary with exactly the top-level keys `state_dict`,
    `model_config`, and `training_metadata`;
  - `model_config` must contain `family="PPGN"`, `order=2`,
    `quadratic_matrix_product=true`, `width`, `blocks`, and `outputs=4`;
  - `training_metadata` must contain `seed`, `trained_on_cuda=true`,
    `cuda_device_name`, positive `optimizer_steps`, `target_names`,
    `target_mean`, and `target_std`.
- `/app/models/regressor.ts`
  - a loadable TorchScript module accepting
    `(atomic_numbers: int64 [batch,29], coordinates: float32 [batch,29,3],
    atom_mask: bool [batch,29])` and returning `float32 [batch,4]` in original
    target units;
  - it must be a faithful export of `training_checkpoint.pt`: its learned
    parameter and buffer tensors must match that checkpoint's `state_dict`,
    and its target-normalization buffers must equal the checkpoint
    `target_mean` and `target_std`.
- `/app/models/ablation_training_checkpoint.pt`
  - the same checkpoint schema for the quadratic-free regressor, with
    `quadratic_matrix_product=false` and 7,050 optimizer steps.
- `/app/models/ablation_regressor.ts`
  - the same stable molecular interface and checkpoint-faithful export
    requirement for the quadratic-free regressor.
- `/app/models/expressivity.ts`
  - a loadable TorchScript module accepting
    `(adjacency: float32 [batch,6,6], node_mask: bool [batch,6])` and returning
    `float32 [batch,2]` logits.
- `/app/models/expressivity_ablation.ts`
  - the same stable interface for the quadratic-free expressivity ablation.
- `/app/artifacts/valid_predictions.npz`
- `/app/artifacts/test_predictions.npz`
- `/app/artifacts/ablation_valid_predictions.npz`
  - each prediction archive must contain exactly `molecule_id` and
    `predictions`; predictions are finite `[rows,4]` values in original units.
- `/app/artifacts/metrics.json`
  - exactly `validation`, `ablation_validation`,
    `quadratic_validation_gain`, `expressivity_accuracy`, and
    `ablation_expressivity_accuracy`;
  - each validation object contains `mae` with the four target-name keys,
    `mean_standardized_mae`, and `rmse_standardized`;
  - `quadratic_validation_gain` is ablation-minus-primary mean standardized
    validation MAE;
  - `rmse_standardized` is the square root of the mean squared validation
    error after dividing each target error by that target's
    `dataset_manifest.json` training-set standard deviation, averaged across
    all validation rows and all four targets.
- `/app/artifacts/training_history.json`
  - exactly `main` and `ablation`; each contains an `epochs` list with exactly
    30 entries and a finite numeric `best_valid` value.
- `/app/artifacts/gpu_execution.json`
  - exactly `cuda_available`, `visible_cuda_devices`, `device_name`,
    `torch_cuda_version`, `seed`, `main_optimizer_steps`,
    `ablation_optimizer_steps`, `expressivity_optimizer_steps`,
    `peak_allocated_bytes`, and `elapsed_seconds`;
  - `visible_cuda_devices` is the integer visible-device count and must equal
    `1`; the file must also report positive optimizer-step counts, peak
    allocated CUDA memory, and elapsed time; the primary and ablation counts
    are each 7,050 and the combined expressivity count is 480.
- `/app/reports/report.md`

The report must cover data provenance, graph construction, model and ablation
design, per-target validation results, the expressivity experiment,
reproducibility, CUDA execution, quadratic compute cost, and limitations. It
must name Revised QM9, its CC BY 4.0 license, and be at least 180 words.

## Acceptance metrics

Successful outputs have hidden-test mean standardized MAE at most `0.45`;
hidden-test MAE at most `0.12` for atomization, `0.012` for HOMO, `0.020` for
LUMO, and `0.85` for dipole; quadratic validation gain at least `0.05`; hidden
expressivity accuracy at least `0.95`; hidden expressivity accuracy gain over
the ablation at least `0.30`; and maximum permutation delta at most `0.0002`.

Internal module layout and parameter names are your choice. The exported
interfaces and artifact schemas above are the stable contract.
