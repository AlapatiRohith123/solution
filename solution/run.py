#!/usr/bin/env python3
"""Train, evaluate, and export the PPGN task artifacts on one CUDA GPU."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from src.ppgn import (
    TARGET_NAMES,
    ExpressivityPPGN,
    MolecularPPGN,
    MoleculeDataset,
    RawUnitRegressor,
    load_manifest,
    load_split,
    set_determinism,
)
from torch import nn
from torch.utils.data import DataLoader

SEED = 1_912_019
BATCH_SIZE = 128
MAIN_EPOCHS = 30
ABLATION_EPOCHS = 30
EXPRESSIVITY_EPOCHS = 240


def data_loader(arrays, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        MoleculeDataset(arrays),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def standardized_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    absolute = np.abs(predictions - targets)
    per_target = {
        name: float(absolute[:, index].mean())
        for index, name in enumerate(TARGET_NAMES)
    }
    standardized = absolute / target_std[None, :]
    return {
        "mae": per_target,
        "mean_standardized_mae": float(standardized.mean()),
        "rmse_standardized": float(np.sqrt(np.mean(standardized * standardized))),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    arrays,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in data_loader(arrays, shuffle=False):
        inputs = [value.to(device, non_blocking=True) for value in batch[:3]]
        chunks.append(model(*inputs).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_regressor(
    *,
    train_arrays,
    valid_arrays,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    epochs: int,
    use_quadratic: bool,
) -> tuple[MolecularPPGN, dict, int]:
    model = MolecularPPGN(use_quadratic=use_quadratic).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-5)
    train = data_loader(train_arrays, shuffle=True)
    best_state = None
    best_valid = float("inf")
    history = []
    steps = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for atomic_numbers, coordinates, atom_mask, targets in train:
            atomic_numbers = atomic_numbers.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)
            atom_mask = atom_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            standardized_targets = (targets - target_mean) / target_std

            optimizer.zero_grad(set_to_none=True)
            outputs = model(atomic_numbers, coordinates, atom_mask)
            loss = torch.nn.functional.smooth_l1_loss(
                outputs, standardized_targets, beta=0.5
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))

        wrapped = RawUnitRegressor(model, target_mean, target_std)
        valid_predictions = predict(wrapped, valid_arrays, device)
        valid_value = standardized_metrics(
            valid_predictions,
            valid_arrays.targets,
            target_std.cpu().numpy(),
        )["mean_standardized_mae"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "valid_mean_standardized_mae": float(valid_value),
            }
        )
        if valid_value < best_valid:
            best_valid = float(valid_value)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {"epochs": history, "best_valid": best_valid}, steps


def cycle_adjacency(nodes: int) -> np.ndarray:
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for index in range(nodes):
        adjacency[index, (index + 1) % nodes] = 1.0
        adjacency[(index + 1) % nodes, index] = 1.0
    return adjacency


def two_cycles_adjacency(nodes_per_cycle: int) -> np.ndarray:
    first = cycle_adjacency(nodes_per_cycle)
    result = np.zeros((nodes_per_cycle * 2, nodes_per_cycle * 2), dtype=np.float32)
    result[:nodes_per_cycle, :nodes_per_cycle] = first
    result[nodes_per_cycle:, nodes_per_cycle:] = first
    return result


def permuted_graphs(
    *,
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bases = [cycle_adjacency(6), two_cycles_adjacency(3)]
    graphs = []
    labels = []
    for label, base in enumerate(bases):
        for _ in range(samples_per_class):
            permutation = rng.permutation(6)
            graphs.append(base[permutation][:, permutation])
            labels.append(label)
    order = rng.permutation(len(graphs))
    adjacency = torch.from_numpy(np.stack(graphs)[order]).float()
    target = torch.tensor(np.asarray(labels)[order], dtype=torch.long)
    mask = torch.ones((len(graphs), 6), dtype=torch.bool)
    return adjacency, mask, target


def train_expressivity_model(
    *,
    device: torch.device,
    use_quadratic: bool,
) -> tuple[ExpressivityPPGN, float, int]:
    rng = np.random.default_rng(SEED + (0 if use_quadratic else 17))
    adjacency, mask, labels = permuted_graphs(samples_per_class=192, rng=rng)
    model = ExpressivityPPGN(use_quadratic=use_quadratic).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-3)
    steps = 0
    for _ in range(EXPRESSIVITY_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(adjacency.to(device), mask.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        steps += 1
    with torch.no_grad():
        predictions = model(adjacency.to(device), mask.to(device)).argmax(dim=1).cpu()
    accuracy = float((predictions == labels).float().mean())
    return model, accuracy, steps


def save_predictions(
    path: Path, molecule_ids: np.ndarray, predictions: np.ndarray
) -> None:
    np.savez_compressed(
        path,
        molecule_id=molecule_ids,
        predictions=predictions.astype(np.float32),
    )


def main() -> None:
    app_dir = Path(os.environ.get("OTTER_APP_DIR", "/app"))
    data_dir = Path(os.environ.get("OTTER_DATA_DIR", app_dir / "data"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This task requires one CUDA GPU; CPU fallback is forbidden."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_determinism(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    for name in ("artifacts", "models", "reports"):
        target = app_dir / name
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifest = load_manifest(data_dir)
    target_mean = torch.tensor(
        manifest["target_mean"], device=device, dtype=torch.float32
    )
    target_std = torch.tensor(
        manifest["target_std"], device=device, dtype=torch.float32
    )
    train_arrays = load_split(data_dir, "train")
    valid_arrays = load_split(data_dir, "valid")
    test_arrays = load_split(data_dir, "test")

    main_model, main_history, main_steps = train_regressor(
        train_arrays=train_arrays,
        valid_arrays=valid_arrays,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        epochs=MAIN_EPOCHS,
        use_quadratic=True,
    )
    raw_main = RawUnitRegressor(main_model, target_mean, target_std).to(device)
    valid_predictions = predict(raw_main, valid_arrays, device)
    test_predictions = predict(raw_main, test_arrays, device)
    valid_metrics = standardized_metrics(
        valid_predictions,
        valid_arrays.targets,
        target_std.cpu().numpy(),
    )
    save_predictions(
        app_dir / "artifacts" / "valid_predictions.npz",
        valid_arrays.molecule_id,
        valid_predictions,
    )
    save_predictions(
        app_dir / "artifacts" / "test_predictions.npz",
        test_arrays.molecule_id,
        test_predictions,
    )

    ablation_model, ablation_history, ablation_steps = train_regressor(
        train_arrays=train_arrays,
        valid_arrays=valid_arrays,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        epochs=ABLATION_EPOCHS,
        use_quadratic=False,
    )
    raw_ablation = RawUnitRegressor(ablation_model, target_mean, target_std).to(device)
    ablation_valid_predictions = predict(raw_ablation, valid_arrays, device)
    ablation_metrics = standardized_metrics(
        ablation_valid_predictions,
        valid_arrays.targets,
        target_std.cpu().numpy(),
    )
    save_predictions(
        app_dir / "artifacts" / "ablation_valid_predictions.npz",
        valid_arrays.molecule_id,
        ablation_valid_predictions,
    )

    expressivity_model, expressivity_accuracy, expression_steps = (
        train_expressivity_model(device=device, use_quadratic=True)
    )
    (
        ablation_expressivity_model,
        ablation_expressivity_accuracy,
        ablation_expression_steps,
    ) = train_expressivity_model(device=device, use_quadratic=False)

    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in main_model.state_dict().items()
        },
        "model_config": {
            "family": "PPGN",
            "order": 2,
            "quadratic_matrix_product": True,
            "width": 48,
            "blocks": 3,
            "outputs": len(TARGET_NAMES),
        },
        "training_metadata": {
            "seed": SEED,
            "trained_on_cuda": True,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "optimizer_steps": main_steps,
            "target_names": list(TARGET_NAMES),
            "target_mean": target_mean.cpu().tolist(),
            "target_std": target_std.cpu().tolist(),
        },
    }
    torch.save(checkpoint, app_dir / "models" / "training_checkpoint.pt")
    ablation_checkpoint = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in ablation_model.state_dict().items()
        },
        "model_config": {
            "family": "PPGN",
            "order": 2,
            "quadratic_matrix_product": False,
            "width": 48,
            "blocks": 3,
            "outputs": len(TARGET_NAMES),
        },
        "training_metadata": {
            "seed": SEED,
            "trained_on_cuda": True,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "optimizer_steps": ablation_steps,
            "target_names": list(TARGET_NAMES),
            "target_mean": target_mean.cpu().tolist(),
            "target_std": target_std.cpu().tolist(),
        },
    }
    torch.save(
        ablation_checkpoint,
        app_dir / "models" / "ablation_training_checkpoint.pt",
    )

    raw_main.eval()
    trace_atoms = torch.from_numpy(valid_arrays.atomic_numbers[:2]).long().to(device)
    trace_coords = torch.from_numpy(valid_arrays.coordinates[:2]).float().to(device)
    trace_mask = torch.from_numpy(valid_arrays.atom_mask[:2]).bool().to(device)
    scripted_regressor = torch.jit.trace(
        raw_main, (trace_atoms, trace_coords, trace_mask)
    )
    torch.jit.save(scripted_regressor, app_dir / "models" / "regressor.ts")
    raw_ablation.eval()
    scripted_ablation_regressor = torch.jit.trace(
        raw_ablation,
        (trace_atoms, trace_coords, trace_mask),
    )
    torch.jit.save(
        scripted_ablation_regressor,
        app_dir / "models" / "ablation_regressor.ts",
    )

    expressivity_model.eval()
    trace_adjacency = (
        torch.from_numpy(np.stack([cycle_adjacency(6), two_cycles_adjacency(3)]))
        .float()
        .to(device)
    )
    trace_graph_mask = torch.ones((2, 6), dtype=torch.bool, device=device)
    scripted_expressivity = torch.jit.trace(
        expressivity_model, (trace_adjacency, trace_graph_mask)
    )
    torch.jit.save(scripted_expressivity, app_dir / "models" / "expressivity.ts")
    ablation_expressivity_model.eval()
    scripted_expressivity_ablation = torch.jit.trace(
        ablation_expressivity_model, (trace_adjacency, trace_graph_mask)
    )
    torch.jit.save(
        scripted_expressivity_ablation,
        app_dir / "models" / "expressivity_ablation.ts",
    )

    metrics = {
        "validation": valid_metrics,
        "ablation_validation": ablation_metrics,
        "quadratic_validation_gain": float(
            ablation_metrics["mean_standardized_mae"]
            - valid_metrics["mean_standardized_mae"]
        ),
        "expressivity_accuracy": expressivity_accuracy,
        "ablation_expressivity_accuracy": ablation_expressivity_accuracy,
    }
    (app_dir / "artifacts" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (app_dir / "artifacts" / "training_history.json").write_text(
        json.dumps(
            {
                "main": main_history,
                "ablation": ablation_history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    gpu_execution = {
        "cuda_available": True,
        "visible_cuda_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "torch_cuda_version": torch.version.cuda,
        "seed": SEED,
        "main_optimizer_steps": main_steps,
        "ablation_optimizer_steps": ablation_steps,
        "expressivity_optimizer_steps": (expression_steps + ablation_expression_steps),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.time() - started,
    }
    (app_dir / "artifacts" / "gpu_execution.json").write_text(
        json.dumps(gpu_execution, indent=2, sort_keys=True) + "\n"
    )

    report = f"""# PPGN Revised-QM9 Results

## Data and provenance

The frozen task subset contains 30,000 training, 3,000 validation, and
3,000 hidden-test molecules sampled deterministically from Revised QM9
(CC BY 4.0, DOI 10.6084/m9.figshare.25266574). Molecule identifiers are
task-local and the test labels are not present in the task inputs.

## Method

Graph construction creates dense atom-pair features from the atom mask,
element identities, interatomic distances, and undirected bonds defined by
the manifest's covalent-radius cutoff rule. Padded atoms are excluded.
The primary model is an order-2 Provably Powerful Graph Network. It applies
feature-wise MLPs to dense atom-by-atom tensors and includes learned
feature-channel matrix products inside every block. The readout combines
diagonal and off-diagonal invariant summaries. The ablation removes only
the quadratic matrix-product path.

## Validation

Primary mean standardized MAE: {valid_metrics["mean_standardized_mae"]:.6f}

Ablation mean standardized MAE:
{ablation_metrics["mean_standardized_mae"]:.6f}

Quadratic validation gain:
{metrics["quadratic_validation_gain"]:.6f}

Per-target validation MAE:
{json.dumps(valid_metrics["mae"], sort_keys=True)}

## Expressivity probe

The permutation-augmented probe distinguishes a six-cycle from two
disconnected triangles, a pair that uniform-label 1-WL cannot separate.
Primary PPGN accuracy: {expressivity_accuracy:.6f}. Quadratic-free ablation
accuracy: {ablation_expressivity_accuracy:.6f}.

## Reproducibility and limitations

The complete pipeline is regenerated by `/app/run.py` with seed {SEED} on
exactly one CUDA GPU. Dense order-2 tensors have quadratic memory cost in
the atom count, so this implementation is appropriate for small molecules
but not directly for large molecular systems.
"""
    (app_dir / "reports" / "report.md").write_text(report)


if __name__ == "__main__":
    main()
