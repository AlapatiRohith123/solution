"""Deterministic verifier for the PPGN Revised-QM9 GPU task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
DATA_DIR = Path(os.environ.get("OTTER_DATA_DIR", APP_DIR / "data"))
TESTS_DIR = Path(__file__).resolve().parent
TARGET_NAMES = ("atomization", "homo", "lumo", "dipole")
EXPECTED_DATA_HASHES = {
    "dataset_manifest.json": (
        "0601c2593bd0c79eae3c1f0d8867dd560dcd2f66c7763997119d00c50d6ee0d4"
    ),
    "test.npz": "d5da324de676ba5c0ca95b853821afb727091cec341eedb307f36b557ab58cbb",
    "train.npz": "e5be34445de001fb32e5da143f515251f48568f33eb755a725f826c30a4c0072",
    "valid.npz": "2e12e2f46960181a621d494ac07a63b39d8de739f6e65b7a3c2f4a112f66dd80",
}
EXPECTED_HIDDEN_HASH = (
    "8d8bbb2377beb4fa71c2ca6dadca9f707612ce379b49399a90d551b53c588b07"
)
REQUIRED_ARTIFACTS = (
    "models/training_checkpoint.pt",
    "models/ablation_training_checkpoint.pt",
    "models/regressor.ts",
    "models/ablation_regressor.ts",
    "models/expressivity.ts",
    "models/expressivity_ablation.ts",
    "artifacts/valid_predictions.npz",
    "artifacts/test_predictions.npz",
    "artifacts/ablation_valid_predictions.npz",
    "artifacts/metrics.json",
    "artifacts/training_history.json",
    "artifacts/gpu_execution.json",
    "reports/report.md",
)
CRITERION_MILESTONES = {
    "data-integrity": "frozen-data-and-regeneration",
    "entrypoint-contract": "frozen-data-and-regeneration",
    "required-artifacts": "frozen-data-and-regeneration",
    "model-checkpoint": "ppgn-model-and-hidden-evaluation",
    "ablation-model": "ppgn-model-and-hidden-evaluation",
    "prediction-and-metric-integrity": "ppgn-model-and-hidden-evaluation",
    "torchscript-and-invariance": "ppgn-model-and-hidden-evaluation",
    "gpu-execution": "ppgn-model-and-hidden-evaluation",
    "report-content": "analysis-and-reproducibility",
}

_CACHE: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise AssertionError(f"Missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON in {path}: {exc}") from exc


def split_archive(split: str) -> dict[str, np.ndarray]:
    cache_key = f"split:{split}"
    if cache_key not in _CACHE:
        path = DATA_DIR / f"{split}.npz"
        if not path.is_file():
            raise AssertionError(f"Missing frozen split: {path}")
        archive = np.load(path, allow_pickle=False)
        _CACHE[cache_key] = {key: archive[key] for key in archive.files}
    return _CACHE[cache_key]


def hidden_test() -> dict[str, np.ndarray]:
    if "hidden" not in _CACHE:
        archive = np.load(TESTS_DIR / "hidden_test.npz", allow_pickle=False)
        _CACHE["hidden"] = {key: archive[key] for key in archive.files}
    return _CACHE["hidden"]


def ordered_predictions(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(f"Missing prediction archive: {path}")
    archive = np.load(path, allow_pickle=False)
    if set(archive.files) != {"molecule_id", "predictions"}:
        raise AssertionError(
            f"{path.name} must contain exactly molecule_id and predictions"
        )
    ids = archive["molecule_id"].astype(str)
    predictions = np.asarray(archive["predictions"], dtype=np.float64)
    if predictions.shape != (len(expected_ids), len(TARGET_NAMES)):
        raise AssertionError(
            f"{path.name} prediction shape {predictions.shape} is invalid"
        )
    if len(set(ids.tolist())) != len(ids):
        raise AssertionError(f"{path.name} contains duplicate molecule IDs")
    expected = expected_ids.astype(str)
    if set(ids.tolist()) != set(expected.tolist()):
        raise AssertionError(f"{path.name} does not exactly cover the frozen split")
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{path.name} contains non-finite predictions")
    position = {value: index for index, value in enumerate(ids)}
    return predictions[
        np.asarray([position[value] for value in expected], dtype=np.int64)
    ]


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, Any]:
    absolute = np.abs(predictions - targets)
    standardized = absolute / target_std[None, :]
    return {
        "mae": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(TARGET_NAMES)
        },
        "mean_standardized_mae": float(standardized.mean()),
        "rmse_standardized": float(np.sqrt(np.mean(standardized * standardized))),
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1.0e-5, abs_tol=1.0e-6):
        raise AssertionError(
            f"{label} mismatch: reported {actual}, recomputed {expected}"
        )


def tensor_fingerprint(
    value: torch.Tensor,
) -> tuple[tuple[int, ...], str, str]:
    tensor = value.detach().cpu().contiguous()
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    )


def tensor_values_match(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if tuple(actual.shape) != tuple(expected.shape):
        return False
    if actual.is_floating_point() or expected.is_floating_point():
        low_precision = {
            torch.float16,
            torch.bfloat16,
        }
        uses_low_precision = (
            actual.dtype in low_precision or expected.dtype in low_precision
        )
        return bool(
            torch.allclose(
                actual.detach().cpu().to(torch.float64),
                expected.detach().cpu().to(torch.float64),
                rtol=5.0e-3 if uses_low_precision else 1.0e-5,
                atol=5.0e-4 if uses_low_precision else 1.0e-6,
            )
        )
    return bool(torch.equal(actual.detach().cpu(), expected.detach().cpu()))


def validate_export_fidelity(
    checkpoint_state: dict[str, Any],
    export_state: dict[str, Any],
    target_buffers: tuple[torch.Tensor, torch.Tensor],
    label: str,
) -> None:
    checkpoint_tensors = [
        value for value in checkpoint_state.values() if torch.is_tensor(value)
    ]
    export_tensors = [
        value for value in export_state.values() if torch.is_tensor(value)
    ]
    checkpoint_fingerprints = Counter(
        tensor_fingerprint(value) for value in checkpoint_tensors
    )
    export_fingerprints = Counter(
        tensor_fingerprint(value) for value in export_tensors
    )
    if any(
        export_fingerprints[fingerprint] < count
        for fingerprint, count in checkpoint_fingerprints.items()
    ):
        raise AssertionError(f"{label} is missing checkpoint tensors")

    remaining_checkpoint = checkpoint_fingerprints.copy()
    extra_tensors: list[torch.Tensor] = []
    for value in export_tensors:
        fingerprint = tensor_fingerprint(value)
        if remaining_checkpoint[fingerprint] > 0:
            remaining_checkpoint[fingerprint] -= 1
        else:
            extra_tensors.append(value)

    missing_targets = [
        target
        for target in target_buffers
        if not any(
            tensor_values_match(value, target) for value in checkpoint_tensors
        )
    ]
    if len(extra_tensors) != len(missing_targets):
        raise AssertionError(f"{label} contains unexpected export tensors")
    for target in missing_targets:
        match = next(
            (
                index
                for index, value in enumerate(extra_tensors)
                if tensor_values_match(value, target)
            ),
            None,
        )
        if match is None:
            raise AssertionError(
                f"{label} is missing a target-normalization buffer"
            )
        extra_tensors.pop(match)


def cycle_adjacency(nodes: int) -> np.ndarray:
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for index in range(nodes):
        adjacency[index, (index + 1) % nodes] = 1.0
        adjacency[(index + 1) % nodes, index] = 1.0
    return adjacency


def two_cycles_adjacency(nodes_per_cycle: int) -> np.ndarray:
    base = cycle_adjacency(nodes_per_cycle)
    result = np.zeros((nodes_per_cycle * 2, nodes_per_cycle * 2), dtype=np.float32)
    result[:nodes_per_cycle, :nodes_per_cycle] = base
    result[nodes_per_cycle:, nodes_per_cycle:] = base
    return result


def hidden_expressivity(model: torch.jit.ScriptModule, device: torch.device) -> float:
    rng = np.random.default_rng(91_919)
    graphs = []
    labels = []
    for label, base in enumerate((cycle_adjacency(6), two_cycles_adjacency(3))):
        for _ in range(128):
            permutation = rng.permutation(6)
            graphs.append(base[permutation][:, permutation])
            labels.append(label)
    adjacency = torch.from_numpy(np.stack(graphs)).float().to(device)
    mask = torch.ones((len(graphs), 6), dtype=torch.bool, device=device)
    with torch.no_grad():
        predictions = model(adjacency, mask).argmax(dim=1).cpu().numpy()
    return float(np.mean(predictions == np.asarray(labels)))


def validate_entrypoint() -> None:
    run_path = APP_DIR / "run.py"
    if not run_path.is_file() or run_path.is_symlink():
        raise AssertionError("Missing required /app/run.py")


def validate_data() -> dict[str, Any]:
    declared_hashes = load_json(DATA_DIR / "checksums.json")
    if declared_hashes != EXPECTED_DATA_HASHES:
        raise AssertionError("checksums.json does not match the frozen task inputs")
    for filename, expected in EXPECTED_DATA_HASHES.items():
        observed = sha256(DATA_DIR / filename)
        if observed != expected:
            raise AssertionError(
                f"Frozen input hash mismatch for {filename}: {observed}"
            )

    manifest = load_json(DATA_DIR / "dataset_manifest.json")
    if manifest.get("source_license") != "CC BY 4.0":
        raise AssertionError("Dataset license metadata is missing")
    if manifest.get("target_names") != list(TARGET_NAMES):
        raise AssertionError("Target order does not match the task contract")
    if manifest.get("split_sizes") != {
        "train": 30000,
        "valid": 3000,
        "test": 3000,
    }:
        raise AssertionError("Frozen split sizes are invalid")

    all_ids: list[set[str]] = []
    for split, size, labeled in (
        ("train", 30000, True),
        ("valid", 3000, True),
        ("test", 3000, False),
    ):
        archive = split_archive(split)
        expected_keys = {
            "molecule_id",
            "atomic_numbers",
            "coordinates",
            "atom_mask",
        }
        if labeled:
            expected_keys.add("targets")
        if set(archive) != expected_keys:
            raise AssertionError(f"{split}.npz has an invalid field set")
        ids = archive["molecule_id"].astype(str)
        if len(ids) != size or len(set(ids.tolist())) != size:
            raise AssertionError(f"{split} IDs are missing or duplicated")
        if archive["atomic_numbers"].shape != (size, 29):
            raise AssertionError(f"{split} atomic-number shape is invalid")
        if archive["atomic_numbers"].dtype != np.uint8:
            raise AssertionError(f"{split} atomic-number dtype is invalid")
        if archive["coordinates"].shape != (size, 29, 3):
            raise AssertionError(f"{split} coordinate shape is invalid")
        if archive["coordinates"].dtype != np.float32:
            raise AssertionError(f"{split} coordinate dtype is invalid")
        if archive["atom_mask"].shape != (size, 29):
            raise AssertionError(f"{split} mask shape is invalid")
        if archive["atom_mask"].dtype != np.bool_:
            raise AssertionError(f"{split} mask dtype is invalid")
        if not np.isfinite(archive["coordinates"]).all():
            raise AssertionError(f"{split} coordinates contain non-finite data")
        if labeled and archive["targets"].shape != (size, 4):
            raise AssertionError(f"{split} target shape is invalid")
        if labeled and archive["targets"].dtype != np.float32:
            raise AssertionError(f"{split} target dtype is invalid")
        all_ids.append(set(ids.tolist()))
    if any(
        all_ids[left] & all_ids[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("Frozen data splits overlap")

    hidden_path = TESTS_DIR / "hidden_test.npz"
    if sha256(hidden_path) != EXPECTED_HIDDEN_HASH:
        raise AssertionError("Hidden test-label fixture hash is invalid")
    hidden = hidden_test()
    if set(hidden) != {"molecule_id", "targets"}:
        raise AssertionError("Hidden test archive schema is invalid")
    if not np.array_equal(
        hidden["molecule_id"].astype(str),
        split_archive("test")["molecule_id"].astype(str),
    ):
        raise AssertionError("Hidden labels do not align with test inputs")
    return manifest


def validate_checkpoint_file(
    path: Path,
    *,
    quadratic_matrix_product: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Training checkpoint is not a dictionary")
    if set(checkpoint) != {"state_dict", "model_config", "training_metadata"}:
        raise AssertionError("Training checkpoint top-level schema is invalid")
    state = checkpoint.get("state_dict")
    config = checkpoint.get("model_config")
    metadata = checkpoint.get("training_metadata")
    if not isinstance(state, dict) or not state:
        raise AssertionError("Checkpoint state_dict is missing or empty")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors or any(not torch.isfinite(value).all() for value in tensors):
        raise AssertionError("Checkpoint tensors are missing or non-finite")
    flattened = torch.cat(
        [value.detach().float().reshape(-1) for value in tensors if value.numel()]
    )
    if flattened.numel() == 0 or torch.unique(flattened).numel() < 2:
        raise AssertionError("Checkpoint is empty or effectively constant")
    if not isinstance(config, dict):
        raise TypeError("Semantic model_config is missing")
    if set(config) != {
        "family",
        "order",
        "quadratic_matrix_product",
        "width",
        "blocks",
        "outputs",
    }:
        raise AssertionError("Checkpoint model_config schema is invalid")
    if (
        config.get("family") != "PPGN"
        or config.get("order") != 2
        or config.get("quadratic_matrix_product") is not quadratic_matrix_product
        or int(config.get("width", 0)) <= 0
        or int(config.get("blocks", 0)) <= 0
        or int(config.get("outputs", 0)) != 4
    ):
        raise AssertionError("Checkpoint model_config is not the required PPGN")
    if not isinstance(metadata, dict):
        raise TypeError("Training metadata is missing")
    if set(metadata) != {
        "seed",
        "trained_on_cuda",
        "cuda_device_name",
        "optimizer_steps",
        "target_names",
        "target_mean",
        "target_std",
    }:
        raise AssertionError("Checkpoint training_metadata schema is invalid")
    manifest = load_json(DATA_DIR / "dataset_manifest.json")
    if (
        int(metadata.get("seed", 0)) != 1_912_019
        or metadata.get("trained_on_cuda") is not True
        or not str(metadata.get("cuda_device_name", "")).strip()
        or int(metadata.get("optimizer_steps", 0)) != 7_050
        or metadata.get("target_names") != list(TARGET_NAMES)
        or not np.allclose(
            np.asarray(metadata.get("target_mean"), dtype=np.float64),
            np.asarray(manifest["target_mean"], dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        or not np.allclose(
            np.asarray(metadata.get("target_std"), dtype=np.float64),
            np.asarray(manifest["target_std"], dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    ):
        raise AssertionError("Checkpoint training metadata is invalid")
    return checkpoint


def validate_checkpoint() -> dict[str, Any]:
    checkpoint = validate_checkpoint_file(
        APP_DIR / "models" / "training_checkpoint.pt",
        quadratic_matrix_product=True,
    )

    history = load_json(APP_DIR / "artifacts" / "training_history.json")
    if set(history) != {"main", "ablation"}:
        raise AssertionError("training_history.json top-level schema is invalid")
    for name in ("main", "ablation"):
        section = history[name]
        if not isinstance(section, dict) or set(section) != {
            "epochs",
            "best_valid",
        }:
            raise AssertionError(f"{name} training-history schema is invalid")
        epochs = section["epochs"]
        if not isinstance(epochs, list) or len(epochs) != 30:
            raise AssertionError(
                "Primary and ablation regressors must each train for 30 epochs"
            )
        if not math.isfinite(float(section["best_valid"])):
            raise AssertionError(f"{name} best_valid is non-finite")
    return checkpoint


def validate_ablation_checkpoint() -> dict[str, Any]:
    return validate_checkpoint_file(
        APP_DIR / "models" / "ablation_training_checkpoint.pt",
        quadratic_matrix_product=False,
    )


def validate_required_artifacts() -> None:
    missing = [
        path
        for path in REQUIRED_ARTIFACTS
        if (
            not (APP_DIR / path).is_file()
            or (APP_DIR / path).is_symlink()
            or (APP_DIR / path).stat().st_size <= 0
        )
    ]
    if missing:
        raise AssertionError(f"Required artifacts are missing or empty: {missing}")


def validate_prediction_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    valid = split_archive("valid")
    test = split_archive("test")
    hidden = hidden_test()
    target_std = np.asarray(manifest["target_std"], dtype=np.float64)

    valid_predictions = ordered_predictions(
        APP_DIR / "artifacts" / "valid_predictions.npz",
        valid["molecule_id"],
    )
    test_predictions = ordered_predictions(
        APP_DIR / "artifacts" / "test_predictions.npz",
        test["molecule_id"],
    )
    ablation_predictions = ordered_predictions(
        APP_DIR / "artifacts" / "ablation_valid_predictions.npz",
        valid["molecule_id"],
    )
    if np.allclose(valid_predictions, ablation_predictions):
        raise AssertionError(
            "Primary and quadratic-free validation predictions are identical"
        )

    valid_metrics = regression_metrics(
        valid_predictions,
        valid["targets"].astype(np.float64),
        target_std,
    )
    ablation_metrics = regression_metrics(
        ablation_predictions,
        valid["targets"].astype(np.float64),
        target_std,
    )
    hidden_metrics = regression_metrics(
        test_predictions,
        hidden["targets"].astype(np.float64),
        target_std,
    )
    reported = load_json(APP_DIR / "artifacts" / "metrics.json")
    if set(reported) != {
        "validation",
        "ablation_validation",
        "quadratic_validation_gain",
        "expressivity_accuracy",
        "ablation_expressivity_accuracy",
    }:
        raise AssertionError("metrics.json top-level schema is invalid")
    for section_name in ("validation", "ablation_validation"):
        section = reported[section_name]
        if set(section) != {"mae", "mean_standardized_mae", "rmse_standardized"}:
            raise AssertionError(f"{section_name} metric schema is invalid")
        if set(section["mae"]) != set(TARGET_NAMES):
            raise AssertionError(f"{section_name} MAE schema is invalid")
    for name in TARGET_NAMES:
        assert_close(
            reported["validation"]["mae"][name],
            valid_metrics["mae"][name],
            f"validation {name} MAE",
        )
        assert_close(
            reported["ablation_validation"]["mae"][name],
            ablation_metrics["mae"][name],
            f"ablation validation {name} MAE",
        )
    for key in ("mean_standardized_mae", "rmse_standardized"):
        assert_close(
            reported["validation"][key],
            valid_metrics[key],
            f"validation {key}",
        )
        assert_close(
            reported["ablation_validation"][key],
            ablation_metrics[key],
            f"ablation validation {key}",
        )
    validation_gain = (
        ablation_metrics["mean_standardized_mae"]
        - valid_metrics["mean_standardized_mae"]
    )
    assert_close(
        reported["quadratic_validation_gain"],
        validation_gain,
        "quadratic validation gain",
    )
    return {
        "valid_predictions": valid_predictions,
        "test_predictions": test_predictions,
        "ablation_predictions": ablation_predictions,
        "valid_metrics": valid_metrics,
        "ablation_metrics": ablation_metrics,
        "hidden_metrics": hidden_metrics,
        "validation_gain": float(validation_gain),
        "reported": reported,
    }


def validate_torchscript(prediction_state: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise AssertionError("Verifier requires CUDA for model replay")
    if torch.cuda.device_count() != 1:
        raise AssertionError("Verifier expected exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    model = torch.jit.load(
        str(APP_DIR / "models" / "regressor.ts"), map_location=device
    )
    model.eval()
    checkpoint = torch.load(
        APP_DIR / "models" / "training_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    target_buffers = (
        torch.tensor(
            checkpoint["training_metadata"]["target_mean"],
            dtype=torch.float32,
        ),
        torch.tensor(
            checkpoint["training_metadata"]["target_std"],
            dtype=torch.float32,
        ),
    )
    validate_export_fidelity(
        checkpoint["state_dict"],
        model.state_dict(),
        target_buffers,
        "TorchScript regressor",
    )
    test = split_archive("test")
    probe = np.arange(64, dtype=np.int64)
    atoms = torch.from_numpy(test["atomic_numbers"][probe]).long().to(device)
    coords = torch.from_numpy(test["coordinates"][probe]).float().to(device)
    mask = torch.from_numpy(test["atom_mask"][probe]).bool().to(device)
    with torch.no_grad():
        replay = model(atoms, coords, mask).cpu().numpy()
    if not np.allclose(
        replay,
        prediction_state["test_predictions"][probe],
        rtol=1.0e-4,
        atol=2.0e-5,
    ):
        raise AssertionError(
            "TorchScript regressor does not reproduce saved test predictions"
        )

    rng = np.random.default_rng(31_337)
    permuted_atoms = atoms.clone()
    permuted_coords = coords.clone()
    permuted_mask = mask.clone()
    for row in range(len(probe)):
        permutation = torch.from_numpy(rng.permutation(29)).long().to(device)
        permuted_atoms[row] = atoms[row, permutation]
        permuted_coords[row] = coords[row, permutation]
        permuted_mask[row] = mask[row, permutation]
    with torch.no_grad():
        permuted = model(permuted_atoms, permuted_coords, permuted_mask).cpu().numpy()
    maximum_delta = float(np.max(np.abs(replay - permuted)))
    if maximum_delta > 2.0e-4:
        raise AssertionError(
            f"Regressor is not permutation invariant; max delta {maximum_delta}"
        )

    ablation_model = torch.jit.load(
        str(APP_DIR / "models" / "ablation_regressor.ts"),
        map_location=device,
    )
    ablation_model.eval()
    valid = split_archive("valid")
    valid_atoms = torch.from_numpy(valid["atomic_numbers"][probe]).long().to(device)
    valid_coords = torch.from_numpy(valid["coordinates"][probe]).float().to(device)
    valid_mask = torch.from_numpy(valid["atom_mask"][probe]).bool().to(device)
    with torch.no_grad():
        ablation_replay = (
            ablation_model(valid_atoms, valid_coords, valid_mask).cpu().numpy()
        )
    if not np.allclose(
        ablation_replay,
        prediction_state["ablation_predictions"][probe],
        rtol=1.0e-4,
        atol=2.0e-5,
    ):
        raise AssertionError(
            "TorchScript ablation regressor does not reproduce its predictions"
        )
    ablation_checkpoint = torch.load(
        APP_DIR / "models" / "ablation_training_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    ablation_target_buffers = (
        torch.tensor(
            ablation_checkpoint["training_metadata"]["target_mean"],
            dtype=torch.float32,
        ),
        torch.tensor(
            ablation_checkpoint["training_metadata"]["target_std"],
            dtype=torch.float32,
        ),
    )
    validate_export_fidelity(
        ablation_checkpoint["state_dict"],
        ablation_model.state_dict(),
        ablation_target_buffers,
        "TorchScript ablation",
    )

    expressivity = torch.jit.load(
        str(APP_DIR / "models" / "expressivity.ts"), map_location=device
    )
    ablation = torch.jit.load(
        str(APP_DIR / "models" / "expressivity_ablation.ts"),
        map_location=device,
    )
    expressivity.eval()
    ablation.eval()
    primary_accuracy = hidden_expressivity(expressivity, device)
    ablation_accuracy = hidden_expressivity(ablation, device)
    return {
        "permutation_max_delta": maximum_delta,
        "expressivity_accuracy": primary_accuracy,
        "ablation_expressivity_accuracy": ablation_accuracy,
        "expressivity_gap": primary_accuracy - ablation_accuracy,
    }


def validate_gpu_execution() -> dict[str, Any]:
    evidence = load_json(APP_DIR / "artifacts" / "gpu_execution.json")
    required = {
        "cuda_available",
        "visible_cuda_devices",
        "device_name",
        "torch_cuda_version",
        "seed",
        "main_optimizer_steps",
        "ablation_optimizer_steps",
        "expressivity_optimizer_steps",
        "peak_allocated_bytes",
        "elapsed_seconds",
    }
    if set(evidence) != required:
        raise AssertionError("gpu_execution.json schema is invalid")
    if (
        evidence["cuda_available"] is not True
        or int(evidence["visible_cuda_devices"]) != 1
        or str(evidence["device_name"]) != torch.cuda.get_device_name(0)
        or str(evidence["torch_cuda_version"]) != str(torch.version.cuda)
        or int(evidence["seed"]) != 1_912_019
        or int(evidence["main_optimizer_steps"]) != 7_050
        or int(evidence["ablation_optimizer_steps"]) != 7_050
        or int(evidence["expressivity_optimizer_steps"]) != 480
        or int(evidence["peak_allocated_bytes"]) <= 0
        or float(evidence["elapsed_seconds"]) <= 0
    ):
        raise AssertionError("GPU execution evidence is invalid")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("Verifier does not have exactly one visible CUDA GPU")
    return evidence


def validate_report() -> None:
    report = (APP_DIR / "reports" / "report.md").read_text(encoding="utf-8")
    required_topics = (
        ("Revised QM9",),
        ("CC BY 4.0",),
        ("graph construction", "bond", "adjacency", "edge"),
        ("order-2", "order 2", "second-order"),
        ("matrix", "bilinear", "tensor product"),
        ("ablation", "quadratic-free", "without quadratic"),
        ("expressivity", "six-cycle", "two disjoint triangles"),
        ("CUDA",),
        ("permutation", "relabel", "node order"),
        ("quadratic", "compute cost", "complexity"),
        ("limitations", "limitation", "caveat"),
    )
    lowered = report.lower()
    missing = [
        "/".join(topic)
        for topic in required_topics
        if not any(term.lower() in lowered for term in topic)
    ]
    if missing:
        raise AssertionError(f"Report is missing required topics: {missing}")
    if len(report.split()) < 180:
        raise AssertionError("Report is too short to cover the required analysis")


def evaluate(context: dict[str, Any]) -> dict[str, Any]:
    del context
    _CACHE.clear()
    criteria: list[dict[str, Any]] = []
    metrics: dict[str, float] = {
        "hidden_mean_standardized_mae": 1.0e30,
        "hidden_atomization_mae": 1.0e30,
        "hidden_homo_mae": 1.0e30,
        "hidden_lumo_mae": 1.0e30,
        "hidden_dipole_mae": 1.0e30,
        "quadratic_validation_gain": -1.0e30,
        "hidden_expressivity_accuracy": -1.0e30,
        "hidden_expressivity_gap": -1.0e30,
        "permutation_max_delta": 1.0e30,
    }

    def check(identifier: str, operation: Callable[[], Any]) -> Any:
        milestone_id = CRITERION_MILESTONES[identifier]
        try:
            value = operation()
            criteria.append(
                {
                    "id": identifier,
                    "milestone_id": milestone_id,
                    "passed": True,
                    "detail": "verified",
                }
            )
            return value
        except Exception as exc:  # noqa: BLE001
            criteria.append(
                {
                    "id": identifier,
                    "milestone_id": milestone_id,
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            return None

    manifest = check("data-integrity", validate_data)
    check("entrypoint-contract", validate_entrypoint)
    check("required-artifacts", validate_required_artifacts)
    check("model-checkpoint", validate_checkpoint)
    check("ablation-model", validate_ablation_checkpoint)

    prediction_state = None
    if manifest is not None:
        prediction_state = check(
            "prediction-and-metric-integrity",
            lambda: validate_prediction_metrics(manifest),
        )
    else:
        criteria.append(
            {
                "id": "prediction-and-metric-integrity",
                "milestone_id": CRITERION_MILESTONES["prediction-and-metric-integrity"],
                "passed": False,
                "detail": "data integrity failed",
            }
        )

    replay_state = None
    if prediction_state is not None:
        replay_state = check(
            "torchscript-and-invariance",
            lambda: validate_torchscript(prediction_state),
        )
    else:
        criteria.append(
            {
                "id": "torchscript-and-invariance",
                "milestone_id": CRITERION_MILESTONES["torchscript-and-invariance"],
                "passed": False,
                "detail": "prediction validation failed",
            }
        )

    check("gpu-execution", validate_gpu_execution)
    check("report-content", validate_report)

    if prediction_state is not None:
        hidden_metrics = prediction_state["hidden_metrics"]
        metrics.update(
            {
                "hidden_mean_standardized_mae": float(
                    hidden_metrics["mean_standardized_mae"]
                ),
                "hidden_atomization_mae": float(hidden_metrics["mae"]["atomization"]),
                "hidden_homo_mae": float(hidden_metrics["mae"]["homo"]),
                "hidden_lumo_mae": float(hidden_metrics["mae"]["lumo"]),
                "hidden_dipole_mae": float(hidden_metrics["mae"]["dipole"]),
                "quadratic_validation_gain": float(prediction_state["validation_gain"]),
            }
        )
    if replay_state is not None:
        metrics.update(
            {
                "hidden_expressivity_accuracy": float(
                    replay_state["expressivity_accuracy"]
                ),
                "hidden_expressivity_gap": float(replay_state["expressivity_gap"]),
                "permutation_max_delta": float(replay_state["permutation_max_delta"]),
            }
        )

    return {"criteria": criteria, "metrics": metrics}
