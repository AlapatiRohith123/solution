"""Behavioral verification for the general-purpose byte trainer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
TRAINING_PATH = Path(
    os.environ.get(
        "OTTER_TRAINING_PATH",
        APP_DIR / "data" / "shakespeare.npz",
    )
)
SHAKESPEARE_HOLDOUT_PATH = Path(
    os.environ.get(
        "OTTER_SHAKESPEARE_HOLDOUT_PATH",
        Path(__file__).with_name("shakespeare_holdout.npz"),
    )
)
RUNNER_PATH = Path(
    os.environ.get("OTTER_RUNNER_PATH", APP_DIR / "run_training.py")
)
FEEDER_PATH = Path(
    os.environ.get("OTTER_FEEDER_PATH", APP_DIR / "data_feeder.py")
)
IBSEN_PATH = Path(
    os.environ.get(
        "OTTER_IBSEN_PATH",
        Path(__file__).with_name("ibsen_complete.npz"),
    )
)
MODEL_BYTE_LIMIT = 65_536
FAST_STEPS = 10_000
CANARY_STEPS = FAST_STEPS
IBSEN_STEPS = 10_000
FULL_STEPS = 100_000
REVERSED_MIN_MACRO_AP = 0.05
CANARY_MAX_SHAKESPEARE_MACRO_AP = 0.05
TRAINING_SHAPE = (61_840, 72)
HOLDOUT_SHAPE = (14_091, 72)
IBSEN_SHAPE = (52_763, 72)
IBSEN_HOLDOUT_SHAPE = (12_461, 72)
TRAINING_SHA256 = "0efb587858b59c7d385a6340ed03de9c8aa283440d65da952a8a904bc9720c91"
SHAKESPEARE_HOLDOUT_SHA256 = "77f6449da5a8477848c831a3499e4059ca6ac5fff9483ab650e57ff795fba622"
RUNNER_SHA256 = "d2aa5c5c856c020f4980ac59ef672aa46938ad3182d65014e06121919d932dd2"
FEEDER_SHA256 = "db3cd2fa70d4405df091e8adead019e1428999f8a915d6828dc1dcf0afcce102"
IBSEN_SHA256 = "ffbacf094863803e74fdc6fa1fda95a3c14e99ff686a92f490c8ff32642bc1fe"


@dataclass(frozen=True)
class TrialSpec:
    """One verifier-selected training problem."""

    name: str
    steps: int
    seed: int
    num_labels: int
    remap: bool
    canary: bool


RUNS = (
    TrialSpec("fast", FAST_STEPS, 0x243F6A88, 28, False, False),
    TrialSpec("reversed", FAST_STEPS, 0x85A308D3, 28, True, False),
    TrialSpec("canary", CANARY_STEPS, 0x13198A2E, 28, False, True),
    TrialSpec("full", FULL_STEPS, 0x03707344, 28, False, False),
)
IBSEN_SPEC = TrialSpec(
    "ibsen",
    IBSEN_STEPS,
    0x299F31D0,
    22,
    False,
    False,
)


@dataclass
class Trial:
    """Verifier-owned measurements from one step-bounded retraining."""

    name: str
    steps: int
    runtime_seconds: float
    num_labels: int
    accuracies: list[float]
    macro_average_precisions: list[float]
    chance_macro_ap: float
    parameter_hashes: list[str]
    prediction_hashes: list[str]
    model_paths: list[str]
    checkpoint_steps: list[int]
    holdout_records: int

    @property
    def final_macro_ap(self) -> float:
        return self.macro_average_precisions[-1]

    @property
    def final_macro_ap_ratio(self) -> float:
        return self.final_macro_ap / self.chance_macro_ap

    @property
    def updated_intervals(self) -> int:
        return sum(
            left != right
            for left, right in zip(
                self.parameter_hashes,
                self.parameter_hashes[1:],
                strict=False,
            )
        )

    @property
    def updated_prediction_intervals(self) -> int:
        return sum(
            left != right
            for left, right in zip(
                self.prediction_hashes,
                self.prediction_hashes[1:],
                strict=False,
            )
        )


def criterion(
    identifier: str,
    milestone: str,
    passed: bool,
    detail: str,
) -> dict[str, Any]:
    """Build one atomic criterion result."""
    return {
        "id": identifier,
        "milestone_id": milestone,
        "passed": bool(passed),
        "detail": detail,
    }


def load_training_archive(path: Path) -> dict[str, np.ndarray]:
    """Load the immutable labeled training lines."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != {"classes", "labels", "samples"}:
        raise RuntimeError("unexpected training arrays")
    if arrays["samples"].dtype != np.uint8:
        raise RuntimeError("training sample type changed")
    if arrays["labels"].shape != (len(arrays["samples"]),):
        raise RuntimeError("training labels changed")
    if arrays["classes"].shape != (28,):
        raise RuntimeError("training classes changed")
    return arrays


def load_holdout_archive(path: Path) -> dict[str, np.ndarray]:
    """Load the protected Shakespeare holdout lines."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != {"classes", "labels", "samples"}:
        raise RuntimeError("unexpected holdout arrays")
    if arrays["samples"].dtype != np.uint8:
        raise RuntimeError("holdout sample type changed")
    if arrays["labels"].shape != (len(arrays["samples"]),):
        raise RuntimeError("holdout labels changed")
    if arrays["classes"].shape != (28,):
        raise RuntimeError("holdout classes changed")
    return arrays


def source_data() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict,
]:
    """Load the fixed Shakespeare training and holdout splits."""
    training_path = TRAINING_PATH
    holdout_path = SHAKESPEARE_HOLDOUT_PATH
    runner_path = RUNNER_PATH
    feeder_path = FEEDER_PATH
    if hashlib.sha256(runner_path.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("fixed training runner changed")
    if hashlib.sha256(feeder_path.read_bytes()).hexdigest() != FEEDER_SHA256:
        raise RuntimeError("fixed data feeder changed")
    if hashlib.sha256(training_path.read_bytes()).hexdigest() != TRAINING_SHA256:
        raise RuntimeError("training corpus digest changed")
    if (
        hashlib.sha256(holdout_path.read_bytes()).hexdigest()
        != SHAKESPEARE_HOLDOUT_SHA256
    ):
        raise RuntimeError("holdout corpus digest changed")
    training = load_training_archive(training_path)
    holdout = load_holdout_archive(holdout_path)
    if training["samples"].shape != TRAINING_SHAPE:
        raise RuntimeError("training corpus shape changed")
    if holdout["samples"].shape != HOLDOUT_SHAPE:
        raise RuntimeError("holdout corpus shape changed")
    if not np.array_equal(training["classes"], holdout["classes"]):
        raise RuntimeError("corpus class maps differ")
    metadata = {
        "shakespeare_holdout_sha256": SHAKESPEARE_HOLDOUT_SHA256,
        "training_sha256": TRAINING_SHA256,
    }
    return training, holdout, metadata


def ibsen_data() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, str],
]:
    """Load the protected public-domain Ibsen split."""
    if hashlib.sha256(IBSEN_PATH.read_bytes()).hexdigest() != IBSEN_SHA256:
        raise RuntimeError("Ibsen corpus digest changed")
    with np.load(IBSEN_PATH, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != {"classes", "holdout", "labels", "samples"}:
        raise RuntimeError("unexpected Ibsen arrays")
    if arrays["samples"].shape != IBSEN_SHAPE:
        raise RuntimeError("Ibsen corpus shape changed")
    if arrays["samples"].dtype != np.uint8:
        raise RuntimeError("Ibsen sample type changed")
    if arrays["labels"].shape != (len(arrays["samples"]),):
        raise RuntimeError("Ibsen labels changed")
    if arrays["holdout"].shape != (len(arrays["samples"]),):
        raise RuntimeError("Ibsen split changed")
    if arrays["holdout"].dtype != np.bool_:
        raise RuntimeError("Ibsen split type changed")
    if arrays["classes"].shape != (IBSEN_SPEC.num_labels,):
        raise RuntimeError("Ibsen classes changed")
    training_rows = ~arrays["holdout"]
    holdout_rows = arrays["holdout"]
    training = {
        "classes": arrays["classes"].copy(),
        "labels": arrays["labels"][training_rows].copy(),
        "samples": arrays["samples"][training_rows].copy(),
    }
    holdout = {
        "classes": arrays["classes"].copy(),
        "labels": arrays["labels"][holdout_rows].copy(),
        "samples": arrays["samples"][holdout_rows].copy(),
    }
    if holdout["samples"].shape != IBSEN_HOLDOUT_SHAPE:
        raise RuntimeError("Ibsen holdout shape changed")
    for split in (training, holdout):
        if not np.array_equal(
            np.unique(split["labels"]),
            np.arange(IBSEN_SPEC.num_labels),
        ):
            raise RuntimeError("Ibsen split lacks a play label")
    return training, holdout, {"ibsen_sha256": IBSEN_SHA256}


def variant(
    training: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    spec: TrialSpec,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one changed candidate set for a fresh run."""
    rng = np.random.default_rng(spec.seed)
    source_count = len(training["classes"])
    if spec.num_labels == source_count:
        selected = np.arange(source_count)
    else:
        selected = np.sort(
            rng.choice(
                source_count,
                size=spec.num_labels,
                replace=False,
            )
        )
    assigned = (
        np.arange(spec.num_labels)[::-1]
        if spec.remap
        else np.arange(spec.num_labels)
    )
    label_map = np.full(source_count, -1, dtype=np.int64)
    label_map[selected] = assigned
    classes = np.empty(spec.num_labels, dtype=training["classes"].dtype)
    classes[assigned] = training["classes"][selected]

    training_rows = np.flatnonzero(label_map[training["labels"]] >= 0)
    rng.shuffle(training_rows)
    training_labels = label_map[training["labels"][training_rows]]
    training_samples = training["samples"][training_rows].copy()
    if spec.canary:
        canary_bytes = (32 + training_labels).astype(np.uint8)
        training_samples[:] = canary_bytes[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=classes,
        labels=training_labels,
        samples=training_samples,
    )

    holdout_rows = np.flatnonzero(label_map[holdout["labels"]] >= 0)
    rng.shuffle(holdout_rows)
    holdout_labels = label_map[holdout["labels"][holdout_rows]]
    holdout_samples = holdout["samples"][holdout_rows].copy()
    if spec.canary:
        canary_bytes = (32 + holdout_labels).astype(np.uint8)
        holdout_samples[:] = canary_bytes[:, None]
    return holdout_samples, holdout_labels


def validate_graph(path: Path) -> ort.InferenceSession:
    """Validate one self-contained standard ONNX graph."""
    if not path.is_file():
        raise RuntimeError(f"missing ONNX file: {path}")
    if not 0 < path.stat().st_size <= MODEL_BYTE_LIMIT:
        raise RuntimeError(f"invalid ONNX size: {path.stat().st_size}")
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise RuntimeError("external tensor data is not allowed")
        if tensor.external_data:
            raise RuntimeError("external tensor references are not allowed")
    domains = {item.domain for item in model.opset_import}
    domains.update(node.domain for node in model.graph.node)
    if not domains <= {"", "ai.onnx"}:
        raise RuntimeError(f"nonstandard ONNX domains: {sorted(domains)}")
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("model must expose one input and one output")
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(uint8)":
        raise RuntimeError("model input must use uint8")
    if model_input.shape[1:] != [72]:
        raise RuntimeError("model input must have 72 byte columns")
    return session


def infer(
    path: Path,
    samples: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """Run a validated graph over raw held-out samples."""
    session = validate_graph(path)
    model_input = session.get_inputs()[0]
    chunks: list[np.ndarray] = []
    for begin in range(0, len(samples), 256):
        batch = samples[begin : begin + 256]
        output = session.run(None, {model_input.name: batch})[0]
        if output.shape != (len(batch), num_labels):
            raise RuntimeError(f"unexpected model output shape: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("model produced non-finite scores")
        chunks.append(output)
    return np.concatenate(chunks)


def probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert finite scores into stable probabilities."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def prediction_probability_rmse(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> float:
    """Measure RMSE between normalized predictions."""
    reference = probabilities(reference_scores.astype(np.float64))
    candidate = probabilities(candidate_scores.astype(np.float64))
    return float(np.sqrt(np.mean(np.square(reference - candidate))))


def macro_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Measure macro one-vs-rest average precision."""
    normalized_scores = probabilities(scores)
    values: list[float] = []
    for label in range(scores.shape[1]):
        targets = labels == label
        order = np.argsort(-normalized_scores[:, label], kind="stable")
        ranked_targets = targets[order]
        ranked_scores = normalized_scores[order, label]
        boundaries = np.flatnonzero(np.diff(ranked_scores)) + 1
        boundaries = np.append(boundaries, len(ranked_scores))
        true_positives = np.cumsum(ranked_targets)[boundaries - 1]
        precision = true_positives / boundaries
        recall = true_positives / np.sum(targets)
        increments = np.diff(np.append(0.0, recall))
        values.append(float(np.sum(increments * precision)))
    return float(np.mean(values))


def accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Measure top-one play accuracy."""
    return float(np.mean(scores.argmax(axis=1) == labels))


def run_trial(
    root: Path,
    training: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    spec: TrialSpec,
) -> tuple[Trial, np.ndarray, np.ndarray]:
    """Run one private problem through submitted code."""
    trial_dir = root / spec.name
    dataset = trial_dir / "train.npz"
    output = trial_dir / "output"
    holdout_samples, holdout_labels = variant(
        training,
        holdout,
        spec,
        dataset,
    )
    command = [
        sys.executable,
        str(RUNNER_PATH),
        str(dataset),
        str(output),
        "--train-steps",
        str(spec.steps),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_seconds = time.monotonic() - started
    (trial_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (trial_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        tail = result.stderr[-2000:]
        raise RuntimeError(
            f"{spec.name} exited {result.returncode}: {tail}"
        )

    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    states = trace.get("states")
    if not isinstance(states, list) or len(states) != 5:
        raise RuntimeError(f"{spec.name} did not produce five timed states")
    if trace.get("num_labels") != spec.num_labels:
        raise RuntimeError(f"{spec.name} used the wrong label count")
    parameter_hashes = [str(state["parameter_sha256"]) for state in states]
    checkpoint_steps = [int(state["steps"]) for state in states]
    model_paths: list[str] = []
    accuracies: list[float] = []
    macro_aps: list[float] = []
    prediction_hashes: list[str] = []
    for state in states:
        filename = str(state["checkpoint"])
        if Path(filename).name != filename:
            raise RuntimeError(f"unsafe checkpoint name: {filename}")
        model_path = output / "checkpoints" / filename
        scores = infer(model_path, holdout_samples, spec.num_labels)
        model_paths.append(str(model_path))
        prediction_hashes.append(hashlib.sha256(scores.tobytes()).hexdigest())
        accuracies.append(accuracy(holdout_labels, scores))
        macro_aps.append(macro_average_precision(holdout_labels, scores))
    trial = Trial(
        name=spec.name,
        steps=spec.steps,
        runtime_seconds=runtime_seconds,
        num_labels=spec.num_labels,
        accuracies=accuracies,
        macro_average_precisions=macro_aps,
        chance_macro_ap=macro_average_precision(
            holdout_labels,
            np.zeros((len(holdout_labels), spec.num_labels)),
        ),
        parameter_hashes=parameter_hashes,
        prediction_hashes=prediction_hashes,
        model_paths=model_paths,
        checkpoint_steps=checkpoint_steps,
        holdout_records=len(holdout_labels),
    )
    return trial, holdout_samples, holdout_labels


def failure_result(message: str) -> dict[str, Any]:
    """Return aligned failing criteria without hiding the cause."""
    details = f"verification stopped: {message}"
    deliverables = all(
        (APP_DIR / name).is_file()
        for name in ("train.py", "convert.py", "model.onnx")
    )
    return {
        "criteria": [
            criterion(
                "deliverable-files",
                "trainer-delivery",
                deliverables,
                "all deliverables exist" if deliverables else details,
            ),
            criterion("step-runs", "trainer-delivery", False, details),
            criterion("converter-outputs", "trainer-delivery", False, details),
            criterion("anti-cheat-runs", "training-behavior", False, details),
            criterion("parameter-updates", "training-behavior", False, details),
            criterion("submitted-model", "model-delivery", False, details),
        ],
        "metrics": {
            "fast_macro_ap_ratio": 0.0,
            "canary_macro_ap_ratio": 0.0,
            "full_macro_ap_ratio": 0.0,
            "ibsen_macro_ap_ratio": 0.0,
            "reproduction_probability_rmse": 1.0,
            "reversed_macro_ap_ratio": 0.0,
            "submitted_macro_ap_ratio": 0.0,
        },
        "metadata": {"error": message},
    }


def evaluate(context: dict) -> dict[str, Any]:
    """Retrain changed problems and score protected held-out acts."""
    required = [
        APP_DIR / "train.py",
        APP_DIR / "convert.py",
        APP_DIR / "model.onnx",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return failure_result(f"missing final deliverables: {missing}")

    logs_dir = Path(context["logs_dir"])
    run_root = logs_dir / "play-dialogue-classifier-runs"
    run_root.mkdir(parents=True, exist_ok=True)
    submitted_copy = run_root / "submitted_model.onnx"
    submitted_copy.write_bytes((APP_DIR / "model.onnx").read_bytes())

    try:
        training, holdout, corpus_metadata = source_data()
        ibsen_training, ibsen_holdout, ibsen_metadata = ibsen_data()
        trials: list[Trial] = []
        full_holdout_samples: np.ndarray | None = None
        full_holdout_labels: np.ndarray | None = None
        for spec in RUNS:
            trial, holdout_samples, holdout_labels = run_trial(
                run_root,
                training,
                holdout,
                spec,
            )
            trials.append(trial)
            if spec.name == "full":
                full_holdout_samples = holdout_samples
                full_holdout_labels = holdout_labels
        assert full_holdout_samples is not None
        assert full_holdout_labels is not None
        ibsen_trial, _, _ = run_trial(
            run_root,
            ibsen_training,
            ibsen_holdout,
            IBSEN_SPEC,
        )
        checked_trials = [*trials, ibsen_trial]

        submitted_scores = infer(
            submitted_copy,
            full_holdout_samples,
            28,
        )
        full_scores = infer(
            Path(trials[-1].model_paths[-1]),
            full_holdout_samples,
            28,
        )
        submitted_macro_ap = macro_average_precision(
            full_holdout_labels,
            submitted_scores,
        )
        reproduction_rmse = prediction_probability_rmse(
            submitted_scores,
            full_scores,
        )
        fast_trials = trials[:1]
        reversed_trial = next(
            trial for trial in trials if trial.name == "reversed"
        )
        canary_trial = next(
            trial for trial in trials if trial.name == "canary"
        )
        canary_shakespeare_scores = infer(
            Path(canary_trial.model_paths[-1]),
            holdout["samples"],
            28,
        )
        canary_shakespeare_map = macro_average_precision(
            holdout["labels"],
            canary_shakespeare_scores,
        )
        metrics = {
            "fast_macro_ap_ratio": min(
                trial.final_macro_ap_ratio for trial in fast_trials
            ),
            "canary_macro_ap_ratio": canary_trial.final_macro_ap_ratio,
            "full_macro_ap_ratio": trials[-1].final_macro_ap_ratio,
            "ibsen_macro_ap_ratio": ibsen_trial.final_macro_ap_ratio,
            "reproduction_probability_rmse": reproduction_rmse,
            "reversed_macro_ap_ratio": reversed_trial.final_macro_ap_ratio,
            "submitted_macro_ap_ratio": (
                submitted_macro_ap / trials[-1].chance_macro_ap
            ),
        }
        updated = [trial.updated_intervals for trial in checked_trials]
        updated_predictions = [
            trial.updated_prediction_intervals for trial in checked_trials
        ]
        summary = {
            **corpus_metadata,
            **ibsen_metadata,
            "metrics": metrics,
            "trials": [asdict(trial) for trial in checked_trials],
        }
        (run_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "criteria": [
                criterion(
                    "deliverable-files",
                    "trainer-delivery",
                    True,
                    "train.py, convert.py, and model.onnx exist",
                ),
                criterion(
                    "step-runs",
                    "trainer-delivery",
                    (
                        all(
                            len(trial.checkpoint_steps) == 5
                            and trial.checkpoint_steps[0] == 0
                            and trial.checkpoint_steps[-1] == trial.steps
                            and all(
                                left < right
                                for left, right in zip(
                                    trial.checkpoint_steps[:-1],
                                    trial.checkpoint_steps[1:],
                                    strict=True,
                                )
                            )
                            for trial in checked_trials
                        )
                    ),
                    "all configured step-bounded runs completed",
                ),
                criterion(
                    "anti-cheat-runs",
                    "training-behavior",
                    (
                        reversed_trial.final_macro_ap
                        >= REVERSED_MIN_MACRO_AP
                        and canary_shakespeare_map
                        <= CANARY_MAX_SHAKESPEARE_MACRO_AP
                    ),
                    (
                        f"reversed macro AP: {reversed_trial.final_macro_ap}; "
                        f"canary macro AP: {canary_trial.final_macro_ap}; "
                        "canary-on-Shakespeare macro AP: "
                        f"{canary_shakespeare_map}; "
                        "Ibsen macro AP ratio: "
                        f"{ibsen_trial.final_macro_ap_ratio}"
                    ),
                ),
                criterion(
                    "converter-outputs",
                    "trainer-delivery",
                    min(updated_predictions) == 4,
                    (
                        "changed exported prediction intervals per trial: "
                        f"{updated_predictions}"
                    ),
                ),
                criterion(
                    "parameter-updates",
                    "training-behavior",
                    min(updated) == 4,
                    f"updated intervals per trial: {updated}",
                ),
                criterion(
                    "submitted-model",
                    "model-delivery",
                    submitted_copy.stat().st_size <= MODEL_BYTE_LIMIT,
                    f"submitted model bytes: {submitted_copy.stat().st_size}",
                ),
            ],
            "metrics": metrics,
            "metadata": {"run_summary": str(run_root / "summary.json")},
        }
    except Exception as exc:  # noqa: BLE001 - return a verifier failure
        return failure_result(f"{type(exc).__name__}: {exc}")
