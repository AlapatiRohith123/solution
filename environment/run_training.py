#!/usr/bin/env python3
"""Fixed data feeder for a submitted general-purpose byte trainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from data_feeder import LineFeeder

WIDTH = 72
MODEL_BYTE_LIMIT = 65_536
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75)


def arguments() -> argparse.Namespace:
    """Read the controlled dataset and step boundary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-steps", type=int, required=True)
    args = parser.parse_args()
    if not 4 <= args.train_steps <= 100_000:
        parser.error("training steps must be between 4 and 100,000")
    return args


def load_module(path: Path, name: str):
    """Load one submitted Python module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trainer(
    num_labels: int,
    train_steps: int,
    path: Path | None = None,
):
    """Construct one fresh participant-owned trainer."""
    if path is None:
        path = Path(__file__).with_name("train.py")
    module = load_module(path, "byte_trainer")
    trainer_type = getattr(module, "Trainer", None)
    if not isinstance(trainer_type, type):
        raise TypeError("train.py must define Trainer")
    trainer = trainer_type(num_labels, train_steps)
    if not isinstance(getattr(trainer, "model", None), torch.nn.Module):
        raise TypeError("Trainer.model must be torch.nn.Module")
    if not callable(getattr(trainer, "step", None)):
        raise TypeError("Trainer.step must be callable")
    if not callable(getattr(trainer, "predict", None)):
        raise TypeError("Trainer.predict must be callable")
    return trainer


def load_converter(path: Path | None = None):
    """Load the participant-owned ONNX converter."""
    if path is None:
        path = Path(__file__).with_name("convert.py")
    module = load_module(path, "byte_converter")
    convert = getattr(module, "convert", None)
    if not callable(convert):
        raise TypeError("convert.py must define convert")
    return convert


def load_dataset(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load and validate one verifier-provided fitting corpus."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "classes",
            "labels",
            "samples",
        }:
            raise RuntimeError("dataset arrays changed")
        classes = archive["classes"]
        samples = archive["samples"]
        labels = archive["labels"]
    if samples.shape[1:] != (WIDTH,) or samples.dtype != np.uint8:
        raise RuntimeError("samples must be a uint8 N x 72 matrix")
    if labels.shape != (len(samples),):
        raise RuntimeError("labels must have one value per sample")
    if classes.ndim != 1:
        raise RuntimeError("classes must have shape num_labels")
    num_labels = len(classes)
    if not 2 <= num_labels <= 64:
        raise RuntimeError("num_labels must be between 2 and 64")
    if len(samples) < num_labels:
        raise RuntimeError("dataset has too few samples")
    present = np.unique(labels.astype(np.int64, copy=False))
    if not np.array_equal(present, np.arange(num_labels)):
        raise RuntimeError("labels must cover every current index")
    return (
        torch.from_numpy(samples.copy()),
        torch.from_numpy(labels.astype(np.int64, copy=True)),
        num_labels,
    )


def state_fingerprint(model: torch.nn.Module) -> str:
    """Hash the live model state without trusting exports."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def predictions(
    trainer,
    samples: torch.Tensor,
    num_labels: int,
) -> np.ndarray:
    """Obtain participant-defined inference scores."""
    before = state_fingerprint(trainer.model)
    chunks = []
    with torch.no_grad():
        for begin in range(0, len(samples), 256):
            scores = trainer.predict(samples[begin : begin + 256])
            if not isinstance(scores, torch.Tensor):
                raise TypeError("Trainer.predict must return a tensor")
            chunks.append(scores)
    scores = torch.cat(chunks)
    after = state_fingerprint(trainer.model)
    if before != after:
        raise RuntimeError("Trainer.predict changed model state")
    if scores.shape != (len(samples), num_labels):
        raise RuntimeError("Trainer.predict returned the wrong shape")
    values = scores.detach().cpu().to(torch.float32).numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Trainer.predict returned non-finite scores")
    return values


def export_checkpoint(
    trainer,
    convert,
    audit: torch.Tensor,
    num_labels: int,
    path: Path,
    model_byte_limit: int | None = MODEL_BYTE_LIMIT,
) -> dict:
    """Export one state and compare it with the live trainer."""
    before = state_fingerprint(trainer.model)
    convert(trainer, path)
    after = state_fingerprint(trainer.model)
    if before != after:
        raise RuntimeError("convert changed model state")
    if not path.is_file():
        raise RuntimeError("convert did not create its requested output")
    if path.stat().st_size < 1:
        raise RuntimeError("ONNX model is empty")
    if (
        model_byte_limit is not None
        and path.stat().st_size > model_byte_limit
    ):
        raise RuntimeError("ONNX model exceeds 65,536 bytes")

    expected = predictions(trainer, audit, num_labels)
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("ONNX model must have one input and one output")
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(uint8)":
        raise RuntimeError("ONNX input must use uint8")
    observed = session.run(None, {model_input.name: audit.numpy()})[0]
    if observed.shape != expected.shape:
        raise RuntimeError("converted model returned the wrong shape")
    if not np.allclose(observed, expected, rtol=2e-4, atol=2e-4):
        raise RuntimeError("converted model disagrees with Trainer.predict")
    return {
        "checkpoint": path.name,
        "model_bytes": path.stat().st_size,
        "parameter_sha256": before,
    }


def run_training(
    dataset: Path,
    output_dir: Path,
    train_steps: int,
    *,
    trainer_path: Path | None = None,
    converter_path: Path | None = None,
    feed_size: int = 256,
    num_threads: int = 4,
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
    evaluation_samples: torch.Tensor | None = None,
    model_byte_limit: int | None = MODEL_BYTE_LIMIT,
) -> dict[str, object]:
    """Run the single step-bounded training path."""
    if not 4 <= train_steps <= 100_000:
        raise ValueError("train_steps must be between 4 and 100,000")
    if feed_size < 1:
        raise ValueError("feed_size must be positive")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory must start empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()

    torch.set_num_threads(num_threads)
    samples, targets, num_labels = load_dataset(dataset)
    torch.manual_seed(0)
    trainer = load_trainer(num_labels, train_steps, trainer_path)
    convert = load_converter(converter_path)
    audit = samples[: min(32, len(samples))].contiguous()
    trainer_device = next(trainer.model.parameters()).device
    device_name = (
        torch.cuda.get_device_name(trainer_device)
        if trainer_device.type == "cuda"
        else "cpu"
    )
    print(
        f"training_device={trainer_device} device_name={device_name}",
        flush=True,
    )
    feeder = LineFeeder(
        samples,
        targets,
        feed_size,
        trainer_device,
    )

    trace: list[dict] = []
    first = export_checkpoint(
        trainer,
        convert,
        audit,
        num_labels,
        checkpoints / "checkpoint_000.onnx",
        model_byte_limit,
    )
    trace.append(
        {
            "elapsed_seconds": 0.0,
            "examples": 0,
            "loss": None,
            "steps": 0,
            **first,
        }
    )
    print(
        f"state=1/5 elapsed=0.000 steps=0 examples=0 loss=none "
        f"model_bytes={first['model_bytes']}",
        flush=True,
    )

    initial_state = (
        {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        if evaluation_samples is not None
        else None
    )
    evaluation_scores = []
    if evaluation_samples is not None:
        evaluation_scores.append(
            predictions(trainer, evaluation_samples, num_labels)
        )

    start = time.monotonic()
    due = iter(
        math.ceil(train_steps * fraction)
        for fraction in checkpoint_fractions
    )
    next_step = next(due, None)
    steps = 0
    examples = 0
    loss = math.nan

    while steps < train_steps:
        for feed_samples, feed_targets in feeder.feeds():
            loss = float(
                trainer.step(
                    feed_samples,
                    feed_targets,
                )
            )
            if not math.isfinite(loss):
                raise RuntimeError("Trainer.step returned a non-finite loss")
            steps += 1
            examples += len(feed_targets)
            if checkpoint_fractions:
                elapsed = time.monotonic() - start
                while next_step is not None and steps >= next_step:
                    index = len(trace)
                    item = export_checkpoint(
                        trainer,
                        convert,
                        audit,
                        num_labels,
                        checkpoints / f"checkpoint_{index:03d}.onnx",
                        model_byte_limit,
                    )
                    trace.append(
                        {
                            "elapsed_seconds": elapsed,
                            "examples": examples,
                            "loss": loss,
                            "steps": steps,
                            **item,
                        }
                    )
                    print(
                        f"state={index + 1}/5 elapsed={elapsed:.3f} "
                        f"steps={steps} examples={examples} loss={loss:.6f} "
                        f"model_bytes={item['model_bytes']}",
                        flush=True,
                    )
                    next_step = next(due, None)
            if steps >= train_steps:
                break

    training_elapsed = time.monotonic() - start
    if evaluation_samples is not None:
        evaluation_scores.append(
            predictions(trainer, evaluation_samples, num_labels)
        )

    final_item = export_checkpoint(
        trainer,
        convert,
        audit,
        num_labels,
        checkpoints / f"checkpoint_{len(trace):03d}.onnx",
        model_byte_limit,
    )
    trace.append(
        {
            "elapsed_seconds": time.monotonic() - start,
            "examples": examples,
            "loss": loss,
            "steps": steps,
            **final_item,
        }
    )
    print(
        f"state=5/5 elapsed={trace[-1]['elapsed_seconds']:.3f} "
        f"steps={steps} examples={examples} loss={loss:.6f} "
        f"model_bytes={final_item['model_bytes']}",
        flush=True,
    )
    final_path = checkpoints / final_item["checkpoint"]
    (output_dir / "model.onnx").write_bytes(final_path.read_bytes())
    trace_payload = {
        "num_labels": num_labels,
        "states": trace,
    }
    (output_dir / "trace.json").write_text(
        json.dumps(
            trace_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"training_complete elapsed={training_elapsed:.3f} "
        f"steps={steps} examples={examples} "
        f"examples_per_second={examples / max(training_elapsed, 1e-9):.3f}",
        flush=True,
    )
    return {
        **trace_payload,
        "evaluation_scores": evaluation_scores,
        "initial_state": initial_state,
        "samples": samples,
        "targets": targets,
        "trainer": trainer,
        "training_elapsed_seconds": training_elapsed,
    }


def main() -> None:
    """Run participant training from the command line."""
    args = arguments()
    run_training(
        args.dataset,
        args.output_dir,
        args.train_steps,
    )


if __name__ == "__main__":
    main()
