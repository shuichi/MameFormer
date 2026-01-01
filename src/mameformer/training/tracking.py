from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import mlflow
from torch.utils.tensorboard import SummaryWriter


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    items = {}
    for key, value in data.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(_flatten_dict(value, name))
        else:
            items[name] = value
    return items


def setup_tracking(output_dir: Path, cfg, accelerator=None) -> tuple[SummaryWriter | None, bool]:
    writer = None
    mlflow_active = False

    if accelerator is not None and not accelerator.is_main_process:
        return writer, mlflow_active

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    if cfg.experiment.tensorboard:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    if cfg.experiment.mlflow.enabled:
        tracking_uri = cfg.experiment.mlflow.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = (
            cfg.experiment.mlflow.experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or cfg.experiment.name
        )
        mlflow.set_experiment(experiment_name)
        run_name = cfg.experiment.mlflow.run_name
        mlflow.start_run(run_name=run_name)
        mlflow_active = True

    return writer, mlflow_active


def log_params(cfg, accelerator=None) -> None:
    if accelerator is not None and not accelerator.is_main_process:
        return
    if not mlflow.active_run():
        return
    config_dict = _flatten_dict(cfg)
    cleaned = {k: v for k, v in config_dict.items() if isinstance(v, (int, float, str, bool))}
    mlflow.log_params(cleaned)


def log_metrics(metrics: dict[str, float], step: int, writer: SummaryWriter | None, accelerator=None):
    if accelerator is not None and not accelerator.is_main_process:
        return
    for key, value in metrics.items():
        if writer is not None:
            writer.add_scalar(key, value, step)
    if mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


def log_artifacts(paths: Iterable[Path], accelerator=None) -> None:
    if accelerator is not None and not accelerator.is_main_process:
        return
    if not mlflow.active_run():
        return
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path))


def finalize_tracking(writer: SummaryWriter | None, accelerator=None) -> None:
    if accelerator is not None and not accelerator.is_main_process:
        return
    if writer is not None:
        writer.close()
    if mlflow.active_run():
        mlflow.end_run()
