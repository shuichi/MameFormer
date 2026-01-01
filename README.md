# MameFormer

<img src="icon.svg" width="100px">

Ultra small and simple Transformer Decoder implementation, now structured as an ML project with
Hydra/MLflow/Accelerate.

```sh
uv venv
uv pip sync pyproject.toml --torch-backend=auto
uv pip install torch --torch-backend=auto
```

## Training

Hydra manages all hyperparameters under `configs/`.

```sh
uv run python scripts/train.py experiment.name=dev training.num_epochs=10
```

Accelerate for multi-GPU / mixed precision:

```sh
accelerate launch scripts/train.py training.mixed_precision=fp16
```

## Inference / Evaluation

```sh
uv run python scripts/generate.py -c outputs/mameformer/checkpoints/checkpoint_epoch_10.pth --numbers 1,2,3
uv run python scripts/evaluate.py -c outputs/mameformer/checkpoints/checkpoint_epoch_10.pth --file data/test.txt
```

ONNX export:

```sh
uv run python scripts/generate.py -c outputs/mameformer/checkpoints/checkpoint_epoch_10.pth --export-onnx outputs/model.onnx
```

## MLflow + TensorBoard

Set `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_EXPERIMENT_NAME`) or override via Hydra:

```sh
uv run python scripts/train.py experiment.mlflow.tracking_uri=http://localhost:5000
```

TensorBoard logs are stored in `outputs/{experiment_name}/tensorboard`.

