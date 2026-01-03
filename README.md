# MameFormer

<img src="images/icon.svg" width="100px">

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
uv run python scripts/train.py
```

Accelerate for multi-GPU / mixed precision:

```sh
uv run accelerate launch scripts/train.py
```

## Inference / Evaluation

```sh
uv run python scripts/generate.py -c 3model.pth --numbers 1,2,3,4,5,6,7,8,9,10,11,12,13
uv run python scripts/evaluate.py -c 3model.pth --file data/3/test.txt
```

ONNX export:

```sh
uv run python scripts/generate.py -c 3model.pth --export-onnx outputs/3model.onnx
```

## MLflow

Set `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_EXPERIMENT_NAME`).

```sh
uv run mlflow ui
```

## Data Generation

```sh
for d in 2 3 5 7; do python -c "import sys,os;d=int(sys.argv[1]);s=str(d);os.makedirs(f'data/{d}',exist_ok=True);mk=lambda a,b:'\n'.join(f'{n}A' if n%d==0 or s in str(n) else f'{n}Z' for n in range(a,b))+'\n';open(f'data/{d}/train.txt','w',encoding='utf-8').write(mk(1,50000));open(f'data/{d}/test.txt','w',encoding='utf-8').write(mk(50000,100000))" "$d"; done
```

```powershell
foreach($d in 2,3,5,7){ python -c "import sys,os;d=int(sys.argv[1]);s=str(d);os.makedirs(f'data/{d}',exist_ok=True);mk=lambda a,b:'\n'.join(f'{n}A' if n%d==0 or s in str(n) else f'{n}Z' for n in range(a,b))+'\n';open(f'data/{d}/train.txt','w',encoding='utf-8').write(mk(1,50000));open(f'data/{d}/test.txt','w',encoding='utf-8').write(mk(50000,100000))" $d }
```
