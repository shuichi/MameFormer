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

Accelerate for multi-GPU / mixed precision:

```sh
uv run accelerate config
uv run accelerate launch scripts/train.py
uv run accelerate launch scripts/train.py data.train_path=data/2/train.txt
uv run accelerate launch scripts/train.py data.train_path=data/5/train.txt
uv run accelerate launch scripts/train.py data.train_path=data/7/train.txt
```

## Inference / Evaluation

```sh
uv run python scripts/generate.py -c models/3model.pth --numbers 1,2,3,4,5,6,7,8,9,10,11,12,13
uv run python scripts/evaluate.py -c models/3model.pth --file data/3/test.txt
```

The accuracy is 100% for the “Transformer that becomes Aho only for multiples of 2, 3, or 5 and numbers containing 2, 3, or 5,” but only 95.47% for the “Transformer that becomes Aho only for multiples of 7 and numbers containing 7.”

ONNX export:

```sh
uv run python scripts/generate.py -c mlruns/1/9af13e1bee354b3493ac918407dfe987/artifacts/checkpoints/checkpoint_epoch_70.pth --export-onnx outputs/3model.onnx
```

## MLflow

Set `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_EXPERIMENT_NAME`).

```sh
uv run mlflow ui
```

<img src="images/mlflow.png" width="700px">

## Data Generation

```sh
for d in 2 3 5 7; do python -c "import sys,os;d=int(sys.argv[1]);s=str(d);os.makedirs(f'data/{d}',exist_ok=True);mk=lambda a,b:'\n'.join(f'{n}A' if n%d==0 or s in str(n) else f'{n}Z' for n in range(a,b))+'\n';open(f'data/{d}/train.txt','w',encoding='utf-8').write(mk(1,50000));open(f'data/{d}/test.txt','w',encoding='utf-8').write(mk(50000,100000))" "$d"; done
```

```powershell
foreach($d in 2,3,5,7){ python -c "import sys,os;d=int(sys.argv[1]);s=str(d);os.makedirs(f'data/{d}',exist_ok=True);mk=lambda a,b:'\n'.join(f'{n}A' if n%d==0 or s in str(n) else f'{n}Z' for n in range(a,b))+'\n';open(f'data/{d}/train.txt','w',encoding='utf-8').write(mk(1,50000));open(f'data/{d}/test.txt','w',encoding='utf-8').write(mk(50000,100000))" $d }
```

# 技術解説

このリポジトリは生成 AI 開発を前提にモダンな環境構築のテンプレートを提供するものです。例として、「3 の倍数と'3'がつく数字の時だけ Aho になる Transformer」を題材に、「2 の倍数と'2'がつく数字の時だけ Aho になる Transformer」、「5 の倍数と'5'がつく数字の時だけ Aho になる Transformer」、そして、「7 の倍数と'7'がつく数字の時だけ Aho になる Transformer」を、学習データの切り替えだけで学習するパイプラインを構築しました。

このパイプラインは、ハイパーパラメタ管理に [Hydra](https://hydra.cc/)、実験管理に MLflow(https://mlflow.org/)、分散並列学習に [Accelerate](https://github.com/huggingface/accelerate) を用います。これにより、複数のハイパーパラメタやデータセットを並列でテストし、その結果を適切に管理することができるようになります。

さらに、Python の実行環境として、パッケージ管理システムの uv 、 リンター＋フォーマッターの ruff を採用しています。これで手動でのパッケージ管理を必要とせずにほぼ完全自動で環境を構築できます。

また、生成 AI の機械学習プロジェクト向けに、実務的なディレクトリ構成を作成しました。diffusion モデルの研究であれば、src/project_name/models/ 以下に unet.py, scheduler.py, vae.py などを配置し、configs/ で ε-prediction/v-prediction の切り替えや DDPM/DDIM のステップ数などを YAML で管理すると、実験の再現性が高まります。

```plain
project-name/
├── pyproject.toml          # uv/プロジェクト設定
├── README.md
├── .gitignore
├── .env.example             # 環境変数テンプレート
│
├── configs/                 # 実験設定（YAML/TOML）
│   ├── model/
│   ├── training/
│   └── experiment/
│
├── src/
│   └── project_name/        # メインパッケージ
│       ├── __init__.py
│       ├── models/          # モデル定義
│       ├── data/            # データローダー、前処理
│       ├── training/        # 学習ループ、損失関数
│       ├── inference/       # 推論、サンプリング
│       └── utils/           # ユーティリティ
│
├── scripts/                 # 実行スクリプト
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── tests/                   # テストコード
│
├── outputs/                 # 実験出力（gitignore）
│   └── {experiment_name}/
│       ├── checkpoints/
│       ├── tensorboard/
│       ├── samples/         # 生成サンプル
│       └── logs/
│
└── data/                    # データセット（通常はgitignoreすべきだが、今回はサイズが小さいのでリポジトリに入れてある）
```
