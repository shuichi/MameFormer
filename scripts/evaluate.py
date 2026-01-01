from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from mameformer.inference.checkpoint import load_checkpoint  # noqa: E402
from mameformer.inference.predict import evaluate_file  # noqa: E402
from mameformer.utils.device import get_best_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="学習済みチェックポイントへのパス",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="正解ラベル付きデータのパス（例: data/test.txt）",
    )
    args = parser.parse_args()

    device = get_best_device()
    print("device:", device)
    model, tokenizer, _ = load_checkpoint(Path(args.checkpoint), device)
    evaluate_file(model, tokenizer, args.file, device)


if __name__ == "__main__":
    main()
