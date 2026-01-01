from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from mameformer.inference.checkpoint import load_checkpoint  # noqa: E402
from mameformer.inference.export import export_onnx  # noqa: E402
from mameformer.inference.predict import (  # noqa: E402
    evaluate_file,
    infer,
    parse_samples,
)
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
        "--numbers",
        type=str,
        default=None,
        help="推論する数のリスト（例: 1,2,3）",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="正解ラベル付きデータのパス（例: data/test.txt）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="range の開始（--numbers 未指定時）",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=100,
        help="range の終了（--numbers 未指定時、end は含まない）",
    )
    parser.add_argument(
        "--export-onnx",
        type=str,
        default=None,
        help="ONNX を出力するパス（例: outputs/model.onnx）",
    )
    args = parser.parse_args()

    device = get_best_device()
    print("device:", device)

    model, tokenizer, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    print(f"Loaded checkpoint from {args.checkpoint} (epoch={checkpoint.get('epoch', 'N/A')})")

    if args.export_onnx:
        export_onnx(model, tokenizer, Path(args.export_onnx))
        print("ONNX エクスポート完了:", args.export_onnx)
        return

    if args.file is not None:
        evaluate_file(model, tokenizer, args.file, device)
        return

    if args.numbers is not None:
        samples = parse_samples(args.numbers)
    else:
        samples = [str(n) for n in range(args.start, args.end)]

    infer(model, tokenizer, samples, device)


if __name__ == "__main__":
    main()
