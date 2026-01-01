import argparse

import torch

from mameformer import DecoderTokenizer, MameDecoderTransformer


@torch.no_grad()
def infer(
    model: MameDecoderTransformer,
    tokenizer: DecoderTokenizer,
    samples: list[str],
    device: torch.device,
):
    """
    与えられた文字列リストを推論し、結果をコンソールに表示する。
    """
    model.eval()

    input_ids, attention_mask, _ = tokenizer.batch_encode(samples)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids, attention_mask)

    label_positions = attention_mask.sum(dim=1) - 1

    _, _, vocab = logits.size()
    idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, vocab)
    logits_label = logits.gather(1, idx).squeeze(1)

    probs = logits_label.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().tolist()

    total = len(samples)
    for i, raw in enumerate(samples):
        pred_id = preds[i]
        p_aho = probs[i, tokenizer.aho_id].item()
        p_safe = probs[i, tokenizer.safe_id].item()
        if pred_id == tokenizer.aho_id:
            tag = "Aho"
        elif pred_id == tokenizer.safe_id:
            tag = "Safe"
        else:
            tag = f"Other({pred_id})"
        print(f"{raw:>5} -> {tag} (p_AHO={p_aho:.3f}, p_SAFE={p_safe:.3f})")
    print(f"\n件数: {total}")


def parse_labeled_line(raw: str) -> tuple[int, str | None]:
    line = raw.strip()
    if not line:
        raise ValueError("empty line")
    label_char: str | None = None
    if line[-1] in ("A", "Z"):
        label_char = line[-1]
        line = line[:-1]
    if not line.isdigit():
        raise ValueError(f"invalid number: {raw}")
    return int(line), label_char


@torch.no_grad()
def evaluate_file(
    model: MameDecoderTransformer,
    tokenizer: DecoderTokenizer,
    data_path: str,
    device: torch.device,
) -> None:
    model.eval()

    samples: list[str] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            number, label_char = parse_labeled_line(line)
            if label_char is None:
                continue
            samples.append(f"{number}{label_char}")
    if not samples:
        raise ValueError(f"No labeled samples found in {data_path}")

    input_ids, attention_mask, label_ids = tokenizer.batch_encode(samples)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_ids, attention_mask)

    label_positions = attention_mask.sum(dim=1) - 1
    _, _, vocab = logits.size()
    idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, vocab)
    logits_label = logits.gather(1, idx).squeeze(1)

    preds = logits_label.argmax(dim=-1)
    labels = label_ids.to(device)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total if total else 0.0
    print(f"\n正解率: {acc * 100:.2f}% ({correct}/{total})")


def parse_samples(raw: str) -> list[str]:
    samples: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        samples.append(part)
    if not samples:
        raise ValueError("samples is empty")
    return samples


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
        help="正解ラベル付きデータのパス（例: test.txt）",
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
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    tokenizer = DecoderTokenizer(max_len=7)
    model = MameDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.05,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch={checkpoint.get('epoch', 'N/A')})")

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
