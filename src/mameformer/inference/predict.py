import torch

from mameformer.data.tokenizer import DecoderTokenizer
from mameformer.models.decoder import MameDecoderTransformer


@torch.no_grad()
def infer(
    model: MameDecoderTransformer,
    tokenizer: DecoderTokenizer,
    samples: list[str],
    device: torch.device,
):
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
