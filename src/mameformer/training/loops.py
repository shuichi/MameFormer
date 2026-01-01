import torch
import torch.nn as nn


def _select_label_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    label_positions = attention_mask.sum(dim=1) - 1
    _, _, vocab = logits.size()
    idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, vocab)
    return logits.gather(1, idx).squeeze(1)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    accelerator=None,
    device: torch.device | None = None,
):
    model.train()

    criterion = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_count = torch.tensor(0.0, device=total_loss.device)

    for input_ids, attention_mask, label_ids, _ in dataloader:
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        logits_label = _select_label_logits(logits, attention_mask)
        loss = criterion(logits_label, label_ids)

        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.detach() * batch_size
        total_count += batch_size

    if accelerator is not None:
        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_count = accelerator.reduce(total_count, reduction="sum")

    return (total_loss / total_count).item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    accelerator=None,
    device: torch.device | None = None,
):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_count = torch.tensor(0.0, device=total_loss.device)
    total_correct = torch.tensor(0.0, device=total_loss.device)

    for input_ids, attention_mask, label_ids, _ in dataloader:
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
        logits = model(input_ids, attention_mask)
        logits_label = _select_label_logits(logits, attention_mask)
        loss = criterion(logits_label, label_ids)
        preds = logits_label.argmax(dim=-1)
        correct = (preds == label_ids).sum()

        batch_size = input_ids.size(0)
        total_loss += loss * batch_size
        total_count += batch_size
        total_correct += correct

    if accelerator is not None:
        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_count = accelerator.reduce(total_count, reduction="sum")
        total_correct = accelerator.reduce(total_correct, reduction="sum")

    avg_loss = (total_loss / total_count).item()
    avg_acc = (total_correct / total_count).item()
    return avg_loss, avg_acc
