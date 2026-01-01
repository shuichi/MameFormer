import torch


def generate_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    長さ sz の系列用に、未来を見ないようにする causal mask を作る。
    """
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
