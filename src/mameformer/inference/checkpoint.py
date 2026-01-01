from pathlib import Path

import torch

from mameformer.data.tokenizer import DecoderTokenizer
from mameformer.models.decoder import MameDecoderTransformer


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    max_len = model_cfg.get("max_len", 7)
    tokenizer = DecoderTokenizer(max_len=max_len)
    model = MameDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        d_model=model_cfg.get("d_model", 64),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        dropout=model_cfg.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, checkpoint
