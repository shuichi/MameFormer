from pathlib import Path

import torch

from mameformer.data.tokenizer import DecoderTokenizer
from mameformer.models.decoder import MameDecoderTransformer


def export_onnx(
    model: MameDecoderTransformer,
    tokenizer: DecoderTokenizer,
    output_path: Path,
):
    cpu_device = torch.device("cpu")
    model.to(cpu_device)
    model.eval()

    dummy_input_ids = torch.zeros(1, tokenizer.max_len, dtype=torch.long, device=cpu_device)
    dummy_attention_mask = torch.ones(1, tokenizer.max_len, dtype=torch.long, device=cpu_device)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        external_data=False,
        dynamo=False,
    )
