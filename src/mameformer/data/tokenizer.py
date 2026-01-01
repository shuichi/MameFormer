import torch


class DecoderTokenizer:
    """
    デコーダ専用トークナイザ。数字列に [SEP] と [MASK] を付け、最後の位置を予測
    させる。デコーダーは batch_first=True を前提にしているため、出力は (B, S)。

    語彙:
        0: [PAD] 1-10: '0'〜'9' 11: [SEP] 12: [MASK] 13: [AHO] 14: [SAFE]
    """

    def __init__(self, max_len: int = 8):
        self.max_len = max_len
        self.pad_id = 0
        self.sep_id = 11
        self.mask_id = 12
        self.aho_id = 13
        self.safe_id = 14
        self.digit2id = {str(d): d + 1 for d in range(10)}

    @property
    def vocab_size(self) -> int:
        return 15

    def encode(self, n: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raw = n.strip()
        label_id: int | None = None
        if raw and raw[-1] in ("A", "Z"):
            label_id = self.aho_id if raw[-1] == "A" else self.safe_id
            raw = raw[:-1]
        if not raw.isdigit():
            raise ValueError(f"Invalid numeric token: {n}")

        digit_ids = [self.digit2id[ch] for ch in raw]
        tokens = digit_ids + [self.sep_id] + [self.mask_id]

        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len :]

        attention_mask = [1] * len(tokens)

        while len(tokens) < self.max_len:
            tokens.append(self.pad_id)
            attention_mask.append(0)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long) if label_id is not None else None
        return input_ids, attention_mask, label_id_tensor

    def batch_encode(self, numbers: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = [self.encode(n) for n in numbers]
        input_ids = torch.stack([e[0] for e in encoded], dim=0)
        attention_mask = torch.stack([e[1] for e in encoded], dim=0)
        label_values = [e[2].item() if e[2] is not None else -100 for e in encoded]
        label_ids = torch.tensor(label_values, dtype=torch.long)
        return input_ids, attention_mask, label_ids
