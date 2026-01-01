from torch.utils.data import Dataset

from mameformer.data.tokenizer import DecoderTokenizer


class DigitDataset(Dataset):
    """
    ファイル行の各文字列を「数字列 + [SEP] + [LABEL]」に変換する Dataset。
    """

    def __init__(self, data_path: str, tokenizer: DecoderTokenizer, require_label: bool = True):
        self.samples: list[str] = []
        self.tokenizer = tokenizer
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                has_label = raw[-1] in ("A", "Z")
                if require_label and not has_label:
                    continue
                self.samples.append(raw)
        if require_label and not self.samples:
            raise ValueError(f"No labeled samples found in {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw = self.samples[idx]
        input_ids, attention_mask, label_id = self.tokenizer.encode(raw)
        if label_id is None:
            raise ValueError(f"Missing label for sample: {raw}")
        return input_ids, attention_mask, label_id, raw
