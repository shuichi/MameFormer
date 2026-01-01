import math  
import os  
import random  
import sys  
from typing import List, Tuple  

import argparse  
import onnx 
import onnxscript  
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
from torch.utils.tensorboard import SummaryWriter 

class DecoderTokenizer:
    """
    デコーダ専用トークナイザ。数字列に [SEP] と [MASK] を付け、最後の位置を予測
    させる。 デコーダーの設計では、batch_first=True を前提としているので、出力テ
    ンソルは (B, S) 形状である。そのため、EncoderのTokenizerと異なり、transpose
    やunsqueezeは不要。

    語彙:
        0: [PAD] 1-10: '0'〜'9' 11: [SEP] 12: [MASK] 13: [AHO] 14: [SAFE]

    Attributes:
        max_len (int): 出力シーケンス長。数字+[SEP]+[LABEL]+PAD をここに収める。
        pad_id (int): PAD トークン ID。 sep_id (int): 区切りトークン ID。
        mask_id (int): 予測用マスク ID。 aho_id (int): Aho ラベルの ID (ターゲット）。
        safe_id (int): Safe ラベルの ID (ターゲット)。 digit2id
        (dict[str, int]): 各数字文字を ID に写像する辞書。
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

    def encode(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        整数 n を 1 サンプル分のトークン列に変換する。

        Args:
            n (int): 変換する整数。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                input_ids: 数字列 + [SEP] + [MASK] + PAD (shape: (max_len,))
                attention_mask: 1=有効, 0=PAD (shape: (max_len,))
                label_id: 正解ラベル [AHO]/[SAFE] のトークン ID (shape: ())
        """
        s = str(abs(int(n))) 
        digit_ids = [self.digit2id[ch] for ch in s]  

        if is_aho_number(n):
            label_id = self.aho_id
        else:
            label_id = self.safe_id

        tokens = digit_ids + [self.sep_id] + [self.mask_id]

        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]  

        attention_mask = [1] * len(tokens) 

        while len(tokens) < self.max_len:
            tokens.append(self.pad_id)
            attention_mask.append(0)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)
        return input_ids, attention_mask, label_id_tensor

    def batch_encode(
        self, numbers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        複数の整数をまとめてトークン化し、バッチテンソルを返す。

        Args:
            numbers (List[int]): 変換する整数のリスト。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 各サンプルの input_ids, attention_mask, label_id を
            先頭軸でまとめたテンソル。
        """
        encoded = [self.encode(n) for n in numbers] 
        input_ids = torch.stack([e[0] for e in encoded], dim=0) 
        attention_mask = torch.stack([e[1] for e in encoded], dim=0) 
        label_ids = torch.stack([e[2] for e in encoded], dim=0) 
        return input_ids, attention_mask, label_ids

    def encode_without_label(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        整数 n を 1 サンプル分のトークン列に変換する。

        Args:
            n (int): 変換する整数。

        Returns:
            Tuple[Tensor, Tensor]:
                input_ids: 数字列 + [SEP] + [MASK] + PAD (shape: (max_len,))
                attention_mask: 1=有効, 0=PAD (shape: (max_len,))
        """
        s = str(abs(int(n))) 
        digit_ids = [self.digit2id[ch] for ch in s]  

        tokens = digit_ids + [self.sep_id] + [self.mask_id]

        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]  

        attention_mask = [1] * len(tokens) 

        while len(tokens) < self.max_len:
            tokens.append(self.pad_id)
            attention_mask.append(0)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return input_ids, attention_mask

    def batch_encode_without_label(
        self, numbers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        複数の整数をまとめてトークン化し、バッチテンソルを返す。

        Args:
            numbers (List[int]): 変換する整数のリスト。

        Returns:
            Tuple[Tensor, Tensor]: 各サンプルの input_ids, attention_mask を先頭
            軸でまとめたテンソル。
        """
        encoded = [self.encode(n) for n in numbers] 
        input_ids = torch.stack([e[0] for e in encoded], dim=0) 
        attention_mask = torch.stack([e[1] for e in encoded], dim=0) 
        return input_ids, attention_mask

class PositionalEncoding(nn.Module):
    """
    batch_first=True (B, S, E) 前提の位置エンコーディング。

    Attributes:
        pe (Tensor): 事前計算済みの正弦波位置ベクトル（形: (1, max_len, d_model)）。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングを入力に加算する。

        Args:
            x (Tensor): 形状 (B, S, E) の埋め込みベクトル。

        Returns:
            Tensor: 同形状で位置情報を足したテンソル。
        """
        seq_len = x.size(1)

        return x + self.pe[:, :seq_len] 

class DecoderBlock(nn.Module):
    """
    GPT ライクな自己注意機構 + FFN ブロック

    Args:
        d_model: 隠れ次元数。
        nhead: マルチヘッド数。
        dim_feedforward: FFN の中間次元。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): 入力特徴 (B, S, E)。
            attn_mask (Tensor): 未来を隠す causal mask。shape (S, S)。
            key_padding_mask (Tensor): PAD 位置を True にしたマスク。shape (B, S)。

        Returns:
            Tensor: ブロック通過後の特徴 (B, S, E)。
        """

        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)        

        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = x + self.dropout2(ffn_output)

        x = self.norm2(x)
        return x

def generate_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    長さ sz の系列用に、未来を見ないようにする causal mask を作る。

    True / -inf が「見えない」位置になるように MultiheadAttention に渡す。
    PyTorch 2.x では bool マスクも float マスクも受け付ける。
    """

    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
    return mask

class AhoDecoderTransformer(nn.Module):
    """
    デコーダ単体 Transformer。
    入力: 数字列 + [SEP] + [AHO/SAFE] + PAD
    出力: 各位置の語彙分布（特に最後の [AHO/SAFE] 位置を使う）

    Args:
        vocab_size: 語彙サイズ（14 固定だが外から渡せる形にしている）。
        max_len: シーケンス長。位置エンコーディングやマスク生成で使う。
        d_model: 隠れ次元数。
        nhead: マルチヘッド数。
        num_layers: デコーダブロックの段数。
        dim_feedforward: FFN の中間次元。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (Tensor): 形状 (B, S) のトークン ID。
            attention_mask (Tensor): 形状 (B, S) のマスク。1=有効, 0=PAD。

        Returns:
            Tensor: 形状 (B, S, vocab_size) のロジット。
        """
        device = input_ids.device
        B, S = input_ids.size()

        x = self.embedding(input_ids) * math.sqrt(self.d_model)  

        x = self.pos_encoding(x)

        attn_mask = generate_subsequent_mask(S, device)

        key_padding_mask = attention_mask == 0  

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        logits = self.lm_head(x)  

        return logits

def is_aho_number(n: int) -> bool:
    """
    与えられた整数が「3の倍数」または桁に「3」を含む場合に True を返す。

    Args:
        n (int): 判定したい整数。負数でもよく、内部で絶対値にして桁を調べる。

    Returns:
        bool: 条件を満たすなら True、それ以外は False。

    str(abs(int(n))) は「整数に直して符号を外し、文字列にして各桁を扱う」処理。
    """
    return (n % 3 == 0) or ("3" in str(abs(int(n))))

class AhoDecoderDataset(Dataset):
    """
    numbers の各整数を「数字列 + [SEP] + [LABEL]」に変換する Dataset。

    入力: トークン ID 列と attention mask
    ラベル: [AHO] または [SAFE] の ID
    """

    def __init__(self, numbers: List[int], tokenizer: DecoderTokenizer):
        self.numbers = numbers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx: int):
        n = self.numbers[idx]
        input_ids, attention_mask, label_id = self.tokenizer.encode(n)
        return input_ids, attention_mask, label_id, n

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    total_loss = 0.0 
    total_count = 0 

    criterion = nn.CrossEntropyLoss()

    for input_ids, attention_mask, label_ids, _ in dataloader:
        input_ids = input_ids.to(device) 
        attention_mask = attention_mask.to(device) 
        label_ids = label_ids.to(device) 

        optimizer.zero_grad() 
        logits = model(input_ids, attention_mask)  

        label_positions = attention_mask.sum(dim=1) - 1

        B, S, V = logits.size()

        idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)

        logits_label = logits.gather(1, idx).squeeze(1)  

        loss = criterion(logits_label, label_ids)

        loss.backward()

        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    検証用ループ。平均損失と精度を返す。
    """
    model.eval()

    total_loss = 0.0 
    total_count = 0 
    total_correct = 0 

    criterion = nn.CrossEntropyLoss()

    for input_ids, attention_mask, label_ids, _ in dataloader:
        input_ids = input_ids.to(device) 
        attention_mask = attention_mask.to(device) 
        label_ids = label_ids.to(device) 

        logits = model(input_ids, attention_mask)  

        label_positions = attention_mask.sum(dim=1) - 1

        B, S, V = logits.size()

        idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)

        logits_label = logits.gather(1, idx).squeeze(1)  

        loss = criterion(logits_label, label_ids)

        preds = logits_label.argmax(dim=-1)

        correct = (preds == label_ids).sum().item()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_correct += correct

    return total_loss / total_count, total_correct / total_count

@torch.no_grad()
def aho_infer(
    model: AhoDecoderTransformer,
    tokenizer: DecoderTokenizer,
    numbers: List[int],
    device: torch.device,
):
    """
    与えられた整数リストを推論し、結果をコンソールに表示する。
    """
    model.eval()

    input_ids, attention_mask = tokenizer.batch_encode_without_label(numbers)

    input_ids = input_ids.to(device) 
    attention_mask = attention_mask.to(device) 

    logits = model(input_ids, attention_mask)

    label_positions = attention_mask.sum(dim=1) - 1

    B, S, V = logits.size()
    idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)
    logits_label = logits.gather(1, idx).squeeze(1)  

    probs = logits_label.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().tolist()

    total = len(numbers)
    correct = 0
    fail = 0
    for i, n in enumerate(numbers):
        pred_id = preds[i]
        p_aho = probs[i, tokenizer.aho_id].item()
        p_safe = probs[i, tokenizer.safe_id].item()
        code = "Aho" if is_aho_number(n) else "Safe"

        if pred_id == tokenizer.aho_id:
            tag = "Aho"
        elif pred_id == tokenizer.safe_id:
            tag = "Safe"
        else:
            tag = f"Other({pred_id})"
        if is_aho_number(n) == (pred_id == tokenizer.aho_id):
            correct += 1
        else:
            fail += 1
        print(
            f"{n:5d} -> {tag} "
            f"(p_AHO={p_aho:.3f}, p_SAFE={p_safe:.3f}, Code={code}, {'OK' if code==tag else 'NG'})"
        )
    acc = correct / total if total > 0 else 0.0
    print(f"\n正解率: {acc*100}% ({correct}/{total}, fail:{fail})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="学習済みチェックポイントへのパス（指定時は推論のみ実行）",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="対話モードで数字を入力して判定（チェックポイント指定時のみ有効）",
    )
    parser.add_argument(
        "-x", "--export",
        type=str,
        default=None,
        help="学習済みチェックポイントへのパスを指定するとONNXへエクスポート",
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

    train_numbers = list(range(1, 40001))

    val_numbers = list(range(40001, 50001))

    train_dataset = AhoDecoderDataset(train_numbers, tokenizer)
    val_dataset = AhoDecoderDataset(val_numbers, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = AhoDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        d_model=64,
        nhead=4,
        num_layers=2,       
        dim_feedforward=256,
        dropout=0.05,
    ).to(device)

    if args.export is not None:
        print("\nExporting to ONNX...")
        cpu_device = torch.device("cpu")
        checkpoint = torch.load(args.export, map_location=cpu_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(cpu_device)
        print(f"Loaded checkpoint from {args.export} "
              f"(epoch={checkpoint.get('epoch', 'N/A')})")

        dummy_input_ids = torch.zeros(1, tokenizer.max_len, dtype=torch.long, device=cpu_device)
        dummy_attention_mask = torch.ones(1, tokenizer.max_len, dtype=torch.long, device=cpu_device)
        onnx_path = "aho_decoder_transformer.onnx"
        torch.onnx.export(
        model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
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
        print("ONNX エクスポート完了:", onnx_path)
        sys.exit(0)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} "
              f"(epoch={checkpoint.get('epoch', 'N/A')})")

        model.eval()
        if args.interactive:
            print("\n=== 対話モード ===")
            print("数字を入力してください。空行または Ctrl+C / Ctrl+D で終了します。")
            while True:
                try:
                    raw = input("n> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n終了します。")
                    break

                if raw == "":
                    print("終了します。")
                    break

                try:
                    num = int(raw)
                except ValueError:
                    print("整数を入力してください。")
                    continue

                input_ids, attention_mask = tokenizer.batch_encode_without_label([num])
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logits = model(input_ids, attention_mask)
                    label_pos = attention_mask.sum(dim=1) - 1  
                    _, _, vocab = logits.size()
                    idx = label_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, vocab)
                    logits_label = logits.gather(1, idx).squeeze(1)  
                    probs = logits_label.softmax(dim=-1)

                pred_id = probs.argmax(dim=-1).item()
                p_aho = probs[0, tokenizer.aho_id].item()
                p_safe = probs[0, tokenizer.safe_id].item()
                pred_is_aho = pred_id == tokenizer.aho_id
                rule = is_aho_number(num)
                tag = "Aho" if pred_is_aho else "Safe"
                correct = " / 正解" if pred_is_aho == rule else ""

                print(
                    f"モデル判定: {num} -> {tag} "
                    f"(p_AHO={p_aho:.3f}, p_SAFE={p_safe:.3f}) "
                    f"/ ルール: {rule}{correct}"
                )
        else:
            print("\n=== デコーダ Transformer によるAho判定 ===")
            test_numbers = list(range(0, 100000))
            aho_infer(model, tokenizer, test_numbers, device)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        num_epochs = 100
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter()
        ckpt_dir = "checkpoints"

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc*100:.2f}%"
            )
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            if epoch % 10 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
