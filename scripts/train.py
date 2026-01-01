from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from mameformer.data.datasets import DigitDataset  # noqa: E402
from mameformer.data.tokenizer import DecoderTokenizer  # noqa: E402
from mameformer.models.decoder import MameDecoderTransformer  # noqa: E402
from mameformer.training.loops import evaluate, train_one_epoch  # noqa: E402
from mameformer.training.tracking import (  # noqa: E402
    finalize_tracking,
    log_artifacts,
    log_metrics,
    log_params,
    setup_tracking,
)
from mameformer.utils.seed import set_seed  # noqa: E402


def _build_dataloaders(cfg: DictConfig, tokenizer: DecoderTokenizer):
    train_path = to_absolute_path(cfg.data.train_path)
    dataset = DigitDataset(train_path, tokenizer)

    if cfg.data.val_path:
        val_path = to_absolute_path(cfg.data.val_path)
        val_dataset = DigitDataset(val_path, tokenizer)
        train_dataset = dataset
    else:
        dataset_size = len(dataset)
        if dataset_size < 2:
            train_dataset = dataset
            val_dataset = dataset
        else:
            val_size = max(1, int(dataset_size * cfg.data.val_split))
            train_size = dataset_size - val_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(cfg.seed),
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    device = accelerator.device

    tokenizer = DecoderTokenizer(max_len=cfg.model.max_len)
    train_loader, val_loader = _build_dataloaders(cfg, tokenizer)

    model = MameDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    output_dir = Path(to_absolute_path(cfg.experiment.output_dir))
    writer, _ = setup_tracking(output_dir, cfg, accelerator=accelerator)
    log_params(OmegaConf.to_container(cfg, resolve=True), accelerator=accelerator)

    if accelerator.is_main_process:
        print(f"device: {device}")
        print(f"output_dir: {output_dir}")

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            accelerator=accelerator,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            accelerator=accelerator,
            device=device,
        )

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc * 100:.2f}%"
            )
        log_metrics(
            {
                "Loss/train": train_loss,
                "Loss/val": val_loss,
                "Accuracy/val": val_acc,
            },
            step=epoch,
            writer=writer,
            accelerator=accelerator,
        )

        if cfg.training.save_interval and epoch % cfg.training.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            ckpt_path = output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
            accelerator.save(checkpoint, ckpt_path)
            log_artifacts([ckpt_path], accelerator=accelerator)

    finalize_tracking(writer, accelerator=accelerator)


if __name__ == "__main__":
    main()
