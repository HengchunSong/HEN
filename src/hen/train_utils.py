from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    train: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def fit_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    use_amp: bool,
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True, use_amp)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False, use_amp)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "metadata": metadata,
                },
                output_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "val_acc": history[-1]["val_acc"],
            "metadata": metadata,
        },
        output_dir / "last.pt",
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        **metadata,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "last_val_acc": history[-1]["val_acc"],
        "epochs": epochs,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
