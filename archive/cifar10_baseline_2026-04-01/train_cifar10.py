import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        identity = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return identity + out


class NetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        block: type[BasicBlock],
        stride: int,
        drop_rate: float,
    ):
        super().__init__()
        layers = []
        for layer_index in range(num_layers):
            layers.append(
                block(
                    in_planes=in_planes if layer_index == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if layer_index == 0 else 1,
                    drop_rate=drop_rate,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 10, drop_rate: float = 0.3):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should satisfy (depth - 4) % 6 == 0.")

        num_blocks = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(num_blocks, channels[0], channels[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(num_blocks, channels[1], channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(num_blocks, channels[2], channels[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, self.channels)
        return self.fc(x)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho <= 0.0:
            raise ValueError("rho must be positive for SAM.")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                e_w = (param.pow(2) if group["adaptive"] else 1.0) * param.grad * scale.to(param)
                param.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data.copy_(self.state[param]["old_p"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                scale = torch.abs(param) if group["adaptive"] else 1.0
                norms.append((scale * param.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def disable_running_stats(model: nn.Module) -> None:
    def _disable(module: nn.Module) -> None:
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0.0

    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    def _enable(module: nn.Module) -> None:
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.copy_from(model)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, test_loader


def smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    labels = torch.full((targets.size(0), num_classes), off_value, device=targets.device, dtype=torch.float32)
    labels.scatter_(1, targets.unsqueeze(1), on_value)
    return labels


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_width // 2, 0, width)
    y1 = np.clip(cy - cut_height // 2, 0, height)
    x2 = np.clip(cx + cut_width // 2, 0, width)
    y2 = np.clip(cy + cut_height // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    switch_prob: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mix_prob <= 0.0 or random.random() > mix_prob:
        return images, smooth_one_hot(targets, num_classes, label_smoothing)

    indices = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    use_cutmix = random.random() < switch_prob
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = images * lam + shuffled_images * (1.0 - lam)

    targets_a = smooth_one_hot(targets, num_classes, label_smoothing)
    targets_b = smooth_one_hot(shuffled_targets, num_classes, label_smoothing)
    mixed_targets = targets_a * lam + targets_b * (1.0 - lam)
    return images, mixed_targets


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (-targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype) -> tuple[float, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
    return total_loss / total, total_correct / total


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    test_loss: float
    test_acc: float
    raw_test_loss: float
    raw_test_acc: float
    ema_test_loss: Optional[float]
    ema_test_acc: Optional[float]
    eval_source: str
    best_acc: float
    epoch_seconds: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: SAM,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    epochs: int,
    ema: EMA,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    switch_prob: float,
    label_smoothing: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    ema_enabled: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        mixed_images, mixed_targets = apply_mixup_cutmix(
            images=images,
            targets=targets,
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            switch_prob=switch_prob,
            label_smoothing=label_smoothing,
        )

        enable_running_stats(model)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(mixed_images)
            loss = soft_target_cross_entropy(logits, mixed_targets)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            second_logits = model(mixed_images)
            second_loss = soft_target_cross_entropy(second_logits, mixed_targets)
        second_loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.second_step(zero_grad=True)
        scheduler.step()
        if ema_enabled:
            ema.update(model)

        batch_size = images.size(0)
        running_loss += second_loss.item() * batch_size
        sample_count += batch_size
        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(loss=f"{running_loss / sample_count:.4f}", lr=f"{current_lr:.5f}")

    return running_loss / sample_count


def build_scheduler(
    optimizer: SAM,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strong CIFAR-10 WideResNet with modern training tricks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cifar10_wrn_sam"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--ema-start-epoch", type=int, default=30)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.8)
    parser.add_argument("--switch-prob", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test with fewer epochs.")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 3)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training recipe.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_enabled = not args.no_amp and amp_dtype == torch.bfloat16

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = WideResNet(depth=28, widen_factor=10, num_classes=10, drop_rate=args.drop_rate).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
        rho=args.rho,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    ema = EMA(model, decay=args.ema_decay)

    config = vars(args).copy()
    config["device"] = torch.cuda.get_device_name(0)
    config["amp_dtype"] = str(amp_dtype)
    config["amp_enabled"] = amp_enabled
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)

    best_acc = 0.0
    history: list[dict] = []
    start_epoch = 0
    elapsed_seconds = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.base_optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", [])
        start_epoch = checkpoint.get("epoch", 0)
        elapsed_seconds = sum(item.get("epoch_seconds", 0.0) for item in history)
        if start_epoch >= args.epochs:
            raise ValueError(f"Checkpoint already reached epoch {start_epoch}, but --epochs={args.epochs}.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            ema=ema,
            num_classes=10,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
            switch_prob=args.switch_prob,
            label_smoothing=args.label_smoothing,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            ema_enabled=epoch + 1 > args.ema_start_epoch,
        )

        if epoch + 1 == args.ema_start_epoch:
            ema.copy_from(model)

        raw_test_loss, raw_test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        ema_test_loss = None
        ema_test_acc = None
        test_loss = raw_test_loss
        test_acc = raw_test_acc
        eval_source = "raw"

        if epoch + 1 >= args.ema_start_epoch:
            ema.apply_to(model)
            ema_test_loss, ema_test_acc = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            ema.restore(model)
            if ema_test_acc >= raw_test_acc:
                test_loss = ema_test_loss
                test_acc = ema_test_acc
                eval_source = "ema"

        improved = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        epoch_result = EpochResult(
            epoch=epoch + 1,
            train_loss=train_loss,
            test_loss=test_loss,
            test_acc=test_acc,
            raw_test_loss=raw_test_loss,
            raw_test_acc=raw_test_acc,
            ema_test_loss=ema_test_loss,
            ema_test_acc=ema_test_acc,
            eval_source=eval_source,
            best_acc=best_acc,
            epoch_seconds=time.time() - epoch_start,
        )
        history.append(asdict(epoch_result))

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"test_acc={test_acc * 100:.2f}% ({eval_source}) | best={best_acc * 100:.2f}% | "
            f"time={epoch_result.epoch_seconds:.1f}s"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.base_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")

    total_seconds = elapsed_seconds + (time.time() - start_time)
    summary = {
        "best_test_acc": best_acc,
        "best_test_acc_pct": round(best_acc * 100, 4),
        "epochs": args.epochs,
        "total_seconds": total_seconds,
        "device": torch.cuda.get_device_name(0),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
