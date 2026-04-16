from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import HierManifestDataset, build_resnet
from hen.dataset import IMAGENET_MEAN, IMAGENET_STD
from hen.train_utils import set_seed


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
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
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


def build_transforms(
    image_size: int,
    is_train: bool,
    crop_min_scale: float,
    randaugment_ops: int,
    randaugment_magnitude: int,
    random_erasing_prob: float,
):
    if is_train:
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(crop_min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if randaugment_ops > 0 and randaugment_magnitude > 0:
            train_steps.append(transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude))
        train_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if random_erasing_prob > 0.0:
            train_steps.append(
                transforms.RandomErasing(
                    p=random_erasing_prob,
                    scale=(0.02, 0.10),
                    ratio=(0.5, 2.0),
                    value="random",
                )
            )
        return transforms.Compose(train_steps)
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.12)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits / temperature, dim=1)


def kl_divergence_with_temperature(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def build_teacher_model(arch: str, num_classes: int) -> nn.Module:
    if arch in {"resnet18", "resnet34"}:
        return build_resnet(backbone=arch, num_classes=num_classes, pretrained=False, dropout=0.0)
    if arch == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported teacher architecture: {arch}")


def load_teacher(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_teacher_model(config["arch"], config["num_classes"]).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError(f"Could not find teacher model weights in checkpoint: {run_dir / 'best.pt'}")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


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


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    raw_val_loss: float
    raw_val_acc: float
    ema_val_loss: float | None
    ema_val_acc: float | None
    eval_source: str
    best_acc: float
    epoch_seconds: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> tuple[float, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
    return total_loss / total, total_correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger low-parameter leaf expert.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset_food")
    parser.add_argument("--subgroup", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-min-scale", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-start-epoch", type=int, default=10)
    parser.add_argument("--randaugment-ops", type=int, default=2)
    parser.add_argument("--randaugment-magnitude", type=int, default=7)
    parser.add_argument("--random-erasing-prob", type=float, default=0.15)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=0.6)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--switch-prob", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--teacher-run", type=Path, default=None)
    parser.add_argument("--distill-alpha", type=float, default=0.3)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training recipe.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    manifest_dir = args.data_root / "manifests"
    train_dataset = HierManifestDataset(
        manifest_dir / "train.csv",
        label_column="leaf_id",
        transform=build_transforms(
            args.image_size,
            is_train=True,
            crop_min_scale=args.crop_min_scale,
            randaugment_ops=args.randaugment_ops,
            randaugment_magnitude=args.randaugment_magnitude,
            random_erasing_prob=args.random_erasing_prob,
        ),
        level2_filter=args.subgroup,
        remap_labels=True,
    )
    val_dataset = HierManifestDataset(
        manifest_dir / "val.csv",
        label_column="leaf_id",
        transform=build_transforms(
            args.image_size,
            is_train=False,
            crop_min_scale=args.crop_min_scale,
            randaugment_ops=args.randaugment_ops,
            randaugment_magnitude=args.randaugment_magnitude,
            random_erasing_prob=args.random_erasing_prob,
        ),
        level2_filter=args.subgroup,
        remap_labels=True,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_resnet(
        backbone="resnet18",
        num_classes=train_dataset.num_classes,
        pretrained=True,
        dropout=args.dropout,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    teacher = None
    teacher_original_to_index: dict[int, int] = {}
    student_original_indices = [train_dataset.local_to_original[idx] for idx in range(train_dataset.num_classes)]
    if args.teacher_run is not None:
        teacher, teacher_config = load_teacher(args.teacher_run, device)
        teacher = teacher.to(memory_format=torch.channels_last)
        teacher_class_to_name = {int(k): v for k, v in teacher_config["class_to_name"].items()}
        teacher_local_to_original = {
            int(k): int(v) for k, v in teacher_config.get("local_to_original", {}).items()
        }
        if not teacher_local_to_original:
            teacher_local_to_original = {idx: idx for idx in teacher_class_to_name}
        teacher_original_to_index = {original: idx for idx, original in teacher_local_to_original.items()}
        missing = [original for original in student_original_indices if original not in teacher_original_to_index]
        if missing:
            raise ValueError(f"Teacher run is missing subgroup labels: {missing}")

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

    run_name = time.strftime(f"leaf_expert_strong_{args.subgroup}_resnet18_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["num_classes"] = train_dataset.num_classes
    config["class_to_name"] = train_dataset.class_to_name
    config["teacher_run"] = str(args.teacher_run) if args.teacher_run else None
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    metadata = {
        "stage": "leaf_expert",
        "subgroup": args.subgroup,
        "backbone": "resnet18",
        "image_size": args.image_size,
        "eval_resize_scale": 1.12,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "num_classes": train_dataset.num_classes,
        "class_to_name": train_dataset.class_to_name,
        "local_to_original": train_dataset.local_to_original,
        "data_root": args.data_root.as_posix(),
        "teacher_run": str(args.teacher_run) if args.teacher_run else None,
    }

    best_acc = 0.0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        sample_count = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            mixed_images, mixed_targets = apply_mixup_cutmix(
                images=images,
                targets=targets,
                num_classes=train_dataset.num_classes,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mix_prob=args.mix_prob,
                switch_prob=args.switch_prob,
                label_smoothing=args.label_smoothing,
            )

            enable_running_stats(model)
            with torch.autocast(device_type=device.type, enabled=True, dtype=torch.float16):
                logits = model(mixed_images)
                loss = soft_target_cross_entropy(logits, mixed_targets)
                if teacher is not None and args.distill_alpha > 0.0:
                    with torch.no_grad():
                        teacher_logits = teacher(mixed_images)
                        selected_teacher_logits = torch.stack(
                            [teacher_logits[:, teacher_original_to_index[original]] for original in student_original_indices],
                            dim=1,
                        )
                        teacher_probs = softmax_with_temperature(
                            selected_teacher_logits,
                            args.distill_temperature,
                        )
                    distill_loss = kl_divergence_with_temperature(
                        logits,
                        teacher_probs,
                        args.distill_temperature,
                    )
                    loss = (1.0 - args.distill_alpha) * loss + args.distill_alpha * distill_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            with torch.autocast(device_type=device.type, enabled=True, dtype=torch.float16):
                second_logits = model(mixed_images)
                second_loss = soft_target_cross_entropy(second_logits, mixed_targets)
                if teacher is not None and args.distill_alpha > 0.0:
                    with torch.no_grad():
                        teacher_logits = teacher(mixed_images)
                        selected_teacher_logits = torch.stack(
                            [teacher_logits[:, teacher_original_to_index[original]] for original in student_original_indices],
                            dim=1,
                        )
                        teacher_probs = softmax_with_temperature(
                            selected_teacher_logits,
                            args.distill_temperature,
                        )
                    distill_loss = kl_divergence_with_temperature(
                        second_logits,
                        teacher_probs,
                        args.distill_temperature,
                    )
                    second_loss = (1.0 - args.distill_alpha) * second_loss + args.distill_alpha * distill_loss
            second_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.second_step(zero_grad=True)
            scheduler.step()

            if epoch + 1 > args.ema_start_epoch:
                ema.update(model)

            batch_size = images.size(0)
            running_loss += second_loss.item() * batch_size
            sample_count += batch_size

        if epoch + 1 == args.ema_start_epoch:
            ema.copy_from(model)

        raw_val_loss, raw_val_acc = evaluate(model, val_loader, device, amp_enabled=True)
        val_loss = raw_val_loss
        val_acc = raw_val_acc
        ema_val_loss = None
        ema_val_acc = None
        eval_source = "raw"

        if epoch + 1 >= args.ema_start_epoch:
            ema.apply_to(model)
            ema_val_loss, ema_val_acc = evaluate(model, val_loader, device, amp_enabled=True)
            ema.restore(model)
            if ema_val_acc >= raw_val_acc:
                val_loss = ema_val_loss
                val_acc = ema_val_acc
                eval_source = "ema"

        best_acc = max(best_acc, val_acc)
        epoch_result = EpochResult(
            epoch=epoch + 1,
            train_loss=running_loss / sample_count,
            val_loss=val_loss,
            val_acc=val_acc,
            raw_val_loss=raw_val_loss,
            raw_val_acc=raw_val_acc,
            ema_val_loss=ema_val_loss,
            ema_val_acc=ema_val_acc,
            eval_source=eval_source,
            best_acc=best_acc,
            epoch_seconds=time.time() - epoch_start,
        )
        history.append(asdict(epoch_result))
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_loss={epoch_result.train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc * 100:.2f}% ({eval_source}) | best={best_acc * 100:.2f}% | "
            f"time={epoch_result.epoch_seconds:.1f}s"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_acc": val_acc,
            "metadata": metadata,
            "optimizer": optimizer.base_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if val_acc >= best_acc:
            torch.save(checkpoint, output_dir / "best.pt")

    summary = {
        "best_val_acc": best_acc,
        "best_val_acc_pct": round(best_acc * 100, 4),
        "epochs": args.epochs,
        "total_seconds": time.time() - start_time,
        "subgroup": args.subgroup,
        "data_root": args.data_root.as_posix(),
        "class_to_name": train_dataset.class_to_name,
        "local_to_original": train_dataset.local_to_original,
        "arch": "resnet18",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
