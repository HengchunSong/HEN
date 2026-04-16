from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import (
    HierarchySpec,
    JointHierManifestDataset,
    build_modular_hen,
    build_transforms,
    transfer_modular_hen_weights,
)
from hen.train_utils import set_seed


@dataclass
class EpochSummary:
    epoch: int
    train_loss: float
    train_top_acc: float
    train_mid_acc: float
    train_leaf_acc: float
    val_loss: float
    val_top_acc: float
    val_mid_acc: float
    val_leaf_acc: float
    best_score: float
    epoch_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modular shared-backbone hierarchical expert network.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset_food")
    parser.add_argument(
        "--backbone",
        choices=[
            "resnet18",
            "resnet34",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
        ],
        default="resnet18",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--level1-loss-weight", type=float, default=0.25)
    parser.add_argument("--level2-loss-weight", type=float, default=0.5)
    parser.add_argument("--leaf-loss-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-from-run", type=Path, default=None)
    parser.add_argument("--scope", choices=["full", "top", "level1_branch", "level2_branch"], default="full")
    parser.add_argument("--target-level1", default=None)
    parser.add_argument("--target-level2", default=None)
    parser.add_argument("--train-backbone", action="store_true")
    return parser.parse_args()


def smoothed_nll_loss(log_probs: torch.Tensor, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.nll_loss(log_probs, targets)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def compute_loss(
    outputs,
    level1_targets: torch.Tensor,
    level2_targets: torch.Tensor,
    leaf_targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    top_loss = F.cross_entropy(outputs.level1_logits, level1_targets, label_smoothing=args.label_smoothing)
    mid_loss = smoothed_nll_loss(outputs.level2_log_probs, level2_targets, smoothing=args.label_smoothing)
    leaf_loss = smoothed_nll_loss(outputs.leaf_log_probs, leaf_targets, smoothing=args.label_smoothing)

    if args.scope == "top":
        return top_loss
    if args.scope == "level1_branch":
        if args.target_level2:
            return args.level2_loss_weight * mid_loss + args.leaf_loss_weight * leaf_loss
        return args.level2_loss_weight * mid_loss
    if args.scope == "level2_branch":
        return leaf_loss
    return (
        args.level1_loss_weight * top_loss
        + args.level2_loss_weight * mid_loss
        + args.leaf_loss_weight * leaf_loss
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    if train and args.scope != "full" and not args.train_backbone:
        # Keep shared trunk batch norm statistics fixed during local branch updates.
        model.feature_extractor.eval()
    total_loss = 0.0
    total_samples = 0
    total_top_correct = 0
    total_mid_correct = 0
    total_leaf_correct = 0

    for images, level1_targets, level2_targets, leaf_targets in loader:
        images = images.to(device, non_blocking=True)
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)
        level1_targets = level1_targets.to(device, non_blocking=True)
        level2_targets = level2_targets.to(device, non_blocking=True)
        leaf_targets = leaf_targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = compute_loss(outputs, level1_targets, level2_targets, leaf_targets, args)

        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top_correct += outputs.level1_logits.argmax(dim=1).eq(level1_targets).sum().item()
        total_mid_correct += outputs.level2_log_probs.argmax(dim=1).eq(level2_targets).sum().item()
        total_leaf_correct += outputs.leaf_log_probs.argmax(dim=1).eq(leaf_targets).sum().item()

    return {
        "loss": total_loss / max(total_samples, 1),
        "top_acc": total_top_correct / max(total_samples, 1),
        "mid_acc": total_mid_correct / max(total_samples, 1),
        "leaf_acc": total_leaf_correct / max(total_samples, 1),
    }


def build_datasets(args: argparse.Namespace):
    level1_filter = args.target_level1 if args.scope == "level1_branch" else None
    level2_filter = args.target_level2 if args.scope == "level2_branch" else None
    manifest_dir = args.data_root / "manifests"
    train_dataset = JointHierManifestDataset(
        manifest_dir / "train.csv",
        transform=build_transforms(args.image_size, is_train=True),
        level1_filter=level1_filter,
        level2_filter=level2_filter,
    )
    val_dataset = JointHierManifestDataset(
        manifest_dir / "val.csv",
        transform=build_transforms(args.image_size, is_train=False),
        level1_filter=level1_filter,
        level2_filter=level2_filter,
    )
    return train_dataset, val_dataset


def initialize_model(args: argparse.Namespace, hierarchy: HierarchySpec, device: torch.device):
    model = build_modular_hen(
        backbone=args.backbone,
        hierarchy=hierarchy,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        adapter_dim=args.adapter_dim,
    ).to(device)

    if args.init_from_run is None:
        return model

    checkpoint = torch.load(args.init_from_run / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    source_hierarchy = HierarchySpec.from_metadata(metadata["hierarchy"])
    source_model = build_modular_hen(
        backbone=metadata["backbone"],
        hierarchy=source_hierarchy,
        pretrained=False,
        dropout=float(metadata.get("dropout", 0.0)),
        adapter_dim=int(metadata.get("adapter_dim", args.adapter_dim)),
    )
    source_model.load_state_dict(checkpoint["model_state_dict"])
    transfer_modular_hen_weights(
        target_model=model,
        source_model=source_model,
        target_hierarchy=hierarchy,
        source_hierarchy=source_hierarchy,
    )
    return model


def validation_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    if args.scope == "top":
        return metrics["top_acc"]
    if args.scope == "level1_branch" and not args.target_level2:
        return metrics["mid_acc"]
    return metrics["leaf_acc"]


def require_targets(args: argparse.Namespace, hierarchy: HierarchySpec) -> tuple[int | None, int | None]:
    level1_id = None
    level2_id = None
    if args.scope == "level1_branch":
        if not args.target_level1:
            raise ValueError("--target-level1 is required for level1_branch scope.")
        level1_id = hierarchy.level1_name_to_id[args.target_level1]
        if args.target_level2:
            level2_id = hierarchy.level2_name_to_id[args.target_level2]
            if level2_id not in hierarchy.level1_to_level2[level1_id]:
                raise ValueError("--target-level2 must belong to --target-level1.")
    if args.scope == "level2_branch":
        if not args.target_level2:
            raise ValueError("--target-level2 is required for level2_branch scope.")
        level2_id = hierarchy.level2_name_to_id[args.target_level2]
    return level1_id, level2_id


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    hierarchy = HierarchySpec.from_data_root(args.data_root)
    level1_id, level2_id = require_targets(args, hierarchy)

    model = initialize_model(args, hierarchy, device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.configure_trainable(
        scope=args.scope,
        level1_id=level1_id,
        level2_id=level2_id,
        train_backbone=args.train_backbone,
        include_leaf_branch=args.scope == "level1_branch" and level2_id is not None,
    )

    train_dataset, val_dataset = build_datasets(args)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected for this scope.")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = time.strftime(f"modular_hen_{args.scope}_{args.backbone}_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    config["use_amp"] = use_amp
    config["hierarchy"] = hierarchy.to_metadata()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    metadata = {
        "stage": "modular_hen",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "dropout": args.dropout,
        "adapter_dim": args.adapter_dim,
        "scope": args.scope,
        "target_level1": args.target_level1,
        "target_level2": args.target_level2,
        "train_backbone": args.train_backbone,
        "label_smoothing": args.label_smoothing,
        "level1_loss_weight": args.level1_loss_weight,
        "level2_loss_weight": args.level2_loss_weight,
        "leaf_loss_weight": args.leaf_loss_weight,
        "data_root": args.data_root.as_posix(),
        "hierarchy": hierarchy.to_metadata(),
    }

    best_score = -1.0
    history: list[dict] = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            args=args,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            args=args,
            train=False,
        )
        scheduler.step()

        score = validation_score(val_metrics, args)
        improved = score > best_score
        best_score = max(best_score, score)
        summary = EpochSummary(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_top_acc=train_metrics["top_acc"],
            train_mid_acc=train_metrics["mid_acc"],
            train_leaf_acc=train_metrics["leaf_acc"],
            val_loss=val_metrics["loss"],
            val_top_acc=val_metrics["top_acc"],
            val_mid_acc=val_metrics["mid_acc"],
            val_leaf_acc=val_metrics["leaf_acc"],
            best_score=best_score,
            epoch_seconds=time.time() - epoch_start,
        )
        history.append(asdict(summary))
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={summary.train_loss:.4f} | val_loss={summary.val_loss:.4f} | "
            f"val_top={summary.val_top_acc * 100:.2f}% | "
            f"val_mid={summary.val_mid_acc * 100:.2f}% | "
            f"val_leaf={summary.val_leaf_acc * 100:.2f}% | "
            f"best={best_score * 100:.2f}% | "
            f"time={summary.epoch_seconds:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "metadata": metadata,
            "best_score": best_score,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")

    final_summary = {
        **metadata,
        "best_score": best_score,
        "best_score_pct": round(best_score * 100, 4),
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - train_start,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
