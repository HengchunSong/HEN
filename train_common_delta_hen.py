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
    build_common_delta_hen,
    build_joint_hen,
    build_transforms,
    transfer_common_delta_hen_weights,
)
from hen.train_utils import set_seed


@dataclass
class EpochSummary:
    epoch: int
    train_loss: float
    train_level1_acc: float
    train_level2_acc: float
    train_leaf_acc: float
    val_loss: float
    val_level1_acc: float
    val_level2_acc: float
    val_leaf_acc: float
    best_leaf_acc: float
    epoch_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a common-delta 3-level hierarchical expert network.")
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
        default="mobilenet_v3_small",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--level1-loss-weight", type=float, default=0.25)
    parser.add_argument("--level2-loss-weight", type=float, default=0.5)
    parser.add_argument("--leaf-loss-weight", type=float, default=1.0)
    parser.add_argument("--split-hidden-dim", type=int, default=256)
    parser.add_argument("--delta-dim", type=int, default=128)
    parser.add_argument(
        "--common-delta-level2",
        nargs="*",
        default=None,
        help="Level-2 branch names that use common-delta leaf heads. Omit to use all level-2 branches.",
    )
    parser.add_argument(
        "--parent-refine-level2",
        nargs="*",
        default=None,
        help="Level-2 branch names that first pass through a parent-specific feature refiner before leaf prediction.",
    )
    parser.add_argument("--parent-refine-hidden-dim", type=int, default=0)
    parser.add_argument("--split-entropy-weight", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init-from-run", type=Path, default=None)
    parser.add_argument("--selected-leaf-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def smoothed_nll_loss(log_probs: torch.Tensor, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.nll_loss(log_probs, targets)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def compute_losses(
    outputs,
    level1_targets: torch.Tensor,
    level2_targets: torch.Tensor,
    leaf_targets: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    level1_loss = F.cross_entropy(outputs.level1_logits, level1_targets, label_smoothing=args.label_smoothing)
    level2_loss = smoothed_nll_loss(outputs.level2_log_probs, level2_targets, smoothing=args.label_smoothing)
    leaf_loss = smoothed_nll_loss(outputs.leaf_log_probs, leaf_targets, smoothing=args.label_smoothing)
    split_loss = outputs.aux_loss if outputs.aux_loss is not None else torch.zeros((), device=level1_targets.device)
    total_loss = (
        args.level1_loss_weight * level1_loss
        + args.level2_loss_weight * level2_loss
        + args.leaf_loss_weight * leaf_loss
        + args.split_entropy_weight * split_loss
    )
    return total_loss, {
        "level1_loss": level1_loss.item(),
        "level2_loss": level2_loss.item(),
        "leaf_loss": leaf_loss.item(),
        "split_loss": float(split_loss.item()),
    }


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
    if train and args.selected_leaf_only:
        model.feature_extractor.eval()
    total_loss = 0.0
    total_samples = 0
    total_level1_correct = 0
    total_level2_correct = 0
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
            loss, _ = compute_losses(
                outputs=outputs,
                level1_targets=level1_targets,
                level2_targets=level2_targets,
                leaf_targets=leaf_targets,
                args=args,
            )

        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_level1_correct += outputs.level1_logits.argmax(dim=1).eq(level1_targets).sum().item()
        total_level2_correct += outputs.level2_log_probs.argmax(dim=1).eq(level2_targets).sum().item()
        total_leaf_correct += outputs.leaf_log_probs.argmax(dim=1).eq(leaf_targets).sum().item()

    return {
        "loss": total_loss / max(total_samples, 1),
        "level1_acc": total_level1_correct / max(total_samples, 1),
        "level2_acc": total_level2_correct / max(total_samples, 1),
        "leaf_acc": total_leaf_correct / max(total_samples, 1),
    }


def initialize_from_run(model, hierarchy: HierarchySpec, init_from_run: Path) -> dict[str, str | None]:
    checkpoint = torch.load(init_from_run / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    source_hierarchy = (
        HierarchySpec.from_metadata(metadata["hierarchy"])
        if "hierarchy" in metadata
        else HierarchySpec.from_data_root(Path(metadata["data_root"]))
    )
    stage = metadata.get("stage")
    if stage == "joint_hen":
        source_model = build_joint_hen(
            backbone=metadata["backbone"],
            hierarchy=source_hierarchy,
            pretrained=False,
            dropout=float(metadata.get("dropout", 0.0)),
        )
    elif stage == "common_delta_hen":
        source_model = build_common_delta_hen(
            backbone=metadata["backbone"],
            hierarchy=source_hierarchy,
            pretrained=False,
            dropout=float(metadata.get("dropout", 0.0)),
            split_hidden_dim=int(metadata.get("split_hidden_dim", 256)),
            delta_dim=int(metadata.get("delta_dim", 128)),
            common_delta_level2_names=tuple(metadata["common_delta_level2"]) if metadata.get("common_delta_level2") else None,
            parent_refine_hidden_dim=int(metadata.get("parent_refine_hidden_dim", 0)),
            parent_refine_level2_names=tuple(metadata["parent_refine_level2"]) if metadata.get("parent_refine_level2") else None,
        )
    else:
        raise ValueError(f"Unsupported init-from-run stage: {stage}")

    source_model.load_state_dict(checkpoint["model_state_dict"])
    transfer_common_delta_hen_weights(
        target_model=model,
        source_model=source_model,
        target_hierarchy=hierarchy,
        source_hierarchy=source_hierarchy,
    )
    return {
        "stage": stage,
        "run_dir": init_from_run.as_posix(),
    }


def configure_selected_leaf_only(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for level1_id in model.parent_refiners:
        for parameter in model.parent_refiners[level1_id].parameters():
            parameter.requires_grad = True
    for level2_id in model.common_delta_level2_ids:
        for parameter in model.leaf_heads[str(level2_id)].parameters():
            parameter.requires_grad = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    manifest_dir = args.data_root / "manifests"
    hierarchy = HierarchySpec.from_data_root(args.data_root)

    train_dataset = JointHierManifestDataset(
        manifest_dir / "train.csv",
        transform=build_transforms(args.image_size, is_train=True),
    )
    val_dataset = JointHierManifestDataset(
        manifest_dir / "val.csv",
        transform=build_transforms(args.image_size, is_train=False),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    effective_parent_refine_level2 = (
        tuple(args.parent_refine_level2)
        if args.parent_refine_level2
        else (
            tuple(args.common_delta_level2)
            if args.parent_refine_hidden_dim > 0 and args.common_delta_level2
            else None
        )
    )
    model = build_common_delta_hen(
        backbone=args.backbone,
        hierarchy=hierarchy,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        split_hidden_dim=args.split_hidden_dim,
        delta_dim=args.delta_dim,
        common_delta_level2_names=tuple(args.common_delta_level2) if args.common_delta_level2 else None,
        parent_refine_hidden_dim=args.parent_refine_hidden_dim,
        parent_refine_level2_names=effective_parent_refine_level2,
    )
    init_metadata = None
    if args.init_from_run is not None:
        init_metadata = initialize_from_run(model, hierarchy, args.init_from_run)
    if args.selected_leaf_only:
        configure_selected_leaf_only(model)
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = time.strftime(f"common_delta_hen_{args.backbone}_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    config["use_amp"] = use_amp
    config["hierarchy"] = hierarchy.to_metadata()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    metadata = {
        "stage": "common_delta_hen",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "level1_loss_weight": args.level1_loss_weight,
        "level2_loss_weight": args.level2_loss_weight,
        "leaf_loss_weight": args.leaf_loss_weight,
        "split_hidden_dim": args.split_hidden_dim,
        "delta_dim": args.delta_dim,
        "common_delta_level2": list(args.common_delta_level2) if args.common_delta_level2 else None,
        "parent_refine_hidden_dim": args.parent_refine_hidden_dim,
        "parent_refine_level2": list(effective_parent_refine_level2) if effective_parent_refine_level2 else None,
        "split_entropy_weight": args.split_entropy_weight,
        "data_root": args.data_root.as_posix(),
        "hierarchy": hierarchy.to_metadata(),
        "init_from_run": init_metadata,
        "selected_leaf_only": args.selected_leaf_only,
    }

    best_leaf_acc = -1.0
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

        improved = val_metrics["leaf_acc"] > best_leaf_acc
        best_leaf_acc = max(best_leaf_acc, val_metrics["leaf_acc"])
        summary = EpochSummary(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_level1_acc=train_metrics["level1_acc"],
            train_level2_acc=train_metrics["level2_acc"],
            train_leaf_acc=train_metrics["leaf_acc"],
            val_loss=val_metrics["loss"],
            val_level1_acc=val_metrics["level1_acc"],
            val_level2_acc=val_metrics["level2_acc"],
            val_leaf_acc=val_metrics["leaf_acc"],
            best_leaf_acc=best_leaf_acc,
            epoch_seconds=time.time() - epoch_start,
        )
        history.append(asdict(summary))
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={summary.train_loss:.4f} | val_loss={summary.val_loss:.4f} | "
            f"val_top={summary.val_level1_acc * 100:.2f}% | "
            f"val_mid={summary.val_level2_acc * 100:.2f}% | "
            f"val_leaf={summary.val_leaf_acc * 100:.2f}% | "
            f"best_leaf={best_leaf_acc * 100:.2f}% | "
            f"time={summary.epoch_seconds:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "metadata": metadata,
            "best_leaf_acc": best_leaf_acc,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")

    final_summary = {
        **metadata,
        "best_leaf_val_acc": best_leaf_acc,
        "best_leaf_val_acc_pct": round(best_leaf_acc * 100, 4),
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - train_start,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
