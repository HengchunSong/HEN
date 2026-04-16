from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

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
    build_coarse_to_fine_hen,
    build_resnet,
    build_transforms,
    transfer_coarse_to_fine_weights,
)
from hen.train_utils import set_seed


@dataclass
class EpochSummary:
    epoch: int
    train_loss: float
    train_top_acc: float
    train_mid_acc: float
    train_leaf_acc: float | None
    val_loss: float
    val_top_acc: float
    val_mid_acc: float
    val_leaf_acc: float | None
    best_score: float
    epoch_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a coarse-to-fine hierarchical expert network.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset_food")
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument(
        "--router-backbone",
        choices=["tiny", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0"],
        default="tiny",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--router-image-size", type=int, default=64)
    parser.add_argument("--router-base-width", type=int, default=32)
    parser.add_argument("--router-hidden-dim", type=int, default=256)
    parser.add_argument("--mid-highres-level1", default=None)
    parser.add_argument("--mid-highres-image-size", type=int, default=None)
    parser.add_argument("--mid-highres-base-width", type=int, default=24)
    parser.add_argument("--mid-highres-hidden-dim", type=int, default=256)
    parser.add_argument("--mid-feature-level1", default=None)
    parser.add_argument("--mid-feature-adapter-dim", type=int, default=256)
    parser.add_argument("--mid-attention-level1", default=None)
    parser.add_argument("--mid-attention-image-size", type=int, default=None)
    parser.add_argument("--mid-attention-base-width", type=int, default=24)
    parser.add_argument("--mid-attention-hidden-dim", type=int, default=256)
    parser.add_argument("--leaf-adapter-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--train-crop-min-scale", type=float, default=0.6)
    parser.add_argument("--level1-loss-weight", type=float, default=0.25)
    parser.add_argument("--level2-loss-weight", type=float, default=0.5)
    parser.add_argument("--leaf-loss-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-from-run", type=Path, default=None)
    parser.add_argument(
        "--scope",
        choices=["full", "top_router", "router", "mid_level1_branch", "level1_branch", "level2_branch"],
        default="full",
    )
    parser.add_argument("--target-level1", default=None)
    parser.add_argument("--target-level2", default=None)
    parser.add_argument("--train-shared-stem", action="store_true")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--teacher-run", type=Path, default=None)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
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
    teacher_top_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    top_loss = F.cross_entropy(outputs.level1_logits, level1_targets, label_smoothing=args.label_smoothing)
    mid_loss = smoothed_nll_loss(outputs.level2_log_probs, level2_targets, smoothing=args.label_smoothing)
    leaf_loss = None
    if outputs.leaf_log_probs is not None:
        leaf_loss = smoothed_nll_loss(outputs.leaf_log_probs, leaf_targets, smoothing=args.label_smoothing)

    if args.scope == "top_router":
        if teacher_top_logits is not None and args.distill_alpha > 0.0:
            temperature = args.distill_temperature
            teacher_probs = F.softmax(teacher_top_logits / temperature, dim=1)
            student_log_probs = F.log_softmax(outputs.level1_logits / temperature, dim=1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
            return (1.0 - args.distill_alpha) * top_loss + args.distill_alpha * distill_loss
        return top_loss
    if args.scope == "router":
        return args.level1_loss_weight * top_loss + args.level2_loss_weight * mid_loss
    if args.scope == "mid_level1_branch":
        total = args.level2_loss_weight * mid_loss
        if args.train_router:
            total = total + args.level1_loss_weight * top_loss
        return total
    if args.scope == "level1_branch":
        total = args.leaf_loss_weight * leaf_loss
        if args.train_router:
            total = total + args.level1_loss_weight * top_loss + args.level2_loss_weight * mid_loss
        return total
    if args.scope == "level2_branch":
        total = args.leaf_loss_weight * leaf_loss
        if args.train_router:
            total = total + args.level1_loss_weight * top_loss + args.level2_loss_weight * mid_loss
        return total
    return (
        args.level1_loss_weight * top_loss
        + args.level2_loss_weight * mid_loss
        + args.leaf_loss_weight * leaf_loss
    )


def set_training_modes(model, args: argparse.Namespace, train: bool, level1_id: int | None) -> None:
    model.train(train)
    if not train:
        return
    if args.scope == "full":
        return

    if not args.train_shared_stem:
        model.shared_stem.eval()
    if not args.train_router and args.scope not in {"top_router", "router", "mid_level1_branch"}:
        model.router.eval()

    for branch in model.level1_experts.values():
        branch.eval()

    if args.scope == "mid_level1_branch" and level1_id is not None:
        model.router.eval()
        model.router.level2_adapters[str(level1_id)].train(True)
        model.router.level2_heads[str(level1_id)].train(True)
        model.router.level2_residual_heads[str(level1_id)].train(True)
        if str(level1_id) in model.router.level2_highres_branches:
            model.router.level2_highres_branches[str(level1_id)].train(True)
        if str(level1_id) in model.mid_feature_branches:
            model.mid_feature_branches[str(level1_id)].train(True)
        if str(level1_id) in model.mid_attention_branches:
            model.mid_attention_branches[str(level1_id)].train(True)
        if args.train_router:
            model.router.level1_head.train(True)

    if args.scope == "level1_branch" and level1_id is not None:
        model.level1_experts[str(level1_id)].train(True)


def forward_for_scope(model, images: torch.Tensor, args: argparse.Namespace):
    if args.scope in {"top_router", "router", "mid_level1_branch"}:
        return model(images, compute_leaf=False)
    return model(images)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    level1_id: int | None,
    teacher_model,
    train: bool,
) -> dict[str, float]:
    set_training_modes(model, args, train, level1_id)
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
            outputs = forward_for_scope(model, images, args)
            teacher_top_logits = None
            if teacher_model is not None and args.scope == "top_router":
                with torch.no_grad():
                    teacher_top_logits = teacher_model(images)
            loss = compute_loss(
                outputs,
                level1_targets,
                level2_targets,
                leaf_targets,
                args,
                teacher_top_logits=teacher_top_logits,
            )

        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top_correct += outputs.level1_logits.argmax(dim=1).eq(level1_targets).sum().item()
        total_mid_correct += outputs.level2_log_probs.argmax(dim=1).eq(level2_targets).sum().item()
        if outputs.leaf_log_probs is not None:
            total_leaf_correct += outputs.leaf_log_probs.argmax(dim=1).eq(leaf_targets).sum().item()

    return {
        "loss": total_loss / max(total_samples, 1),
        "top_acc": total_top_correct / max(total_samples, 1),
        "mid_acc": total_mid_correct / max(total_samples, 1),
        "leaf_acc": None if args.scope in {"top_router", "router", "mid_level1_branch"} else total_leaf_correct / max(total_samples, 1),
    }


def build_datasets(args: argparse.Namespace):
    level1_filter = args.target_level1 if args.scope in {"mid_level1_branch", "level1_branch"} else None
    level2_filter = args.target_level2 if args.scope == "level2_branch" else None
    manifest_dir = args.data_root / "manifests"
    train_dataset = JointHierManifestDataset(
        manifest_dir / "train.csv",
        transform=build_transforms(args.image_size, is_train=True, train_crop_min_scale=args.train_crop_min_scale),
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
    model = build_coarse_to_fine_hen(
        backbone=args.backbone,
        hierarchy=hierarchy,
        pretrained=not args.no_pretrained,
        router_backbone=args.router_backbone,
        router_image_size=args.router_image_size,
        router_base_width=args.router_base_width,
        router_hidden_dim=args.router_hidden_dim,
        mid_highres_level1=args.mid_highres_level1,
        mid_highres_image_size=args.mid_highres_image_size,
        mid_highres_base_width=args.mid_highres_base_width,
        mid_highres_hidden_dim=args.mid_highres_hidden_dim,
        mid_feature_level1=args.mid_feature_level1,
        mid_feature_adapter_dim=args.mid_feature_adapter_dim,
        mid_attention_level1=args.mid_attention_level1,
        mid_attention_image_size=args.mid_attention_image_size,
        mid_attention_base_width=args.mid_attention_base_width,
        mid_attention_hidden_dim=args.mid_attention_hidden_dim,
        leaf_adapter_dim=args.leaf_adapter_dim,
        dropout=args.dropout,
    ).to(device)

    if args.init_from_run is None:
        return model

    checkpoint = torch.load(args.init_from_run / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    source_hierarchy = HierarchySpec.from_metadata(metadata["hierarchy"])
    source_model = build_coarse_to_fine_hen(
        backbone=metadata["backbone"],
        hierarchy=source_hierarchy,
        pretrained=False,
        router_backbone=metadata.get("router_backbone", "tiny"),
        router_image_size=int(metadata["router_image_size"]),
        router_base_width=int(metadata["router_base_width"]),
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        mid_highres_level1=metadata.get("mid_highres_level1"),
        mid_highres_image_size=metadata.get("mid_highres_image_size"),
        mid_highres_base_width=int(metadata.get("mid_highres_base_width", 24)),
        mid_highres_hidden_dim=int(metadata.get("mid_highres_hidden_dim", 256)),
        mid_feature_level1=metadata.get("mid_feature_level1"),
        mid_feature_adapter_dim=int(metadata.get("mid_feature_adapter_dim", 256)),
        mid_attention_level1=metadata.get("mid_attention_level1"),
        mid_attention_image_size=metadata.get("mid_attention_image_size"),
        mid_attention_base_width=int(metadata.get("mid_attention_base_width", 24)),
        mid_attention_hidden_dim=int(metadata.get("mid_attention_hidden_dim", 256)),
        leaf_adapter_dim=int(metadata["leaf_adapter_dim"]),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    source_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    transfer_coarse_to_fine_weights(
        target_model=model,
        source_model=source_model,
        target_hierarchy=hierarchy,
        source_hierarchy=source_hierarchy,
    )
    return model


def load_top_teacher(args: argparse.Namespace, device: torch.device):
    if args.teacher_run is None:
        return None

    checkpoint = torch.load(args.teacher_run / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    state_dict = checkpoint["model_state_dict"]
    uses_dropout_head = any(key.startswith("fc.1.") for key in state_dict)
    model = build_resnet(
        backbone=metadata["backbone"],
        num_classes=metadata["num_classes"],
        pretrained=False,
        dropout=0.1 if uses_dropout_head else 0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def validation_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    if args.scope == "top_router":
        return metrics["top_acc"]
    if args.scope in {"router", "mid_level1_branch"}:
        return metrics["mid_acc"]
    return metrics["leaf_acc"]


def format_pct(metric: float | None) -> str:
    if metric is None:
        return "--"
    return f"{metric * 100:.2f}%"


def require_targets(args: argparse.Namespace, hierarchy: HierarchySpec) -> tuple[int | None, int | None]:
    level1_id = None
    level2_id = None
    if args.scope in {"mid_level1_branch", "level1_branch"}:
        if not args.target_level1:
            raise ValueError("--target-level1 is required for this scope.")
        level1_id = hierarchy.level1_name_to_id[args.target_level1]
    if args.scope == "level2_branch":
        if not args.target_level2:
            raise ValueError("--target-level2 is required for level2_branch scope.")
        level2_id = hierarchy.level2_name_to_id[args.target_level2]
        level1_id = hierarchy.level2_to_level1[level2_id]
    return level1_id, level2_id


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    hierarchy = HierarchySpec.from_data_root(args.data_root)
    level1_id, level2_id = require_targets(args, hierarchy)

    model = initialize_model(args, hierarchy, device)
    teacher_model = load_top_teacher(args, device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.configure_trainable(
        scope=args.scope,
        level1_id=level1_id,
        level2_id=level2_id,
        train_shared_stem=args.train_shared_stem,
        train_router=args.train_router,
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

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected for this scope.")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = time.strftime(f"c2f_hen_{args.scope}_{args.backbone}_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    config["use_amp"] = use_amp
    config["hierarchy"] = hierarchy.to_metadata()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    metadata = {
        "stage": "coarse_to_fine_hen",
        "backbone": args.backbone,
        "router_backbone": args.router_backbone,
        "image_size": args.image_size,
        "router_image_size": args.router_image_size,
        "router_base_width": args.router_base_width,
        "router_hidden_dim": args.router_hidden_dim,
        "mid_highres_level1": args.mid_highres_level1,
        "mid_highres_image_size": args.mid_highres_image_size,
        "mid_highres_base_width": args.mid_highres_base_width,
        "mid_highres_hidden_dim": args.mid_highres_hidden_dim,
        "mid_feature_level1": args.mid_feature_level1,
        "mid_feature_adapter_dim": args.mid_feature_adapter_dim,
        "mid_attention_level1": args.mid_attention_level1,
        "mid_attention_image_size": args.mid_attention_image_size,
        "mid_attention_base_width": args.mid_attention_base_width,
        "mid_attention_hidden_dim": args.mid_attention_hidden_dim,
        "leaf_adapter_dim": args.leaf_adapter_dim,
        "dropout": args.dropout,
        "scope": args.scope,
        "target_level1": args.target_level1,
        "target_level2": args.target_level2,
        "train_shared_stem": args.train_shared_stem,
        "train_router": args.train_router,
        "teacher_run": str(args.teacher_run) if args.teacher_run else None,
        "distill_alpha": args.distill_alpha,
        "distill_temperature": args.distill_temperature,
        "label_smoothing": args.label_smoothing,
        "train_crop_min_scale": args.train_crop_min_scale,
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
            level1_id=level1_id,
            teacher_model=teacher_model,
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
            level1_id=level1_id,
            teacher_model=teacher_model,
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
            f"val_top={format_pct(summary.val_top_acc)} | "
            f"val_mid={format_pct(summary.val_mid_acc)} | "
            f"val_leaf={format_pct(summary.val_leaf_acc)} | "
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
