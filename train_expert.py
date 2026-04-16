from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import HierManifestDataset, build_resnet, build_transforms
from hen.train_utils import fit_classifier, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one top-level expert on the ImageNet 27-class subtree.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset")
    parser.add_argument("--group", required=True)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_dir = args.data_root / "manifests"
    train_dataset = HierManifestDataset(
        manifest_dir / "train.csv",
        label_column="leaf_id",
        transform=build_transforms(args.image_size, is_train=True),
        level1_filter=args.group,
        remap_labels=True,
    )
    val_dataset = HierManifestDataset(
        manifest_dir / "val.csv",
        label_column="leaf_id",
        transform=build_transforms(args.image_size, is_train=False),
        level1_filter=args.group,
        remap_labels=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet(
        backbone=args.backbone,
        num_classes=train_dataset.num_classes,
        pretrained=not args.no_pretrained,
        dropout=0.2,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    run_name = datetime.now().strftime(f"expert_{args.group}_{args.backbone}_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / run_name)
    metadata = {
        "stage": "expert",
        "group": args.group,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "num_classes": train_dataset.num_classes,
        "class_to_name": train_dataset.class_to_name,
        "local_to_original": train_dataset.local_to_original,
        "data_root": args.data_root.as_posix(),
    }
    summary = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        use_amp=device.type == "cuda",
        metadata=metadata,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
