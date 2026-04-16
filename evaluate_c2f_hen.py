from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import HierarchySpec, JointHierManifestDataset, build_coarse_to_fine_hen, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a coarse-to-fine HEN run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs" / "c2f_hen_eval_summary.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    data_root = args.data_root or Path(metadata["data_root"])
    hierarchy = HierarchySpec.from_data_root(data_root)

    model = build_coarse_to_fine_hen(
        backbone=metadata["backbone"],
        hierarchy=hierarchy,
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dataset = JointHierManifestDataset(
        data_root / "manifests" / "val.csv",
        transform=build_transforms(int(metadata["image_size"]), is_train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    total_samples = 0
    total_top_correct = 0
    total_mid_correct = 0
    total_leaf_correct = 0
    router_only = metadata.get("scope") in {"top_router", "router", "mid_level1_branch"}
    per_level1 = {
        name: {"samples": 0, "mid_correct": 0, "leaf_correct": 0}
        for name in hierarchy.level1_names
    }
    per_level2 = {
        name: {"samples": 0, "leaf_correct": 0}
        for name in hierarchy.level2_names
    }

    with torch.no_grad():
        for images, level1_targets, level2_targets, leaf_targets in loader:
            images = images.to(device, non_blocking=True)
            level1_targets = level1_targets.to(device, non_blocking=True)
            level2_targets = level2_targets.to(device, non_blocking=True)
            leaf_targets = leaf_targets.to(device, non_blocking=True)

            if router_only:
                outputs = model(images, compute_leaf=False)
                top_preds = outputs.level1_logits.argmax(dim=1)
                mid_preds = outputs.level2_log_probs.argmax(dim=1)
                leaf_preds = None
            else:
                outputs = model(images)
                top_preds = outputs.level1_logits.argmax(dim=1)
                mid_preds = outputs.level2_log_probs.argmax(dim=1)
                leaf_preds = outputs.leaf_log_probs.argmax(dim=1)

            total_samples += images.size(0)
            total_top_correct += top_preds.eq(level1_targets).sum().item()
            total_mid_correct += mid_preds.eq(level2_targets).sum().item()
            if leaf_preds is not None:
                total_leaf_correct += leaf_preds.eq(leaf_targets).sum().item()

            for idx in range(images.size(0)):
                level1_name = hierarchy.level1_names[int(level1_targets[idx].item())]
                level2_name = hierarchy.level2_names[int(level2_targets[idx].item())]
                per_level1[level1_name]["samples"] += 1
                per_level1[level1_name]["mid_correct"] += int(mid_preds[idx] == level2_targets[idx])
                if leaf_preds is not None:
                    per_level1[level1_name]["leaf_correct"] += int(leaf_preds[idx] == leaf_targets[idx])
                per_level2[level2_name]["samples"] += 1
                if leaf_preds is not None:
                    per_level2[level2_name]["leaf_correct"] += int(leaf_preds[idx] == leaf_targets[idx])

    summary = {
        "total_samples": total_samples,
        "top_acc": total_top_correct / max(total_samples, 1),
        "mid_acc": total_mid_correct / max(total_samples, 1),
        "leaf_acc": None if router_only else total_leaf_correct / max(total_samples, 1),
        "per_level1": {
            name: {
                "samples": stats["samples"],
                "mid_acc": stats["mid_correct"] / max(stats["samples"], 1),
                "leaf_acc": None if router_only else stats["leaf_correct"] / max(stats["samples"], 1),
            }
            for name, stats in per_level1.items()
        },
        "per_level2": {
            name: {
                "samples": stats["samples"],
                "leaf_acc": None if router_only else stats["leaf_correct"] / max(stats["samples"], 1),
            }
            for name, stats in per_level2.items()
        },
    }
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
