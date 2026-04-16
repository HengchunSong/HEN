from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import build_resnet, build_transforms
from hen.dataset import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class ValRow:
    image_path: Path
    level1_id: int
    level1_name: str
    level2_id: int
    level2_name: str
    leaf_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a true three-level HEN pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset_food")
    parser.add_argument("--top-router-dir", type=Path, required=True)
    parser.add_argument("--mid-router", action="append", required=True, help="group_name=path_to_run_dir")
    parser.add_argument("--leaf-expert", action="append", required=True, help="level2_name=path_to_run_dir")
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs" / "hen_eval_3level_summary.json")
    return parser.parse_args()


def parse_mapping(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid mapping: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid mapping: {item}")
        mapping[name] = Path(raw_path)
    return mapping


def load_rows(manifest_path: Path) -> list[ValRow]:
    dataset_root = manifest_path.parent.parent
    rows: list[ValRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ValRow(
                    image_path=dataset_root / row["image_path"],
                    level1_id=int(row["level1_id"]),
                    level1_name=row["level1_name"],
                    level2_id=int(row["level2_id"]),
                    level2_name=row["level2_name"],
                    leaf_id=int(row["leaf_id"]),
                )
            )
    return rows


def build_eval_transform_for_run(run_dir: Path, metadata: dict):
    resize_scale = metadata.get("eval_resize_scale")
    if resize_scale is not None:
        return transforms.Compose(
            [
                transforms.Resize(int(metadata["image_size"] * float(resize_scale))),
                transforms.CenterCrop(metadata["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if any(key in config for key in ("randaugment_ops", "mix_prob", "random_erasing_prob")):
            return transforms.Compose(
                [
                    transforms.Resize(int(metadata["image_size"] * 1.12)),
                    transforms.CenterCrop(metadata["image_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    return build_transforms(metadata["image_size"], is_train=False)


def load_stage_model(run_dir: Path):
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("fc.1.") for key in state_dict):
        dropout = 0.2
    else:
        dropout = 0.0
    model = build_resnet(
        backbone=metadata["backbone"],
        num_classes=metadata["num_classes"],
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata, build_eval_transform_for_run(run_dir, metadata)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_router, top_meta, top_transform = load_stage_model(args.top_router_dir)
    top_router.to(device)
    top_class_to_name = {int(k): v for k, v in top_meta["class_to_name"].items()}

    mid_router_dirs = parse_mapping(args.mid_router)
    mid_routers = {}
    for group, run_dir in mid_router_dirs.items():
        model, meta, transform = load_stage_model(run_dir)
        model.to(device)
        mid_routers[group] = {
            "model": model,
            "meta": meta,
            "transform": transform,
            "class_to_name": {int(k): v for k, v in meta["class_to_name"].items()},
        }

    leaf_expert_dirs = parse_mapping(args.leaf_expert)
    leaf_experts = {}
    for subgroup, run_dir in leaf_expert_dirs.items():
        model, meta, transform = load_stage_model(run_dir)
        model.to(device)
        leaf_experts[subgroup] = {
            "model": model,
            "meta": meta,
            "transform": transform,
            "local_to_original": {int(k): int(v) for k, v in meta["local_to_original"].items()},
        }

    rows = load_rows(args.data_root / "manifests" / "val.csv")
    total = len(rows)
    top_correct = 0
    mid_correct = 0
    leaf_oracle_correct = 0
    end_to_end_correct = 0
    per_level1 = {name: {"samples": 0, "mid_acc": 0, "leaf_oracle": 0, "end_to_end": 0} for name in mid_router_dirs}
    per_level2 = {name: {"samples": 0, "leaf_oracle": 0, "end_to_end": 0} for name in leaf_expert_dirs}

    with torch.no_grad():
        for row in rows:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")

            top_x = top_transform(image).unsqueeze(0).to(device)
            top_pred = int(top_router(top_x).argmax(dim=1).item())
            top_correct += int(top_pred == row.level1_id)

            true_mid_router = mid_routers[row.level1_name]
            mid_true_x = true_mid_router["transform"](image).unsqueeze(0).to(device)
            mid_true_pred = int(true_mid_router["model"](mid_true_x).argmax(dim=1).item())
            mid_true_name = true_mid_router["class_to_name"][mid_true_pred]
            mid_hit = int(mid_true_name == row.level2_name)
            mid_correct += mid_hit

            true_leaf_expert = leaf_experts[row.level2_name]
            leaf_true_x = true_leaf_expert["transform"](image).unsqueeze(0).to(device)
            leaf_true_pred = int(true_leaf_expert["model"](leaf_true_x).argmax(dim=1).item())
            leaf_true_id = true_leaf_expert["local_to_original"][leaf_true_pred]
            leaf_hit = int(leaf_true_id == row.leaf_id)
            leaf_oracle_correct += leaf_hit

            pred_level1_name = top_class_to_name[top_pred]
            routed_mid_router = mid_routers[pred_level1_name]
            mid_routed_x = routed_mid_router["transform"](image).unsqueeze(0).to(device)
            mid_routed_pred = int(routed_mid_router["model"](mid_routed_x).argmax(dim=1).item())
            pred_level2_name = routed_mid_router["class_to_name"][mid_routed_pred]

            routed_leaf_expert = leaf_experts[pred_level2_name]
            leaf_routed_x = routed_leaf_expert["transform"](image).unsqueeze(0).to(device)
            leaf_routed_pred = int(routed_leaf_expert["model"](leaf_routed_x).argmax(dim=1).item())
            pred_leaf_id = routed_leaf_expert["local_to_original"][leaf_routed_pred]
            e2e_hit = int(pred_leaf_id == row.leaf_id)
            end_to_end_correct += e2e_hit

            per_level1[row.level1_name]["samples"] += 1
            per_level1[row.level1_name]["mid_acc"] += mid_hit
            per_level1[row.level1_name]["leaf_oracle"] += leaf_hit
            per_level1[row.level1_name]["end_to_end"] += e2e_hit

            per_level2[row.level2_name]["samples"] += 1
            per_level2[row.level2_name]["leaf_oracle"] += leaf_hit
            per_level2[row.level2_name]["end_to_end"] += e2e_hit

    summary = {
        "total_samples": total,
        "top_router_acc": top_correct / total,
        "mid_router_oracle_acc": mid_correct / total,
        "leaf_expert_oracle_acc": leaf_oracle_correct / total,
        "end_to_end_leaf_acc": end_to_end_correct / total,
        "per_level1": {
            name: {
                "samples": stats["samples"],
                "mid_router_oracle_acc": stats["mid_acc"] / max(stats["samples"], 1),
                "leaf_expert_oracle_acc": stats["leaf_oracle"] / max(stats["samples"], 1),
                "end_to_end_acc": stats["end_to_end"] / max(stats["samples"], 1),
            }
            for name, stats in per_level1.items()
        },
        "per_level2": {
            name: {
                "samples": stats["samples"],
                "leaf_expert_oracle_acc": stats["leaf_oracle"] / max(stats["samples"], 1),
                "end_to_end_acc": stats["end_to_end"] / max(stats["samples"], 1),
            }
            for name, stats in per_level2.items()
        },
    }
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
