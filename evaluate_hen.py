from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hen import build_resnet, build_transforms


@dataclass
class ValRow:
    image_path: Path
    level1_id: int
    level1_name: str
    leaf_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end HEN accuracy.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "imagenet_subset")
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Expert mapping in the form group_name=path_to_run_dir. Repeat once per top-level group.",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs" / "hen_eval_summary.json")
    return parser.parse_args()


def parse_expert_dirs(items: list[str]) -> dict[str, Path]:
    expert_dirs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --expert value: {item}")
        group, raw_path = item.split("=", 1)
        group = group.strip()
        if not group:
            raise ValueError(f"Invalid --expert value: {item}")
        expert_dirs[group] = Path(raw_path)
    return expert_dirs


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
                    leaf_id=int(row["leaf_id"]),
                )
            )
    return rows


def load_stage_model(run_dir: Path):
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    state_dict = checkpoint["model_state_dict"]
    uses_dropout_head = any(key.startswith("fc.1.") for key in state_dict)
    model = build_resnet(
        backbone=metadata["backbone"],
        num_classes=metadata["num_classes"],
        pretrained=False,
        dropout=0.1 if uses_dropout_head else 0.0,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router_model, router_meta = load_stage_model(args.router_dir)
    router_model.to(device)
    router_transform = build_transforms(router_meta["image_size"], is_train=False)
    router_class_to_name = {int(key): value for key, value in router_meta["class_to_name"].items()}

    expert_dirs = parse_expert_dirs(args.expert)
    experts = {}
    for group, run_dir in expert_dirs.items():
        model, meta = load_stage_model(run_dir)
        model.to(device)
        experts[group] = {
            "model": model,
            "meta": meta,
            "transform": build_transforms(meta["image_size"], is_train=False),
            "local_to_original": {int(k): int(v) for k, v in meta["local_to_original"].items()},
        }

    rows = load_rows(args.data_root / "manifests" / "val.csv")
    total = len(rows)
    router_correct = 0
    oracle_correct = 0
    end_to_end_correct = 0
    per_group = {group: {"total": 0, "oracle_correct": 0, "end_to_end_correct": 0} for group in expert_dirs}

    with torch.no_grad():
        for row in rows:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")

            router_tensor = router_transform(image).unsqueeze(0).to(device)
            router_pred = int(router_model(router_tensor).argmax(dim=1).item())
            router_correct += int(router_pred == row.level1_id)

            actual_expert = experts[row.level1_name]
            actual_tensor = actual_expert["transform"](image).unsqueeze(0).to(device)
            oracle_local_pred = int(actual_expert["model"](actual_tensor).argmax(dim=1).item())
            oracle_leaf_pred = actual_expert["local_to_original"][oracle_local_pred]
            oracle_hit = int(oracle_leaf_pred == row.leaf_id)
            oracle_correct += oracle_hit

            routed_group = router_class_to_name[router_pred]
            routed_expert = experts[routed_group]
            routed_tensor = routed_expert["transform"](image).unsqueeze(0).to(device)
            routed_local_pred = int(routed_expert["model"](routed_tensor).argmax(dim=1).item())
            routed_leaf_pred = routed_expert["local_to_original"][routed_local_pred]
            e2e_hit = int(routed_leaf_pred == row.leaf_id)
            end_to_end_correct += e2e_hit

            per_group[row.level1_name]["total"] += 1
            per_group[row.level1_name]["oracle_correct"] += oracle_hit
            per_group[row.level1_name]["end_to_end_correct"] += e2e_hit

    summary = {
        "total_samples": total,
        "router_acc": router_correct / total,
        "expert_oracle_acc": oracle_correct / total,
        "end_to_end_leaf_acc": end_to_end_correct / total,
        "per_group": {
            group: {
                "samples": stats["total"],
                "oracle_acc": stats["oracle_correct"] / max(stats["total"], 1),
                "end_to_end_acc": stats["end_to_end_correct"] / max(stats["total"], 1),
            }
            for group, stats in per_group.items()
        },
    }
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
