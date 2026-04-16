from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HierarchySpec:
    level1_names: tuple[str, ...]
    level2_names: tuple[str, ...]
    leaf_names: tuple[str, ...]
    level2_to_level1: tuple[int, ...]
    leaf_to_level1: tuple[int, ...]
    leaf_to_level2: tuple[int, ...]
    level1_to_level2: dict[int, tuple[int, ...]]
    level2_to_leaf: dict[int, tuple[int, ...]]

    @classmethod
    def from_data_root(cls, data_root: Path) -> "HierarchySpec":
        manifest_dir = Path(data_root) / "manifests"
        class_tree_path = manifest_dir / "class_tree.json"
        if class_tree_path.exists():
            payload = json.loads(class_tree_path.read_text(encoding="utf-8"))
            records = payload["classes"]
        else:
            train_manifest = manifest_dir / "train.csv"
            if not train_manifest.exists():
                raise FileNotFoundError(f"Could not find hierarchy metadata in {manifest_dir}")
            seen_leaf_ids: set[int] = set()
            records = []
            with train_manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    leaf_id = int(row["leaf_id"])
                    if leaf_id in seen_leaf_ids:
                        continue
                    seen_leaf_ids.add(leaf_id)
                    records.append(
                        {
                            "leaf_id": leaf_id,
                            "leaf_name": row["leaf_name"],
                            "level1_id": int(row["level1_id"]),
                            "level1_name": row["level1_name"],
                            "level2_id": int(row["level2_id"]),
                            "level2_name": row["level2_name"],
                        }
                    )
        return cls.from_records(records)

    @classmethod
    def from_metadata(cls, payload: dict) -> "HierarchySpec":
        return cls(
            level1_names=tuple(payload["level1_names"]),
            level2_names=tuple(payload["level2_names"]),
            leaf_names=tuple(payload["leaf_names"]),
            level2_to_level1=tuple(int(value) for value in payload["level2_to_level1"]),
            leaf_to_level1=tuple(int(value) for value in payload["leaf_to_level1"]),
            leaf_to_level2=tuple(int(value) for value in payload["leaf_to_level2"]),
            level1_to_level2={
                int(key): tuple(int(item) for item in value)
                for key, value in payload["level1_to_level2"].items()
            },
            level2_to_leaf={
                int(key): tuple(int(item) for item in value)
                for key, value in payload["level2_to_leaf"].items()
            },
        )

    @classmethod
    def from_records(cls, records: list[dict]) -> "HierarchySpec":
        if not records:
            raise ValueError("Cannot build a hierarchy from an empty record set.")

        level1_name_by_id: dict[int, str] = {}
        level2_name_by_id: dict[int, str] = {}
        leaf_name_by_id: dict[int, str] = {}
        level2_to_level1: dict[int, int] = {}
        leaf_to_level1: dict[int, int] = {}
        leaf_to_level2: dict[int, int] = {}

        for record in records:
            level1_id = int(record["level1_id"])
            level2_id = int(record["level2_id"])
            leaf_id = int(record["leaf_id"])

            level1_name_by_id.setdefault(level1_id, record["level1_name"])
            level2_name_by_id.setdefault(level2_id, record["level2_name"])
            leaf_name_by_id.setdefault(leaf_id, record["leaf_name"])
            level2_to_level1.setdefault(level2_id, level1_id)
            leaf_to_level1.setdefault(leaf_id, level1_id)
            leaf_to_level2.setdefault(leaf_id, level2_id)

        level1_names = tuple(name for _, name in sorted(level1_name_by_id.items()))
        level2_names = tuple(name for _, name in sorted(level2_name_by_id.items()))
        leaf_names = tuple(name for _, name in sorted(leaf_name_by_id.items()))

        level1_to_level2: dict[int, list[int]] = {level1_id: [] for level1_id in range(len(level1_names))}
        for level2_id, level1_id in sorted(level2_to_level1.items()):
            level1_to_level2[level1_id].append(level2_id)

        level2_to_leaf: dict[int, list[int]] = {level2_id: [] for level2_id in range(len(level2_names))}
        for leaf_id, level2_id in sorted(leaf_to_level2.items()):
            level2_to_leaf[level2_id].append(leaf_id)

        return cls(
            level1_names=level1_names,
            level2_names=level2_names,
            leaf_names=leaf_names,
            level2_to_level1=tuple(level2_to_level1[idx] for idx in range(len(level2_names))),
            leaf_to_level1=tuple(leaf_to_level1[idx] for idx in range(len(leaf_names))),
            leaf_to_level2=tuple(leaf_to_level2[idx] for idx in range(len(leaf_names))),
            level1_to_level2={idx: tuple(children) for idx, children in level1_to_level2.items()},
            level2_to_leaf={idx: tuple(children) for idx, children in level2_to_leaf.items()},
        )

    @property
    def num_level1(self) -> int:
        return len(self.level1_names)

    @property
    def num_level2(self) -> int:
        return len(self.level2_names)

    @property
    def num_leaf(self) -> int:
        return len(self.leaf_names)

    @property
    def level1_name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.level1_names)}

    @property
    def level2_name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.level2_names)}

    @property
    def leaf_name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.leaf_names)}

    def to_metadata(self) -> dict:
        return {
            "level1_names": list(self.level1_names),
            "level2_names": list(self.level2_names),
            "leaf_names": list(self.leaf_names),
            "level2_to_level1": list(self.level2_to_level1),
            "leaf_to_level1": list(self.leaf_to_level1),
            "leaf_to_level2": list(self.leaf_to_level2),
            "level1_to_level2": {str(key): list(value) for key, value in self.level1_to_level2.items()},
            "level2_to_leaf": {str(key): list(value) for key, value in self.level2_to_leaf.items()},
        }
