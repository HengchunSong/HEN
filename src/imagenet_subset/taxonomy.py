from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClassRecord:
    label_id: int
    synset: str
    leaf_id: int
    leaf_name: str
    level1_id: int
    level1_name: str
    level2_id: int
    level2_name: str


@dataclass(frozen=True)
class Taxonomy:
    name: str
    dataset_repo: str
    level1_names: list[str]
    level2_names: list[str]
    classes: list[ClassRecord]

    @property
    def label_to_record(self) -> dict[int, ClassRecord]:
        return {record.label_id: record for record in self.classes}

    @property
    def synset_to_record(self) -> dict[str, ClassRecord]:
        return {record.synset: record for record in self.classes}

    @property
    def target_labels(self) -> set[int]:
        return {record.label_id for record in self.classes}

    def to_tree_payload(self, repo_id: str | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_repo": repo_id or self.dataset_repo,
            "level1_names": self.level1_names,
            "level2_names": self.level2_names,
            "classes": [asdict(record) for record in self.classes],
        }


def _ordered_unique(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def load_taxonomy(config_path: Path, repo_id_override: str | None = None) -> Taxonomy:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    class_specs = payload["classes"]

    label_ids = [int(item["label_id"]) for item in class_specs]
    synsets = [item["synset"] for item in class_specs]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("Duplicate label_id values found in taxonomy config.")
    if len(synsets) != len(set(synsets)):
        raise ValueError("Duplicate synset values found in taxonomy config.")

    level1_names = _ordered_unique([item["level1"] for item in class_specs])
    level2_names = _ordered_unique([item["level2"] for item in class_specs])
    level1_to_id = {name: idx for idx, name in enumerate(level1_names)}
    level2_to_id = {name: idx for idx, name in enumerate(level2_names)}

    records: list[ClassRecord] = []
    for leaf_id, item in enumerate(class_specs):
        records.append(
            ClassRecord(
                label_id=int(item["label_id"]),
                synset=item["synset"],
                leaf_id=leaf_id,
                leaf_name=item["leaf_name"],
                level1_id=level1_to_id[item["level1"]],
                level1_name=item["level1"],
                level2_id=level2_to_id[item["level2"]],
                level2_name=item["level2"],
            )
        )

    return Taxonomy(
        name=payload.get("name", config_path.stem),
        dataset_repo=repo_id_override or payload["dataset_repo"],
        level1_names=level1_names,
        level2_names=level2_names,
        classes=records,
    )
