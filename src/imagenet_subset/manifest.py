from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .taxonomy import Taxonomy


MANIFEST_FIELDS = [
    "split",
    "image_path",
    "filename",
    "synset",
    "label_id",
    "leaf_id",
    "leaf_name",
    "level1_id",
    "level1_name",
    "level2_id",
    "level2_name",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_class_tree(manifests_dir: Path, taxonomy: Taxonomy) -> Path:
    path = manifests_dir / "class_tree.json"
    write_json(path, taxonomy.to_tree_payload())
    return path


def write_selected_shards(manifests_dir: Path, shard_selection: dict[str, list[str]]) -> Path:
    path = manifests_dir / "selected_shards.json"
    write_json(path, shard_selection)
    return path


def write_download_summary(manifests_dir: Path, summary: dict) -> Path:
    path = manifests_dir / "download_summary.json"
    write_json(path, summary)
    return path


@contextmanager
def manifest_writer(path: Path) -> Iterator[csv.DictWriter]:
    ensure_parent(path)
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    try:
        yield writer
    finally:
        csv_file.close()
