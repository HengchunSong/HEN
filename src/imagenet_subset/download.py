from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import fsspec
import pyarrow.parquet as pq
from datasets import Image, load_dataset
from huggingface_hub import HfApi

from .manifest import manifest_writer
from .taxonomy import ClassRecord, Taxonomy


SPLITS = ("train", "val")
_SYNSET_PATTERN = re.compile(r"(n\d{8})")


def list_split_files(repo_id: str, split: str) -> list[str]:
    prefix = f"data/{split}-"
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = sorted(path for path in repo_files if path.startswith(prefix) and path.endswith(".parquet"))
    return [f"hf://datasets/{repo_id}/{path}" for path in parquet_files]


def _label_column_index(parquet_file: pq.ParquetFile) -> int:
    row_group = parquet_file.metadata.row_group(0)
    for index in range(row_group.num_columns):
        if row_group.column(index).path_in_schema == "label":
            return index
    raise ValueError("Could not find a 'label' column in the parquet file.")


def parquet_contains_target(path: str, targets: set[int]) -> bool:
    with fsspec.open(path, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        label_column = _label_column_index(parquet_file)
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            stats = parquet_file.metadata.row_group(row_group_idx).column(label_column).statistics
            if stats is None:
                return True
            if any(stats.min <= label <= stats.max for label in targets):
                return True
    return False


def select_shards(repo_id: str, split: str, targets: set[int]) -> list[str]:
    split_files = list_split_files(repo_id, split)
    selected = [path for path in split_files if parquet_contains_target(path, targets)]
    if not selected:
        raise RuntimeError(f"No shards selected for split={split}.")
    return selected


def select_all_shards(repo_id: str, targets: set[int]) -> dict[str, list[str]]:
    return {split: select_shards(repo_id, split, targets) for split in SPLITS}


def _extract_synset(image_path: str) -> str | None:
    match = _SYNSET_PATTERN.search(Path(image_path).name)
    return match.group(1) if match else None


def _build_manifest_row(split: str, filename: str, record: ClassRecord) -> dict[str, str | int]:
    relative_path = Path("images") / split / record.synset / filename
    return {
        "split": split,
        "image_path": relative_path.as_posix(),
        "filename": filename,
        "synset": record.synset,
        "label_id": record.label_id,
        "leaf_id": record.leaf_id,
        "leaf_name": record.leaf_name,
        "level1_id": record.level1_id,
        "level1_name": record.level1_name,
        "level2_id": record.level2_id,
        "level2_name": record.level2_name,
    }


def download_split(
    repo_id: str,
    split: str,
    shards: Iterable[str],
    taxonomy: Taxonomy,
    images_dir: Path,
    manifests_dir: Path,
    max_per_class: int | None = None,
) -> dict:
    shards = list(shards)
    label_to_record = taxonomy.label_to_record
    counts_by_label: dict[int, int] = defaultdict(int)
    counts_by_synset: dict[str, int] = defaultdict(int)
    total_rows = 0
    matched_rows = 0
    skipped_existing = 0
    synset_mismatches = 0

    dataset = load_dataset(
        "parquet",
        data_files={split: shards},
        split=split,
        streaming=True,
    ).cast_column("image", Image(decode=False))

    manifest_path = manifests_dir / f"{split}.csv"
    with manifest_writer(manifest_path) as writer:
        for sample in dataset:
            total_rows += 1
            label_id = int(sample["label"])
            record = label_to_record.get(label_id)
            if record is None:
                continue
            if max_per_class is not None and counts_by_label[label_id] >= max_per_class:
                continue

            image_record = sample["image"]
            filename = Path(image_record["path"]).name
            parsed_synset = _extract_synset(image_record["path"])
            if parsed_synset is not None and parsed_synset != record.synset:
                synset_mismatches += 1
                raise ValueError(
                    f"Synset mismatch for label {label_id}: config={record.synset}, filename={parsed_synset}"
                )

            output_path = images_dir / split / record.synset / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                skipped_existing += 1
            else:
                output_path.write_bytes(image_record["bytes"])

            writer.writerow(_build_manifest_row(split, filename, record))
            counts_by_label[label_id] += 1
            counts_by_synset[record.synset] += 1
            matched_rows += 1

    return {
        "split": split,
        "repo_id": repo_id,
        "selected_shards": shards,
        "selected_shard_count": len(shards),
        "total_streamed_rows": total_rows,
        "matched_rows": matched_rows,
        "skipped_existing": skipped_existing,
        "synset_mismatches": synset_mismatches,
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_synset": dict(sorted(counts_by_synset.items())),
        "manifest": manifest_path.as_posix(),
    }
