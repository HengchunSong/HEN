from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: int
    original_label: int
    synset: str
    leaf_name: str
    leaf_id: int
    level1_name: str
    level1_id: int
    level2_name: str
    level2_id: int


def build_transforms(image_size: int, is_train: bool, train_crop_min_scale: float = 0.6):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(train_crop_min_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HierManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        label_column: str,
        transform=None,
        level1_filter: str | None = None,
        level2_filter: str | None = None,
        remap_labels: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent.parent
        self.transform = transform
        self.label_column = label_column
        self.level1_filter = level1_filter
        self.level2_filter = level2_filter
        self.remap_labels = remap_labels
        self.records: list[SampleRecord] = []
        self.class_to_name: dict[int, str] = {}
        self.original_to_local: dict[int, int] = {}
        self.local_to_original: dict[int, int] = {}
        self._load_records()

    def _load_records(self) -> None:
        rows = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self.level1_filter and row["level1_name"] != self.level1_filter:
                    continue
                if self.level2_filter and row["level2_name"] != self.level2_filter:
                    continue
                rows.append(row)

        if not rows:
            raise ValueError(f"No rows left after filtering manifest: {self.manifest_path}")

        original_labels = sorted({int(row[self.label_column]) for row in rows})
        if self.remap_labels:
            self.original_to_local = {label: idx for idx, label in enumerate(original_labels)}
            self.local_to_original = {idx: label for label, idx in self.original_to_local.items()}
        else:
            self.original_to_local = {label: label for label in original_labels}
            self.local_to_original = {label: label for label in original_labels}

        name_field = "leaf_name"
        if self.label_column == "level1_id":
            name_field = "level1_name"
        elif self.label_column == "level2_id":
            name_field = "level2_name"

        for row in rows:
            original_label = int(row[self.label_column])
            local_label = self.original_to_local[original_label]
            self.class_to_name[local_label] = row[name_field]
            self.records.append(
                SampleRecord(
                    image_path=self.dataset_root / row["image_path"],
                    label=local_label,
                    original_label=original_label,
                    synset=row["synset"],
                    leaf_name=row["leaf_name"],
                    leaf_id=int(row["leaf_id"]),
                    level1_name=row["level1_name"],
                    level1_id=int(row["level1_id"]),
                    level2_name=row["level2_name"],
                    level2_id=int(row["level2_id"]),
                )
            )

    @property
    def num_classes(self) -> int:
        return len(self.class_to_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


class JointHierManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        transform=None,
        level1_filter: str | None = None,
        level2_filter: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent.parent
        self.transform = transform
        self.level1_filter = level1_filter
        self.level2_filter = level2_filter
        self.records: list[SampleRecord] = []
        self._load_records()

    def _load_records(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self.level1_filter and row["level1_name"] != self.level1_filter:
                    continue
                if self.level2_filter and row["level2_name"] != self.level2_filter:
                    continue
                self.records.append(
                    SampleRecord(
                        image_path=self.dataset_root / row["image_path"],
                        label=int(row["leaf_id"]),
                        original_label=int(row["leaf_id"]),
                        synset=row["synset"],
                        leaf_name=row["leaf_name"],
                        leaf_id=int(row["leaf_id"]),
                        level1_name=row["level1_name"],
                        level1_id=int(row["level1_id"]),
                        level2_name=row["level2_name"],
                        level2_id=int(row["level2_id"]),
                    )
                )

        if not self.records:
            raise ValueError(f"No rows left after filtering manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.level1_id, record.level2_id, record.leaf_id
