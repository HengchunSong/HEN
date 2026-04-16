from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imagenet_subset.download import download_split, select_all_shards
from imagenet_subset.manifest import write_class_tree, write_download_summary, write_selected_shards
from imagenet_subset.taxonomy import load_taxonomy


DEFAULT_CONFIG = ROOT / "configs" / "tree_27cls.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a configurable ImageNet-1K subtree from a Hugging Face mirror.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a taxonomy config JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "imagenet_subset",
        help="Directory where images and manifests will be stored.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional override for the Hugging Face dataset repo in the config file.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only select shards and write metadata files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    images_dir = output_root / "images"
    manifests_dir = output_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.config, repo_id_override=args.repo_id)
    write_class_tree(manifests_dir, taxonomy)

    shard_selection = select_all_shards(taxonomy.dataset_repo, taxonomy.target_labels)
    write_selected_shards(manifests_dir, shard_selection)

    if args.skip_download:
        print(
            json.dumps(
                {
                    "taxonomy": taxonomy.name,
                    "dataset_repo": taxonomy.dataset_repo,
                    "selected_shards": shard_selection,
                    "output_root": output_root.as_posix(),
                },
                indent=2,
            )
        )
        return

    summaries = {}
    for split in ("train", "val"):
        summaries[split] = download_split(
            repo_id=taxonomy.dataset_repo,
            split=split,
            shards=shard_selection[split],
            taxonomy=taxonomy,
            images_dir=images_dir,
            manifests_dir=manifests_dir,
            max_per_class=args.max_per_class,
        )

    write_download_summary(manifests_dir, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
