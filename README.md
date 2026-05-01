# HEN

Hierarchical Expert Networks experiments on a balanced 27-class ImageNet subset arranged as `3 x 3 x 3`.

## Open the report

- Rendered HTML report: [Flat CNN vs HEN Comparative Report](https://raw.githack.com/HengchunSong/HEN/main/outputs/flat_vs_hen_report_20260501_en.html)
- Report source file: [outputs/flat_vs_hen_report_20260501_en.html](https://github.com/HengchunSong/HEN/blob/main/outputs/flat_vs_hen_report_20260501_en.html)

## Repository contents

- `src/hen/`: model definitions and hierarchy utilities
- `train_*.py`: training entrypoints for flat, joint, modular, coarse-to-fine, and common-delta variants
- `evaluate_*.py`: evaluation scripts for the corresponding model families
- `outputs/`: experiment summaries, exported reports, and review artifacts

## Current headline result

- Best flat accuracy: `96.89%` with `ConvNeXt-Tiny`
- Best 95%+ efficiency result: `95.63%` with `Joint HEN + MobileNetV3-Large`

