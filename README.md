# Diabetic Retinopathy Detection

This project trains and evaluates a CNN with residual blocks to classify retinal images into diabetic retinopathy severity classes.

## Refactored Structure

The codebase is now split into focused modules:

- `diabetic.py`: backward-compatible entry point (`python diabetic.py`).
- `drd/config.py`: runtime configuration defaults.
- `drd/data.py`: dataset scanning, stratified split, and data generators.
- `drd/model.py`: residual block and model architecture.
- `drd/evaluate.py`: prediction mapping, metrics, and confusion matrix inputs.
- `drd/visualize.py`: plotting helpers for EDA and predictions.
- `drd/pipeline.py`: orchestration (train + evaluate).

## Dataset Layout

Arrange the dataset as class subfolders:

```text
your_data_dir/
  Mild/
    img1.jpg
    img2.jpg
  Moderate/
  No_DR/
  Proliferate_DR/
  Severe/
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn pillow
```

## Run

### Option 1 (legacy command)

```bash
python diabetic.py --data-dir "C:/path/to/train" --output-dir "outputs"
```

### Option 2 (module command)

```bash
python -m drd.pipeline --data-dir "C:/path/to/train" --output-dir "outputs"
```

Useful flags:

- `--epochs 20`
- `--batch-size 32`
- `--image-size 256 256`
- `--seed 42`
- `--no-plots` (for headless training runs)

Model checkpoints are saved to:

- `outputs/best_weights.keras` (or your chosen `--output-dir`)

## What Was Fixed During Refactor

- Removed hardcoded absolute paths and replaced them with CLI arguments.
- Fixed checkpoint consistency (train/save/load now use the same file).
- Fixed class label mapping by deriving labels from generator `class_indices`.
- Fixed random sampling bug in qualitative prediction plotting.
- Added stratified train/test split for better class balance.
- Separated validation generator from augmentation pipeline.
- Removed truncating `steps_per_epoch` usage that could drop data.

## Notes

- This remains a research/training project; no API or deployment service is included.
- If you want, the next step can be adding `requirements.txt` and lightweight tests for data loading and label mapping.
