#!/usr/bin/env python3
"""Create ChestX-ray14 Strategy 2 splits for realistic multi-label MEDAF training.

This script implements the realistic splitting strategy described in split.md:
1. Use official NIH train_val_list.txt and test_list.txt for patient-disjoint splits
2. NO FILTERING - Include ALL images in training (even those with new labels)
3. During training: Only supervise on 8 known labels, ignore new labels
4. Split unfiltered train_val into train/validation with patient-disjoint split
5. Use full test_list.txt for evaluation (both known and unknown samples)

Key differences from Strategy 1:
- No filtering of new labels from training data
- More realistic - mimics real-world scenarios with unlabeled pathologies
- Model learns to focus on known features while being robust to unknown "noise"

Outputs:
  * chestxray_strategy2_train.csv - Training data with ALL images (known + unknown labels)
  * chestxray_strategy2_test.csv - Test data with both known and unknown samples

Usage:
    python utils/create_chestxray_strategy2_splits.py \
        --source datasets/data/NIH/Data_Entry_2017.csv \
        --test-list datasets/data/NIH/test_list.txt \
        --output-dir datasets/data/NIH \
        --val-ratio 0.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Labels used for training the base model
KNOWN_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

# Labels that will be treated as novel/open-set targets
NEW_LABELS = [
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/data/NIH/Data_Entry_2017.csv"),
        help="Path to the NIH ChestX-ray14 metadata CSV",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=Path("datasets/data/NIH/test_list.txt"),
        help="Path to the official NIH test_list.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/data/NIH"),
        help="Directory where the split CSVs will be written",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train_val data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random split",
    )
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def extract_labels(label_str: str) -> set[str]:
    """Extract individual labels from the Finding Labels string."""
    if not isinstance(label_str, str) or not label_str or label_str == "nan":
        return set()
    if label_str == "No Finding":
        return set()
    return {part.strip() for part in label_str.split("|") if part.strip()}


def load_test_images(test_list_path: Path) -> set[str]:
    """Load the official test image list."""
    with open(test_list_path, "r") as f:
        test_images = {line.strip() for line in f if line.strip()}
    return test_images


def create_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add label analysis columns to the dataframe."""
    df = df.copy()
    labels = df["Finding Labels"].apply(extract_labels)

    # Create binary columns for each known and new label
    for label in KNOWN_LABELS:
        df[label] = labels.apply(lambda s: 1 if label in s else 0)

    for label in NEW_LABELS:
        df[label] = labels.apply(lambda s: 1 if label in s else 0)

    # Add analysis columns
    df["known_labels"] = labels.apply(lambda s: sorted(s.intersection(KNOWN_LABELS)))
    df["new_labels"] = labels.apply(lambda s: sorted(s.intersection(NEW_LABELS)))
    df["has_known"] = df["known_labels"].apply(bool)
    df["has_new"] = df["new_labels"].apply(bool)
    df["is_no_finding"] = df["Finding Labels"] == "No Finding"

    return df


def create_patient_disjoint_split(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient-disjoint train/validation split."""
    if df.empty:
        return df.copy(), df.copy()

    print(
        f"Creating patient-disjoint train/validation split with {val_ratio} validation ratio"
    )
    # Use GroupShuffleSplit to ensure patient disjointness
    splitter = GroupShuffleSplit(test_size=val_ratio, n_splits=1, random_state=seed)

    train_idx, val_idx = next(splitter.split(df, groups=df["Patient ID"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    return train_df, val_df


def verify_patient_disjointness(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Verify that patient IDs don't overlap between splits."""
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    if train_val_overlap:
        print(
            f"⚠️  Warning: {len(train_val_overlap)} patients overlap between train and val"
        )
    if train_test_overlap:
        print(
            f"⚠️  Warning: {len(train_test_overlap)} patients overlap between train and test"
        )
    if val_test_overlap:
        print(
            f"⚠️  Warning: {len(val_test_overlap)} patients overlap between val and test"
        )

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✅ Patient disjointness verified: No patient overlap between splits")


def print_label_statistics(df: pd.DataFrame, split_name: str) -> None:
    """Print label distribution statistics for a split."""
    print(f"\n📊 {split_name} Label Statistics:")
    print(f"Total samples: {len(df):,}")

    # Known label statistics
    print("\nKnown Labels:")
    for label in KNOWN_LABELS:
        count = df[label].sum()
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    # New label statistics
    print("\nNew Labels:")
    for label in NEW_LABELS:
        count = df[label].sum()
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    # Special cases
    no_finding_count = df["is_no_finding"].sum()
    no_finding_pct = (no_finding_count / len(df)) * 100 if len(df) > 0 else 0
    print(f"\nNo Finding: {no_finding_count:,} ({no_finding_pct:.1f}%)")

    has_new_count = df["has_new"].sum()
    has_new_pct = (has_new_count / len(df)) * 100 if len(df) > 0 else 0
    print(f"Has New Labels: {has_new_count:,} ({has_new_pct:.1f}%)")

    # Mixed samples (both known and new labels)
    mixed_count = df["has_known"].sum() & df["has_new"].sum()
    mixed_pct = (mixed_count / len(df)) * 100 if len(df) > 0 else 0
    print(f"Mixed (Known + New): {mixed_count:,} ({mixed_pct:.1f}%)")


def print_training_notes() -> None:
    """Print important notes about Strategy 2 training approach."""
    print("\n" + "=" * 80)
    print("🎯 STRATEGY 2 TRAINING NOTES")
    print("=" * 80)
    print("📋 Key Differences from Strategy 1:")
    print("  • NO FILTERING: All images included in training (even with new labels)")
    print("  • REALISTIC: Mimics real-world scenarios with unlabeled pathologies")
    print("  • ROBUST: Model learns to focus on known features while handling unknowns")
    print()
    print("🔧 Training Implementation:")
    print("  • Loss: Only compute BCE on 8 known labels (mask new labels)")
    print("  • Labels: Use only known_labels columns for supervision")
    print("  • Ignore: New labels treated as 'background noise'")
    print(
        "  • Example: Image with 'Atelectasis|Edema' → supervise only on 'Atelectasis'"
    )
    print()
    print("📈 Expected Benefits:")
    print("  • Better generalization to mixed novelties")
    print("  • More robust to real-world data contamination")
    print("  • Improved open-set AUROC on complex cases")
    print("  • Slightly lower closed-set accuracy (trade-off)")
    print("=" * 80)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if not args.source.exists():
        print(f"❌ Source CSV not found: {args.source}", file=sys.stderr)
        return 1

    if not args.test_list.exists():
        print(f"❌ Test list not found: {args.test_list}", file=sys.stderr)
        return 1

    ensure_output_dir(args.output_dir)

    print("🔄 Loading ChestX-ray14 dataset...")
    df = pd.read_csv(args.source)
    print(f"✅ Loaded {len(df):,} total samples")

    # Load official test images
    print("🔄 Loading official test list...")
    test_images = load_test_images(args.test_list)
    print(f"✅ Loaded {len(test_images):,} test images")

    # Add label analysis columns
    print("🔄 Processing labels...")
    df = create_label_columns(df)

    # Split into train_val and test using official splits
    print("🔄 Creating official train/test split...")
    train_val_mask = ~df["Image Index"].isin(test_images)
    df_train_val = df[train_val_mask].copy()
    df_test = df[~train_val_mask].copy()

    print(f"✅ Train/Val: {len(df_train_val):,} samples")
    print(f"✅ Test: {len(df_test):,} samples")

    # STRATEGY 2: NO FILTERING - Include ALL images in training
    print("🔄 Strategy 2: Including ALL images in training (no filtering)...")
    print(
        f"✅ Unfiltered train_val: {len(df_train_val):,} samples (includes images with new labels)"
    )

    # Create patient-disjoint train/validation split
    print("🔄 Creating patient-disjoint train/validation split...")
    df_train, df_val = create_patient_disjoint_split(
        df_train_val, args.val_ratio, args.seed
    )

    print(f"✅ Train: {len(df_train):,} samples")
    print(f"✅ Validation: {len(df_val):,} samples")

    # Verify patient disjointness
    verify_patient_disjointness(df_train, df_val, df_test)

    # Print statistics
    print_label_statistics(df_train, "Training Set")
    print_label_statistics(df_val, "Validation Set")
    print_label_statistics(df_test, "Test Set")

    # Print training notes
    print_training_notes()

    # Save splits
    train_path = args.output_dir / "chestxray_strategy2_train.csv"
    test_path = args.output_dir / "chestxray_strategy2_test.csv"

    # For training, combine train and val (validation used for threshold calibration)
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)

    df_train_combined.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\n✅ Wrote {len(df_train_combined):,} rows to {train_path}")
    print(f"✅ Wrote {len(df_test):,} rows to {test_path}")

    print("\n🎯 Strategy 2 Split Summary:")
    print(
        f"  • Training: {len(df_train_combined):,} samples (ALL images, including new labels)"
    )
    print(f"  • Test: {len(df_test):,} samples (both known and unknown samples)")
    print(f"  • Patient-disjoint splits: ✅")
    print(f"  • Realistic training: ✅ (includes unlabeled pathologies)")
    print(f"  • Label masking required: ✅ (supervise only on known labels)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
