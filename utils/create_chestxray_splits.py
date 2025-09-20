#!/usr/bin/env python3
"""Utility to split the ChestX-ray14 metadata into known vs new label CSV files.

Outputs three CSVs:
  * datasets/splits/chestxray_train_known.csv
  * datasets/splits/chestxray_train_new.csv
  * datasets/splits/chestxray_test_new.csv

Usage:
    python utils/create_chestxray_splits.py \
        --source datasets/data/NIH/Data_Entry_2017.csv \
        --output-dir datasets/data/NIH \
        --test-ratio 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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
        "--output-dir",
        type=Path,
        default=Path("datasets/splits"),
        help="Directory where the split CSVs will be written",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of novel-only patients held out for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random split",
    )
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_labels(label_str: str) -> set[str]:
    if not isinstance(label_str, str) or not label_str or label_str == "nan":
        return set()
    if label_str == "No Finding":
        return set()
    return {part.strip() for part in label_str.split("|") if part.strip()}


def assign_splits(
    df: pd.DataFrame, seed: int, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df["Finding Labels"].apply(extract_labels)

    df = df.copy()
    df["known_labels"] = labels.apply(lambda s: sorted(s.intersection(KNOWN_LABELS)))
    df["new_labels"] = labels.apply(lambda s: sorted(s.intersection(NEW_LABELS)))
    df["has_known"] = df["known_labels"].apply(bool)
    df["has_new"] = df["new_labels"].apply(bool)
    df["is_no_finding"] = df["Finding Labels"] == "No Finding"

    # Base training data: rows containing at least one known pathology or negatives
    train_known = df[df["has_known"] | df["is_no_finding"]].copy()

    # Novel pools
    new_pool = df[df["has_new"]].copy()
    new_only = df[df["has_new"] & ~df["has_known"]].copy()

    if not new_only.empty:
        rng = np.random.default_rng(seed)
        unique_patients = new_only["Patient ID"].unique()
        n_test = max(1, int(len(unique_patients) * test_ratio))
        if n_test >= len(unique_patients):
            test_patients = set(unique_patients)
        else:
            test_patients = set(rng.choice(unique_patients, size=n_test, replace=False))
        test_new = new_only[new_only["Patient ID"].isin(test_patients)].copy()
    else:
        test_new = new_only

    if test_new.empty:
        train_new = new_pool.copy()
    else:
        test_indices = set(test_new["Image Index"])
        train_new = new_pool[~new_pool["Image Index"].isin(test_indices)].copy()

    def finalize(df_out: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "Image Index",
            "Finding Labels",
            "known_labels",
            "new_labels",
            "has_known",
            "has_new",
            "is_no_finding",
            "Follow-up #",
            "Patient ID",
            "Patient Age",
            "Patient Gender",
            "View Position",
        ]
        extra_cols = [
            "OriginalImage[Width",
            "Height]",
            "OriginalImagePixelSpacing[x",
            "y]",
        ]
        cols.extend([c for c in extra_cols if c in df_out.columns])
        return df_out.loc[:, cols]

    return finalize(train_known), finalize(train_new), finalize(test_new)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if not args.source.exists():
        print(f"❌ Source CSV not found: {args.source}", file=sys.stderr)
        return 1

    ensure_output_dir(args.output_dir)

    df = pd.read_csv(args.source)
    train_known, train_new, test_new = assign_splits(df, args.seed, args.test_ratio)

    train_known_path = args.output_dir / "chestxray_train_known.csv"
    train_new_path = args.output_dir / "chestxray_train_new.csv"
    test_new_path = args.output_dir / "chestxray_test_new.csv"

    train_known.to_csv(train_known_path, index=False)
    train_new.to_csv(train_new_path, index=False)
    test_new.to_csv(test_new_path, index=False)

    print(f"✅ Wrote {len(train_known):,} rows to {train_known_path}")
    print(f"✅ Wrote {len(train_new):,} rows to {train_new_path}")
    print(f"✅ Wrote {len(test_new):,} rows to {test_new_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
    """_summary_
    datasets/splits/chestxray_train_known.csv – 106,993 rows with at least one known label (or “No Finding”) while retaining any extra novel tags for masking.
    datasets/splits/chestxray_train_new.csv – 12,714 rows that include the six novel labels (rows landing in the test split are excluded so there's no leakage).
    datasets/splits/chestxray_test_new.csv – 1,025 novel-only rows for evaluating generalization to unseen labels.
    """
