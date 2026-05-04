"""Generate three synthetic datasets to demonstrate the framework (Phase 7).

Datasets written to data/raw/:

  clean_dataset.csv        — no quality or leakage issues (control)
  dirty_dataset.csv        — quality issues: missing values, duplicates, outliers,
                             class imbalance, constant column
  leaky_dataset.csv        — leakage issues: perfect proxy feature, ID column,
                             plus the quality issues from dirty_dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw"


def _make_clean(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """500 rows with 5 numeric features, balanced binary target, no issues."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(5, 2, n)
    f3 = rng.normal(-2, 1.5, n)
    f4 = rng.normal(10, 3, n)
    f5 = rng.normal(0, 0.5, n)
    # target depends on a linear combination of features
    score = 0.4 * f1 - 0.3 * f2 + 0.5 * f3 + rng.normal(0, 0.5, n)
    target = (score > score.mean()).astype(int)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "target": target})


def _make_dirty(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Introduces quality issues into an otherwise clean dataset."""
    rng = np.random.default_rng(seed)
    df = _make_clean(n, seed=seed)

    # Missing values — 25 % of f1 and f2
    missing_idx = rng.choice(n, size=int(n * 0.25), replace=False)
    df.loc[missing_idx, "f1"] = np.nan
    missing_idx2 = rng.choice(n, size=int(n * 0.10), replace=False)
    df.loc[missing_idx2, "f2"] = np.nan

    # Duplicates — repeat first 30 rows
    dupes = df.iloc[:30].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # Outliers in f3
    outlier_idx = rng.choice(len(df), size=10, replace=False)
    df.loc[outlier_idx, "f3"] = rng.choice([500.0, -500.0], size=10)

    # Class imbalance — oversample class 0 to make minority class ~5 %
    class1 = df[df["target"] == 1]
    class0 = df[df["target"] == 0].sample(n=len(class1) * 19, replace=True, random_state=seed)
    df = pd.concat([class1, class0], ignore_index=True).sample(frac=1, random_state=seed)

    # Constant column
    df["constant_col"] = 0

    # Low-variance column
    df["near_constant"] = 1.0 + rng.uniform(-0.0001, 0.0001, len(df))

    return df.reset_index(drop=True)


def _make_leaky(n: int = 500, seed: int = 99) -> pd.DataFrame:
    """Adds leakage issues on top of the dirty dataset."""
    df = _make_dirty(n, seed=seed)

    # Perfect proxy — identical to target (target leakage)
    df["target_proxy"] = df["target"].astype(float)

    # Noisy proxy — highly correlated with target (≈ 0.98 correlation)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.05, len(df))
    df["noisy_proxy"] = df["target"].astype(float) + noise

    # ID column (string UUID-like)
    df["user_id"] = [f"UID_{i:06d}" for i in range(len(df))]

    # Temporal disorder — add a date column that is NOT sorted
    from datetime import date, timedelta
    base_date = date(2022, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(len(df))]
    rng.shuffle(dates)
    df["event_date"] = dates

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "clean_dataset.csv": _make_clean(),
        "dirty_dataset.csv": _make_dirty(),
        "leaky_dataset.csv": _make_leaky(),
    }

    for filename, df in datasets.items():
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"  Written: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    print("\nAll synthetic datasets generated successfully.")


if __name__ == "__main__":
    sys.exit(main())
