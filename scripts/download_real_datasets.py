"""Download two real-world datasets via OpenML (same data as Kaggle) for framework testing.

Datasets:
  titanic.csv       — Titanic survival prediction (Kaggle competition dataset)
  diabetes.csv      — Pima Indians Diabetes (UCI/Kaggle classic)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.datasets import fetch_openml


def download_titanic(out_dir: Path) -> Path:
    print("Downloading Titanic dataset from OpenML…")
    data = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Normalise target: may be categorical '0'/'1' or 'no'/'yes'
    survived_str = df["survived"].astype(str).str.strip().str.lower()
    df["survived"] = survived_str.map(
        {"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1}
    ).fillna(0).astype(int)
    # Drop columns with >80 % missing (boat, body, home.dest, cabin)
    threshold = 0.80
    keep = [c for c in df.columns if df[c].isna().mean() < threshold]
    df = df[keep]

    path = out_dir / "titanic.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Target distribution:\n{df['survived'].value_counts().to_string()}")
    return path


def download_diabetes(out_dir: Path) -> Path:
    print("\nDownloading Pima Indians Diabetes dataset from OpenML…")
    data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target column is 'class' with values 'tested_negative'/'tested_positive'
    df["class"] = df["class"].astype(str).str.strip().map(
        {"tested_negative": 0, "tested_positive": 1}
    ).fillna(0).astype(int)

    path = out_dir / "diabetes.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Target distribution:\n{df['class'].value_counts().to_string()}")
    return path


def main() -> int:
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    download_titanic(out_dir)
    download_diabetes(out_dir)
    print("\nAll datasets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
