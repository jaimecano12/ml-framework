"""Download additional UCI datasets via OpenML for extended evaluation.

Datasets:
  adult.csv         — Adult Census Income (48 842 rows, income prediction)
  heart_disease.csv — Cleveland Heart Disease (303 rows, cardiac diagnosis)
  german_credit.csv — German Credit (1 000 rows, credit risk)
  wine_quality.csv  — Wine Quality Red (1 599 rows, quality score binarised)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.datasets import fetch_openml


def download_adult(out_dir: Path) -> Path:
    print("Downloading Adult Census Income dataset from OpenML…")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: '>50K' → 1, '<=50K' → 0
    target_raw = df["class"].astype(str).str.strip().str.replace(".", "", regex=False)
    df["income"] = target_raw.map({"<=50K": 0, ">50K": 1}).fillna(0).astype(int)
    df = df.drop(columns=["class"])

    path = out_dir / "adult.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Target distribution:\n{df['income'].value_counts().to_string()}")
    return path


def download_heart_disease(out_dir: Path) -> Path:
    print("\nDownloading Heart Disease (Cleveland) dataset from OpenML…")
    data = fetch_openml("heart-c", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: 'num' column — 0 = no disease, 1/2/3/4 = disease present
    target_raw = pd.to_numeric(df["num"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    df["disease"] = (target_raw > 0).astype(int)
    df = df.drop(columns=["num"])

    path = out_dir / "heart_disease.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Target distribution:\n{df['disease'].value_counts().to_string()}")
    return path


def download_german_credit(out_dir: Path) -> Path:
    print("\nDownloading German Credit dataset from OpenML…")
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: 'good' → 0, 'bad' → 1
    target_raw = df["class"].astype(str).str.strip().str.lower()
    df["credit_risk"] = target_raw.map({"good": 0, "bad": 1}).fillna(0).astype(int)
    df = df.drop(columns=["class"])

    path = out_dir / "german_credit.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Target distribution:\n{df['credit_risk'].value_counts().to_string()}")
    return path


def download_wine_quality(out_dir: Path) -> Path:
    print("\nDownloading Wine Quality (Red) dataset from OpenML…")
    data = fetch_openml("wine-quality-red", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: quality score; binarise at >= 6 (good vs not good)
    quality = pd.to_numeric(df["class"], errors="coerce")
    df["quality_good"] = (quality >= 6).astype(int)
    df = df.drop(columns=["class"])

    path = out_dir / "wine_quality.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Target distribution:\n{df['quality_good'].value_counts().to_string()}")
    return path


def main() -> int:
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for fn in [download_adult, download_heart_disease, download_german_credit, download_wine_quality]:
        try:
            fn(out_dir)
        except Exception as exc:
            print(f"  WARNING: {fn.__name__} failed: {exc}")
            errors.append(fn.__name__)

    print(f"\nDone. {4 - len(errors)}/4 datasets downloaded successfully.")
    if errors:
        print(f"  Failed: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
