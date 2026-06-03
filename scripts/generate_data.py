"""Generate synthetic datasets to demonstrate the framework.

Phase 7 datasets (data/raw/):
  clean_dataset.csv        — no quality or leakage issues (control)
  dirty_dataset.csv        — quality issues: missing values, duplicates, outliers,
                             class imbalance, constant column
  leaky_dataset.csv        — leakage issues: perfect proxy feature, ID column,
                             plus the quality issues from dirty_dataset

Extended scenarios (data/raw/):
  proxy_leakage.csv        — noisy proxy (low/medium/high noise), indirect computed
                             feature, and group-based ID leakage
  temporal_leakage.csv     — future-information feature (computed after the event)
  multitype_leakage.csv    — combines proxy + temporal + ID in one realistic dataset
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


def _make_proxy_leakage(n: int = 1000, seed: int = 13) -> pd.DataFrame:
    """Proxy leakage + indirect computed feature + group ID leakage.

    Realistic scenario: a credit-scoring dataset where:
    - ``proxy_low_noise``: corr ≈ 0.99 (nearly perfect proxy — obvious leak)
    - ``proxy_med_noise``: corr ≈ 0.90 (high but not obvious — harder to detect)
    - ``indirect_feature``: target * income_scale + noise (indirect computation)
    - ``group_id``: 20 groups each repeated 50 times (entity-level leakage)
    """
    rng = np.random.default_rng(seed)
    income = rng.normal(50_000, 15_000, n)
    age = rng.integers(20, 65, n).astype(float)
    debt_ratio = rng.beta(2, 5, n)
    score = 0.5 * (income / 50_000) - 0.3 * debt_ratio + rng.normal(0, 0.3, n)
    target = (score > score.mean()).astype(int)

    proxy_low = target.astype(float) + rng.normal(0, 0.02, n)   # corr ≈ 0.99
    proxy_med = target.astype(float) + rng.normal(0, 0.45, n)   # corr ≈ 0.90

    # Indirect feature: scaled by income (a post-hoc aggregation leak)
    indirect = target * (income / income.mean()) + rng.normal(0, 0.1, n)

    # Group ID — 20 entities, each appears 50 times (group leakage if not split by group)
    group_id = np.tile(np.arange(20), n // 20 + 1)[:n]
    rng.shuffle(group_id)

    return pd.DataFrame({
        "income":        income,
        "age":           age,
        "debt_ratio":    debt_ratio,
        "proxy_low_noise": proxy_low,
        "proxy_med_noise": proxy_med,
        "indirect_feature": indirect,
        "group_id":      group_id,
        "target":        target,
    })


def _make_temporal_leakage(n: int = 2000, seed: int = 77) -> pd.DataFrame:
    """Temporal leakage: a feature contains information from the future.

    Simulates a customer churn dataset where ``future_usage`` is computed
    *after* the churn event — a classic temporal leakage in production systems.
    The dataset is intentionally NOT sorted by date to compound the issue.
    """
    from datetime import date, timedelta

    rng = np.random.default_rng(seed)
    base_date = date(2021, 1, 1)
    dates = sorted([base_date + timedelta(days=int(d)) for d in rng.integers(0, 730, n)])

    tenure_months = rng.integers(1, 60, n).astype(float)
    monthly_spend = rng.normal(80, 30, n)
    support_calls = rng.poisson(2, n).astype(float)
    score = -0.02 * tenure_months + 0.01 * support_calls + rng.normal(0, 0.5, n)
    target = (score > score.mean()).astype(int)  # 1 = churned

    # Future information: usage in the 30 days AFTER the observation window
    future_usage = np.where(target == 1, rng.normal(5, 1, n), rng.normal(25, 3, n))

    # Shuffle dates to break chronological order
    dates_shuffled = list(dates)
    rng.shuffle(dates_shuffled)

    return pd.DataFrame({
        "event_date":    dates_shuffled,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "support_calls": support_calls,
        "future_usage":  future_usage,   # the leaky feature
        "target":        target,
    })


def _make_multitype_leakage(n: int = 1500, seed: int = 55) -> pd.DataFrame:
    """Combines proxy + temporal + ID leakage in one realistic medical dataset.

    Models an ICU mortality prediction scenario with multiple leakage sources:
    - ``discharge_code``: assigned at discharge (after outcome — temporal leak)
    - ``mortality_score``: direct proxy for target with low noise
    - ``patient_id``: 300 patients each appearing 5 times (entity leakage)
    - ``ward_id``: string ID with high cardinality
    """
    from datetime import date, timedelta

    rng = np.random.default_rng(seed)
    age = rng.integers(30, 90, n).astype(float)
    apache_score = rng.normal(15, 8, n)
    comorbidities = rng.poisson(2, n).astype(float)
    icu_hours = rng.integers(4, 240, n).astype(float)

    score = 0.04 * apache_score + 0.1 * comorbidities + rng.normal(0, 1, n)
    target = (score > score.mean()).astype(int)  # 1 = mortality

    mortality_score = target * 0.8 + rng.normal(0, 0.1, n)       # near-perfect proxy
    discharge_code  = np.where(target == 1, rng.integers(900, 999, n),
                                             rng.integers(100, 299, n))  # temporal

    patient_id = np.tile(np.arange(300), n // 300 + 1)[:n]
    rng.shuffle(patient_id)
    ward_id = [f"WARD_{rng.integers(0, 200):03d}" for _ in range(n)]

    base = date(2020, 1, 1)
    dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 1000, n)]

    return pd.DataFrame({
        "admission_date":  dates,
        "age":             age,
        "apache_score":    apache_score,
        "comorbidities":   comorbidities,
        "icu_hours":       icu_hours,
        "mortality_score": mortality_score,
        "discharge_code":  discharge_code,
        "patient_id":      patient_id,
        "ward_id":         ward_id,
        "target":          target,
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "clean_dataset.csv":       _make_clean(),
        "dirty_dataset.csv":       _make_dirty(),
        "leaky_dataset.csv":       _make_leaky(),
        "proxy_leakage.csv":       _make_proxy_leakage(),
        "temporal_leakage_ext.csv": _make_temporal_leakage(),
        "multitype_leakage.csv":   _make_multitype_leakage(),
    }

    for filename, df in datasets.items():
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"  Written: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    print("\nAll synthetic datasets generated successfully.")


if __name__ == "__main__":
    sys.exit(main())
