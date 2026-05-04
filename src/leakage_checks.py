"""Data leakage detection checks (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import CheckResult, logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association statistic between two categorical series."""
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = stats.chi2_contingency(confusion, correction=False)
    n = len(x)
    r, k = confusion.shape
    denom = n * min(r - 1, k - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _feature_target_association(feature: pd.Series, target: pd.Series) -> float:
    """Return an association score in [0, 1] between *feature* and *target*.

    - Numeric features: absolute Pearson correlation against integer-encoded target.
    - Categorical features: Cramér's V.

    Returns 0.0 when data is insufficient or computation fails.
    """
    clean = pd.DataFrame({"f": feature, "t": target}).dropna()
    if len(clean) < 4:
        return 0.0

    f, t = clean["f"], clean["t"]

    if pd.api.types.is_numeric_dtype(f):
        t_enc = (
            pd.factorize(t)[0]
            if not pd.api.types.is_numeric_dtype(t)
            else t.values.astype(float)
        )
        try:
            corr = np.corrcoef(f.values.astype(float), t_enc)[0, 1]
            return 0.0 if np.isnan(corr) else float(abs(corr))
        except Exception:
            return 0.0
    else:
        try:
            return _cramers_v(f.astype(str), t.astype(str))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_target_leakage(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Detect features that are suspiciously associated with the target.

    Flags any feature whose association score (Pearson |r| for numeric features,
    Cramér's V for categorical) exceeds *config['correlation_threshold']*.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``leakage_checks.target_leakage`` config block.

    Returns:
        Passed CheckResult when no feature exceeds the threshold, failed otherwise.

    Raises:
        ValueError: If *target_col* is not present in *df*.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    threshold: float = config.get("correlation_threshold", 0.95)
    leaky: dict[str, float] = {}

    for col in df.columns:
        if col == target_col:
            continue
        assoc = _feature_target_association(df[col], df[target_col])
        if assoc >= threshold:
            leaky[col] = round(assoc, 4)

    if not leaky:
        return CheckResult(
            check_name="target_leakage",
            passed=True,
            severity="info",
            message=f"No features exceed the leakage threshold ({threshold}).",
            details={"correlation_threshold": threshold},
        )

    return CheckResult(
        check_name="target_leakage",
        passed=False,
        severity="error",
        message=(
            f"{len(leaky)} feature(s) show suspiciously high association with "
            f"'{target_col}' (threshold={threshold})."
        ),
        details={"correlation_threshold": threshold, "flagged_features": leaky},
        affected_columns=list(leaky.keys()),
    )


def check_train_test_overlap(df: pd.DataFrame, config: dict) -> CheckResult:
    """Detect rows that appear in both train and test after a simulated split.

    Overlap is only possible when duplicate rows exist. The check simulates a
    random split and counts rows present in both halves.

    Args:
        df: Input DataFrame.
        config: The ``leakage_checks.train_test_overlap`` config block.

    Returns:
        Passed CheckResult when no overlap is found, failed otherwise.
    """
    from sklearn.model_selection import train_test_split

    test_size: float = config.get("test_size", 0.2)
    random_state: int = config.get("random_state", 42)

    if not df.duplicated().any():
        return CheckResult(
            check_name="train_test_overlap",
            passed=True,
            severity="info",
            message="No duplicate rows — train/test overlap is impossible.",
            details={"overlap_rows": 0},
        )

    train_idx, test_idx = train_test_split(
        list(range(len(df))), test_size=test_size, random_state=random_state
    )
    train_dedup = df.iloc[train_idx].drop_duplicates()
    test_dedup = df.iloc[test_idx].drop_duplicates()

    overlap = pd.merge(train_dedup, test_dedup, how="inner")
    n_overlap = len(overlap)

    if n_overlap == 0:
        return CheckResult(
            check_name="train_test_overlap",
            passed=True,
            severity="info",
            message="Duplicates exist but none span the simulated train/test boundary.",
            details={"overlap_rows": 0, "test_size": test_size},
        )

    overlap_rate = round(n_overlap / len(df), 4)
    severity = "error" if overlap_rate > 0.05 else "warning"
    return CheckResult(
        check_name="train_test_overlap",
        passed=False,
        severity=severity,
        message=(
            f"{n_overlap} row(s) ({overlap_rate:.1%}) appear in both the simulated "
            f"train and test splits."
        ),
        details={
            "overlap_rows": n_overlap,
            "overlap_rate": overlap_rate,
            "test_size": test_size,
        },
    )


def check_temporal_leakage(df: pd.DataFrame, config: dict) -> CheckResult:
    """Detect temporal leakage when a date column is configured.

    Verifies that the dataset is sorted chronologically. An unsorted date column
    means a naïve sequential split would mix past and future observations.

    Args:
        df: Input DataFrame.
        config: The ``leakage_checks.temporal_leakage`` config block.

    Returns:
        Passed CheckResult when data is properly ordered (or no date configured),
        failed otherwise.

    Raises:
        ValueError: If the configured date column is not present in *df*.
    """
    date_col: str | None = config.get("date_column")

    if not date_col:
        return CheckResult(
            check_name="temporal_leakage",
            passed=True,
            severity="info",
            message="No date column configured — temporal leakage check skipped.",
            details={"date_column": None},
        )

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    n_nat = int(dates.isna().sum())
    is_sorted = bool(dates.dropna().is_monotonic_increasing)

    issues: list[str] = []
    if n_nat > 0:
        issues.append(f"{n_nat} unparseable date value(s) in '{date_col}'.")
    if not is_sorted:
        issues.append(
            f"Dataset is not chronologically sorted by '{date_col}' — "
            "a sequential split would mix past and future observations."
        )

    if not issues:
        return CheckResult(
            check_name="temporal_leakage",
            passed=True,
            severity="info",
            message=f"Dataset is properly sorted by '{date_col}'.",
            details={"date_column": date_col, "is_sorted": True, "unparseable_dates": 0},
        )

    severity = "error" if not is_sorted else "warning"
    return CheckResult(
        check_name="temporal_leakage",
        passed=False,
        severity=severity,
        message=" ".join(issues),
        details={
            "date_column": date_col,
            "is_sorted": is_sorted,
            "unparseable_dates": n_nat,
        },
        affected_columns=[date_col],
    )


def check_id_column_leakage(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Detect high-cardinality columns that may act as row identifiers.

    A column where almost every value is unique (e.g. a primary key or UUID)
    gives the model a way to memorise training samples instead of generalising.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column (excluded from the check).
        config: The ``leakage_checks.id_column_leakage`` config block.

    Returns:
        Passed CheckResult when no ID-like columns are found, failed otherwise.
    """
    threshold: float = config.get("cardinality_threshold", 0.95)
    n_rows = len(df)

    if n_rows == 0:
        return CheckResult(
            check_name="id_column_leakage",
            passed=True,
            severity="info",
            message="Empty DataFrame — ID column check skipped.",
            details={"cardinality_threshold": threshold},
        )

    id_cols: dict[str, float] = {}
    for col in df.columns:
        if col == target_col:
            continue
        # Continuous floats are expected to have high cardinality; only string/integer
        # columns are plausible ID carriers (e.g. UUIDs, sequential keys).
        if pd.api.types.is_float_dtype(df[col]):
            continue
        unique_ratio = round(df[col].nunique() / n_rows, 4)
        if unique_ratio >= threshold:
            id_cols[col] = unique_ratio

    if not id_cols:
        return CheckResult(
            check_name="id_column_leakage",
            passed=True,
            severity="info",
            message=f"No high-cardinality (>= {threshold:.0%}) columns detected.",
            details={"cardinality_threshold": threshold},
        )

    return CheckResult(
        check_name="id_column_leakage",
        passed=False,
        severity="warning",
        message=(
            f"{len(id_cols)} column(s) have unique-value ratio >= {threshold:.0%} "
            "and may act as row identifiers."
        ),
        details={"cardinality_threshold": threshold, "flagged_columns": id_cols},
        affected_columns=list(id_cols.keys()),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_leakage_checks(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> list[CheckResult]:
    """Run all enabled leakage-detection checks and return their results.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The full ``leakage_checks`` config block.

    Returns:
        List of :class:`~src.utils.CheckResult`, one per executed check.
    """
    if not config.get("enabled", True):
        logger.info("Leakage checks disabled — skipping.")
        return []

    results: list[CheckResult] = []

    def _run(key: str, fn):
        cfg = config.get(key, {})
        if not cfg.get("enabled", True):
            logger.debug(f"Leakage check '{key}' disabled — skipping.")
            return
        logger.debug(f"Running leakage check: {key}")
        r: CheckResult = fn(cfg)
        results.append(r)
        (logger.warning if not r.passed else logger.debug)(f"[{key}] {r.message}")

    _run("target_leakage",     lambda cfg: check_target_leakage(df, target_col, cfg))
    _run("train_test_overlap", lambda cfg: check_train_test_overlap(df, cfg))
    _run("temporal_leakage",   lambda cfg: check_temporal_leakage(df, cfg))
    _run("id_column_leakage",  lambda cfg: check_id_column_leakage(df, target_col, cfg))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Leakage checks complete: {passed}/{len(results)} passed.")
    return results
