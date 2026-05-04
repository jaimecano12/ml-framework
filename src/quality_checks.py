"""Data quality checks (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import CheckResult, logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_outliers_iqr(series: pd.Series, threshold: float) -> pd.Series:
    """Boolean mask of outliers via IQR method. Returns all-False for constant series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    return (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)


def _detect_outliers_zscore(series: pd.Series, threshold: float) -> pd.Series:
    """Boolean mask of outliers via z-score method. Returns all-False for constant series."""
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    return ((series - series.mean()) / std).abs() > threshold


def _coefficient_of_variation(series: pd.Series) -> float:
    """Coefficient of variation (std / |mean|). Falls back to absolute std when mean ≈ 0.

    A near-constant column (e.g. all values ≈ 1.0 ± 0.0001) has CV ≈ 0.0001,
    while a column with genuine spread has CV >> 0.01.
    """
    std = float(series.std())
    mean_abs = abs(float(series.mean()))
    if mean_abs < 1e-10:
        return std
    return std / mean_abs


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_missing_values(df: pd.DataFrame, config: dict) -> CheckResult:
    """Flag columns whose missing-value rate exceeds *config['threshold']*.

    Args:
        df: Input DataFrame.
        config: The ``quality_checks.missing_values`` config block.

    Returns:
        Passed CheckResult when no column exceeds the threshold, failed otherwise.
    """
    threshold: float = config.get("threshold", 0.05)
    missing_rates = df.isnull().mean()
    flagged = missing_rates[missing_rates > threshold].sort_values(ascending=False)

    if flagged.empty:
        return CheckResult(
            check_name="missing_values",
            passed=True,
            severity="info",
            message="No columns exceed the missing-value threshold.",
            details={"threshold": threshold, "max_missing_rate": round(float(missing_rates.max()), 4)},
        )

    severity = "error" if float(flagged.max()) > 0.5 else "warning"
    return CheckResult(
        check_name="missing_values",
        passed=False,
        severity=severity,
        message=f"{len(flagged)} column(s) exceed {threshold:.0%} missing-value rate.",
        details={
            "threshold": threshold,
            "flagged_columns": {col: round(rate, 4) for col, rate in flagged.items()},
        },
        affected_columns=flagged.index.tolist(),
    )


def check_duplicates(df: pd.DataFrame, config: dict) -> CheckResult:  # noqa: ARG001
    """Detect exact duplicate rows.

    Args:
        df: Input DataFrame.
        config: The ``quality_checks.duplicates`` config block (currently unused).

    Returns:
        Passed CheckResult when no duplicates are found, failed otherwise.
    """
    duplicate_mask = df.duplicated()
    n_dupes = int(duplicate_mask.sum())

    if n_dupes == 0:
        return CheckResult(
            check_name="duplicates",
            passed=True,
            severity="info",
            message="No duplicate rows detected.",
            details={"duplicate_count": 0},
        )

    dup_rate = round(n_dupes / len(df), 4)
    severity = "error" if dup_rate > 0.1 else "warning"
    return CheckResult(
        check_name="duplicates",
        passed=False,
        severity=severity,
        message=f"{n_dupes} duplicate row(s) found ({dup_rate:.1%} of dataset).",
        details={"duplicate_count": n_dupes, "duplicate_rate": dup_rate},
    )


def check_outliers(df: pd.DataFrame, config: dict) -> CheckResult:
    """Detect outliers in numeric columns using IQR or z-score.

    Args:
        df: Input DataFrame.
        config: The ``quality_checks.outliers`` config block.

    Returns:
        Passed CheckResult when no outliers are detected, failed otherwise.
    """
    method: str = config.get("method", "iqr")
    threshold: float = config.get("threshold", 3.0)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    detector = _detect_outliers_iqr if method == "iqr" else _detect_outliers_zscore
    outlier_counts: dict[str, int] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        n_outliers = int(detector(series, threshold).sum())
        if n_outliers > 0:
            outlier_counts[col] = n_outliers

    if not outlier_counts:
        return CheckResult(
            check_name="outliers",
            passed=True,
            severity="info",
            message=f"No outliers detected (method={method}, threshold={threshold}).",
            details={"method": method, "threshold": threshold},
        )

    total = sum(outlier_counts.values())
    return CheckResult(
        check_name="outliers",
        passed=False,
        severity="warning",
        message=f"{total} outlier(s) across {len(outlier_counts)} column(s) "
                f"(method={method}, threshold={threshold}).",
        details={"method": method, "threshold": threshold, "flagged_columns": outlier_counts},
        affected_columns=list(outlier_counts.keys()),
    )


def check_class_imbalance(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Check whether the target column is heavily imbalanced.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``quality_checks.class_imbalance`` config block.

    Returns:
        Passed CheckResult when class distribution is balanced, failed otherwise.

    Raises:
        ValueError: If *target_col* is not present in *df*.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    threshold: float = config.get("threshold", 0.1)
    counts = df[target_col].value_counts(normalize=True).sort_values()
    minority_ratio = float(counts.iloc[0])
    minority_class = counts.index[0]

    if minority_ratio >= threshold:
        return CheckResult(
            check_name="class_imbalance",
            passed=True,
            severity="info",
            message="Class distribution is within acceptable bounds.",
            details={
                "threshold": threshold,
                "class_distribution": {str(k): round(v, 4) for k, v in counts.items()},
            },
        )

    severity = "error" if minority_ratio < 0.05 else "warning"
    return CheckResult(
        check_name="class_imbalance",
        passed=False,
        severity=severity,
        message=(
            f"Class imbalance detected: minority class '{minority_class}' "
            f"represents only {minority_ratio:.1%} of samples (threshold={threshold:.0%})."
        ),
        details={
            "threshold": threshold,
            "minority_class": str(minority_class),
            "minority_ratio": round(minority_ratio, 4),
            "class_distribution": {str(k): round(v, 4) for k, v in counts.items()},
        },
        affected_columns=[target_col],
    )


def check_constant_features(df: pd.DataFrame, config: dict) -> CheckResult:  # noqa: ARG001
    """Detect columns with a single unique non-null value.

    Args:
        df: Input DataFrame.
        config: The ``quality_checks.constant_features`` config block (currently unused).

    Returns:
        Passed CheckResult when no constant columns are found, failed otherwise.
    """
    constant_cols = [col for col in df.columns if df[col].dropna().nunique() <= 1]

    if not constant_cols:
        return CheckResult(
            check_name="constant_features",
            passed=True,
            severity="info",
            message="No constant features detected.",
            details={"constant_columns": []},
        )

    return CheckResult(
        check_name="constant_features",
        passed=False,
        severity="warning",
        message=f"{len(constant_cols)} constant column(s) detected (zero information).",
        details={"constant_columns": constant_cols},
        affected_columns=constant_cols,
    )


def check_low_variance(df: pd.DataFrame, config: dict) -> CheckResult:
    """Detect numeric columns with low normalised variance.

    Variance is computed after min-max normalisation so the threshold is
    scale-independent. Constant columns (caught by :func:`check_constant_features`)
    are skipped.

    Args:
        df: Input DataFrame.
        config: The ``quality_checks.low_variance`` config block.

    Returns:
        Passed CheckResult when no low-variance columns are found, failed otherwise.
    """
    threshold: float = config.get("threshold", 0.01)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    low_var: dict[str, float] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        var = _coefficient_of_variation(series)
        if 0.0 < var < threshold:  # skip constants (var == 0)
            low_var[col] = round(var, 6)

    if not low_var:
        return CheckResult(
            check_name="low_variance",
            passed=True,
            severity="info",
            message="No low-variance numeric columns detected.",
            details={"threshold": threshold},
        )

    return CheckResult(
        check_name="low_variance",
        passed=False,
        severity="warning",
        message=(
            f"{len(low_var)} column(s) have coefficient of variation below {threshold} "
            f"(likely near-constant)."
        ),
        details={"threshold": threshold, "flagged_columns": low_var},
        affected_columns=list(low_var.keys()),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_CHECK_REGISTRY: dict[str, str] = {
    "missing_values": "check_missing_values",
    "duplicates": "check_duplicates",
    "outliers": "check_outliers",
    "constant_features": "check_constant_features",
    "low_variance": "check_low_variance",
}


def run_all_quality_checks(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> list[CheckResult]:
    """Run all enabled quality checks and return their results.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column (used by class_imbalance).
        config: The full ``quality_checks`` config block.

    Returns:
        List of :class:`~src.utils.CheckResult`, one per executed check.
    """
    if not config.get("enabled", True):
        logger.info("Quality checks disabled — skipping.")
        return []

    results: list[CheckResult] = []
    module = globals()

    for key, func_name in _CHECK_REGISTRY.items():
        check_cfg = config.get(key, {})
        if not check_cfg.get("enabled", True):
            logger.debug(f"Quality check '{key}' disabled — skipping.")
            continue
        logger.debug(f"Running quality check: {key}")
        result: CheckResult = module[func_name](df, check_cfg)
        results.append(result)
        level = "warning" if not result.passed else "debug"
        getattr(logger, level)(f"[{key}] {result.message}")

    # class_imbalance needs the target column explicitly
    ci_cfg = config.get("class_imbalance", {})
    if ci_cfg.get("enabled", True):
        logger.debug("Running quality check: class_imbalance")
        result = check_class_imbalance(df, target_col, ci_cfg)
        results.append(result)
        level = "warning" if not result.passed else "debug"
        getattr(logger, level)(f"[class_imbalance] {result.message}")

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Quality checks complete: {passed}/{len(results)} passed.")
    return results
