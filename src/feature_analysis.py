"""Feature analysis checks: correlation, relevance, and distribution shape (Phase 9)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from .utils import CheckResult, logger


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_feature_correlation(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> CheckResult:
    """Detect pairs of numeric features that are highly correlated with each other.

    High inter-feature correlation (multicollinearity) means redundant information,
    longer training times, and unstable coefficients in linear models.

    Args:
        df: Input DataFrame.
        target_col: Target column — excluded from the pairwise correlation.
        config: The ``feature_analysis.feature_correlation`` config block.

    Returns:
        Passed CheckResult when no pair exceeds the threshold, failed otherwise.
    """
    threshold: float = config.get("correlation_threshold", 0.90)
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    if len(feature_cols) < 2:
        return CheckResult(
            check_name="feature_correlation",
            passed=True,
            severity="info",
            message="Fewer than 2 numeric features — correlation check skipped.",
            details={"threshold": threshold},
        )

    corr_matrix = df[feature_cols].corr(method="pearson").abs()
    correlated_pairs: list[dict] = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = float(corr_matrix.iloc[i, j])
            if np.isnan(r):
                continue
            if r >= threshold:
                correlated_pairs.append({
                    "feature_a": feature_cols[i],
                    "feature_b": feature_cols[j],
                    "correlation": round(r, 4),
                })

    if not correlated_pairs:
        return CheckResult(
            check_name="feature_correlation",
            passed=True,
            severity="info",
            message=f"No highly correlated feature pairs detected (threshold={threshold}).",
            details={"threshold": threshold, "features_checked": len(feature_cols)},
        )

    affected = sorted(set(
        p["feature_a"] for p in correlated_pairs
    ) | set(
        p["feature_b"] for p in correlated_pairs
    ))

    return CheckResult(
        check_name="feature_correlation",
        passed=False,
        severity="warning",
        message=(
            f"{len(correlated_pairs)} highly correlated feature pair(s) detected "
            f"(|r| ≥ {threshold}). Consider removing redundant features or applying PCA."
        ),
        details={"threshold": threshold, "correlated_pairs": correlated_pairs},
        affected_columns=affected,
    )


def check_feature_relevance(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> CheckResult:
    """Detect numeric features with near-zero mutual information with the target.

    Features with MI ≈ 0 carry no signal for the target and act as noise,
    increasing overfitting risk without improving predictions.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``feature_analysis.feature_relevance`` config block.

    Returns:
        Passed CheckResult when all features have sufficient MI, failed otherwise.
    """
    threshold: float = config.get("mi_threshold", 0.01)
    random_state: int = config.get("random_state", 42)

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]
    if not feature_cols:
        return CheckResult(
            check_name="feature_relevance",
            passed=True,
            severity="info",
            message="No numeric features available for relevance analysis.",
            details={"threshold": threshold},
        )

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Impute NaN with column median for MI computation
    X = X.fillna(X.median())

    if len(X) < 10:
        return CheckResult(
            check_name="feature_relevance",
            passed=True,
            severity="info",
            message="Not enough rows for reliable mutual information estimation.",
            details={"threshold": threshold},
        )

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.fillna(y.median()).values

    n_classes = len(np.unique(y))
    is_classification = n_classes <= 20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_classification:
            raw_mi = mutual_info_classif(X, y, random_state=random_state)
        else:
            raw_mi = mutual_info_regression(X, y, random_state=random_state)

    max_mi = float(np.max(raw_mi)) if np.max(raw_mi) > 0 else 1.0
    normalized_mi = raw_mi / max_mi

    all_scores = {col: round(float(s), 4) for col, s in zip(feature_cols, normalized_mi)}
    low_relevance = {col: score for col, score in all_scores.items() if score < threshold}

    if not low_relevance:
        return CheckResult(
            check_name="feature_relevance",
            passed=True,
            severity="info",
            message=f"All features have normalised MI ≥ {threshold} relative to the target.",
            details={"threshold": threshold, "all_scores": all_scores},
        )

    return CheckResult(
        check_name="feature_relevance",
        passed=False,
        severity="warning",
        message=(
            f"{len(low_relevance)} feature(s) have near-zero mutual information with "
            f"the target (normalised MI < {threshold}). They may be pure noise."
        ),
        details={
            "threshold": threshold,
            "low_relevance_features": low_relevance,
            "all_scores": all_scores,
        },
        affected_columns=list(low_relevance.keys()),
    )


def check_distribution_shape(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> CheckResult:
    """Flag numeric features with extreme skewness or heavy tails.

    Highly non-normal distributions can reduce the effectiveness of distance-based
    and linear models. Variance-stabilising transforms often improve performance.

    Args:
        df: Input DataFrame.
        target_col: Target column — excluded from the check.
        config: The ``feature_analysis.distribution_shape`` config block.

    Returns:
        Passed CheckResult when all distributions are within bounds, failed otherwise.
    """
    skew_threshold: float = config.get("skewness_threshold", 2.0)
    kurt_threshold: float = config.get("kurtosis_threshold", 7.0)

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    flagged: dict[str, dict] = {}
    for col in feature_cols:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        skewness = float(series.skew())
        kurtosis = float(series.kurtosis())  # excess kurtosis
        if abs(skewness) > skew_threshold or kurtosis > kurt_threshold:
            flagged[col] = {
                "skewness": round(skewness, 3),
                "excess_kurtosis": round(kurtosis, 3),
            }

    if not flagged:
        return CheckResult(
            check_name="distribution_shape",
            passed=True,
            severity="info",
            message=(
                f"All numeric features are within shape bounds "
                f"(|skew| ≤ {skew_threshold}, kurtosis ≤ {kurt_threshold})."
            ),
            details={"skewness_threshold": skew_threshold, "kurtosis_threshold": kurt_threshold},
        )

    return CheckResult(
        check_name="distribution_shape",
        passed=False,
        severity="warning",
        message=(
            f"{len(flagged)} feature(s) have highly non-normal distributions "
            f"(|skew| > {skew_threshold} or excess kurtosis > {kurt_threshold}). "
            "Consider variance-stabilising transforms."
        ),
        details={
            "skewness_threshold": skew_threshold,
            "kurtosis_threshold": kurt_threshold,
            "flagged_features": flagged,
        },
        affected_columns=list(flagged.keys()),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_feature_checks(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> list[CheckResult]:
    """Run all enabled feature analysis checks.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The full ``feature_analysis`` config block.

    Returns:
        List of :class:`~src.utils.CheckResult`, one per executed check.
    """
    if not config.get("enabled", True):
        logger.info("Feature analysis disabled — skipping.")
        return []

    results: list[CheckResult] = []

    def _run(key: str, fn):
        cfg = config.get(key, {})
        if not cfg.get("enabled", True):
            logger.debug(f"Feature check '{key}' disabled — skipping.")
            return
        logger.debug(f"Running feature check: {key}")
        r = fn(cfg)
        results.append(r)
        (logger.warning if not r.passed else logger.debug)(f"[{key}] {r.message}")

    _run("feature_correlation", lambda cfg: check_feature_correlation(df, target_col, cfg))
    _run("feature_relevance",   lambda cfg: check_feature_relevance(df, target_col, cfg))
    _run("distribution_shape",  lambda cfg: check_distribution_shape(df, target_col, cfg))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Feature analysis complete: {passed}/{len(results)} passed.")
    return results
