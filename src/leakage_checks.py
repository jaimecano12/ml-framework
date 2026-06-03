"""Data leakage detection checks (Phase 4 + unified risk score)."""

from __future__ import annotations

import warnings
from typing import Any

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
# Unified leakage risk score
# ---------------------------------------------------------------------------

def compute_leakage_risk_score(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> dict[str, Any]:
    """Compute a per-feature leakage risk score combining three signals.

    Signals:
        - **Correlation** (Pearson |r| or Cramér's V): linear/categorical association.
        - **Mutual information** (normalised by max MI): nonlinear association.
        - **Performance inflation**: how much accuracy a single feature achieves
          above the majority-class baseline (proxy for direct label predictability).

    Each signal is normalised to [0, 1] before combining.  The unified score is
    a weighted sum::

        risk = w_corr * corr + w_mi * mi + w_perf * perf_inflation

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``leakage_checks.leakage_risk_score`` config block.
            Relevant keys: ``weights`` (list[float, float, float]),
            ``cv_folds`` (int, default 3).

    Returns:
        Dict with keys ``risk_scores``, ``corr_scores``, ``mi_scores``,
        ``perf_scores``, and ``weights``.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        return {"risk_scores": {}, "corr_scores": {}, "mi_scores": {}, "perf_scores": {}, "weights": {}}

    # Encode target to integers
    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        target_enc = pd.Series(
            LabelEncoder().fit_transform(target.astype(str).fillna("__missing__")),
            name=target_col,
        )
    else:
        target_enc = target.fillna(target.median()).astype(int)

    # Prepare X: encode categoricals, impute
    X_raw = df[feature_cols].copy()
    for col in X_raw.select_dtypes(include=["object", "category"]).columns:
        X_raw[col] = pd.factorize(X_raw[col])[0].astype(float)
    X_raw = X_raw.astype(float)
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(X_raw),
        columns=feature_cols,
    )

    # 1. Correlation scores
    corr_scores = {col: _feature_target_association(df[col], df[target_col]) for col in feature_cols}

    # 2. Mutual information (normalised by max)
    mi_raw = mutual_info_classif(X_imp.values, target_enc.values, random_state=42)
    mi_max = max(float(mi_raw.max()), 1e-10)
    mi_scores = {col: round(float(mi_raw[i]) / mi_max, 4) for i, col in enumerate(feature_cols)}

    # 3. Performance inflation: single-feature LR accuracy vs majority-class baseline
    majority_baseline = float(target_enc.value_counts(normalize=True).max())
    range_above = max(1.0 - majority_baseline, 1e-10)
    cv_folds = min(config.get("cv_folds", 3), len(df) // 2)

    perf_scores: dict[str, float] = {}
    for col in feature_cols:
        x_single = X_imp[[col]].values
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=42, n_jobs=1)),
        ])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    pipe, x_single, target_enc.values,
                    cv=cv_folds, scoring="accuracy", n_jobs=1,
                )
            inflation = max(0.0, float(scores.mean()) - majority_baseline) / range_above
            perf_scores[col] = round(min(1.0, inflation), 4)
        except Exception:
            perf_scores[col] = 0.0

    # Unified weighted score
    weights = config.get("weights", [0.35, 0.35, 0.30])
    w_corr, w_mi, w_perf = weights[0], weights[1], weights[2]
    risk_scores = {
        col: round(min(1.0, w_corr * corr_scores[col] + w_mi * mi_scores[col] + w_perf * perf_scores[col]), 4)
        for col in feature_cols
    }

    return {
        "risk_scores": risk_scores,
        "corr_scores": {c: round(v, 4) for c, v in corr_scores.items()},
        "mi_scores": mi_scores,
        "perf_scores": perf_scores,
        "weights": {"correlation": w_corr, "mutual_information": w_mi, "performance": w_perf},
    }


def check_leakage_risk_score(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Unified leakage risk check — combines correlation, MI, and performance inflation.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``leakage_checks.leakage_risk_score`` config block.

    Returns:
        Passed CheckResult when no feature exceeds the risk threshold, failed otherwise.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    threshold: float = config.get("threshold", 0.7)
    result = compute_leakage_risk_score(df, target_col, config)
    risk_scores = result["risk_scores"]

    high_risk = {col: score for col, score in risk_scores.items() if score >= threshold}

    if not high_risk:
        return CheckResult(
            check_name="leakage_risk_score",
            passed=True,
            severity="info",
            message=f"No features exceed the unified leakage risk threshold ({threshold}).",
            details={**result, "threshold": threshold},
        )

    top_score = max(high_risk.values())
    return CheckResult(
        check_name="leakage_risk_score",
        passed=False,
        severity="error" if top_score >= 0.9 else "warning",
        message=(
            f"{len(high_risk)} feature(s) have unified leakage risk score >= {threshold} "
            f"(max={top_score:.3f})."
        ),
        details={**result, "threshold": threshold, "high_risk_features": high_risk},
        affected_columns=list(high_risk.keys()),
    )


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

    _run("target_leakage",      lambda cfg: check_target_leakage(df, target_col, cfg))
    _run("train_test_overlap",  lambda cfg: check_train_test_overlap(df, cfg))
    _run("temporal_leakage",    lambda cfg: check_temporal_leakage(df, cfg))
    _run("id_column_leakage",   lambda cfg: check_id_column_leakage(df, target_col, cfg))
    _run("leakage_risk_score",  lambda cfg: check_leakage_risk_score(df, target_col, cfg))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Leakage checks complete: {passed}/{len(results)} passed.")
    return results
