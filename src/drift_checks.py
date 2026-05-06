"""Covariate and label drift detection between dataset halves (Phase 14)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from .utils import CheckResult, logger

# Population Stability Index thresholds
_PSI_MODERATE = 0.1
_PSI_SEVERE   = 0.25


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric arrays.

    Adaptive binning prevents inflated PSI values on small samples
    (rule of thumb: at least 20 observations per bin).
    """
    n_adaptive = max(5, min(n_bins, min(len(expected), len(actual)) // 20))
    bins = np.percentile(expected, np.linspace(0, 100, n_adaptive + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual,   bins=bins)[0]

    exp_pct = np.maximum(exp_counts / len(expected), 1e-6)
    act_pct = np.maximum(act_counts / len(actual),   1e-6)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _split_dataframe(
    df: pd.DataFrame, date_col: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into two halves for drift comparison.

    If *date_col* is provided the dataset is sorted chronologically first;
    otherwise a sequential index split is used.
    """
    if date_col and date_col in df.columns:
        df_s = df.sort_values(date_col).reset_index(drop=True)
    else:
        df_s = df.reset_index(drop=True)

    mid = len(df_s) // 2
    return df_s.iloc[:mid], df_s.iloc[mid:]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_covariate_drift(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> CheckResult:
    """Detect distributional shift in features between two dataset halves.

    Numeric features are assessed with the KS test and Population Stability Index.
    A Bonferroni correction is applied to the significance threshold to control the
    family-wise error rate.

    Args:
        df: Input DataFrame.
        target_col: Target column — excluded from drift analysis.
        config: The ``drift_checks.covariate_drift`` config block.
    """
    alpha: float     = config.get("alpha", 0.05)
    date_col: str | None = config.get("date_column")

    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if not feature_cols:
        return CheckResult(
            check_name="covariate_drift",
            passed=True,
            severity="info",
            message="No numeric features available for drift detection.",
        )

    n_tests = len(feature_cols)
    corrected_alpha = alpha / max(n_tests, 1)   # Bonferroni

    df1, df2 = _split_dataframe(df, date_col)
    split_label = f"by '{date_col}'" if date_col else "by index (first/second half)"

    drifted: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in feature_cols:
            s1 = df1[col].dropna().values
            s2 = df2[col].dropna().values
            if len(s1) < 10 or len(s2) < 10:
                continue
            _, p_val = ks_2samp(s1, s2)
            psi_val  = _psi(s1, s2)
            if p_val < corrected_alpha or psi_val > _PSI_MODERATE:
                drifted[col] = {
                    "ks_p_value":  round(float(p_val), 6),
                    "psi":         round(psi_val, 4),
                    "severity":    "high" if psi_val > _PSI_SEVERE else "moderate",
                }

    if not drifted:
        return CheckResult(
            check_name="covariate_drift",
            passed=True,
            severity="info",
            message=f"No significant feature drift detected (split {split_label}).",
            details={"split_method": split_label, "features_checked": n_tests,
                     "corrected_alpha": round(corrected_alpha, 6)},
        )

    high_drift = sum(1 for v in drifted.values() if v["severity"] == "high")
    severity   = "error" if high_drift > 0 else "warning"
    return CheckResult(
        check_name="covariate_drift",
        passed=False,
        severity=severity,
        message=(
            f"{len(drifted)} feature(s) show distributional shift between dataset halves "
            f"(split {split_label}). {high_drift} with high PSI (> {_PSI_SEVERE})."
        ),
        details={
            "split_method": split_label,
            "corrected_alpha": round(corrected_alpha, 6),
            "drifted_features": drifted,
        },
        affected_columns=list(drifted.keys()),
    )


def check_label_drift(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> CheckResult:
    """Detect shift in the target label distribution between dataset halves.

    Uses a chi-square test of independence on the class frequencies.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        config: The ``drift_checks.label_drift`` config block.
    """
    if target_col not in df.columns:
        return CheckResult(
            check_name="label_drift",
            passed=True,
            severity="info",
            message=f"Target column '{target_col}' not found — check skipped.",
        )

    alpha:    float       = config.get("alpha", 0.05)
    date_col: str | None  = config.get("date_column")

    df1, df2 = _split_dataframe(df, date_col)
    split_label = f"by '{date_col}'" if date_col else "by index"

    dist1 = df1[target_col].value_counts()
    dist2 = df2[target_col].value_counts()
    all_classes = sorted(set(dist1.index) | set(dist2.index))

    observed = np.array([
        [dist1.get(c, 0) for c in all_classes],
        [dist2.get(c, 0) for c in all_classes],
    ])

    if observed.min() == 0:
        # Some class appears only in one half — definite drift
        return CheckResult(
            check_name="label_drift",
            passed=False,
            severity="error",
            message=f"Target class distribution changed between halves (split {split_label}): "
                    "some classes appear in only one half.",
            details={"split_method": split_label,
                     "first_half": {str(k): int(v) for k, v in dist1.items()},
                     "second_half": {str(k): int(v) for k, v in dist2.items()}},
            affected_columns=[target_col],
        )

    try:
        chi2, p_val, _, _ = chi2_contingency(observed, correction=False)
    except Exception:
        p_val = 1.0
        chi2  = 0.0

    if p_val >= alpha:
        return CheckResult(
            check_name="label_drift",
            passed=True,
            severity="info",
            message=f"Target distribution is stable between halves (χ²={chi2:.2f}, p={p_val:.4f}).",
            details={"split_method": split_label, "chi2": round(chi2, 4),
                     "p_value": round(p_val, 6)},
        )

    return CheckResult(
        check_name="label_drift",
        passed=False,
        severity="warning",
        message=(
            f"Target distribution shifted significantly between halves "
            f"(χ²={chi2:.2f}, p={p_val:.4f}, split {split_label})."
        ),
        details={"split_method": split_label, "chi2": round(chi2, 4),
                 "p_value": round(p_val, 6),
                 "first_half":  {str(k): int(v) for k, v in dist1.items()},
                 "second_half": {str(k): int(v) for k, v in dist2.items()}},
        affected_columns=[target_col],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_drift_checks(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
) -> list[CheckResult]:
    """Run all enabled drift-detection checks.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        config: The full ``drift_checks`` config block.

    Returns:
        List of :class:`~src.utils.CheckResult`, one per executed check.
    """
    if not config.get("enabled", True):
        logger.info("Drift checks disabled — skipping.")
        return []

    results: list[CheckResult] = []

    def _run(key: str, fn):
        cfg = config.get(key, {})
        if not cfg.get("enabled", True):
            logger.debug(f"Drift check '{key}' disabled — skipping.")
            return
        logger.debug(f"Running drift check: {key}")
        r = fn(cfg)
        results.append(r)
        (logger.warning if not r.passed else logger.debug)(f"[{key}] {r.message}")

    _run("covariate_drift", lambda cfg: check_covariate_drift(df, target_col, cfg))
    _run("label_drift",     lambda cfg: check_label_drift(df, target_col, cfg))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Drift checks complete: {passed}/{len(results)} passed.")
    return results
