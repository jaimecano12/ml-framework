"""Statistical sufficiency checks — validates dataset size and stability (Phase 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import CheckResult, FrameworkReport, logger


def check_sample_size(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Check whether the dataset has a sufficient number of rows.

    A minimum of 100 rows is required for any meaningful ML experiment.
    The n/p ratio (rows per feature) indicates how comfortable the margin is.

    Args:
        df: Input DataFrame.
        target_col: Target column (excluded from feature count).
        config: The ``sufficiency_checks.sample_size`` config block.
    """
    min_rows: int = config.get("min_rows", 100)
    comfortable_ratio: float = config.get("comfortable_ratio", 50.0)

    n_rows   = len(df)
    n_feat   = max(len(df.columns) - 1, 1)
    ratio    = n_rows / n_feat

    if n_rows < min_rows:
        return CheckResult(
            check_name="sample_size",
            passed=False,
            severity="error",
            message=(
                f"Dataset has only {n_rows} rows (minimum recommended: {min_rows}). "
                "Most models cannot generalise reliably from this."
            ),
            details={"n_rows": n_rows, "n_features": n_feat, "n_p_ratio": round(ratio, 1),
                     "min_rows": min_rows},
        )

    if ratio < comfortable_ratio:
        return CheckResult(
            check_name="sample_size",
            passed=False,
            severity="warning",
            message=(
                f"n/p ratio is {ratio:.1f} (n={n_rows}, p={n_feat}). "
                f"Recommended ≥ {comfortable_ratio:.0f} for comfortable margin against overfitting."
            ),
            details={"n_rows": n_rows, "n_features": n_feat, "n_p_ratio": round(ratio, 1),
                     "comfortable_ratio": comfortable_ratio},
        )

    return CheckResult(
        check_name="sample_size",
        passed=True,
        severity="info",
        message=f"Dataset size is adequate (n={n_rows}, n/p={ratio:.1f}).",
        details={"n_rows": n_rows, "n_features": n_feat, "n_p_ratio": round(ratio, 1)},
    )


def check_class_support(df: pd.DataFrame, target_col: str, config: dict) -> CheckResult:
    """Verify each class has enough samples for stable cross-validation.

    With k-fold CV, each class needs at least k examples to appear in every fold.
    Fewer than 30 samples per class makes per-class metrics unreliable.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        config: The ``sufficiency_checks.class_support`` config block.
    """
    if target_col not in df.columns:
        return CheckResult(
            check_name="class_support",
            passed=True,
            severity="info",
            message=f"Target column '{target_col}' not found — check skipped.",
        )

    min_samples: int = config.get("min_samples_per_class", 30)
    counts = df[target_col].value_counts()

    if len(counts) > 50:
        return CheckResult(
            check_name="class_support",
            passed=True,
            severity="info",
            message="Target appears to be continuous (> 50 unique values) — class support check skipped.",
            details={"unique_values": int(len(counts))},
        )

    insufficient = {str(k): int(v) for k, v in counts.items() if v < min_samples}

    if not insufficient:
        return CheckResult(
            check_name="class_support",
            passed=True,
            severity="info",
            message=f"All classes have ≥ {min_samples} samples. CV estimates will be stable.",
            details={"min_samples_per_class": min_samples,
                     "class_counts": {str(k): int(v) for k, v in counts.items()}},
        )

    severity = "error" if any(v < 5 for v in insufficient.values()) else "warning"
    return CheckResult(
        check_name="class_support",
        passed=False,
        severity=severity,
        message=(
            f"{len(insufficient)} class(es) have fewer than {min_samples} samples: "
            + ", ".join(f"'{k}'={v}" for k, v in insufficient.items())
        ),
        details={"min_samples_per_class": min_samples, "insufficient_classes": insufficient,
                 "class_counts": {str(k): int(v) for k, v in counts.items()}},
        affected_columns=[target_col],
    )


def check_cv_stability(report: FrameworkReport, config: dict) -> CheckResult:
    """Assess whether cross-validation estimates are stable (low standard deviation).

    High CV std suggests the dataset is too small or too noisy for reliable evaluation.
    Reads std values from impact_results — skips if no impact analysis was run.

    Args:
        report: FrameworkReport with populated impact_results.
        config: The ``sufficiency_checks.cv_stability`` config block.
    """
    max_std: float = config.get("max_cv_std", 0.10)

    if not report.impact_results:
        return CheckResult(
            check_name="cv_stability",
            passed=True,
            severity="info",
            message="No impact analysis results available — CV stability check skipped.",
            details={"max_cv_std": max_std},
        )

    unstable: dict[str, float] = {}
    for r in report.impact_results:
        std = r.details.get("baseline_std", 0.0)
        if std > max_std:
            model = r.details.get("model", r.check_name)
            unstable[model] = round(float(std), 4)

    if not unstable:
        return CheckResult(
            check_name="cv_stability",
            passed=True,
            severity="info",
            message=f"CV estimates are stable (all std ≤ {max_std}).",
            details={"max_cv_std": max_std,
                     "model_stds": {r.details.get("model", r.check_name):
                                    round(float(r.details.get("baseline_std", 0)), 4)
                                    for r in report.impact_results}},
        )

    return CheckResult(
        check_name="cv_stability",
        passed=False,
        severity="warning",
        message=(
            f"{len(unstable)} model(s) show high CV standard deviation (> {max_std}), "
            "indicating unstable performance estimates."
        ),
        details={"max_cv_std": max_std, "unstable_models": unstable},
    )


def check_feature_to_sample_ratio(
    df: pd.DataFrame, target_col: str, config: dict
) -> CheckResult:
    """Flag datasets where the number of features is high relative to sample count.

    A p/n ratio above 0.1 (10 % as many features as rows) creates serious overfitting
    risk, especially for linear models and SVMs.

    Args:
        df: Input DataFrame.
        target_col: Target column (excluded from count).
        config: The ``sufficiency_checks.feature_to_sample_ratio`` config block.
    """
    max_ratio: float = config.get("max_ratio", 0.10)

    n_rows = len(df)
    n_feat = max(len(df.columns) - 1, 1)
    ratio  = n_feat / max(n_rows, 1)

    if ratio <= max_ratio:
        return CheckResult(
            check_name="feature_to_sample_ratio",
            passed=True,
            severity="info",
            message=f"Feature-to-sample ratio is acceptable (p/n={ratio:.3f}, threshold={max_ratio}).",
            details={"p_n_ratio": round(ratio, 4), "n_features": n_feat, "n_rows": n_rows},
        )

    severity = "error" if ratio > 0.5 else "warning"
    return CheckResult(
        check_name="feature_to_sample_ratio",
        passed=False,
        severity=severity,
        message=(
            f"p/n ratio is {ratio:.3f} (p={n_feat}, n={n_rows}), "
            f"exceeding threshold {max_ratio}. High overfitting risk — consider feature selection."
        ),
        details={"p_n_ratio": round(ratio, 4), "n_features": n_feat,
                 "n_rows": n_rows, "max_ratio": max_ratio},
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_sufficiency_checks(
    df: pd.DataFrame,
    target_col: str,
    report: FrameworkReport,
    config: dict,
) -> list[CheckResult]:
    """Run all enabled sufficiency checks.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        report: FrameworkReport — used by cv_stability to read impact_results.
        config: The full ``sufficiency_checks`` config block.

    Returns:
        List of :class:`~src.utils.CheckResult`, one per executed check.
    """
    if not config.get("enabled", True):
        logger.info("Sufficiency checks disabled — skipping.")
        return []

    results: list[CheckResult] = []

    def _run(key: str, fn):
        cfg = config.get(key, {})
        if not cfg.get("enabled", True):
            logger.debug(f"Sufficiency check '{key}' disabled — skipping.")
            return
        logger.debug(f"Running sufficiency check: {key}")
        r = fn(cfg)
        results.append(r)
        (logger.warning if not r.passed else logger.debug)(f"[{key}] {r.message}")

    _run("sample_size",              lambda cfg: check_sample_size(df, target_col, cfg))
    _run("class_support",            lambda cfg: check_class_support(df, target_col, cfg))
    _run("cv_stability",             lambda cfg: check_cv_stability(report, cfg))
    _run("feature_to_sample_ratio",  lambda cfg: check_feature_to_sample_ratio(df, target_col, cfg))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Sufficiency checks complete: {passed}/{len(results)} passed.")
    return results
