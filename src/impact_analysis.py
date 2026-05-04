"""Impact analysis — measures performance delta caused by quality/leakage issues (Phase 5)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import CheckResult, FrameworkReport, logger

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

_SIGNIFICANCE_THRESHOLD = 0.05  # drop of > 5 pp is considered significant

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_problem_columns(report: FrameworkReport) -> set[str]:
    """Return columns flagged as problematic by target_leakage, id_column_leakage,
    or constant_features checks."""
    target_checks = {"target_leakage", "id_column_leakage", "constant_features"}
    cols: set[str] = set()
    for r in report.quality_results + report.leakage_results:
        if not r.passed and r.check_name in target_checks:
            cols.update(r.affected_columns)
    return cols


def _prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: set[str] | None = None,
    remove_dupes: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for sklearn.

    - Drops problem columns.
    - Encodes categorical features as integer codes.
    - Encodes target if non-numeric.
    - Optionally removes duplicate rows.
    """
    data = df.copy()
    if remove_dupes:
        data = data.drop_duplicates()
    if drop_cols:
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    X = data.drop(columns=[target_col])
    y = data[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0].astype(float)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=target_col)

    return X.astype(float), y


def _build_pipeline(model: Any) -> Pipeline:
    """SimpleImputer → StandardScaler → model."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _resolve_scorer(metric: str, n_classes: int) -> str:
    """Map config metric name to a valid sklearn scorer string."""
    if n_classes > 2:
        if metric == "roc_auc":
            return "roc_auc_ovr_weighted"
        if metric == "f1":
            return "f1_weighted"
    return metric


def _cv_score(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    metric: str,
) -> tuple[float, float]:
    """Return (mean, std) cross-validated score. Returns (nan, nan) on failure."""
    scorer = _resolve_scorer(metric, int(y.nunique()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scorer, n_jobs=1)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as exc:
        logger.warning(f"CV scoring failed ({metric}): {exc}")
        return float("nan"), float("nan")


def _get_model(name: str, random_state: int) -> Any:
    """Instantiate an sklearn-compatible model by name."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    if name == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=100, random_state=random_state,
            eval_metric="logloss", verbosity=0,
        )
    raise ValueError(f"Unknown model '{name}'. Valid: logistic_regression, random_forest, xgboost.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_impact_analysis(
    df: pd.DataFrame,
    target_col: str,
    report: FrameworkReport,
    config: dict,
) -> list[CheckResult]:
    """Measure the performance delta caused by quality and leakage issues.

    For each configured model:

    1. **Baseline** — train on the full dataset.
    2. **Cleaned** — drop columns flagged as problematic and remove duplicates.
    3. Compare CV scores and report the delta.

    A negative delta (cleaned < baseline) reveals that the model was relying on
    leaked or low-quality features to inflate its apparent performance.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        report: Partially populated FrameworkReport (quality + leakage results).
        config: The full ``impact_analysis`` config block.

    Returns:
        One :class:`~src.utils.CheckResult` per model evaluated.
    """
    if not config.get("enabled", True):
        logger.info("Impact analysis disabled — skipping.")
        return []

    if target_col not in df.columns:
        logger.warning(f"Impact analysis: target column '{target_col}' not in DataFrame — skipping.")
        return []

    models: list[str] = config.get("models", ["logistic_regression"])
    cv_folds: int = config.get("cv_folds", 5)
    random_state: int = config.get("random_state", 42)
    metrics: list[str] = config.get("metrics", ["accuracy"])
    primary_metric: str = metrics[0]

    problem_cols = _extract_problem_columns(report)
    has_dupes = any(
        r.check_name == "duplicates" and not r.passed
        for r in report.quality_results
    )

    logger.info(
        f"Impact analysis: {len(models)} model(s), metric='{primary_metric}', "
        f"columns_to_drop={sorted(problem_cols) or 'none'}, remove_dupes={has_dupes}"
    )

    try:
        X_base, y_base = _prepare_xy(df, target_col)
        X_clean, y_clean = _prepare_xy(
            df, target_col, drop_cols=problem_cols, remove_dupes=has_dupes
        )
    except Exception as exc:
        logger.error(f"Impact analysis: dataset preparation failed: {exc}")
        return []

    if len(X_base) < cv_folds * 2:
        logger.warning("Impact analysis: not enough rows for cross-validation — skipping.")
        return []

    results: list[CheckResult] = []

    for model_name in models:
        logger.debug(f"Impact analysis: evaluating {model_name}")
        try:
            pipe_base = _build_pipeline(_get_model(model_name, random_state))
            pipe_clean = _build_pipeline(_get_model(model_name, random_state))
        except ImportError as exc:
            logger.warning(f"Skipping {model_name}: {exc}")
            continue

        base_mean, base_std = _cv_score(pipe_base, X_base, y_base, cv_folds, primary_metric)
        clean_mean, clean_std = _cv_score(pipe_clean, X_clean, y_clean, cv_folds, primary_metric)

        if np.isnan(base_mean) or np.isnan(clean_mean):
            logger.warning(f"Impact analysis: scoring failed for {model_name} — skipping.")
            continue

        delta = clean_mean - base_mean
        significant_drop = delta < -_SIGNIFICANCE_THRESHOLD

        results.append(CheckResult(
            check_name=f"impact_{model_name}",
            passed=not significant_drop,
            severity="warning" if significant_drop else "info",
            message=(
                f"{model_name}: baseline={base_mean:.3f}±{base_std:.3f}, "
                f"cleaned={clean_mean:.3f}±{clean_std:.3f}, Δ={delta:+.3f}"
            ),
            details={
                "model": model_name,
                "metric": primary_metric,
                "baseline_score": round(base_mean, 4),
                "baseline_std": round(base_std, 4),
                "cleaned_score": round(clean_mean, 4),
                "cleaned_std": round(clean_std, 4),
                "delta": round(delta, 4),
                "dropped_columns": sorted(problem_cols),
                "n_baseline_rows": len(X_base),
                "n_cleaned_rows": len(X_clean),
            },
        ))
        level = "warning" if significant_drop else "debug"
        getattr(logger, level)(f"[impact_{model_name}] {results[-1].message}")

    passed_count = sum(1 for r in results if r.passed)
    logger.info(f"Impact analysis complete: {passed_count}/{len(results)} models unaffected.")
    return results
