"""YAML configuration loader with validation and typed access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import logger

# ---------------------------------------------------------------------------
# Default configuration (single source of truth for optional fields)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "dataset": {
        "separator": ",",
        "encoding": "utf-8",
    },
    "logging": {
        "level": "INFO",
        "file": None,
    },
    "quality_checks": {
        "enabled": True,
        "missing_values": {"enabled": True, "threshold": 0.05},
        "duplicates": {"enabled": True},
        "outliers": {"enabled": True, "method": "iqr", "threshold": 3.0},
        "class_imbalance": {"enabled": True, "threshold": 0.1},
        "constant_features": {"enabled": True},
        "low_variance": {"enabled": True, "threshold": 0.01},
    },
    "leakage_checks": {
        "enabled": True,
        "target_leakage": {"enabled": True, "correlation_threshold": 0.95},
        "train_test_overlap": {"enabled": True, "test_size": 0.2, "random_state": 42},
        "temporal_leakage": {"enabled": True, "date_column": None},
        "id_column_leakage": {"enabled": True, "cardinality_threshold": 0.95},
    },
    "impact_analysis": {
        "enabled": True,
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "cv_folds": 5,
        "test_size": 0.2,
        "random_state": 42,
        "metrics": ["accuracy", "roc_auc", "f1"],
    },
    "feature_analysis": {
        "enabled": True,
        "feature_correlation": {"enabled": True, "correlation_threshold": 0.90},
        "feature_relevance":   {"enabled": True, "mi_threshold": 0.01, "random_state": 42},
        "distribution_shape":  {"enabled": True, "skewness_threshold": 2.0, "kurtosis_threshold": 7.0},
    },
    "sufficiency_checks": {
        "enabled": True,
        "sample_size":             {"enabled": True, "min_rows": 100, "comfortable_ratio": 50.0},
        "class_support":           {"enabled": True, "min_samples_per_class": 30},
        "cv_stability":            {"enabled": True, "max_cv_std": 0.10},
        "feature_to_sample_ratio": {"enabled": True, "max_ratio": 0.10},
    },
    "drift_checks": {
        "enabled": True,
        "covariate_drift": {"enabled": True, "alpha": 0.05, "date_column": None},
        "label_drift":     {"enabled": True, "alpha": 0.05, "date_column": None},
    },
    "plugins": [],
    "reporting": {
        "output_dir": "reports/",
        "format": "html",
        "include_plots": True,
    },
}

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_VALID_OUTLIER_METHODS = {"iqr", "zscore"}
_VALID_REPORT_FORMATS = {"html"}
_VALID_MODELS = {"logistic_regression", "random_forest", "xgboost"}
_VALID_METRICS = {"accuracy", "roc_auc", "f1", "precision", "recall"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Missing optional keys are filled from :data:`_DEFAULTS`.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully-populated, validated configuration dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required keys are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    config = _deep_merge(_DEFAULTS, raw)
    _validate(config)

    logger.info(f"Config loaded from '{path}'")
    return config


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a top-level section, raising KeyError if absent.

    Args:
        config: Full configuration dictionary returned by :func:`load_config`.
        section: Top-level key (e.g. ``'quality_checks'``).

    Returns:
        The section sub-dictionary.
    """
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config.")
    return config[section]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate(config: dict[str, Any]) -> None:
    """Raise ValueError if the config contains invalid values."""
    errors: list[str] = []

    # dataset
    ds = config.get("dataset", {})
    if not ds.get("path"):
        errors.append("dataset.path is required.")
    if not ds.get("target_column"):
        errors.append("dataset.target_column is required.")

    # logging
    log_level = config.get("logging", {}).get("level", "INFO")
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"logging.level must be one of {_VALID_LOG_LEVELS}, got '{log_level}'.")

    # quality_checks
    qc = config.get("quality_checks", {})
    outlier_method = qc.get("outliers", {}).get("method", "iqr")
    if outlier_method not in _VALID_OUTLIER_METHODS:
        errors.append(
            f"quality_checks.outliers.method must be one of {_VALID_OUTLIER_METHODS}, got '{outlier_method}'."
        )
    for threshold_key in ("missing_values", "class_imbalance", "low_variance"):
        t = qc.get(threshold_key, {}).get("threshold")
        if t is not None and not (0.0 <= t <= 1.0):
            errors.append(f"quality_checks.{threshold_key}.threshold must be in [0, 1], got {t}.")

    # leakage_checks
    lc = config.get("leakage_checks", {})
    corr_t = lc.get("target_leakage", {}).get("correlation_threshold")
    if corr_t is not None and not (0.0 <= corr_t <= 1.0):
        errors.append(
            f"leakage_checks.target_leakage.correlation_threshold must be in [0, 1], got {corr_t}."
        )

    # impact_analysis
    ia = config.get("impact_analysis", {})
    for model in ia.get("models", []):
        if model not in _VALID_MODELS:
            errors.append(f"impact_analysis.models: unknown model '{model}'. Valid: {_VALID_MODELS}.")
    for metric in ia.get("metrics", []):
        if metric not in _VALID_METRICS:
            errors.append(f"impact_analysis.metrics: unknown metric '{metric}'. Valid: {_VALID_METRICS}.")
    test_size = ia.get("test_size", 0.2)
    if not (0.0 < test_size < 1.0):
        errors.append(f"impact_analysis.test_size must be in (0, 1), got {test_size}.")
    cv_folds = ia.get("cv_folds", 5)
    if not isinstance(cv_folds, int) or cv_folds < 2:
        errors.append(f"impact_analysis.cv_folds must be an integer >= 2, got {cv_folds}.")

    # feature_analysis
    fa = config.get("feature_analysis", {})
    fa_corr_t = fa.get("feature_correlation", {}).get("correlation_threshold")
    if fa_corr_t is not None and not (0.0 <= fa_corr_t <= 1.0):
        errors.append(
            f"feature_analysis.feature_correlation.correlation_threshold must be in [0, 1], got {fa_corr_t}."
        )
    fa_mi_t = fa.get("feature_relevance", {}).get("mi_threshold")
    if fa_mi_t is not None and not (0.0 <= fa_mi_t <= 1.0):
        errors.append(
            f"feature_analysis.feature_relevance.mi_threshold must be in [0, 1], got {fa_mi_t}."
        )

    # reporting
    fmt = config.get("reporting", {}).get("format", "html")
    if fmt not in _VALID_REPORT_FORMATS:
        errors.append(f"reporting.format must be one of {_VALID_REPORT_FORMATS}, got '{fmt}'.")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
