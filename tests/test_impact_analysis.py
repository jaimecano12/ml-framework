"""Tests for src/impact_analysis.py (Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.impact_analysis import (
    _build_pipeline,
    _cv_score,
    _extract_problem_columns,
    _get_model,
    _prepare_xy,
    _resolve_scorer,
    run_impact_analysis,
)
from src.utils import CheckResult, FrameworkReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_clean() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 150
    feature = rng.normal(0, 1, n)
    target = (feature > 0).astype(int)
    return pd.DataFrame({
        "feature": feature,
        "noise": rng.normal(0, 1, n),
        "target": target,
    })


@pytest.fixture()
def df_with_proxy() -> pd.DataFrame:
    """Dataset with a perfect proxy feature (target leakage)."""
    rng = np.random.default_rng(0)
    n = 120
    target = rng.integers(0, 2, n)
    return pd.DataFrame({
        "proxy": target.astype(float),
        "noise1": rng.normal(0, 1, n),
        "noise2": rng.normal(0, 1, n),
        "target": target,
    })


@pytest.fixture()
def report_clean(df_clean: pd.DataFrame) -> FrameworkReport:
    return FrameworkReport(dataset_name="test", metadata={"target_col": "target"})


@pytest.fixture()
def report_with_leakage() -> FrameworkReport:
    report = FrameworkReport(dataset_name="test", metadata={"target_col": "target"})
    report.leakage_results = [
        CheckResult(
            check_name="target_leakage",
            passed=False,
            severity="error",
            message="proxy leaks target",
            affected_columns=["proxy"],
        )
    ]
    return report


@pytest.fixture()
def report_with_constant() -> FrameworkReport:
    report = FrameworkReport(dataset_name="test", metadata={"target_col": "target"})
    report.quality_results = [
        CheckResult(
            check_name="constant_features",
            passed=False,
            severity="warning",
            affected_columns=["const_col"],
        )
    ]
    return report


def _fast_config(models=None) -> dict:
    return {
        "enabled": True,
        "models": models or ["logistic_regression"],
        "cv_folds": 3,
        "random_state": 42,
        "metrics": ["accuracy"],
    }


# ---------------------------------------------------------------------------
# _extract_problem_columns
# ---------------------------------------------------------------------------

class TestExtractProblemColumns:
    def test_empty_report_returns_empty_set(self):
        report = FrameworkReport(dataset_name="t")
        assert _extract_problem_columns(report) == set()

    def test_extracts_target_leakage_columns(self, report_with_leakage):
        cols = _extract_problem_columns(report_with_leakage)
        assert "proxy" in cols

    def test_extracts_constant_feature_columns(self, report_with_constant):
        cols = _extract_problem_columns(report_with_constant)
        assert "const_col" in cols

    def test_does_not_extract_passed_checks(self):
        report = FrameworkReport(dataset_name="t")
        report.leakage_results = [
            CheckResult("target_leakage", passed=True, affected_columns=["safe_col"])
        ]
        assert "safe_col" not in _extract_problem_columns(report)

    def test_does_not_extract_outliers_columns(self):
        report = FrameworkReport(dataset_name="t")
        report.quality_results = [
            CheckResult("outliers", passed=False, affected_columns=["col_a"])
        ]
        assert "col_a" not in _extract_problem_columns(report)


# ---------------------------------------------------------------------------
# _prepare_xy
# ---------------------------------------------------------------------------

class TestPrepareXy:
    def test_returns_dataframe_and_series(self, df_clean):
        X, y = _prepare_xy(df_clean, "target")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, df_clean):
        X, _ = _prepare_xy(df_clean, "target")
        assert "target" not in X.columns

    def test_drop_cols_removed(self, df_with_proxy):
        X, _ = _prepare_xy(df_with_proxy, "target", drop_cols={"proxy"})
        assert "proxy" not in X.columns

    def test_duplicates_removed_when_requested(self):
        base = pd.DataFrame({"a": [1, 2], "t": [0, 1]})
        df = pd.concat([base, base], ignore_index=True)
        X, y = _prepare_xy(df, "t", remove_dupes=True)
        assert len(X) == 2

    def test_categorical_feature_encoded(self):
        df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "target": [0, 1, 0, 1]})
        X, _ = _prepare_xy(df, "target")
        assert pd.api.types.is_float_dtype(X["cat"])

    def test_categorical_target_encoded(self):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "target": ["yes", "no", "yes", "no"]})
        _, y = _prepare_xy(df, "target")
        assert pd.api.types.is_numeric_dtype(y)


# ---------------------------------------------------------------------------
# _resolve_scorer
# ---------------------------------------------------------------------------

class TestResolveScorer:
    def test_accuracy_unchanged(self):
        assert _resolve_scorer("accuracy", 2) == "accuracy"

    def test_roc_auc_binary_unchanged(self):
        assert _resolve_scorer("roc_auc", 2) == "roc_auc"

    def test_roc_auc_multiclass_ovr(self):
        assert _resolve_scorer("roc_auc", 3) == "roc_auc_ovr_weighted"

    def test_f1_binary_unchanged(self):
        assert _resolve_scorer("f1", 2) == "f1"

    def test_f1_multiclass_weighted(self):
        assert _resolve_scorer("f1", 5) == "f1_weighted"


# ---------------------------------------------------------------------------
# _get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_logistic_regression(self):
        model = _get_model("logistic_regression", 42)
        assert isinstance(model, type(model))
        assert hasattr(model, "fit")

    def test_random_forest(self):
        model = _get_model("random_forest", 42)
        assert hasattr(model, "fit")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            _get_model("svm", 42)


# ---------------------------------------------------------------------------
# _cv_score
# ---------------------------------------------------------------------------

class TestCvScore:
    def test_returns_mean_and_std(self, df_clean):
        X, y = _prepare_xy(df_clean, "target")
        model = _get_model("logistic_regression", 42)
        pipe = _build_pipeline(model)
        mean, std = _cv_score(pipe, X, y, cv_folds=3, metric="accuracy")
        assert 0.0 <= mean <= 1.0
        assert std >= 0.0

    def test_returns_nan_on_invalid_metric(self, df_clean):
        X, y = _prepare_xy(df_clean, "target")
        pipe = _build_pipeline(_get_model("logistic_regression", 42))
        mean, std = _cv_score(pipe, X, y, cv_folds=3, metric="nonexistent_metric")
        assert np.isnan(mean)


# ---------------------------------------------------------------------------
# run_impact_analysis
# ---------------------------------------------------------------------------

class TestRunImpactAnalysis:
    def test_disabled_returns_empty(self, df_clean, report_clean):
        results = run_impact_analysis(df_clean, "target", report_clean, {"enabled": False})
        assert results == []

    def test_missing_target_col_returns_empty(self, df_clean, report_clean):
        results = run_impact_analysis(df_clean, "ghost", report_clean, _fast_config())
        assert results == []

    def test_returns_one_result_per_model(self, df_clean, report_clean):
        results = run_impact_analysis(df_clean, "target", report_clean, _fast_config())
        assert len(results) == 1
        assert results[0].check_name == "impact_logistic_regression"

    def test_details_contain_required_keys(self, df_clean, report_clean):
        results = run_impact_analysis(df_clean, "target", report_clean, _fast_config())
        d = results[0].details
        for key in ("model", "metric", "baseline_score", "cleaned_score", "delta",
                    "dropped_columns", "n_baseline_rows", "n_cleaned_rows"):
            assert key in d, f"Key '{key}' missing from details"

    def test_clean_report_both_scores_similar(self, df_clean, report_clean):
        """No problematic columns → cleaned = baseline, delta ≈ 0."""
        results = run_impact_analysis(df_clean, "target", report_clean, _fast_config())
        assert len(results) == 1
        assert abs(results[0].details["delta"]) < 0.1

    def test_leakage_causes_negative_delta(self, df_with_proxy, report_with_leakage):
        """Removing a perfect proxy drops performance significantly."""
        results = run_impact_analysis(
            df_with_proxy, "target", report_with_leakage, _fast_config()
        )
        assert len(results) == 1
        assert results[0].details["delta"] < 0

    def test_significant_drop_fails_check(self, df_with_proxy, report_with_leakage):
        results = run_impact_analysis(
            df_with_proxy, "target", report_with_leakage, _fast_config()
        )
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].severity == "warning"

    def test_dropped_columns_listed_in_details(self, df_with_proxy, report_with_leakage):
        results = run_impact_analysis(
            df_with_proxy, "target", report_with_leakage, _fast_config()
        )
        assert "proxy" in results[0].details["dropped_columns"]

    def test_multiple_models_return_multiple_results(self, df_clean, report_clean):
        cfg = _fast_config(models=["logistic_regression", "random_forest"])
        results = run_impact_analysis(df_clean, "target", report_clean, cfg)
        assert len(results) == 2
        names = {r.check_name for r in results}
        assert "impact_logistic_regression" in names
        assert "impact_random_forest" in names

    def test_too_few_rows_returns_empty(self, report_clean):
        df_tiny = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
        results = run_impact_analysis(df_tiny, "target", report_clean, _fast_config())
        assert results == []
