"""Tests for src/quality_checks.py (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.quality_checks import (
    check_class_imbalance,
    check_constant_features,
    check_duplicates,
    check_low_variance,
    check_missing_values,
    check_outliers,
    run_all_quality_checks,
)
from src.utils import CheckResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_clean() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num_a": rng.normal(0, 1, 100),
        "num_b": rng.normal(5, 2, 100),
        "cat":   ["A"] * 50 + ["B"] * 50,
        "target": [0] * 50 + [1] * 50,
    })


@pytest.fixture()
def df_missing() -> pd.DataFrame:
    values = list(range(100))
    col_high = [None if i % 2 == 0 else i for i in values]  # 50 % missing
    col_low  = [None if i == 0 else i for i in values]       #  1 % missing
    return pd.DataFrame({
        "high_missing": col_high,
        "low_missing":  col_low,
        "complete":     values,
        "target":       [0] * 50 + [1] * 50,
    })


@pytest.fixture()
def df_dupes() -> pd.DataFrame:
    base = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    return pd.concat([base, base.iloc[:3]], ignore_index=True)


@pytest.fixture()
def df_outliers_iqr() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normal_vals = rng.normal(loc=10, scale=2, size=95)   # spread so IQR > 0
    outlier_vals = np.array([200.0, 300.0, 400.0, 500.0, 600.0])
    values = np.concatenate([normal_vals, outlier_vals])
    return pd.DataFrame({"x": values, "y": values, "target": [0, 1] * 50})


@pytest.fixture()
def df_imbalanced() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": range(100),
        "target": [1] * 3 + [0] * 97,
    })


@pytest.fixture()
def df_constant() -> pd.DataFrame:
    return pd.DataFrame({
        "constant_col": [42] * 50,
        "normal_col":   list(range(50)),
        "target":       [0, 1] * 25,
    })


@pytest.fixture()
def df_low_var() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # CV = std/mean ≈ 1e-4/1.0 = 1e-4  →  well below threshold=0.01
    near_const = 1.0 + rng.uniform(-0.0001, 0.0001, 50)
    # CV ≈ 1.0/|mean| — mean will be non-zero, CV >> 0.01
    normal_var = rng.normal(loc=10, scale=3, size=50)
    return pd.DataFrame({
        "near_constant": near_const,
        "normal_var":    normal_var,
        "target":        [0, 1] * 25,
    })


def _default_qc_config() -> dict:
    return {
        "enabled": True,
        "missing_values":    {"enabled": True, "threshold": 0.05},
        "duplicates":        {"enabled": True},
        "outliers":          {"enabled": True, "method": "iqr", "threshold": 3.0},
        "class_imbalance":   {"enabled": True, "threshold": 0.1},
        "constant_features": {"enabled": True},
        "low_variance":      {"enabled": True, "threshold": 0.01},
    }


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

class TestCheckMissingValues:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_missing_values(df_clean, {"threshold": 0.05})
        assert r.passed
        assert r.severity == "info"
        assert r.check_name == "missing_values"

    def test_fails_when_column_exceeds_threshold(self, df_missing: pd.DataFrame):
        r = check_missing_values(df_missing, {"threshold": 0.05})
        assert not r.passed
        assert "high_missing" in r.affected_columns

    def test_low_missing_column_not_flagged(self, df_missing: pd.DataFrame):
        r = check_missing_values(df_missing, {"threshold": 0.05})
        assert "low_missing" not in r.affected_columns

    def test_severity_error_when_majority_missing(self):
        df = pd.DataFrame({"a": [None] * 80 + [1] * 20})
        r = check_missing_values(df, {"threshold": 0.05})
        assert r.severity == "error"

    def test_severity_warning_for_moderate_missing(self, df_missing: pd.DataFrame):
        # 50 % missing — but not > 50 % so warning
        r = check_missing_values(df_missing, {"threshold": 0.05})
        assert r.severity in {"warning", "error"}

    def test_details_contain_flagged_columns(self, df_missing: pd.DataFrame):
        r = check_missing_values(df_missing, {"threshold": 0.05})
        assert "flagged_columns" in r.details
        assert "high_missing" in r.details["flagged_columns"]

    def test_threshold_respected(self, df_missing: pd.DataFrame):
        # With threshold=0.6, even high_missing (50 %) should pass
        r = check_missing_values(df_missing, {"threshold": 0.6})
        assert r.passed

    def test_all_nan_column_flagged(self):
        df = pd.DataFrame({"empty": [None] * 10, "ok": range(10)})
        r = check_missing_values(df, {"threshold": 0.05})
        assert not r.passed
        assert "empty" in r.affected_columns


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicates:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_duplicates(df_clean, {})
        assert r.passed
        assert r.details["duplicate_count"] == 0

    def test_fails_when_duplicates_present(self, df_dupes: pd.DataFrame):
        r = check_duplicates(df_dupes, {})
        assert not r.passed
        assert r.details["duplicate_count"] == 3

    def test_severity_error_for_high_dup_rate(self):
        base = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df = pd.concat([base] * 10, ignore_index=True)
        r = check_duplicates(df, {})
        assert r.severity == "error"

    def test_severity_warning_for_low_dup_rate(self):
        # 1 duplicate in 20 rows = 5 % < 10 % threshold → warning
        base = pd.DataFrame({"a": list(range(20)), "b": list(range(20))})
        df = pd.concat([base, base.iloc[:1]], ignore_index=True)
        r = check_duplicates(df, {})
        assert r.severity == "warning"

    def test_details_contain_rate(self, df_dupes: pd.DataFrame):
        r = check_duplicates(df_dupes, {})
        assert "duplicate_rate" in r.details
        assert 0 < r.details["duplicate_rate"] < 1


# ---------------------------------------------------------------------------
# check_outliers
# ---------------------------------------------------------------------------

class TestCheckOutliers:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_outliers(df_clean, {"method": "iqr", "threshold": 3.0})
        assert r.passed

    def test_iqr_detects_extreme_values(self, df_outliers_iqr: pd.DataFrame):
        r = check_outliers(df_outliers_iqr, {"method": "iqr", "threshold": 3.0})
        assert not r.passed
        assert "x" in r.affected_columns

    def test_zscore_detects_extreme_values(self, df_outliers_iqr: pd.DataFrame):
        r = check_outliers(df_outliers_iqr, {"method": "zscore", "threshold": 3.0})
        assert not r.passed

    def test_non_numeric_columns_skipped(self):
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "x"], "num": [1, 2, 3, 4, 1000]})
        r = check_outliers(df, {"method": "iqr", "threshold": 3.0})
        assert not r.passed
        assert "text" not in r.affected_columns

    def test_details_contain_flagged_columns(self, df_outliers_iqr: pd.DataFrame):
        r = check_outliers(df_outliers_iqr, {"method": "iqr", "threshold": 3.0})
        assert "flagged_columns" in r.details
        assert r.details["flagged_columns"]["x"] > 0

    def test_columns_with_fewer_than_4_values_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 100]})  # 3 values — skip
        r = check_outliers(df, {"method": "iqr", "threshold": 1.5})
        assert r.passed

    def test_constant_column_not_flagged(self):
        df = pd.DataFrame({"a": [5] * 20, "b": list(range(20))})
        r = check_outliers(df, {"method": "iqr", "threshold": 3.0})
        assert "a" not in r.affected_columns


# ---------------------------------------------------------------------------
# check_class_imbalance
# ---------------------------------------------------------------------------

class TestCheckClassImbalance:
    def test_passes_on_balanced_target(self, df_clean: pd.DataFrame):
        r = check_class_imbalance(df_clean, "target", {"threshold": 0.1})
        assert r.passed

    def test_fails_on_imbalanced_target(self, df_imbalanced: pd.DataFrame):
        r = check_class_imbalance(df_imbalanced, "target", {"threshold": 0.1})
        assert not r.passed
        assert "target" in r.affected_columns

    def test_severity_error_for_extreme_imbalance(self, df_imbalanced: pd.DataFrame):
        r = check_class_imbalance(df_imbalanced, "target", {"threshold": 0.1})
        assert r.severity == "error"

    def test_details_contain_distribution(self, df_imbalanced: pd.DataFrame):
        r = check_class_imbalance(df_imbalanced, "target", {"threshold": 0.1})
        assert "class_distribution" in r.details
        assert "minority_ratio" in r.details

    def test_missing_target_column_raises(self, df_clean: pd.DataFrame):
        with pytest.raises(ValueError, match="ghost"):
            check_class_imbalance(df_clean, "ghost", {"threshold": 0.1})

    def test_multiclass_detects_minority(self):
        df = pd.DataFrame({"target": ["A"] * 80 + ["B"] * 15 + ["C"] * 5})
        r = check_class_imbalance(df, "target", {"threshold": 0.1})
        assert not r.passed
        assert r.details["minority_class"] == "C"

    def test_threshold_respected(self, df_imbalanced: pd.DataFrame):
        # minority is ~3 %, threshold=0.01 → should pass
        r = check_class_imbalance(df_imbalanced, "target", {"threshold": 0.01})
        assert r.passed


# ---------------------------------------------------------------------------
# check_constant_features
# ---------------------------------------------------------------------------

class TestCheckConstantFeatures:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_constant_features(df_clean, {})
        assert r.passed

    def test_detects_constant_column(self, df_constant: pd.DataFrame):
        r = check_constant_features(df_constant, {})
        assert not r.passed
        assert "constant_col" in r.affected_columns

    def test_normal_column_not_flagged(self, df_constant: pd.DataFrame):
        r = check_constant_features(df_constant, {})
        assert "normal_col" not in r.affected_columns

    def test_all_nan_column_flagged(self):
        df = pd.DataFrame({"empty": [None, None, None], "ok": [1, 2, 3]})
        r = check_constant_features(df, {})
        assert not r.passed
        assert "empty" in r.affected_columns

    def test_details_list_columns(self, df_constant: pd.DataFrame):
        r = check_constant_features(df_constant, {})
        assert "constant_col" in r.details["constant_columns"]


# ---------------------------------------------------------------------------
# check_low_variance
# ---------------------------------------------------------------------------

class TestCheckLowVariance:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_low_variance(df_clean, {"threshold": 0.01})
        assert r.passed

    def test_detects_near_constant_column(self, df_low_var: pd.DataFrame):
        r = check_low_variance(df_low_var, {"threshold": 0.01})
        assert not r.passed
        assert "near_constant" in r.affected_columns

    def test_normal_var_column_not_flagged(self, df_low_var: pd.DataFrame):
        r = check_low_variance(df_low_var, {"threshold": 0.01})
        assert "normal_var" not in r.affected_columns

    def test_constant_column_not_flagged(self, df_constant: pd.DataFrame):
        # Constant column has var == 0.0 — should be skipped (handled by constant_features)
        r = check_low_variance(df_constant, {"threshold": 0.01})
        assert "constant_col" not in r.affected_columns

    def test_non_numeric_columns_skipped(self):
        df = pd.DataFrame({"text": ["a"] * 50, "num": list(range(50))})
        r = check_low_variance(df, {"threshold": 0.01})
        assert r.passed

    def test_details_contain_variance_values(self, df_low_var: pd.DataFrame):
        r = check_low_variance(df_low_var, {"threshold": 0.01})
        assert "near_constant" in r.details["flagged_columns"]


# ---------------------------------------------------------------------------
# run_all_quality_checks (orchestrator)
# ---------------------------------------------------------------------------

class TestRunAllQualityChecks:
    def test_returns_list_of_check_results(self, df_clean: pd.DataFrame):
        results = run_all_quality_checks(df_clean, "target", _default_qc_config())
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)

    def test_all_six_checks_run_by_default(self, df_clean: pd.DataFrame):
        results = run_all_quality_checks(df_clean, "target", _default_qc_config())
        names = {r.check_name for r in results}
        expected = {"missing_values", "duplicates", "outliers",
                    "class_imbalance", "constant_features", "low_variance"}
        assert names == expected

    def test_disabled_top_level_returns_empty(self, df_clean: pd.DataFrame):
        cfg = _default_qc_config()
        cfg["enabled"] = False
        results = run_all_quality_checks(df_clean, "target", cfg)
        assert results == []

    def test_individual_check_can_be_disabled(self, df_clean: pd.DataFrame):
        cfg = _default_qc_config()
        cfg["duplicates"]["enabled"] = False
        results = run_all_quality_checks(df_clean, "target", cfg)
        names = {r.check_name for r in results}
        assert "duplicates" not in names

    def test_detects_issues_in_dirty_data(self, df_missing: pd.DataFrame):
        results = run_all_quality_checks(df_missing, "target", _default_qc_config())
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1
        assert any(r.check_name == "missing_values" for r in failed)

    def test_all_pass_on_clean_data(self, df_clean: pd.DataFrame):
        results = run_all_quality_checks(df_clean, "target", _default_qc_config())
        assert all(r.passed for r in results)
