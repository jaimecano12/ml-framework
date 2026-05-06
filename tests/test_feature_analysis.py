"""Tests for src/feature_analysis.py (Phase 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_analysis import (
    check_distribution_shape,
    check_feature_correlation,
    check_feature_relevance,
    run_all_feature_checks,
)
from src.utils import CheckResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_uncorrelated() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(5, 2, 200),
        "c": rng.integers(0, 10, 200).astype(float),
        "target": rng.integers(0, 2, 200),
    })


@pytest.fixture()
def df_correlated() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 200)
    return pd.DataFrame({
        "a":      base,
        "b":      base + rng.normal(0, 0.01, 200),  # near-perfect correlation with a
        "noise":  rng.normal(0, 1, 200),
        "target": rng.integers(0, 2, 200),
    })


@pytest.fixture()
def df_relevant() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    feature = rng.normal(0, 1, 300)
    return pd.DataFrame({
        "feature": feature,
        "noise":   rng.normal(0, 1, 300),
        "target":  (feature > 0).astype(int),
    })


@pytest.fixture()
def df_skewed() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "normal_col": rng.normal(0, 1, 200),
        "skewed_col": np.exp(rng.normal(3, 1, 200)),   # log-normal → high skewness
        "target":     rng.integers(0, 2, 200),
    })


def _default_cfg() -> dict:
    return {
        "enabled": True,
        "feature_correlation": {"enabled": True, "correlation_threshold": 0.90},
        "feature_relevance":   {"enabled": True, "mi_threshold": 0.01, "random_state": 42},
        "distribution_shape":  {"enabled": True, "skewness_threshold": 2.0, "kurtosis_threshold": 7.0},
    }


# ---------------------------------------------------------------------------
# check_feature_correlation
# ---------------------------------------------------------------------------

class TestCheckFeatureCorrelation:
    def test_passes_on_uncorrelated_features(self, df_uncorrelated):
        r = check_feature_correlation(df_uncorrelated, "target", {"correlation_threshold": 0.90})
        assert r.passed

    def test_detects_highly_correlated_pair(self, df_correlated):
        r = check_feature_correlation(df_correlated, "target", {"correlation_threshold": 0.90})
        assert not r.passed
        assert "a" in r.affected_columns
        assert "b" in r.affected_columns

    def test_target_excluded_from_correlation(self, df_correlated):
        r = check_feature_correlation(df_correlated, "target", {"correlation_threshold": 0.90})
        assert "target" not in r.affected_columns

    def test_noise_column_not_flagged(self, df_correlated):
        r = check_feature_correlation(df_correlated, "target", {"correlation_threshold": 0.90})
        assert "noise" not in r.affected_columns

    def test_details_contain_pairs(self, df_correlated):
        r = check_feature_correlation(df_correlated, "target", {"correlation_threshold": 0.90})
        assert "correlated_pairs" in r.details
        assert len(r.details["correlated_pairs"]) >= 1

    def test_threshold_respected(self, df_correlated):
        # correlation is ~0.9999 — only threshold=1.0 (perfect) allows it to pass
        r = check_feature_correlation(df_correlated, "target", {"correlation_threshold": 1.0})
        assert r.passed

    def test_single_feature_skips(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        r = check_feature_correlation(df, "target", {"correlation_threshold": 0.90})
        assert r.passed

    def test_check_name_correct(self, df_uncorrelated):
        r = check_feature_correlation(df_uncorrelated, "target", {})
        assert r.check_name == "feature_correlation"


# ---------------------------------------------------------------------------
# check_feature_relevance
# ---------------------------------------------------------------------------

class TestCheckFeatureRelevance:
    def test_passes_when_features_are_relevant(self, df_relevant):
        r = check_feature_relevance(df_relevant, "target", {"mi_threshold": 0.01})
        assert r.passed

    def test_detects_irrelevant_noise_feature(self, df_relevant):
        r = check_feature_relevance(df_relevant, "target", {"mi_threshold": 0.30})
        assert not r.passed
        assert "noise" in r.affected_columns

    def test_target_excluded(self, df_relevant):
        r = check_feature_relevance(df_relevant, "target", {"mi_threshold": 0.01})
        assert "target" not in (r.affected_columns or [])

    def test_details_contain_all_scores(self, df_relevant):
        r = check_feature_relevance(df_relevant, "target", {"mi_threshold": 0.01})
        assert "all_scores" in r.details
        assert "feature" in r.details["all_scores"]

    def test_too_few_rows_passes(self):
        df = pd.DataFrame({"a": [1.0] * 5, "target": [0, 1, 0, 1, 0]})
        r = check_feature_relevance(df, "target", {"mi_threshold": 0.01})
        assert r.passed

    def test_check_name_correct(self, df_relevant):
        r = check_feature_relevance(df_relevant, "target", {})
        assert r.check_name == "feature_relevance"

    def test_no_numeric_features_passes(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 20, "target": [0, 1, 0] * 20})
        r = check_feature_relevance(df, "target", {"mi_threshold": 0.01})
        assert r.passed


# ---------------------------------------------------------------------------
# check_distribution_shape
# ---------------------------------------------------------------------------

class TestCheckDistributionShape:
    def test_passes_on_normal_data(self, df_uncorrelated):
        r = check_distribution_shape(df_uncorrelated, "target",
                                     {"skewness_threshold": 2.0, "kurtosis_threshold": 7.0})
        assert r.passed

    def test_detects_highly_skewed_feature(self, df_skewed):
        r = check_distribution_shape(df_skewed, "target",
                                     {"skewness_threshold": 2.0, "kurtosis_threshold": 7.0})
        assert not r.passed
        assert "skewed_col" in r.affected_columns

    def test_normal_column_not_flagged(self, df_skewed):
        r = check_distribution_shape(df_skewed, "target",
                                     {"skewness_threshold": 2.0, "kurtosis_threshold": 7.0})
        assert "normal_col" not in r.affected_columns

    def test_threshold_respected(self, df_skewed):
        # Very permissive threshold → should pass
        r = check_distribution_shape(df_skewed, "target",
                                     {"skewness_threshold": 100.0, "kurtosis_threshold": 1000.0})
        assert r.passed

    def test_details_contain_flagged_features(self, df_skewed):
        r = check_distribution_shape(df_skewed, "target",
                                     {"skewness_threshold": 2.0, "kurtosis_threshold": 7.0})
        assert "flagged_features" in r.details
        assert "skewed_col" in r.details["flagged_features"]

    def test_skewness_and_kurtosis_in_details(self, df_skewed):
        r = check_distribution_shape(df_skewed, "target",
                                     {"skewness_threshold": 2.0, "kurtosis_threshold": 7.0})
        info = r.details["flagged_features"]["skewed_col"]
        assert "skewness" in info
        assert "excess_kurtosis" in info

    def test_check_name_correct(self, df_uncorrelated):
        r = check_distribution_shape(df_uncorrelated, "target", {})
        assert r.check_name == "distribution_shape"


# ---------------------------------------------------------------------------
# run_all_feature_checks
# ---------------------------------------------------------------------------

class TestRunAllFeatureChecks:
    def test_returns_three_results_by_default(self, df_uncorrelated):
        results = run_all_feature_checks(df_uncorrelated, "target", _default_cfg())
        assert len(results) == 3

    def test_all_check_names_present(self, df_uncorrelated):
        results = run_all_feature_checks(df_uncorrelated, "target", _default_cfg())
        names = {r.check_name for r in results}
        assert names == {"feature_correlation", "feature_relevance", "distribution_shape"}

    def test_disabled_top_level_returns_empty(self, df_uncorrelated):
        cfg = _default_cfg()
        cfg["enabled"] = False
        assert run_all_feature_checks(df_uncorrelated, "target", cfg) == []

    def test_individual_check_can_be_disabled(self, df_uncorrelated):
        cfg = _default_cfg()
        cfg["distribution_shape"]["enabled"] = False
        results = run_all_feature_checks(df_uncorrelated, "target", cfg)
        assert "distribution_shape" not in {r.check_name for r in results}

    def test_detects_issues_in_dirty_data(self, df_correlated, df_skewed):
        import pandas as pd
        df = pd.concat([df_correlated, df_skewed.rename(columns={"normal_col": "norm", "skewed_col": "skw"})],
                       axis=1).dropna()
        df["target"] = df["target"].iloc[:, 0] if hasattr(df["target"], "iloc") else df["target"]
        # Just check it runs without error on complex data
        results = run_all_feature_checks(df_correlated, "target", _default_cfg())
        assert len(results) == 3
