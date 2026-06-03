"""Tests for src/leakage_checks.py (Phase 4)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.leakage_checks import (
    _cramers_v,
    _feature_target_association,
    check_id_column_leakage,
    check_leakage_risk_score,
    check_target_leakage,
    check_temporal_leakage,
    check_train_test_overlap,
    compute_leakage_risk_score,
    run_all_leakage_checks,
)
from src.utils import CheckResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_clean() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":    rng.integers(18, 80, 200),
        "income": rng.normal(50_000, 10_000, 200),
        "target": rng.integers(0, 2, 200),
    })


@pytest.fixture()
def df_leaky() -> pd.DataFrame:
    """A feature that is a near-perfect proxy of the target."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 100)
    return pd.DataFrame({
        "proxy":   target.astype(float),      # corr = 1.0
        "noise":   rng.normal(0, 1, 100),
        "target":  target,
    })


@pytest.fixture()
def df_with_duplicates() -> pd.DataFrame:
    """All rows are exact duplicates of 3 unique rows — any split will have overlap."""
    base = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "target": [0, 1, 0]})
    return pd.concat([base] * 12, ignore_index=True)   # 36 rows, 3 unique


@pytest.fixture()
def df_temporal_sorted() -> pd.DataFrame:
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(50)]
    return pd.DataFrame({
        "date":    dates,
        "feature": range(50),
        "target":  [0, 1] * 25,
    })


@pytest.fixture()
def df_temporal_unsorted(df_temporal_sorted: pd.DataFrame) -> pd.DataFrame:
    return df_temporal_sorted.sample(frac=1, random_state=7).reset_index(drop=True)


@pytest.fixture()
def df_with_id() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [f"UID_{i}" for i in range(100)],   # 100 % unique
        "age":     list(range(20, 70)) * 2,              # 50 % unique
        "target":  [0, 1] * 50,
    })


def _default_leakage_config() -> dict:
    return {
        "enabled": True,
        "target_leakage":     {"enabled": True, "correlation_threshold": 0.95},
        "train_test_overlap": {"enabled": True, "test_size": 0.2, "random_state": 42},
        "temporal_leakage":   {"enabled": True, "date_column": None},
        "id_column_leakage":  {"enabled": True, "cardinality_threshold": 0.95},
        "leakage_risk_score": {"enabled": True, "threshold": 0.7, "weights": [0.35, 0.35, 0.30], "cv_folds": 3},
    }


# ---------------------------------------------------------------------------
# Helpers (_cramers_v, _feature_target_association)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_cramers_v_perfect_association(self):
        x = pd.Series(["A", "A", "B", "B"])
        y = pd.Series(["X", "X", "Y", "Y"])
        assert _cramers_v(x, y) == pytest.approx(1.0)

    def test_cramers_v_no_association(self):
        x = pd.Series(["A", "B"] * 50)
        y = pd.Series(["X", "X"] * 50)
        assert _cramers_v(x, y) == pytest.approx(0.0)

    def test_association_numeric_numeric_perfect(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _feature_target_association(s, s) == pytest.approx(1.0)

    def test_association_returns_zero_for_tiny_series(self):
        s = pd.Series([1.0, 2.0])
        assert _feature_target_association(s, s) == 0.0

    def test_association_handles_nans(self):
        f = pd.Series([1.0, 2.0, None, 4.0, 5.0])
        t = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        score = _feature_target_association(f, t)
        assert 0.0 <= score <= 1.0

    def test_association_categorical_feature(self):
        f = pd.Series(["A", "B", "A", "B"] * 10)
        t = pd.Series([0, 1, 0, 1] * 10)
        score = _feature_target_association(f, t)
        assert score > 0.9  # near-perfect association


# ---------------------------------------------------------------------------
# check_target_leakage
# ---------------------------------------------------------------------------

class TestCheckTargetLeakage:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_target_leakage(df_clean, "target", {"correlation_threshold": 0.95})
        assert r.passed
        assert r.severity == "info"

    def test_detects_perfect_proxy(self, df_leaky: pd.DataFrame):
        r = check_target_leakage(df_leaky, "target", {"correlation_threshold": 0.95})
        assert not r.passed
        assert "proxy" in r.affected_columns
        assert r.severity == "error"

    def test_noise_feature_not_flagged(self, df_leaky: pd.DataFrame):
        r = check_target_leakage(df_leaky, "target", {"correlation_threshold": 0.95})
        assert "noise" not in r.affected_columns

    def test_target_column_excluded(self, df_leaky: pd.DataFrame):
        r = check_target_leakage(df_leaky, "target", {"correlation_threshold": 0.95})
        assert "target" not in r.affected_columns

    def test_details_contain_flagged_features(self, df_leaky: pd.DataFrame):
        r = check_target_leakage(df_leaky, "target", {"correlation_threshold": 0.95})
        assert "flagged_features" in r.details
        assert r.details["flagged_features"]["proxy"] == pytest.approx(1.0, abs=1e-3)

    def test_threshold_respected(self, df_leaky: pd.DataFrame):
        # With threshold=1.1 (impossible) nothing should be flagged
        r = check_target_leakage(df_leaky, "target", {"correlation_threshold": 1.1})
        assert r.passed

    def test_categorical_target_handled(self):
        df = pd.DataFrame({
            "proxy":  [0, 0, 1, 1] * 20,
            "target": ["neg", "neg", "pos", "pos"] * 20,
        })
        r = check_target_leakage(df, "target", {"correlation_threshold": 0.95})
        assert not r.passed
        assert "proxy" in r.affected_columns

    def test_missing_target_raises(self, df_clean: pd.DataFrame):
        with pytest.raises(ValueError, match="ghost"):
            check_target_leakage(df_clean, "ghost", {})

    def test_constant_feature_does_not_crash(self):
        df = pd.DataFrame({"const": [5] * 50, "target": [0, 1] * 25})
        r = check_target_leakage(df, "target", {"correlation_threshold": 0.95})
        assert isinstance(r, CheckResult)


# ---------------------------------------------------------------------------
# check_train_test_overlap
# ---------------------------------------------------------------------------

class TestCheckTrainTestOverlap:
    def test_passes_on_data_without_duplicates(self, df_clean: pd.DataFrame):
        r = check_train_test_overlap(df_clean, {"test_size": 0.2, "random_state": 42})
        assert r.passed
        assert r.details["overlap_rows"] == 0

    def test_fails_when_all_rows_are_duplicates(self, df_with_duplicates: pd.DataFrame):
        r = check_train_test_overlap(df_with_duplicates, {"test_size": 0.2, "random_state": 42})
        assert not r.passed
        assert r.details["overlap_rows"] > 0

    def test_severity_error_for_high_overlap_rate(self, df_with_duplicates: pd.DataFrame):
        r = check_train_test_overlap(df_with_duplicates, {"test_size": 0.2, "random_state": 42})
        assert r.severity == "error"

    def test_details_contain_overlap_rate(self, df_with_duplicates: pd.DataFrame):
        r = check_train_test_overlap(df_with_duplicates, {"test_size": 0.2, "random_state": 42})
        assert "overlap_rate" in r.details
        assert 0 < r.details["overlap_rate"] <= 1.0

    def test_no_duplicates_message_mentions_impossibility(self, df_clean: pd.DataFrame):
        r = check_train_test_overlap(df_clean, {})
        assert "impossible" in r.message.lower()

    def test_single_unique_row_repeated(self):
        df = pd.DataFrame({"a": [42] * 20, "b": [99] * 20, "target": [1] * 20})
        r = check_train_test_overlap(df, {"test_size": 0.2, "random_state": 0})
        assert not r.passed


# ---------------------------------------------------------------------------
# check_temporal_leakage
# ---------------------------------------------------------------------------

class TestCheckTemporalLeakage:
    def test_passes_when_no_date_column_configured(self, df_clean: pd.DataFrame):
        r = check_temporal_leakage(df_clean, {"date_column": None})
        assert r.passed
        assert "skipped" in r.message.lower()

    def test_passes_on_sorted_dates(self, df_temporal_sorted: pd.DataFrame):
        r = check_temporal_leakage(df_temporal_sorted, {"date_column": "date"})
        assert r.passed
        assert r.details["is_sorted"] is True

    def test_fails_on_unsorted_dates(self, df_temporal_unsorted: pd.DataFrame):
        r = check_temporal_leakage(df_temporal_unsorted, {"date_column": "date"})
        assert not r.passed
        assert r.details["is_sorted"] is False
        assert r.severity == "error"

    def test_date_column_in_affected_columns(self, df_temporal_unsorted: pd.DataFrame):
        r = check_temporal_leakage(df_temporal_unsorted, {"date_column": "date"})
        assert "date" in r.affected_columns

    def test_missing_date_column_raises(self, df_clean: pd.DataFrame):
        with pytest.raises(ValueError, match="nonexistent_date"):
            check_temporal_leakage(df_clean, {"date_column": "nonexistent_date"})

    def test_detects_unparseable_dates(self):
        df = pd.DataFrame({
            "date": ["2023-01-01", "not-a-date", "2023-01-03"],
            "feature": [1, 2, 3],
        })
        r = check_temporal_leakage(df, {"date_column": "date"})
        assert not r.passed
        assert r.details["unparseable_dates"] == 1

    def test_details_contain_sort_flag(self, df_temporal_sorted: pd.DataFrame):
        r = check_temporal_leakage(df_temporal_sorted, {"date_column": "date"})
        assert "is_sorted" in r.details
        assert "unparseable_dates" in r.details


# ---------------------------------------------------------------------------
# check_id_column_leakage
# ---------------------------------------------------------------------------

class TestCheckIdColumnLeakage:
    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_id_column_leakage(df_clean, "target", {"cardinality_threshold": 0.95})
        assert r.passed

    def test_detects_string_id_column(self, df_with_id: pd.DataFrame):
        r = check_id_column_leakage(df_with_id, "target", {"cardinality_threshold": 0.95})
        assert not r.passed
        assert "user_id" in r.affected_columns

    def test_low_cardinality_column_not_flagged(self, df_with_id: pd.DataFrame):
        r = check_id_column_leakage(df_with_id, "target", {"cardinality_threshold": 0.95})
        assert "age" not in r.affected_columns

    def test_target_column_excluded(self, df_with_id: pd.DataFrame):
        r = check_id_column_leakage(df_with_id, "target", {"cardinality_threshold": 0.95})
        assert "target" not in r.affected_columns

    def test_threshold_respected(self, df_with_id: pd.DataFrame):
        # Lower threshold to 0.3 — "age" (50 % unique) should be flagged too
        r = check_id_column_leakage(df_with_id, "target", {"cardinality_threshold": 0.3})
        assert not r.passed
        assert "age" in r.affected_columns

    def test_details_contain_unique_ratios(self, df_with_id: pd.DataFrame):
        r = check_id_column_leakage(df_with_id, "target", {"cardinality_threshold": 0.95})
        assert "flagged_columns" in r.details
        assert r.details["flagged_columns"]["user_id"] == pytest.approx(1.0)

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame({"a": [], "target": []})
        r = check_id_column_leakage(df, "target", {"cardinality_threshold": 0.95})
        assert r.passed


# ---------------------------------------------------------------------------
# run_all_leakage_checks (orchestrator)
# ---------------------------------------------------------------------------

class TestRunAllLeakageChecks:
    def test_returns_list_of_check_results(self, df_clean: pd.DataFrame):
        results = run_all_leakage_checks(df_clean, "target", _default_leakage_config())
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)

    def test_all_five_checks_run_by_default(self, df_clean: pd.DataFrame):
        results = run_all_leakage_checks(df_clean, "target", _default_leakage_config())
        names = {r.check_name for r in results}
        expected = {
            "target_leakage", "train_test_overlap", "temporal_leakage",
            "id_column_leakage", "leakage_risk_score",
        }
        assert names == expected

    def test_disabled_top_level_returns_empty(self, df_clean: pd.DataFrame):
        cfg = _default_leakage_config()
        cfg["enabled"] = False
        assert run_all_leakage_checks(df_clean, "target", cfg) == []

    def test_individual_check_can_be_disabled(self, df_clean: pd.DataFrame):
        cfg = _default_leakage_config()
        cfg["temporal_leakage"]["enabled"] = False
        results = run_all_leakage_checks(df_clean, "target", cfg)
        assert "temporal_leakage" not in {r.check_name for r in results}

    def test_detects_leakage_in_leaky_dataset(self, df_leaky: pd.DataFrame):
        results = run_all_leakage_checks(df_leaky, "target", _default_leakage_config())
        failed = [r for r in results if not r.passed]
        assert any(r.check_name == "target_leakage" for r in failed)

    def test_all_pass_on_clean_data(self, df_clean: pd.DataFrame):
        results = run_all_leakage_checks(df_clean, "target", _default_leakage_config())
        assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# check_leakage_risk_score / compute_leakage_risk_score
# ---------------------------------------------------------------------------

class TestLeakageRiskScore:
    _cfg = {"threshold": 0.7, "weights": [0.35, 0.35, 0.30], "cv_folds": 3}

    def test_passes_on_clean_data(self, df_clean: pd.DataFrame):
        r = check_leakage_risk_score(df_clean, "target", self._cfg)
        assert r.passed
        assert r.check_name == "leakage_risk_score"

    def test_detects_perfect_proxy(self, df_leaky: pd.DataFrame):
        r = check_leakage_risk_score(df_leaky, "target", self._cfg)
        assert not r.passed
        assert "proxy" in r.affected_columns

    def test_severity_error_when_score_above_09(self, df_leaky: pd.DataFrame):
        r = check_leakage_risk_score(df_leaky, "target", self._cfg)
        assert r.severity == "error"

    def test_details_contain_all_score_components(self, df_clean: pd.DataFrame):
        r = check_leakage_risk_score(df_clean, "target", self._cfg)
        assert "risk_scores" in r.details
        assert "corr_scores" in r.details
        assert "mi_scores" in r.details
        assert "perf_scores" in r.details
        assert "weights" in r.details

    def test_threshold_respected(self, df_leaky: pd.DataFrame):
        # threshold=1.1 → nothing flagged
        r = check_leakage_risk_score(df_leaky, "target", {**self._cfg, "threshold": 1.1})
        assert r.passed

    def test_compute_returns_all_features(self, df_clean: pd.DataFrame):
        result = compute_leakage_risk_score(df_clean, "target", self._cfg)
        feature_cols = [c for c in df_clean.columns if c != "target"]
        assert set(result["risk_scores"].keys()) == set(feature_cols)
        assert set(result["corr_scores"].keys()) == set(feature_cols)
        assert set(result["mi_scores"].keys()) == set(feature_cols)
        assert set(result["perf_scores"].keys()) == set(feature_cols)

    def test_scores_in_unit_interval(self, df_clean: pd.DataFrame):
        result = compute_leakage_risk_score(df_clean, "target", self._cfg)
        for col, score in result["risk_scores"].items():
            assert 0.0 <= score <= 1.0, f"risk_score for '{col}' out of range: {score}"

    def test_missing_target_raises(self, df_clean: pd.DataFrame):
        with pytest.raises(ValueError, match="ghost"):
            check_leakage_risk_score(df_clean, "ghost", self._cfg)

    def test_noise_feature_low_risk(self, df_leaky: pd.DataFrame):
        result = compute_leakage_risk_score(df_leaky, "target", self._cfg)
        assert result["risk_scores"]["noise"] < 0.5
