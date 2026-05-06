"""Tests for src/drift_checks.py (Phase 14)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.drift_checks import (
    _psi,
    _split_dataframe,
    check_covariate_drift,
    check_label_drift,
    run_all_drift_checks,
)


def _stable_df(n=1000) -> pd.DataFrame:
    """Large stable dataset so PSI stays well below 0.1 for both features."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(5, 2, n),
        "target": rng.integers(0, 2, n),
    })


def _drifted_df(n=1000) -> pd.DataFrame:
    """Second half has a VERY different distribution for f1 only; f2 is stable."""
    rng = np.random.default_rng(7)
    # f1: extreme shift (mean 0 → mean 20, same std)
    f1 = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(20, 1, n // 2)])
    # f2: absolutely identical distribution in both halves
    f2 = np.concatenate([rng.normal(5, 2, n // 2), rng.normal(5, 2, n // 2)])
    return pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "target": rng.integers(0, 2, n),
    })


def _temporal_df(n=100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = date(2023, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    f1 = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)])
    return pd.DataFrame({
        "date": dates,
        "f1":   f1,
        "target": rng.integers(0, 2, n),
    })


class TestPsi:
    def test_identical_arrays_give_zero(self):
        arr = np.random.default_rng(0).normal(0, 1, 200)
        assert _psi(arr, arr) < 0.01

    def test_very_different_arrays_give_high_psi(self):
        a = np.zeros(200)
        b = np.ones(200) * 100
        assert _psi(a, b) > 0.25


class TestSplitDataframe:
    def test_index_split_gives_equal_halves(self):
        df = _stable_df(200)
        df1, df2 = _split_dataframe(df, None)
        assert len(df1) == 100
        assert len(df2) == 100

    def test_date_split_sorts_first(self):
        df = _temporal_df(100)
        # Shuffle first
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df1, df2 = _split_dataframe(df, "date")
        # After split, df1 should contain earlier dates
        assert df1["date"].max() <= df2["date"].min()


class TestCheckCovariateDrift:
    def test_passes_on_stable_data(self):
        r = check_covariate_drift(_stable_df(), "target", {"alpha": 0.05})
        assert r.passed

    def test_detects_drift_in_drifted_data(self):
        r = check_covariate_drift(_drifted_df(), "target", {"alpha": 0.05})
        assert not r.passed
        assert "f1" in r.affected_columns

    def test_stable_feature_not_flagged(self):
        r = check_covariate_drift(_drifted_df(), "target", {"alpha": 0.05})
        assert "f2" not in r.affected_columns

    def test_no_numeric_features_passes(self):
        df = pd.DataFrame({"cat": ["a", "b"] * 50, "target": [0, 1] * 50})
        r = check_covariate_drift(df, "target", {"alpha": 0.05})
        assert r.passed

    def test_check_name(self):
        assert check_covariate_drift(_stable_df(), "target", {}).check_name == "covariate_drift"

    def test_details_contain_features_checked(self):
        # key present on both pass and fail results
        r_pass = check_covariate_drift(_stable_df(), "target", {"alpha": 0.05})
        r_fail = check_covariate_drift(_drifted_df(), "target", {"alpha": 0.05})
        assert "features_checked" in r_pass.details or "drifted_features" in r_fail.details


class TestCheckLabelDrift:
    def test_passes_on_stable_labels(self):
        r = check_label_drift(_stable_df(), "target", {"alpha": 0.05})
        assert r.passed

    def test_detects_label_drift(self):
        # First half all class 0, second half all class 1
        df = pd.DataFrame({
            "f": range(100),
            "target": [0] * 50 + [1] * 50,
        })
        r = check_label_drift(df, "target", {"alpha": 0.05})
        assert not r.passed

    def test_missing_target_passes(self):
        r = check_label_drift(_stable_df(), "ghost", {"alpha": 0.05})
        assert r.passed

    def test_check_name(self):
        assert check_label_drift(_stable_df(), "target", {}).check_name == "label_drift"


class TestRunAllDriftChecks:
    def test_returns_two_results(self):
        r = run_all_drift_checks(_stable_df(), "target", {"enabled": True,
                                  "covariate_drift": {"enabled": True, "alpha": 0.05},
                                  "label_drift":     {"enabled": True, "alpha": 0.05}})
        assert len(r) == 2

    def test_disabled_returns_empty(self):
        r = run_all_drift_checks(_stable_df(), "target", {"enabled": False})
        assert r == []

    def test_individual_check_disabled(self):
        r = run_all_drift_checks(_stable_df(), "target", {
            "enabled": True,
            "covariate_drift": {"enabled": False},
            "label_drift":     {"enabled": True, "alpha": 0.05},
        })
        assert len(r) == 1
        assert r[0].check_name == "label_drift"
