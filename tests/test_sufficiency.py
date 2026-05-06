"""Tests for src/sufficiency.py (Phase 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sufficiency import (
    check_class_support,
    check_cv_stability,
    check_feature_to_sample_ratio,
    check_sample_size,
    run_all_sufficiency_checks,
)
from src.utils import CheckResult, FrameworkReport


def _make_df(n=200, n_feat=5, n_classes=2) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.normal(0, 1, n) for i in range(n_feat)}
    data["target"] = rng.integers(0, n_classes, n)
    return pd.DataFrame(data)


def _empty_report() -> FrameworkReport:
    return FrameworkReport("ds", metadata={"shape": [200, 6]})


def _default_cfg() -> dict:
    return {
        "enabled": True,
        "sample_size":             {"enabled": True, "min_rows": 100, "comfortable_ratio": 50.0},
        "class_support":           {"enabled": True, "min_samples_per_class": 30},
        "cv_stability":            {"enabled": True, "max_cv_std": 0.10},
        "feature_to_sample_ratio": {"enabled": True, "max_ratio": 0.10},
    }


# ---------------------------------------------------------------------------
# check_sample_size
# ---------------------------------------------------------------------------

class TestCheckSampleSize:
    def test_passes_with_comfortable_ratio(self):
        df = _make_df(n=500, n_feat=5)
        r = check_sample_size(df, "target", {"min_rows": 100, "comfortable_ratio": 50.0})
        assert r.passed

    def test_fails_below_min_rows(self):
        df = _make_df(n=50, n_feat=5)
        r = check_sample_size(df, "target", {"min_rows": 100, "comfortable_ratio": 50.0})
        assert not r.passed
        assert r.severity == "error"

    def test_warns_when_ratio_below_comfortable(self):
        df = _make_df(n=200, n_feat=10)   # ratio = 200/10 = 20 < 50
        r = check_sample_size(df, "target", {"min_rows": 100, "comfortable_ratio": 50.0})
        assert not r.passed
        assert r.severity == "warning"

    def test_details_contain_ratio(self):
        df = _make_df(n=300, n_feat=5)
        r = check_sample_size(df, "target", {"min_rows": 100, "comfortable_ratio": 50.0})
        assert "n_p_ratio" in r.details

    def test_check_name(self):
        assert check_sample_size(_make_df(), "target", {}).check_name == "sample_size"


# ---------------------------------------------------------------------------
# check_class_support
# ---------------------------------------------------------------------------

class TestCheckClassSupport:
    def test_passes_with_enough_samples(self):
        df = _make_df(n=300, n_classes=2)
        r = check_class_support(df, "target", {"min_samples_per_class": 30})
        assert r.passed

    def test_fails_when_minority_class_too_small(self):
        df = pd.DataFrame({"f": range(100), "target": [1] * 3 + [0] * 97})
        r = check_class_support(df, "target", {"min_samples_per_class": 30})
        assert not r.passed

    def test_severity_error_for_very_small_class(self):
        df = pd.DataFrame({"f": range(100), "target": [1] * 2 + [0] * 98})
        r = check_class_support(df, "target", {"min_samples_per_class": 30})
        assert r.severity == "error"

    def test_skips_continuous_target(self):
        df = pd.DataFrame({"f": range(100), "target": np.random.normal(0, 1, 100)})
        r = check_class_support(df, "target", {"min_samples_per_class": 30})
        assert r.passed  # skipped, no error

    def test_missing_target_passes(self):
        df = _make_df()
        r = check_class_support(df, "ghost", {})
        assert r.passed

    def test_check_name(self):
        assert check_class_support(_make_df(), "target", {}).check_name == "class_support"


# ---------------------------------------------------------------------------
# check_cv_stability
# ---------------------------------------------------------------------------

class TestCheckCvStability:
    def _report_with_cv(self, std: float) -> FrameworkReport:
        from src.utils import CheckResult, FrameworkReport
        r = FrameworkReport("ds")
        r.impact_results = [
            CheckResult("impact_lr", passed=True,
                        details={"model": "lr", "baseline_std": std, "baseline_score": 0.80}),
        ]
        return r

    def test_passes_with_low_std(self):
        r = check_cv_stability(self._report_with_cv(0.03), {"max_cv_std": 0.10})
        assert r.passed

    def test_fails_with_high_std(self):
        r = check_cv_stability(self._report_with_cv(0.15), {"max_cv_std": 0.10})
        assert not r.passed

    def test_skips_when_no_impact_results(self):
        r = check_cv_stability(_empty_report(), {"max_cv_std": 0.10})
        assert r.passed
        assert "skipped" in r.message.lower()

    def test_details_contain_model_stds(self):
        r = check_cv_stability(self._report_with_cv(0.03), {"max_cv_std": 0.10})
        assert "model_stds" in r.details

    def test_check_name(self):
        assert check_cv_stability(_empty_report(), {}).check_name == "cv_stability"


# ---------------------------------------------------------------------------
# check_feature_to_sample_ratio
# ---------------------------------------------------------------------------

class TestCheckFeatureToSampleRatio:
    def test_passes_with_low_ratio(self):
        df = _make_df(n=500, n_feat=5)   # p/n = 5/500 = 0.01 < 0.10
        r = check_feature_to_sample_ratio(df, "target", {"max_ratio": 0.10})
        assert r.passed

    def test_fails_with_high_ratio(self):
        df = _make_df(n=50, n_feat=20)   # p/n = 20/50 = 0.40 > 0.10
        r = check_feature_to_sample_ratio(df, "target", {"max_ratio": 0.10})
        assert not r.passed

    def test_severity_error_above_half(self):
        df = _make_df(n=30, n_feat=20)   # p/n > 0.5
        r = check_feature_to_sample_ratio(df, "target", {"max_ratio": 0.10})
        assert r.severity == "error"

    def test_details_contain_ratio(self):
        df = _make_df()
        r = check_feature_to_sample_ratio(df, "target", {})
        assert "p_n_ratio" in r.details

    def test_check_name(self):
        assert check_feature_to_sample_ratio(_make_df(), "target", {}).check_name == "feature_to_sample_ratio"


# ---------------------------------------------------------------------------
# run_all_sufficiency_checks
# ---------------------------------------------------------------------------

class TestRunAllSufficiencyChecks:
    def test_returns_four_results(self):
        df = _make_df()
        r = run_all_sufficiency_checks(df, "target", _empty_report(), _default_cfg())
        assert len(r) == 4

    def test_disabled_returns_empty(self):
        cfg = _default_cfg()
        cfg["enabled"] = False
        assert run_all_sufficiency_checks(_make_df(), "target", _empty_report(), cfg) == []

    def test_individual_check_can_be_disabled(self):
        cfg = _default_cfg()
        cfg["cv_stability"]["enabled"] = False
        r = run_all_sufficiency_checks(_make_df(), "target", _empty_report(), cfg)
        assert "cv_stability" not in {x.check_name for x in r}

    def test_all_pass_on_good_data(self):
        df = _make_df(n=500, n_feat=5)
        r = run_all_sufficiency_checks(df, "target", _empty_report(), _default_cfg())
        assert all(x.passed for x in r)
