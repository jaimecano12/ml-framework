"""Tests for src/checker.py (Phase 13 — Python SDK)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.checker import DatasetChecker
from src.utils import FrameworkReport


@pytest.fixture()
def df_simple() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 150
    feature = rng.normal(0, 1, n)
    return pd.DataFrame({
        "feature": feature,
        "noise":   rng.normal(0, 1, n),
        "target":  (feature > 0).astype(int),
    })


class TestDatasetChecker:
    def test_run_returns_framework_report(self, df_simple):
        checker = DatasetChecker()
        report = checker.run(df_simple, target_col="target", skip_impact=True)
        assert isinstance(report, FrameworkReport)

    def test_score_available_after_run(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        assert isinstance(checker.score, float)
        assert 0 <= checker.score <= 100

    def test_grade_is_letter(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        assert checker.grade in {"A", "B", "C", "D", "F"}

    def test_score_none_before_run(self):
        assert DatasetChecker().score is None

    def test_grade_none_before_run(self):
        assert DatasetChecker().grade is None

    def test_failed_checks_returns_list(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        failed = checker.failed_checks()
        assert isinstance(failed, list)

    def test_top_recommendations_filtered_by_priority(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        recs = checker.top_recommendations(priority="high")
        assert all(r.priority == "high" for r in recs)

    def test_top_recommendations_capped_by_n(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        recs = checker.top_recommendations(n=2)
        assert len(recs) <= 2

    def test_summary_returns_dict(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        s = checker.summary()
        assert "total_checks" in s
        assert "passed" in s

    def test_to_dict_contains_all_sections(self, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        d = checker.to_dict()
        for key in ("summary", "quality_results", "leakage_results",
                    "feature_results", "sufficiency_results", "drift_results",
                    "recommendations", "readiness_score"):
            assert key in d

    def test_set_overrides_config(self, df_simple):
        checker = DatasetChecker()
        checker.set(leakage_checks__target_leakage__correlation_threshold=0.50)
        assert checker._config["leakage_checks"]["target_leakage"]["correlation_threshold"] == 0.50

    def test_run_without_target_raises(self, df_simple):
        checker = DatasetChecker()
        with pytest.raises((ValueError, KeyError)):
            checker.run(df_simple)

    def test_run_from_file(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 50, "target": [0, 1, 0] * 50})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        checker = DatasetChecker()
        report = checker.run(str(path), target_col="target", skip_impact=True)
        assert isinstance(report, FrameworkReport)

    def test_save_report_creates_html(self, tmp_path, df_simple):
        checker = DatasetChecker()
        checker.run(df_simple, target_col="target", skip_impact=True)
        path = checker.save_report(tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_failed_checks_before_run_raises(self):
        with pytest.raises(RuntimeError, match="run()"):
            DatasetChecker().failed_checks()
