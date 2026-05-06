"""Tests for src/recommendations.py (Phase 8)."""

from __future__ import annotations

import pytest

from src.recommendations import generate_recommendations
from src.utils import CheckResult, FrameworkReport, Recommendation


def _failed(check_name: str, severity: str = "warning", **details) -> CheckResult:
    return CheckResult(check_name, passed=False, severity=severity,
                       details=details, affected_columns=details.pop("affected_columns", []))


def _passed(check_name: str) -> CheckResult:
    return CheckResult(check_name, passed=True, severity="info")


class TestGenerateRecommendations:
    def test_empty_report_returns_empty(self):
        assert generate_recommendations(FrameworkReport("ds")) == []

    def test_passed_checks_produce_no_recommendations(self):
        report = FrameworkReport("ds")
        report.quality_results = [_passed("missing_values"), _passed("duplicates")]
        assert generate_recommendations(report) == []

    def test_missing_values_produces_recommendation(self):
        report = FrameworkReport("ds")
        report.quality_results = [CheckResult(
            "missing_values", passed=False, severity="warning",
            details={"flagged_columns": {"age": 0.20, "income": 0.60}},
        )]
        recs = generate_recommendations(report)
        assert len(recs) >= 1
        assert all(r.check_name == "missing_values" for r in recs)

    def test_high_missing_gets_high_priority(self):
        report = FrameworkReport("ds")
        report.quality_results = [CheckResult(
            "missing_values", passed=False, severity="error",
            details={"flagged_columns": {"col": 0.80}},
        )]
        recs = generate_recommendations(report)
        assert any(r.priority == "high" for r in recs)

    def test_target_leakage_gets_high_priority(self):
        report = FrameworkReport("ds")
        report.leakage_results = [CheckResult(
            "target_leakage", passed=False, severity="error",
            details={"flagged_features": {"proxy": 0.99}},
        )]
        recs = generate_recommendations(report)
        assert len(recs) == 1
        assert recs[0].priority == "high"

    def test_constant_features_gets_low_priority(self):
        report = FrameworkReport("ds")
        report.quality_results = [CheckResult(
            "constant_features", passed=False, severity="warning",
            details={"constant_columns": ["const_col"]},
        )]
        recs = generate_recommendations(report)
        assert recs[0].priority == "low"

    def test_sorted_high_before_medium_before_low(self):
        report = FrameworkReport("ds")
        report.quality_results = [
            CheckResult("constant_features", passed=False, severity="warning",
                        details={"constant_columns": ["c"]}),
            CheckResult("class_imbalance", passed=False, severity="error",
                        details={"minority_ratio": 0.03}),
        ]
        report.leakage_results = [
            CheckResult("id_column_leakage", passed=False, severity="warning",
                        details={"flagged_columns": {"id": 1.0}}),
        ]
        recs = generate_recommendations(report)
        priorities = [r.priority for r in recs]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        assert priorities == sorted(priorities, key=lambda p: priority_order[p])

    def test_recommendations_have_non_empty_action(self):
        report = FrameworkReport("ds")
        report.quality_results = [CheckResult(
            "duplicates", passed=False, severity="warning",
            details={"duplicate_count": 5, "duplicate_rate": 0.05},
        )]
        recs = generate_recommendations(report)
        assert all(r.action for r in recs)

    def test_recommendations_have_non_empty_rationale(self):
        report = FrameworkReport("ds")
        report.leakage_results = [CheckResult(
            "temporal_leakage", passed=False, severity="error",
            details={"date_column": "event_date", "is_sorted": False},
        )]
        recs = generate_recommendations(report)
        assert all(r.rationale for r in recs)

    def test_recommendations_have_code_snippets(self):
        report = FrameworkReport("ds")
        report.quality_results = [CheckResult(
            "outliers", passed=False, severity="warning",
            details={"flagged_columns": {"fare": 99}},
        )]
        recs = generate_recommendations(report)
        assert all(r.code_snippet for r in recs)

    def test_feature_results_included(self):
        report = FrameworkReport("ds")
        report.feature_results = [CheckResult(
            "feature_correlation", passed=False, severity="warning",
            details={"correlated_pairs": [{"feature_a": "a", "feature_b": "b", "correlation": 0.95}]},
        )]
        recs = generate_recommendations(report)
        assert len(recs) == 1
        assert recs[0].check_name == "feature_correlation"

    def test_all_known_checks_have_handlers(self):
        check_names = [
            "missing_values", "duplicates", "outliers", "class_imbalance",
            "constant_features", "low_variance",
            "target_leakage", "train_test_overlap", "temporal_leakage", "id_column_leakage",
            "feature_correlation", "feature_relevance", "distribution_shape",
        ]
        for name in check_names:
            report = FrameworkReport("ds")
            report.quality_results = [CheckResult(name, passed=False, severity="warning", details={})]
            recs = generate_recommendations(report)
            assert len(recs) >= 1, f"No recommendation handler for check '{name}'"

    def test_recommendation_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="priority must be one of"):
            Recommendation("x", priority="critical", action="a", rationale="b")
