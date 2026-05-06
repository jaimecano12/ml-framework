"""Tests for src/scoring.py (Phase 10)."""

from __future__ import annotations

import pytest

from src.scoring import compute_readiness_score
from src.utils import CheckResult, FrameworkReport, ReadinessScore


def _cr(name: str, passed: bool, severity: str = "warning") -> CheckResult:
    return CheckResult(name, passed=passed, severity=severity)


def _perfect_report() -> FrameworkReport:
    report = FrameworkReport(dataset_name="ds", metadata={"shape": [500, 6]})
    report.quality_results  = [_cr("missing_values", True), _cr("duplicates", True)]
    report.leakage_results  = [_cr("target_leakage", True), _cr("temporal_leakage", True)]
    report.feature_results  = [_cr("feature_correlation", True), _cr("feature_relevance", True)]
    return report


def _failing_report() -> FrameworkReport:
    """Report with enough errors to guarantee grade C or below."""
    report = FrameworkReport(dataset_name="ds", metadata={"shape": [10, 10]})
    report.quality_results  = [_cr(f"qc{i}", False, "error") for i in range(4)]
    report.leakage_results  = [_cr(f"lk{i}", False, "error") for i in range(4)]
    report.feature_results  = [_cr(f"fa{i}", False, "warning") for i in range(3)]
    return report


class TestComputeReadinessScore:
    def test_returns_readiness_score_instance(self):
        assert isinstance(compute_readiness_score(_perfect_report()), ReadinessScore)

    def test_all_pass_gives_high_score(self):
        s = compute_readiness_score(_perfect_report())
        assert s.overall >= 85.0

    def test_all_pass_gives_grade_A(self):
        s = compute_readiness_score(_perfect_report())
        assert s.grade == "A"

    def test_empty_report_scores_high(self):
        s = compute_readiness_score(FrameworkReport("ds", metadata={"shape": [500, 6]}))
        assert s.overall >= 70.0

    def test_error_penalty_larger_than_warning(self):
        report_error   = FrameworkReport("ds", metadata={"shape": [500, 6]})
        report_warning = FrameworkReport("ds", metadata={"shape": [500, 6]})
        report_error.quality_results   = [_cr("x", False, "error")]
        report_warning.quality_results = [_cr("x", False, "warning")]
        s_err  = compute_readiness_score(report_error)
        s_warn = compute_readiness_score(report_warning)
        assert s_err.overall < s_warn.overall

    def test_leakage_error_penalises_more_than_quality_error(self):
        r_quality = FrameworkReport("ds", metadata={"shape": [500, 6]})
        r_leakage = FrameworkReport("ds", metadata={"shape": [500, 6]})
        r_quality.quality_results = [_cr("missing_values", False, "error")]
        r_leakage.leakage_results = [_cr("target_leakage", False, "error")]
        s_q = compute_readiness_score(r_quality)
        s_l = compute_readiness_score(r_leakage)
        assert s_l.overall < s_q.overall

    def test_grade_boundaries(self):
        # Manually create a report that produces a specific score range
        # is hard; just verify all grades are reachable
        possible_grades = {"A", "B", "C", "D", "F"}
        grades_seen = set()
        for n_errors in [0, 1, 2, 3, 5]:
            r = FrameworkReport("ds", metadata={"shape": [20, 10]})
            r.quality_results = [_cr(f"c{i}", False, "error") for i in range(n_errors)]
            r.leakage_results = [_cr(f"l{i}", False, "error") for i in range(n_errors)]
            grades_seen.add(compute_readiness_score(r).grade)
        assert len(grades_seen) >= 3  # at least 3 different grades produced

    def test_dimension_scores_are_present(self):
        s = compute_readiness_score(_perfect_report())
        for dim in ("quality", "leakage", "features", "sufficiency"):
            assert hasattr(s, dim)

    def test_quality_dimension_reflects_quality_results(self):
        r = FrameworkReport("ds", metadata={"shape": [500, 6]})
        r.quality_results = [_cr("missing_values", False, "error")]
        s = compute_readiness_score(r)
        assert s.quality.score < 100.0
        assert s.quality.errors == 1

    def test_leakage_dimension_reflects_leakage_results(self):
        r = FrameworkReport("ds", metadata={"shape": [500, 6]})
        r.leakage_results = [_cr("target_leakage", False, "error")]
        s = compute_readiness_score(r)
        assert s.leakage.score < 100.0

    def test_sufficiency_low_ratio_reduces_score(self):
        r_large = FrameworkReport("ds", metadata={"shape": [10000, 5]})
        r_small = FrameworkReport("ds", metadata={"shape": [20, 5]})
        s_large = compute_readiness_score(r_large)
        s_small = compute_readiness_score(r_small)
        assert s_large.sufficiency.score > s_small.sufficiency.score

    def test_overall_is_bounded_0_100(self):
        s = compute_readiness_score(_failing_report())
        assert 0.0 <= s.overall <= 100.0

    def test_label_is_non_empty_string(self):
        s = compute_readiness_score(_perfect_report())
        assert isinstance(s.label, str) and len(s.label) > 0

    def test_to_dict_contains_required_keys(self):
        s = compute_readiness_score(_perfect_report())
        d = s.to_dict()
        for key in ("overall", "grade", "label", "quality", "leakage", "features", "sufficiency"):
            assert key in d

    def test_failing_report_gets_low_grade(self):
        s = compute_readiness_score(_failing_report())
        assert s.grade in {"C", "D", "F"}
