"""Tests for src/reporting.py (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporting import (
    _fig_to_b64,
    _plot_check_summary,
    _plot_impact_comparison,
    _plot_severity_distribution,
    generate_report,
)
from src.utils import CheckResult, FrameworkReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_report() -> FrameworkReport:
    report = FrameworkReport(
        dataset_name="test_ds",
        metadata={"shape": [200, 5], "target_col": "label"},
    )
    report.quality_results = [
        CheckResult("missing_values", passed=True, severity="info", message="OK"),
        CheckResult("duplicates", passed=False, severity="warning", message="3 dupes",
                    affected_columns=["id"]),
    ]
    report.leakage_results = [
        CheckResult("target_leakage", passed=False, severity="error",
                    message="proxy leaks", affected_columns=["proxy"]),
        CheckResult("temporal_leakage", passed=True, severity="info", message="OK"),
    ]
    report.impact_results = [
        CheckResult(
            "impact_logistic_regression", passed=False, severity="warning",
            message="lr: baseline=0.95, cleaned=0.60, Δ=-0.35",
            details={
                "model": "logistic_regression", "metric": "accuracy",
                "baseline_score": 0.95, "baseline_std": 0.01,
                "cleaned_score": 0.60, "cleaned_std": 0.05,
                "delta": -0.35, "dropped_columns": ["proxy"],
            },
        ),
    ]
    return report


def _minimal_report() -> FrameworkReport:
    return FrameworkReport(dataset_name="minimal")


# ---------------------------------------------------------------------------
# _fig_to_b64
# ---------------------------------------------------------------------------

class TestFigToB64:
    def test_returns_non_empty_string(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _fig_to_b64(fig)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_result_is_valid_base64(self):
        import base64
        import matplotlib.pyplot as plt
        fig, _ = plt.subplots()
        result = _fig_to_b64(fig)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

class TestPlotHelpers:
    def test_summary_plot_returns_b64_string(self):
        report = _make_full_report()
        result = _plot_check_summary(report)
        assert isinstance(result, str) and len(result) > 0

    def test_summary_plot_empty_report_returns_empty(self):
        result = _plot_check_summary(_minimal_report())
        assert result == ""

    def test_severity_plot_returns_b64_string(self):
        result = _plot_severity_distribution(_make_full_report())
        assert isinstance(result, str) and len(result) > 0

    def test_severity_plot_empty_report_returns_empty(self):
        result = _plot_severity_distribution(_minimal_report())
        assert result == ""

    def test_impact_plot_returns_b64_string(self):
        result = _plot_impact_comparison(_make_full_report())
        assert isinstance(result, str) and len(result) > 0

    def test_impact_plot_no_impact_results_returns_empty(self):
        report = _make_full_report()
        report.impact_results = []
        result = _plot_impact_comparison(report)
        assert result == ""


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_creates_html_file(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        assert path.exists()
        assert path.suffix == ".html"

    def test_output_in_specified_dir(self, tmp_path: Path):
        out = tmp_path / "my_reports"
        path = generate_report(_minimal_report(), out, {"include_plots": False})
        assert path.parent == out

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        out = tmp_path / "new" / "nested" / "dir"
        generate_report(_minimal_report(), out, {"include_plots": False})
        assert out.exists()

    def test_html_contains_dataset_name(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "test_ds" in content

    def test_html_contains_summary_section(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "Summary" in content

    def test_html_contains_quality_section(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "Quality Checks" in content
        assert "missing_values" in content

    def test_html_contains_leakage_section(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "Leakage Checks" in content
        assert "target_leakage" in content

    def test_html_contains_impact_section(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "Impact Analysis" in content
        assert "logistic_regression" in content

    def test_plots_embedded_when_enabled(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": True})
        content = path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_no_plots_when_disabled(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        content = path.read_text(encoding="utf-8")
        assert "data:image/png;base64," not in content

    def test_filename_contains_dataset_name(self, tmp_path: Path):
        report = _make_full_report()
        path = generate_report(report, tmp_path, {"include_plots": False})
        assert "test_ds" in path.name

    def test_minimal_report_generates_without_error(self, tmp_path: Path):
        path = generate_report(_minimal_report(), tmp_path, {"include_plots": False})
        assert path.exists()
