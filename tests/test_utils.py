"""Smoke tests for src/utils.py (Phase 1)."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.utils import CheckResult, FrameworkReport, load_dataset, setup_logger


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------

class TestSetupLogger:
    def test_runs_without_error(self):
        setup_logger(log_level="DEBUG")

    def test_accepts_file_sink(self, tmp_path: Path):
        log_file = tmp_path / "test.log"
        setup_logger(log_level="INFO", log_file=str(log_file))


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def _make_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "sample.csv"
        p.write_text("a,b,c\n1,2,3\n4,5,6\n")
        return p

    def test_loads_csv(self, tmp_path: Path):
        path = self._make_csv(tmp_path)
        df = load_dataset(path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)

    def test_loads_parquet(self, tmp_path: Path):
        p = tmp_path / "sample.parquet"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_parquet(p, index=False)
        df = load_dataset(p)
        assert df.shape == (2, 2)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "ghost.csv")

    def test_unsupported_format_raises(self, tmp_path: Path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(p)

    def test_kwargs_forwarded(self, tmp_path: Path):
        path = self._make_csv(tmp_path)
        df = load_dataset(path, nrows=1)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_passed_check(self):
        r = CheckResult(check_name="missing_values", passed=True, severity="info", message="No missing values.")
        assert r.passed is True
        assert r.severity == "info"
        assert r.affected_columns == []
        assert r.details == {}

    def test_failed_check_with_details(self):
        r = CheckResult(
            check_name="duplicates",
            passed=False,
            severity="warning",
            message="3 duplicate rows found.",
            details={"duplicate_count": 3},
            affected_columns=["id"],
        )
        assert r.passed is False
        assert r.details["duplicate_count"] == 3

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity must be one of"):
            CheckResult(check_name="x", passed=True, severity="critical")

    def test_to_dict_keys(self):
        r = CheckResult(check_name="test", passed=True)
        d = r.to_dict()
        assert set(d.keys()) == {"check_name", "passed", "severity", "message", "details", "affected_columns"}


# ---------------------------------------------------------------------------
# FrameworkReport
# ---------------------------------------------------------------------------

class TestFrameworkReport:
    def _make_report(self) -> FrameworkReport:
        quality = [
            CheckResult("missing_values", passed=True, severity="info"),
            CheckResult("duplicates", passed=False, severity="warning", message="Found dupes"),
        ]
        leakage = [
            CheckResult("target_leakage", passed=False, severity="error", message="Leak detected"),
        ]
        return FrameworkReport(
            dataset_name="test_dataset",
            quality_results=quality,
            leakage_results=leakage,
        )

    def test_default_timestamp_is_datetime(self):
        r = FrameworkReport(dataset_name="ds")
        assert isinstance(r.run_timestamp, datetime)

    def test_all_results_aggregates(self):
        report = self._make_report()
        assert len(report.all_results()) == 3

    def test_failed_checks(self):
        report = self._make_report()
        failed = report.failed_checks()
        assert len(failed) == 2
        assert all(not r.passed for r in failed)

    def test_summary_counts(self):
        report = self._make_report()
        s = report.summary()
        assert s["total_checks"] == 3
        assert s["passed"] == 1
        assert s["failed"] == 2
        assert s["errors"] == 1
        assert s["warnings"] == 1

    def test_summary_contains_dataset_name(self):
        report = FrameworkReport(dataset_name="my_data")
        assert report.summary()["dataset_name"] == "my_data"
