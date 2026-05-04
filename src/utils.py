"""Shared utilities: logging, data loading, and core result/report dataclasses."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure loguru logger for the framework.

    Args:
        log_level: Minimum severity level to emit (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a file sink. When None only stderr is used.
    """
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level=log_level, colorize=True)
    if log_file:
        logger.add(log_file, format=fmt, level=log_level, rotation="10 MB", retention="7 days")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, or Excel into a DataFrame.

    Args:
        path: File path to the dataset.
        **kwargs: Extra keyword arguments forwarded to the underlying pandas reader.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: {list(readers)}")

    df = readers[suffix](path, **kwargs)
    logger.info(f"Loaded dataset '{path.name}': {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Core result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result produced by any single quality or leakage check.

    Attributes:
        check_name: Identifier of the check (e.g. 'missing_values').
        passed: True when no issue was detected.
        severity: One of 'info', 'warning', 'error'.
        message: Human-readable summary of the finding.
        details: Arbitrary structured data (column names, counts, …).
        affected_columns: Columns implicated in the finding.
    """

    check_name: str
    passed: bool
    severity: str = "info"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    affected_columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_severities = {"info", "warning", "error"}
        if self.severity not in valid_severities:
            raise ValueError(f"severity must be one of {valid_severities}, got '{self.severity}'")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "affected_columns": self.affected_columns,
        }


# ---------------------------------------------------------------------------
# Framework report dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrameworkReport:
    """Aggregated report produced by the full framework run.

    Attributes:
        dataset_name: Display name for the analysed dataset.
        run_timestamp: UTC timestamp of when the run was started.
        quality_results: Results from the quality-check phase.
        leakage_results: Results from the leakage-detection phase.
        impact_results: Results from the impact-analysis phase.
        metadata: Arbitrary top-level metadata (shape, dtypes summary, …).
    """

    dataset_name: str
    run_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_results: list[CheckResult] = field(default_factory=list)
    leakage_results: list[CheckResult] = field(default_factory=list)
    impact_results: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def all_results(self) -> list[CheckResult]:
        """Return every CheckResult across all phases."""
        return self.quality_results + self.leakage_results + self.impact_results

    def failed_checks(self) -> list[CheckResult]:
        """Return only the checks that did not pass."""
        return [r for r in self.all_results() if not r.passed]

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary dictionary."""
        all_r = self.all_results()
        return {
            "dataset_name": self.dataset_name,
            "run_timestamp": self.run_timestamp.isoformat(),
            "total_checks": len(all_r),
            "passed": sum(1 for r in all_r if r.passed),
            "failed": sum(1 for r in all_r if not r.passed),
            "errors": sum(1 for r in all_r if r.severity == "error"),
            "warnings": sum(1 for r in all_r if r.severity == "warning"),
        }
