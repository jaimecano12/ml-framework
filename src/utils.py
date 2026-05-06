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
    """Configure loguru logger for the framework."""
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
    """Load a dataset from CSV, Parquet, or Excel into a DataFrame."""
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
    """Result produced by any single quality, leakage, or feature check."""

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
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "affected_columns": self.affected_columns,
        }


# ---------------------------------------------------------------------------
# Recommendation dataclass  (Phase 8)
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """Actionable suggestion generated from a failed CheckResult.

    Attributes:
        check_name: The check that triggered this recommendation.
        priority: 'high', 'medium', or 'low'.
        action: Short title of the corrective action.
        rationale: Why this action is needed.
        code_snippet: Runnable Python example (optional).
    """

    check_name: str
    priority: str
    action: str
    rationale: str
    code_snippet: str = ""

    def __post_init__(self) -> None:
        valid = {"high", "medium", "low"}
        if self.priority not in valid:
            raise ValueError(f"priority must be one of {valid}, got '{self.priority}'")


# ---------------------------------------------------------------------------
# Readiness score dataclasses  (Phase 10)
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score and statistics for a single readiness dimension."""

    score: float        # 0–100
    checks_total: int
    checks_passed: int
    errors: int
    warnings: int


@dataclass
class ReadinessScore:
    """Composite dataset readiness score across all analysis dimensions.

    Attributes:
        overall: Weighted overall score from 0 (unusable) to 100 (ready).
        grade: Letter grade A–F derived from the overall score.
        label: Human-readable interpretation of the grade.
        quality: Dimension score for data quality checks.
        leakage: Dimension score for leakage checks (higher weight).
        features: Dimension score for feature analysis checks.
        sufficiency: Dimension score derived from dataset size vs. complexity.
    """

    overall: float
    grade: str
    label: str
    quality: DimensionScore
    leakage: DimensionScore
    features: DimensionScore
    sufficiency: DimensionScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "grade": self.grade,
            "label": self.label,
            "quality":     vars(self.quality),
            "leakage":     vars(self.leakage),
            "features":    vars(self.features),
            "sufficiency": vars(self.sufficiency),
        }


# ---------------------------------------------------------------------------
# Framework report dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrameworkReport:
    """Aggregated report produced by the full framework run."""

    dataset_name: str
    run_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_results:      list[CheckResult]    = field(default_factory=list)
    leakage_results:      list[CheckResult]    = field(default_factory=list)
    feature_results:      list[CheckResult]    = field(default_factory=list)    # Phase 9
    sufficiency_results:  list[CheckResult]    = field(default_factory=list)    # Phase 11
    drift_results:        list[CheckResult]    = field(default_factory=list)    # Phase 14
    impact_results:       list[CheckResult]    = field(default_factory=list)
    recommendations:      list[Recommendation] = field(default_factory=list)    # Phase 8
    readiness_score:      Optional[ReadinessScore] = field(default=None)        # Phase 10
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_results(self) -> list[CheckResult]:
        """Return every CheckResult across all phases."""
        return (self.quality_results + self.leakage_results + self.feature_results
                + self.sufficiency_results + self.drift_results + self.impact_results)

    def failed_checks(self) -> list[CheckResult]:
        return [r for r in self.all_results() if not r.passed]

    def summary(self) -> dict[str, Any]:
        all_r = self.all_results()
        s: dict[str, Any] = {
            "dataset_name":   self.dataset_name,
            "run_timestamp":  self.run_timestamp.isoformat(),
            "total_checks":   len(all_r),
            "passed":  sum(1 for r in all_r if r.passed),
            "failed":  sum(1 for r in all_r if not r.passed),
            "errors":  sum(1 for r in all_r if r.severity == "error"),
            "warnings": sum(1 for r in all_r if r.severity == "warning"),
        }
        if self.readiness_score is not None:
            s["readiness_score"] = self.readiness_score.overall
            s["readiness_grade"] = self.readiness_score.grade
        return s
