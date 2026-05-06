"""DatasetChecker — high-level programmatic API for the framework (Phase 13)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import _DEFAULTS, _deep_merge, _validate, get_section
from .drift_checks import run_all_drift_checks
from .feature_analysis import run_all_feature_checks
from .impact_analysis import run_impact_analysis
from .leakage_checks import run_all_leakage_checks
from .plugins import load_plugins
from .quality_checks import run_all_quality_checks
from .recommendations import generate_recommendations
from .reporting import generate_report
from .scoring import compute_readiness_score
from .sufficiency import run_all_sufficiency_checks
from .utils import CheckResult, FrameworkReport, Recommendation, ReadinessScore, load_dataset, logger


class DatasetChecker:
    """Programmatic interface to the ml-framework pipeline.

    Example usage::

        from src.checker import DatasetChecker

        checker = DatasetChecker("configs/config.yaml")
        report  = checker.run("data/titanic.csv", target_col="survived")

        print(f"Score: {checker.score}/100  Grade: {checker.grade}")
        for rec in checker.top_recommendations(priority="high"):
            print(f"  [{rec.check_name}] {rec.action}")

        checker.save_report("reports/")

    Args:
        config_path: Path to a YAML config file. When *None* the built-in
            defaults are used (only *target_col* is then required at run time).
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            from .config import load_config
            self._config: dict[str, Any] = load_config(config_path)
        else:
            self._config = _deep_merge(_DEFAULTS, {})
        self._report: FrameworkReport | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set(self, **kwargs: Any) -> "DatasetChecker":
        """Override config values using dot-notation keyword arguments.

        Examples::

            checker.set(target_col="survived")
            checker.set(leakage_checks__target_leakage__correlation_threshold=0.90)
        """
        for key, value in kwargs.items():
            keys = key.split("__")
            node = self._config
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        return self

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: pd.DataFrame | str | Path,
        target_col: str | None = None,
        *,
        skip_impact: bool = False,
    ) -> FrameworkReport:
        """Execute the full analysis pipeline.

        Args:
            dataset: DataFrame or path to CSV / Parquet / Excel file.
            target_col: Target column name. Overrides the config value if given.
            skip_impact: Skip the (slow) cross-validation impact analysis.

        Returns:
            Populated :class:`~src.utils.FrameworkReport`.
        """
        # Load data
        if isinstance(dataset, (str, Path)):
            df = load_dataset(dataset)
            self._config.setdefault("dataset", {})["path"] = str(dataset)
        else:
            df = dataset.copy()

        if target_col:
            self._config.setdefault("dataset", {})["target_column"] = target_col

        tc: str = self._config.get("dataset", {}).get("target_column", "")
        if not tc:
            raise ValueError("target_col must be provided either in config or as argument.")

        # Load any plugins declared in config
        plugin_paths = self._config.get("plugins", [])
        if plugin_paths:
            load_plugins(plugin_paths)

        report = FrameworkReport(
            dataset_name=Path(self._config.get("dataset", {}).get("path", "dataset")).stem
                         if isinstance(dataset, (str, Path)) else "dataset",
            metadata={"shape": list(df.shape), "columns": df.columns.tolist(), "target_col": tc},
        )

        report.quality_results     = run_all_quality_checks(df, tc, get_section(self._config, "quality_checks"))
        report.leakage_results     = run_all_leakage_checks(df, tc, get_section(self._config, "leakage_checks"))
        report.feature_results     = run_all_feature_checks(df, tc, get_section(self._config, "feature_analysis"))
        report.sufficiency_results = run_all_sufficiency_checks(df, tc, report, get_section(self._config, "sufficiency_checks"))
        report.drift_results       = run_all_drift_checks(df, tc, get_section(self._config, "drift_checks"))

        if not skip_impact:
            report.impact_results = run_impact_analysis(df, tc, report, get_section(self._config, "impact_analysis"))

        report.recommendations = generate_recommendations(report)
        report.readiness_score = compute_readiness_score(report)
        self._report = report
        return report

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def _require_report(self, method: str) -> FrameworkReport:
        if self._report is None:
            raise RuntimeError(f"Call .run() before accessing .{method}()")
        return self._report

    @property
    def score(self) -> float | None:
        """Overall readiness score (0–100), or *None* if not run yet."""
        if self._report and self._report.readiness_score:
            return self._report.readiness_score.overall
        return None

    @property
    def grade(self) -> str | None:
        """Letter grade A–F, or *None* if not run yet."""
        if self._report and self._report.readiness_score:
            return self._report.readiness_score.grade
        return None

    def failed_checks(self) -> list[CheckResult]:
        """Return all failed CheckResults across every phase."""
        return self._require_report("failed_checks").failed_checks()

    def top_recommendations(
        self, priority: str | None = None, n: int | None = None
    ) -> list[Recommendation]:
        """Return recommendations, optionally filtered by *priority* and capped to *n*.

        Args:
            priority: 'high', 'medium', or 'low'. When *None* all are returned.
            n: Maximum number of recommendations to return.
        """
        recs = self._require_report("top_recommendations").recommendations
        if priority:
            recs = [r for r in recs if r.priority == priority]
        return recs[:n] if n else recs

    def summary(self) -> dict[str, Any]:
        """Return the high-level summary dictionary."""
        return self._require_report("summary").summary()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_report(self, output_dir: str | Path = "reports/") -> Path:
        """Render and save the HTML report, returning its path."""
        report = self._require_report("save_report")
        return generate_report(
            report, output_dir, self._config.get("reporting", {"include_plots": True})
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise all results to a plain dictionary (JSON-friendly)."""
        report = self._require_report("to_dict")
        return {
            "summary":             report.summary(),
            "quality_results":     [r.to_dict() for r in report.quality_results],
            "leakage_results":     [r.to_dict() for r in report.leakage_results],
            "feature_results":     [r.to_dict() for r in report.feature_results],
            "sufficiency_results": [r.to_dict() for r in report.sufficiency_results],
            "drift_results":       [r.to_dict() for r in report.drift_results],
            "impact_results":      [r.to_dict() for r in report.impact_results],
            "recommendations":     [vars(r) for r in report.recommendations],
            "readiness_score":     report.readiness_score.to_dict()
                                   if report.readiness_score else None,
        }
