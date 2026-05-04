"""Data quality checks (Phase 3)."""

from __future__ import annotations

import pandas as pd

from .utils import CheckResult


def run_all_quality_checks(df: pd.DataFrame, config: dict) -> list[CheckResult]:
    """Run all configured quality checks on *df*.

    Args:
        df: Input DataFrame to analyse.
        config: Quality-check configuration block from the YAML file.

    Returns:
        List of CheckResult, one per check executed.
    """
    raise NotImplementedError("Quality checks will be implemented in Phase 3.")
