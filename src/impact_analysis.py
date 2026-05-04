"""Impact analysis: measures how quality/leakage issues affect ML performance (Phase 5)."""

from __future__ import annotations

import pandas as pd

from .utils import CheckResult, FrameworkReport


def run_impact_analysis(
    df: pd.DataFrame,
    report: FrameworkReport,
    config: dict,
) -> list[CheckResult]:
    """Quantify the performance impact of detected issues.

    Args:
        df: Input DataFrame.
        report: Partially populated FrameworkReport (quality + leakage results).
        config: Impact-analysis configuration block from the YAML file.

    Returns:
        List of CheckResult describing performance deltas.
    """
    raise NotImplementedError("Impact analysis will be implemented in Phase 5.")
