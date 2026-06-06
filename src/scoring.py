"""Dataset Readiness Score — composite 0-100 score across all dimensions (Phase 10)."""

from __future__ import annotations
from typing import Any

from .utils import CheckResult, DimensionScore, FrameworkReport, ReadinessScore, logger

# ---------------------------------------------------------------------------
# Default penalty weights (overridable via config.yaml scoring block)
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY_PENALTY: dict[str, float] = {"error": 15.0, "warning": 5.0, "info": 0.0}

_DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "quality":     0.25,
    "leakage":     0.35,
    "features":    0.25,
    "sufficiency": 0.15,
}

_DEFAULT_LEAKAGE_MULTIPLIER: float = 1.5

_DEFAULT_GRADE_THRESHOLDS: list[tuple[int, str, str]] = [
    (85, "A", "Dataset is ready for ML training"),
    (70, "B", "Dataset is mostly ready — minor issues detected"),
    (55, "C", "Significant issues — address before training"),
    (40, "D", "Major issues — unreliable model performance expected"),
    (0,  "F", "Dataset is not suitable for ML training"),
]


def _resolve_config(config: dict[str, Any] | None) -> tuple[
    dict[str, float], dict[str, float], float, list[tuple[int, str, str]]
]:
    """Extract scoring parameters from config block, falling back to defaults."""
    cfg = config or {}
    penalties = {**_DEFAULT_SEVERITY_PENALTY,
                 **cfg.get("severity_penalties", {})}
    weights = {**_DEFAULT_DIMENSION_WEIGHTS,
               **cfg.get("dimension_weights", {})}
    multiplier: float = float(cfg.get("leakage_multiplier", _DEFAULT_LEAKAGE_MULTIPLIER))
    raw_thr = cfg.get("grade_thresholds", {})
    if raw_thr:
        grade_map = {"A": 85, "B": 70, "C": 55, "D": 40, **raw_thr}
        thresholds = [
            (int(grade_map["A"]), "A", "Dataset is ready for ML training"),
            (int(grade_map["B"]), "B", "Dataset is mostly ready --- minor issues detected"),
            (int(grade_map["C"]), "C", "Significant issues --- address before training"),
            (int(grade_map["D"]), "D", "Major issues --- unreliable model performance expected"),
            (0,                   "F", "Dataset is not suitable for ML training"),
        ]
    else:
        thresholds = _DEFAULT_GRADE_THRESHOLDS
    return penalties, weights, multiplier, thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_results(
    results: list[CheckResult],
    multiplier: float = 1.0,
    severity_penalty: dict[str, float] | None = None,
) -> DimensionScore:
    """Convert a list of CheckResults into a DimensionScore."""
    penalties = severity_penalty if severity_penalty is not None else _DEFAULT_SEVERITY_PENALTY
    if not results:
        return DimensionScore(
            score=100.0, checks_total=0, checks_passed=0, errors=0, warnings=0
        )

    penalty = sum(
        penalties.get(r.severity, 0.0) * multiplier
        for r in results
        if not r.passed
    )
    score = max(0.0, round(100.0 - penalty, 1))

    return DimensionScore(
        score=score,
        checks_total=len(results),
        checks_passed=sum(1 for r in results if r.passed),
        errors=sum(1 for r in results if not r.passed and r.severity == "error"),
        warnings=sum(1 for r in results if not r.passed and r.severity == "warning"),
    )


def _sufficiency_score(metadata: dict) -> DimensionScore:
    """Derive a data-sufficiency score from dataset shape and structure.

    Rules (heuristic, inspired by common ML practice):
      - n / p ≥ 100 → 100 pts  (very comfortable)
      - n / p ≥  50 → 85 pts
      - n / p ≥  20 → 70 pts
      - n / p ≥  10 → 50 pts
      - n / p <  10 → 25 pts  (likely overfitting territory)

    where n = number of rows, p = number of features (columns − 1).
    """
    shape = metadata.get("shape", [0, 0])
    n_rows  = shape[0] if len(shape) > 0 else 0
    n_cols  = shape[1] if len(shape) > 1 else 0
    n_feat  = max(n_cols - 1, 1)

    if n_rows == 0:
        return DimensionScore(score=50.0, checks_total=1, checks_passed=0, errors=0, warnings=1)

    ratio = n_rows / n_feat

    if ratio >= 100:
        score, passed = 100.0, True
    elif ratio >= 50:
        score, passed = 85.0, True
    elif ratio >= 20:
        score, passed = 70.0, True
    elif ratio >= 10:
        score, passed = 50.0, False
    else:
        score, passed = 25.0, False

    return DimensionScore(
        score=score,
        checks_total=1,
        checks_passed=1 if passed else 0,
        errors=0,
        warnings=0 if passed else 1,
    )


def _derive_grade(
    score: float,
    thresholds: list[tuple[int, str, str]] | None = None,
) -> tuple[str, str]:
    thr = thresholds if thresholds is not None else _DEFAULT_GRADE_THRESHOLDS
    for threshold, grade, label in thr:
        if score >= threshold:
            return grade, label
    return "F", thr[-1][2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_readiness_score(
    report: FrameworkReport,
    config: dict[str, Any] | None = None,
) -> ReadinessScore:
    """Compute the composite Dataset Readiness Score from a FrameworkReport.

    All weights and penalties are configurable via the ``scoring`` block in
    ``config.yaml`` (or the ``config`` argument).  When not supplied the
    built-in defaults are used:

    - **Quality (25 %)**: penalises dataset quality failures.
    - **Leakage (35 %)**: penalises leakage detections with 1.5× multiplier.
    - **Features (25 %)**: penalises redundant or irrelevant features.
    - **Sufficiency (15 %)**: derived from the rows/features ratio.

    Penalty per failed check: error = −15 pts, warning = −5 pts (before multiplier).

    Args:
        report: Fully populated FrameworkReport.
        config: Optional ``scoring`` config block.  Keys: ``dimension_weights``,
            ``severity_penalties``, ``leakage_multiplier``, ``grade_thresholds``.

    Returns:
        :class:`~src.utils.ReadinessScore` with overall score, letter grade,
        and per-dimension breakdown.
    """
    penalties, weights, leakage_mult, grade_thr = _resolve_config(config)

    quality     = _score_results(report.quality_results,  1.0,           penalties)
    leakage     = _score_results(report.leakage_results,  leakage_mult,  penalties)
    features    = _score_results(report.feature_results,  1.0,           penalties)
    sufficiency = (
        _score_results(report.sufficiency_results, 1.0, penalties)
        if report.sufficiency_results
        else _sufficiency_score(report.metadata)
    )

    overall = round(
        quality.score       * weights["quality"]
        + leakage.score     * weights["leakage"]
        + features.score    * weights["features"]
        + sufficiency.score * weights["sufficiency"],
        1,
    )
    grade, label = _derive_grade(overall, grade_thr)

    score = ReadinessScore(
        overall=overall,
        grade=grade,
        label=label,
        quality=quality,
        leakage=leakage,
        features=features,
        sufficiency=sufficiency,
    )

    logger.info(
        f"Readiness Score: {overall}/100  Grade: {grade}  — {label}"
    )
    return score
