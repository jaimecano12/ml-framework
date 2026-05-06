"""Dataset Readiness Score — composite 0-100 score across all dimensions (Phase 10)."""

from __future__ import annotations

from .utils import CheckResult, DimensionScore, FrameworkReport, ReadinessScore, logger

# ---------------------------------------------------------------------------
# Penalty weights
# ---------------------------------------------------------------------------

_SEVERITY_PENALTY = {"error": 15.0, "warning": 5.0, "info": 0.0}

# Leakage issues are weighted more heavily: hidden leakage is worse than known quality issues
_DIMENSION_MULTIPLIERS = {
    "quality":     1.0,
    "leakage":     1.5,
    "features":    1.0,
}

# Contribution of each dimension to the overall score
_DIMENSION_WEIGHTS = {
    "quality":     0.25,
    "leakage":     0.35,
    "features":    0.25,
    "sufficiency": 0.15,
}

_GRADE_THRESHOLDS = [
    (85, "A", "Dataset is ready for ML training"),
    (70, "B", "Dataset is mostly ready — minor issues detected"),
    (55, "C", "Significant issues — address before training"),
    (40, "D", "Major issues — unreliable model performance expected"),
    (0,  "F", "Dataset is not suitable for ML training"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_results(
    results: list[CheckResult],
    multiplier: float = 1.0,
) -> DimensionScore:
    """Convert a list of CheckResults into a DimensionScore."""
    if not results:
        return DimensionScore(
            score=100.0, checks_total=0, checks_passed=0, errors=0, warnings=0
        )

    penalty = sum(
        _SEVERITY_PENALTY.get(r.severity, 0.0) * multiplier
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


def _derive_grade(score: float) -> tuple[str, str]:
    for threshold, grade, label in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade, label
    return "F", _GRADE_THRESHOLDS[-1][2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_readiness_score(report: FrameworkReport) -> ReadinessScore:
    """Compute the composite Dataset Readiness Score from a FrameworkReport.

    The score aggregates four weighted dimensions:

    - **Quality (25 %)**: penalises dataset quality failures.
    - **Leakage (35 %)**: penalises leakage detections with 1.5× multiplier.
    - **Features (25 %)**: penalises redundant or irrelevant features.
    - **Sufficiency (15 %)**: derived from the rows/features ratio.

    Penalty per failed check: error = −15 pts, warning = −5 pts (before multiplier).

    Args:
        report: Fully populated FrameworkReport.

    Returns:
        :class:`~src.utils.ReadinessScore` with overall score, letter grade,
        and per-dimension breakdown.
    """
    quality     = _score_results(report.quality_results,  _DIMENSION_MULTIPLIERS["quality"])
    leakage     = _score_results(report.leakage_results,  _DIMENSION_MULTIPLIERS["leakage"])
    features    = _score_results(report.feature_results,  _DIMENSION_MULTIPLIERS["features"])
    # Phase 11: use actual sufficiency check results when available
    sufficiency = (
        _score_results(report.sufficiency_results)
        if report.sufficiency_results
        else _sufficiency_score(report.metadata)
    )

    overall = round(
        quality.score     * _DIMENSION_WEIGHTS["quality"]
        + leakage.score   * _DIMENSION_WEIGHTS["leakage"]
        + features.score  * _DIMENSION_WEIGHTS["features"]
        + sufficiency.score * _DIMENSION_WEIGHTS["sufficiency"],
        1,
    )
    grade, label = _derive_grade(overall)

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
