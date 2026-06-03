"""LLM-assisted semantic leakage analysis (Phase 16).

Uses GPT-4o-mini via Azure OpenAI to detect *implicit* leakage that purely
statistical methods cannot catch: features whose *names* or *semantics*
suggest they encode future or post-hoc information about the target.

Setup (Azure):
    export AZURE_OPENAI_API_KEY="<your-key>"
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"   # or your deployment name

Fallback:
    If Azure credentials are not configured the module returns a
    ``CheckResult`` with ``passed=True`` and a message explaining that the
    LLM analysis was skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .utils import CheckResult, logger


# ---------------------------------------------------------------------------
# Dataclass for per-feature LLM assessment
# ---------------------------------------------------------------------------

@dataclass
class SemanticRiskAssessment:
    """LLM verdict for a single feature."""

    feature_name: str
    risk_level: str          # "none" | "low" | "medium" | "high"
    reasoning: str
    leakage_type: str        # "temporal" | "proxy" | "post_hoc" | "indirect" | "none"
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "risk_level":   self.risk_level,
            "leakage_type": self.leakage_type,
            "reasoning":    self.reasoning,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a machine learning data quality expert specialising in data leakage detection.
Your task is to analyse feature names and dataset context to identify features that may implicitly
contain future or post-hoc information about the prediction target — a form of semantic data leakage
that purely statistical methods (correlation, mutual information) cannot always detect.

For each feature you must assess:
1. Whether the feature name suggests it encodes information that would only be available AFTER
   the prediction event (temporal leakage).
2. Whether the feature may have been computed FROM the target variable (indirect/proxy leakage).
3. Whether the feature is a post-hoc label or administrative code assigned after the outcome.

Return ONLY a valid JSON array. Each element has:
  {
    "feature_name": "<name>",
    "risk_level": "none" | "low" | "medium" | "high",
    "leakage_type": "none" | "temporal" | "proxy" | "post_hoc" | "indirect",
    "reasoning": "<1-2 sentence explanation>",
    "recommendation": "<brief action if risk > none, else empty string>"
  }
Do not include markdown fences or any text outside the JSON array."""


def _build_user_prompt(
    feature_names: list[str],
    target_col: str,
    dataset_description: str,
    sample_values: dict[str, list],
) -> str:
    samples_str = json.dumps(
        {k: v[:5] for k, v in sample_values.items()},
        default=str,
        indent=2,
    )
    return (
        f"Dataset description: {dataset_description}\n\n"
        f"Target column: '{target_col}'\n\n"
        f"Features to analyse: {feature_names}\n\n"
        f"Sample values (first 5 rows per feature):\n{samples_str}\n\n"
        "Analyse each feature for semantic leakage risk."
    )


# ---------------------------------------------------------------------------
# Azure OpenAI client (optional dependency)
# ---------------------------------------------------------------------------

def _get_client():
    """Return an AzureOpenAI client or raise ImportError / EnvironmentError."""
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package not installed. Run: python -m pip install openai"
        ) from exc

    api_key  = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        raise EnvironmentError(
            "Azure OpenAI credentials not configured. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-01",
    )


def _call_llm(prompt_user: str, deployment: str, max_tokens: int = 2000) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_user},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def _parse_llm_response(raw: str) -> list[SemanticRiskAssessment]:
    """Parse JSON array from LLM response into SemanticRiskAssessment objects."""
    # Strip possible markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    items = json.loads(cleaned)
    assessments = []
    for item in items:
        assessments.append(SemanticRiskAssessment(
            feature_name=item.get("feature_name", ""),
            risk_level=item.get("risk_level", "none"),
            leakage_type=item.get("leakage_type", "none"),
            reasoning=item.get("reasoning", ""),
            recommendation=item.get("recommendation", ""),
        ))
    return assessments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_semantic_leakage(
    df: pd.DataFrame,
    target_col: str,
    config: dict,
    dataset_description: str = "",
) -> CheckResult:
    """Detect implicit semantic leakage using an LLM.

    The LLM is given feature names, their sample values, and a dataset
    description, and asked to flag features that semantically suggest
    future information or post-hoc computation.

    Args:
        df: Input DataFrame.
        target_col: Name of the target / label column.
        config: The ``semantic_leakage`` config block.
            Relevant keys: ``deployment`` (str, default "gpt-4o-mini"),
            ``max_features`` (int, default 30), ``risk_threshold`` ("medium"|"high").
        dataset_description: Free-text description of the dataset context (optional).

    Returns:
        A :class:`~src.utils.CheckResult` with semantic leakage findings,
        or a passing result if LLM is unavailable / disabled.
    """
    if not config.get("enabled", True):
        return CheckResult(
            check_name="semantic_leakage",
            passed=True,
            severity="info",
            message="Semantic leakage analysis disabled.",
            details={},
        )

    feature_cols = [c for c in df.columns if c != target_col]
    max_features: int = config.get("max_features", 30)
    feature_cols = feature_cols[:max_features]

    deployment: str = config.get("deployment", os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"))
    risk_threshold: str = config.get("risk_threshold", "medium")

    sample_values = {col: df[col].dropna().head(5).tolist() for col in feature_cols}

    try:
        logger.info(f"Running LLM semantic leakage analysis ({len(feature_cols)} features)…")
        user_prompt = _build_user_prompt(feature_cols, target_col, dataset_description, sample_values)
        raw_response = _call_llm(user_prompt, deployment)
        assessments = _parse_llm_response(raw_response)
    except (ImportError, EnvironmentError) as exc:
        logger.warning(f"Semantic leakage analysis skipped: {exc}")
        return CheckResult(
            check_name="semantic_leakage",
            passed=True,
            severity="info",
            message=f"Semantic leakage analysis skipped: {exc}",
            details={"skipped_reason": str(exc)},
        )
    except Exception as exc:
        logger.error(f"Semantic leakage analysis failed: {exc}")
        return CheckResult(
            check_name="semantic_leakage",
            passed=False,
            severity="warning",
            message=f"Semantic leakage analysis failed with error: {exc}",
            details={"error": str(exc)},
        )

    # Determine threshold index
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    threshold_idx = risk_order.get(risk_threshold, 2)

    flagged = [a for a in assessments if risk_order.get(a.risk_level, 0) >= threshold_idx]
    flagged_cols = [a.feature_name for a in flagged]

    details = {
        "assessments": [a.to_dict() for a in assessments],
        "risk_threshold": risk_threshold,
        "deployment": deployment,
        "total_features_analysed": len(assessments),
    }

    if not flagged:
        return CheckResult(
            check_name="semantic_leakage",
            passed=True,
            severity="info",
            message=(
                f"LLM found no semantic leakage risk >= '{risk_threshold}' "
                f"across {len(assessments)} features."
            ),
            details=details,
        )

    high_risk = [a for a in flagged if a.risk_level == "high"]
    severity = "error" if high_risk else "warning"

    return CheckResult(
        check_name="semantic_leakage",
        passed=False,
        severity=severity,
        message=(
            f"LLM flagged {len(flagged)} feature(s) with semantic leakage risk "
            f">= '{risk_threshold}': {flagged_cols}"
        ),
        details=details,
        affected_columns=flagged_cols,
    )
