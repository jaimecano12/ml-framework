"""Tests for src/semantic_leakage.py (Phase 16 — LLM semantic analysis)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.semantic_leakage import (
    SemanticRiskAssessment,
    _build_user_prompt,
    _parse_llm_response,
    analyse_semantic_leakage,
)
from src.utils import CheckResult


@pytest.fixture()
def df_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age":            rng.integers(20, 70, n).astype(float),
        "income":         rng.normal(50_000, 10_000, n),
        "discharge_code": rng.integers(100, 999, n),   # semantically post-hoc
        "future_usage":   rng.normal(10, 2, n),        # semantically temporal
        "target":         rng.integers(0, 2, n),
    })


_MOCK_LLM_RESPONSE = json.dumps([
    {
        "feature_name": "age",
        "risk_level": "none",
        "leakage_type": "none",
        "reasoning": "Patient age is available at admission time.",
        "recommendation": "",
    },
    {
        "feature_name": "income",
        "risk_level": "none",
        "leakage_type": "none",
        "reasoning": "Income is a pre-event demographic feature.",
        "recommendation": "",
    },
    {
        "feature_name": "discharge_code",
        "risk_level": "high",
        "leakage_type": "post_hoc",
        "reasoning": "Discharge codes are assigned at the end of the visit, after the outcome.",
        "recommendation": "Remove discharge_code from the feature set.",
    },
    {
        "feature_name": "future_usage",
        "risk_level": "high",
        "leakage_type": "temporal",
        "reasoning": "Name suggests usage measured after the prediction window.",
        "recommendation": "Verify that future_usage is computed before the event date.",
    },
])


class TestParseResponse:
    def test_parses_valid_json_array(self):
        assessments = _parse_llm_response(_MOCK_LLM_RESPONSE)
        assert len(assessments) == 4
        assert all(isinstance(a, SemanticRiskAssessment) for a in assessments)

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{_MOCK_LLM_RESPONSE}\n```"
        assessments = _parse_llm_response(fenced)
        assert len(assessments) == 4

    def test_risk_levels_parsed_correctly(self):
        assessments = _parse_llm_response(_MOCK_LLM_RESPONSE)
        risk_map = {a.feature_name: a.risk_level for a in assessments}
        assert risk_map["age"] == "none"
        assert risk_map["discharge_code"] == "high"
        assert risk_map["future_usage"] == "high"

    def test_leakage_types_parsed(self):
        assessments = _parse_llm_response(_MOCK_LLM_RESPONSE)
        types = {a.feature_name: a.leakage_type for a in assessments}
        assert types["discharge_code"] == "post_hoc"
        assert types["future_usage"] == "temporal"


class TestBuildPrompt:
    def test_contains_feature_names(self, df_sample: pd.DataFrame):
        features = [c for c in df_sample.columns if c != "target"]
        prompt = _build_user_prompt(features, "target", "ICU dataset", {f: [] for f in features})
        for feat in features:
            assert feat in prompt

    def test_contains_target_col(self, df_sample: pd.DataFrame):
        features = [c for c in df_sample.columns if c != "target"]
        prompt = _build_user_prompt(features, "target", "description", {f: [] for f in features})
        assert "target" in prompt

    def test_contains_dataset_description(self, df_sample: pd.DataFrame):
        features = [c for c in df_sample.columns if c != "target"]
        prompt = _build_user_prompt(features, "target", "ICU mortality study", {f: [] for f in features})
        assert "ICU mortality study" in prompt


class TestAnalyseSemanticLeakage:
    _cfg = {"enabled": True, "deployment": "gpt-4o-mini",
            "max_features": 30, "risk_threshold": "medium"}

    def test_disabled_returns_passing_result(self, df_sample: pd.DataFrame):
        r = analyse_semantic_leakage(df_sample, "target", {"enabled": False})
        assert r.passed
        assert r.check_name == "semantic_leakage"

    def test_skips_gracefully_when_no_credentials(self, df_sample: pd.DataFrame):
        import os
        env_backup = {k: os.environ.pop(k, None) for k in
                      ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]}
        try:
            r = analyse_semantic_leakage(df_sample, "target", self._cfg)
            assert r.passed
            assert "skipped" in r.message.lower()
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    @patch("src.semantic_leakage._call_llm", return_value=_MOCK_LLM_RESPONSE)
    def test_detects_high_risk_features(self, mock_llm, df_sample: pd.DataFrame):
        r = analyse_semantic_leakage(df_sample, "target", self._cfg)
        assert not r.passed
        assert "discharge_code" in r.affected_columns
        assert "future_usage" in r.affected_columns

    @patch("src.semantic_leakage._call_llm", return_value=_MOCK_LLM_RESPONSE)
    def test_safe_features_not_flagged(self, mock_llm, df_sample: pd.DataFrame):
        r = analyse_semantic_leakage(df_sample, "target", self._cfg)
        assert "age" not in r.affected_columns
        assert "income" not in r.affected_columns

    @patch("src.semantic_leakage._call_llm", return_value=_MOCK_LLM_RESPONSE)
    def test_severity_error_when_high_risk_found(self, mock_llm, df_sample: pd.DataFrame):
        r = analyse_semantic_leakage(df_sample, "target", self._cfg)
        assert r.severity == "error"

    @patch("src.semantic_leakage._call_llm", return_value=_MOCK_LLM_RESPONSE)
    def test_details_contain_assessments(self, mock_llm, df_sample: pd.DataFrame):
        r = analyse_semantic_leakage(df_sample, "target", self._cfg)
        assert "assessments" in r.details
        assert len(r.details["assessments"]) == 4

    @patch("src.semantic_leakage._call_llm", side_effect=Exception("API error"))
    def test_llm_error_returns_failed_result(self, mock_llm, df_sample: pd.DataFrame):
        import os
        os.environ["AZURE_OPENAI_API_KEY"] = "dummy"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dummy.openai.azure.com/"
        try:
            r = analyse_semantic_leakage(df_sample, "target", self._cfg)
            assert not r.passed
            assert "error" in r.details
        finally:
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

    @patch("src.semantic_leakage._call_llm", return_value=_MOCK_LLM_RESPONSE)
    def test_high_threshold_reduces_flagged_features(self, mock_llm, df_sample: pd.DataFrame):
        cfg = {**self._cfg, "risk_threshold": "high"}
        r = analyse_semantic_leakage(df_sample, "target", cfg)
        # Only "high" risk should be flagged (no "medium")
        assert all(
            a["risk_level"] == "high"
            for a in r.details.get("assessments", [])
            if a["feature_name"] in r.affected_columns
        )
