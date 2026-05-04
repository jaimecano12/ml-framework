"""Tests for src/config.py (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import _deep_merge, get_section, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


def _minimal(tmp_path: Path) -> Path:
    """Minimal valid config — only required fields."""
    return _write_config(tmp_path, {
        "dataset": {"path": "data/raw/ds.csv", "target_column": "label"}
    })


# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_file(self, tmp_path: Path):
        cfg = load_config(_minimal(tmp_path))
        assert cfg["dataset"]["path"] == "data/raw/ds.csv"
        assert cfg["dataset"]["target_column"] == "label"

    def test_defaults_are_applied(self, tmp_path: Path):
        cfg = load_config(_minimal(tmp_path))
        assert cfg["dataset"]["separator"] == ","
        assert cfg["dataset"]["encoding"] == "utf-8"
        assert cfg["logging"]["level"] == "INFO"
        assert cfg["logging"]["file"] is None

    def test_all_top_level_sections_present(self, tmp_path: Path):
        cfg = load_config(_minimal(tmp_path))
        for section in ("dataset", "logging", "quality_checks", "leakage_checks",
                        "impact_analysis", "reporting"):
            assert section in cfg, f"Section '{section}' missing from config"

    def test_override_replaces_default(self, tmp_path: Path):
        p = _write_config(tmp_path, {
            "dataset": {"path": "d.csv", "target_column": "y"},
            "logging": {"level": "DEBUG"},
        })
        cfg = load_config(p)
        assert cfg["logging"]["level"] == "DEBUG"

    def test_partial_section_override_preserves_sibling_defaults(self, tmp_path: Path):
        p = _write_config(tmp_path, {
            "dataset": {"path": "d.csv", "target_column": "y"},
            "quality_checks": {"missing_values": {"threshold": 0.10}},
        })
        cfg = load_config(p)
        assert cfg["quality_checks"]["missing_values"]["threshold"] == 0.10
        assert cfg["quality_checks"]["duplicates"]["enabled"] is True

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "ghost.yaml")

    def test_empty_yaml_raises_validation_error(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="dataset.path is required"):
            load_config(p)

    def test_loads_project_config_yaml(self):
        """Smoke-test that the checked-in configs/config.yaml is valid."""
        cfg = load_config(Path("configs/config.yaml"))
        assert cfg["dataset"]["target_column"] == "target"


# ---------------------------------------------------------------------------
# load_config — validation errors
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _make(self, tmp_path: Path, overrides: dict) -> Path:
        base = {"dataset": {"path": "d.csv", "target_column": "y"}}
        base.update(overrides)
        return _write_config(tmp_path, base)

    def test_missing_path_raises(self, tmp_path: Path):
        p = _write_config(tmp_path, {"dataset": {"target_column": "y"}})
        with pytest.raises(ValueError, match="dataset.path"):
            load_config(p)

    def test_missing_target_column_raises(self, tmp_path: Path):
        p = _write_config(tmp_path, {"dataset": {"path": "d.csv"}})
        with pytest.raises(ValueError, match="dataset.target_column"):
            load_config(p)

    def test_invalid_log_level_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"logging": {"level": "VERBOSE"}})
        with pytest.raises(ValueError, match="logging.level"):
            load_config(p)

    def test_invalid_outlier_method_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"quality_checks": {"outliers": {"method": "mad"}}})
        with pytest.raises(ValueError, match="outliers.method"):
            load_config(p)

    def test_threshold_out_of_range_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"quality_checks": {"missing_values": {"threshold": 1.5}}})
        with pytest.raises(ValueError, match="missing_values.threshold"):
            load_config(p)

    def test_invalid_model_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"impact_analysis": {"models": ["svm"]}})
        with pytest.raises(ValueError, match="unknown model"):
            load_config(p)

    def test_invalid_metric_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"impact_analysis": {"metrics": ["mse"]}})
        with pytest.raises(ValueError, match="unknown metric"):
            load_config(p)

    def test_test_size_out_of_range_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"impact_analysis": {"test_size": 1.5}})
        with pytest.raises(ValueError, match="test_size"):
            load_config(p)

    def test_cv_folds_less_than_2_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"impact_analysis": {"cv_folds": 1}})
        with pytest.raises(ValueError, match="cv_folds"):
            load_config(p)

    def test_invalid_report_format_raises(self, tmp_path: Path):
        p = self._make(tmp_path, {"reporting": {"format": "pdf"}})
        with pytest.raises(ValueError, match="reporting.format"):
            load_config(p)

    def test_multiple_errors_reported_together(self, tmp_path: Path):
        p = _write_config(tmp_path, {
            "dataset": {},
            "logging": {"level": "NOPE"},
        })
        with pytest.raises(ValueError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        assert "dataset.path" in msg
        assert "logging.level" in msg


# ---------------------------------------------------------------------------
# get_section
# ---------------------------------------------------------------------------

class TestGetSection:
    def test_returns_section(self, tmp_path: Path):
        cfg = load_config(_minimal(tmp_path))
        section = get_section(cfg, "quality_checks")
        assert "missing_values" in section

    def test_missing_section_raises_key_error(self, tmp_path: Path):
        cfg = load_config(_minimal(tmp_path))
        with pytest.raises(KeyError, match="nonexistent"):
            get_section(cfg, "nonexistent")


# ---------------------------------------------------------------------------
# _deep_merge (internal utility)
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_override_wins_on_scalar(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_base_key_preserved_when_not_overridden(self):
        result = _deep_merge({"a": 1, "b": 2}, {"a": 99})
        assert result["b"] == 2

    def test_nested_dicts_merged_recursively(self):
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 99, "c": 3}}
        result = _deep_merge(base, over)
        assert result["x"] == {"a": 1, "b": 99, "c": 3}

    def test_base_is_not_mutated(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1

    def test_list_is_replaced_not_merged(self):
        result = _deep_merge({"a": [1, 2]}, {"a": [3]})
        assert result["a"] == [3]
