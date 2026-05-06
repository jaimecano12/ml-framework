"""Tests for src/plugins.py (Phase 15)."""

from __future__ import annotations

import pytest

from src.plugins import (
    _registry,
    clear_registry,
    get_checks,
    load_plugins,
    register_check,
    run_plugin_checks,
)
from src.utils import CheckResult


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure the registry is empty before every test."""
    clear_registry()
    yield
    clear_registry()


class TestRegisterCheck:
    def test_registers_function_in_correct_phase(self):
        @register_check(phase="quality", name="my_check")
        def dummy(df, target_col, config):
            return CheckResult("my_check", passed=True)

        assert any(e["name"] == "my_check" for e in get_checks("quality"))

    def test_registered_function_is_callable(self):
        @register_check(phase="quality", name="test_fn")
        def fn(df, target_col, config):
            return CheckResult("test_fn", passed=True)

        checks = get_checks("quality")
        entry = next(e for e in checks if e["name"] == "test_fn")
        assert callable(entry["fn"])

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="phase must be one of"):
            @register_check(phase="nonexistent", name="x")
            def fn(df, tc, cfg):
                pass

    def test_duplicate_name_overwrites(self):
        @register_check(phase="quality", name="dup")
        def fn1(df, tc, cfg):
            return CheckResult("dup", passed=True)

        @register_check(phase="quality", name="dup")
        def fn2(df, tc, cfg):
            return CheckResult("dup", passed=False)

        checks = get_checks("quality")
        entries = [e for e in checks if e["name"] == "dup"]
        assert len(entries) == 1  # only one after overwrite


class TestGetChecks:
    def test_returns_empty_for_unregistered_phase(self):
        assert get_checks("quality") == []

    def test_returns_registered_checks(self):
        @register_check(phase="leakage", name="lk_check")
        def fn(df, tc, cfg):
            pass

        assert len(get_checks("leakage")) == 1

    def test_does_not_cross_contaminate_phases(self):
        @register_check(phase="quality", name="q_check")
        def fn(df, tc, cfg):
            pass

        assert get_checks("leakage") == []


class TestRunPluginChecks:
    def test_runs_registered_check(self):
        import pandas as pd

        @register_check(phase="quality", name="always_pass")
        def fn(df, tc, cfg):
            return CheckResult("always_pass", passed=True, severity="info")

        df = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
        results = run_plugin_checks("quality", df, "target", {})
        assert len(results) == 1
        assert results[0].check_name == "always_pass"

    def test_disabled_check_skipped(self):
        import pandas as pd

        @register_check(phase="quality", name="skipped_check")
        def fn(df, tc, cfg):
            return CheckResult("skipped_check", passed=True)

        df = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
        results = run_plugin_checks("quality", df, "target",
                                    {"skipped_check": {"enabled": False}})
        assert len(results) == 0

    def test_exception_in_plugin_does_not_crash(self):
        import pandas as pd

        @register_check(phase="quality", name="crashing_check")
        def fn(df, tc, cfg):
            raise RuntimeError("intentional error")

        df = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
        # Should not raise
        results = run_plugin_checks("quality", df, "target", {})
        assert len(results) == 0


class TestLoadPlugins:
    def test_load_nonexistent_module_logs_warning(self):
        # Should not raise, just log a warning
        load_plugins(["nonexistent_module_xyz"])
