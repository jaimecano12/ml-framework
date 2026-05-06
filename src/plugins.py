"""Plugin registry — allows registering custom checks without modifying framework code (Phase 15).

Usage (user plugin file, e.g. my_project/custom_checks.py):

    from ml_framework.src.plugins import register_check

    @register_check(phase="quality", name="phone_format")
    def check_phone_format(df, target_col, config):
        bad = df["phone"].str.match(r"^\\+\\d{7,15}$") == False
        if bad.sum() == 0:
            return CheckResult("phone_format", passed=True, severity="info",
                               message="All phone numbers are valid.")
        return CheckResult("phone_format", passed=False, severity="warning",
                           message=f"{bad.sum()} invalid phone numbers.",
                           affected_columns=["phone"])

Then in config.yaml:

    plugins:
      - my_project.custom_checks

    quality_checks:
      phone_format:
        enabled: true
"""

from __future__ import annotations

import importlib
from typing import Callable

from .utils import CheckResult, logger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_VALID_PHASES = {"quality", "leakage", "feature", "sufficiency", "drift"}

_registry: dict[str, list[dict]] = {phase: [] for phase in _VALID_PHASES}


def register_check(phase: str, name: str) -> Callable:
    """Decorator to register a custom check function.

    Args:
        phase: Pipeline phase to attach to (quality, leakage, feature, sufficiency, drift).
        name: Unique check identifier used as key in config.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If *phase* is not a valid phase name.
    """
    if phase not in _VALID_PHASES:
        raise ValueError(f"phase must be one of {_VALID_PHASES}, got '{phase}'")

    def decorator(fn: Callable) -> Callable:
        existing = [e["name"] for e in _registry[phase]]
        if name in existing:
            logger.warning(f"Plugin check '{name}' in phase '{phase}' is already registered — overwriting.")
            _registry[phase] = [e for e in _registry[phase] if e["name"] != name]
        _registry[phase].append({"name": name, "fn": fn})
        logger.debug(f"Registered plugin check '{name}' for phase '{phase}'")
        return fn

    return decorator


def get_checks(phase: str) -> list[dict]:
    """Return all registered checks for *phase*.

    Each entry is a dict with keys ``name`` (str) and ``fn`` (callable).
    """
    return list(_registry.get(phase, []))


def clear_registry() -> None:
    """Remove all registered checks. Mainly useful in tests."""
    for phase in _registry:
        _registry[phase].clear()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_plugins(plugin_paths: list[str]) -> None:
    """Import plugin modules so their @register_check decorators execute.

    Args:
        plugin_paths: List of dotted module paths (e.g. ``['my_pkg.checks']``).
    """
    for path in plugin_paths:
        try:
            importlib.import_module(path)
            logger.info(f"Plugin module loaded: '{path}'")
        except ImportError as exc:
            logger.warning(f"Could not load plugin module '{path}': {exc}")


# ---------------------------------------------------------------------------
# Extension point for orchestrators
# ---------------------------------------------------------------------------

def run_plugin_checks(
    phase: str,
    df,
    target_col: str,
    config: dict,
) -> list[CheckResult]:
    """Run all registered plugin checks for *phase*.

    Called at the end of each orchestrator (run_all_quality_checks, etc.) so
    plugin results are included in the phase's result list.

    Args:
        phase: Phase name matching the orchestrator.
        df: Input DataFrame.
        target_col: Target column name.
        config: The full phase config block (plugin configs are nested under their name).

    Returns:
        List of CheckResult from plugin checks; empty if no plugins registered.
    """
    results: list[CheckResult] = []
    for entry in get_checks(phase):
        name = entry["name"]
        fn   = entry["fn"]
        cfg  = config.get(name, {})
        if not cfg.get("enabled", True):
            logger.debug(f"Plugin check '{name}' disabled — skipping.")
            continue
        try:
            logger.debug(f"Running plugin check: '{name}' (phase={phase})")
            r = fn(df, target_col, cfg)
            results.append(r)
            (logger.warning if not r.passed else logger.debug)(f"[{name}] {r.message}")
        except Exception as exc:
            logger.error(f"Plugin check '{name}' raised an exception: {exc}")
    return results
