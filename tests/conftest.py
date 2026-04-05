"""Shared test fixtures for the isnad-graph-ingestion test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ENVIRONMENT=test for all tests."""
    monkeypatch.setenv("ENVIRONMENT", "test")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the cached settings singleton between tests."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
