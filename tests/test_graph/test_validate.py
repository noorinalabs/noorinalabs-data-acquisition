"""Tests for graph validation queries (.cypher files) and result classification."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.graph.validate import (
    _CLASSIFIER_REGISTRY,
    ValidationResult,
    _classify,
    register_classifier,
    run_validation,
)

# The validation queries live in queries/validation/*.cypher
QUERIES_DIR = Path(__file__).resolve().parents[2] / "queries" / "validation"


class TestValidationQueryFiles:
    def test_orphan_narrators_exists(self) -> None:
        path = QUERIES_DIR / "orphan_narrators.cypher"
        assert path.exists(), f"Missing {path}"

    def test_chain_integrity_exists(self) -> None:
        path = QUERIES_DIR / "chain_integrity.cypher"
        assert path.exists(), f"Missing {path}"

    def test_collection_coverage_exists(self) -> None:
        path = QUERIES_DIR / "collection_coverage.cypher"
        assert path.exists(), f"Missing {path}"

    def test_cypher_files_are_not_empty(self) -> None:
        for cypher_file in QUERIES_DIR.glob("*.cypher"):
            content = cypher_file.read_text().strip()
            assert len(content) > 0, f"{cypher_file.name} is empty"

    def test_cypher_files_contain_match_or_return(self) -> None:
        for cypher_file in QUERIES_DIR.glob("*.cypher"):
            content = cypher_file.read_text().upper()
            assert "MATCH" in content or "RETURN" in content, (
                f"{cypher_file.name} does not contain MATCH or RETURN"
            )


class TestOrphanCheckClassification:
    """Orphan check: 0 results = pass (no orphan narrators)."""

    def test_zero_results_is_pass(self) -> None:
        result = _classify("orphan_narrators", [])
        assert result.passed is True
        assert result.row_count == 0

    def test_nonzero_results_is_fail(self) -> None:
        rows: list[dict[str, object]] = [
            {"narrator_id": "nar:orphan-1", "name": "Orphan Narrator"},
        ]
        result = _classify("orphan_narrators", rows)
        assert result.passed is False
        assert result.row_count == 1


class TestChainIntegrityClassification:
    """Chain integrity: 0 cycles = pass."""

    def test_zero_cycles_is_pass(self) -> None:
        result = _classify("chain_integrity", [])
        assert result.passed is True
        assert result.row_count == 0

    def test_cycles_detected_is_fail(self) -> None:
        rows: list[dict[str, object]] = [
            {"narrator_id": "nar:cycle-node", "cycle_length": 3},
        ]
        result = _classify("chain_integrity", rows)
        assert result.passed is False
        assert result.row_count == 1


class TestCollectionCoverageClassification:
    """Collection coverage: deviation within threshold = pass."""

    def test_within_threshold_is_pass(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "collection_id": "col:bukhari",
                "expected": 7563,
                "actual": 7500,
                "deviation_pct": 0.83,
            },
            {
                "collection_id": "col:muslim",
                "expected": 5362,
                "actual": 5300,
                "deviation_pct": 1.16,
            },
        ]
        result = _classify("collection_coverage", rows)
        assert result.passed is True

    def test_exceeds_threshold_is_fail(self) -> None:
        rows: list[dict[str, object]] = [
            {"collection_id": "col:bad", "expected": 1000, "actual": 500, "deviation_pct": 50.0},
        ]
        result = _classify("collection_coverage", rows)
        assert result.passed is False

    def test_null_expected_is_pass(self) -> None:
        rows: list[dict[str, object]] = [
            {"collection_id": "col:unknown", "expected": None, "actual": 42, "deviation_pct": None},
        ]
        result = _classify("collection_coverage", rows)
        assert result.passed is True


class TestCypherFileLoading:
    """Test that .cypher files can be read and used as query strings."""

    def test_load_orphan_query(self) -> None:
        path = QUERIES_DIR / "orphan_narrators.cypher"
        query = path.read_text().strip()
        assert "Narrator" in query
        assert "MATCH" in query

    def test_load_chain_integrity_query(self) -> None:
        path = QUERIES_DIR / "chain_integrity.cypher"
        query = path.read_text().strip()
        assert "TRANSMITTED_TO" in query

    def test_load_collection_coverage_query(self) -> None:
        path = QUERIES_DIR / "collection_coverage.cypher"
        query = path.read_text().strip()
        assert "APPEARS_IN" in query
        assert "Collection" in query


class TestRegistryPattern:
    """Parametrized coverage: every registered classifier is tested for pass and fail."""

    @pytest.mark.parametrize("name", list(_CLASSIFIER_REGISTRY.keys()))
    def test_registered_classifier_pass_on_empty(self, name: str) -> None:
        result = _classify(name, [])
        assert result.passed is True
        assert result.row_count == 0

    @pytest.mark.parametrize(
        ("name", "rows"),
        [
            ("orphan_narrators", [{"narrator_id": "nar:x"}]),
            ("chain_integrity", [{"narrator_id": "nar:y", "cycle_length": 2}]),
            (
                "collection_coverage",
                [{"collection_id": "col:z", "expected": 100, "actual": 10, "deviation_pct": 90.0}],
            ),
        ],
    )
    def test_registered_classifier_fail_on_bad_rows(
        self, name: str, rows: list[dict[str, object]]
    ) -> None:
        result = _classify(name, rows)
        assert result.passed is False

    def test_unknown_classifier_uses_default(self) -> None:
        result = _classify("nonexistent_check", [])
        assert result.passed is True

    def test_unknown_classifier_fails_with_rows(self) -> None:
        result = _classify("nonexistent_check", [{"x": 1}])
        assert result.passed is False

    def test_register_custom_classifier(self) -> None:
        def _custom(name: str, rows: list[dict[str, object]], threshold: float) -> ValidationResult:
            return ValidationResult(name, passed=True, details="custom", row_count=len(rows))

        register_classifier("custom_test", _custom)
        try:
            result = _classify("custom_test", [{"a": 1}])
            assert result.passed is True
            assert result.details == "custom"
        finally:
            del _CLASSIFIER_REGISTRY["custom_test"]


# --- da#259: bounded validation ------------------------------------------------


class _FakeClient:
    """Minimal stand-in for Neo4jClient.execute_read used by run_validation.

    Configurable to return rows quickly, sleep past the timeout budget, or raise
    an error (optionally carrying a Neo4j-style timeout ``code``).
    """

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]] | None = None,
        sleep: float = 0.0,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._rows = rows or []
        self._sleep = sleep
        self._raise = raise_exc
        self.timeouts_seen: list[float | None] = []
        self._release = threading.Event()

    def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        self.timeouts_seen.append(timeout)
        if self._raise is not None:
            raise self._raise
        if self._sleep:
            # Wait until released or the (test-generous) cap elapses so the daemon
            # worker thread does not linger indefinitely after the test asserts.
            self._release.wait(timeout=self._sleep)
        return self._rows

    def release(self) -> None:
        self._release.set()


class _FakeNeo4jTimeout(Exception):
    """Mimics a Neo4j server-side statement/transaction timeout (has a ``code``)."""

    code = "Neo.ClientError.Transaction.TransactionTimedOut"


def _write_query(queries_dir: Path, name: str = "chain_integrity") -> None:
    validation = queries_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (validation / f"{name}.cypher").write_text(
        "MATCH (n:Narrator)-[:TRANSMITTED_TO*1..20]->(n) RETURN n.id LIMIT 100",
        encoding="utf-8",
    )


class TestValidationResultStatus:
    """The tri-state status/is_fatal contract that keeps warnings non-fatal."""

    def test_pass(self) -> None:
        r = ValidationResult("q", passed=True, details="ok", row_count=0)
        assert r.status == "PASS"
        assert r.is_fatal is False
        assert r.warning is False

    def test_fail_is_fatal(self) -> None:
        r = ValidationResult("q", passed=False, details="bad", row_count=3)
        assert r.status == "FAIL"
        assert r.is_fatal is True

    def test_warning_is_not_fatal(self) -> None:
        r = ValidationResult("q", passed=False, details="timed out", row_count=0, warning=True)
        assert r.status == "WARN"
        assert r.is_fatal is False


class TestBoundedValidation:
    """da#259: a slow/cyclic validation query must warn, never hang the loader."""

    def test_fast_graph_validates_normally(self, tmp_path: Path) -> None:
        _write_query(tmp_path)
        client = _FakeClient(rows=[])  # 0 cycles -> chain_integrity passes
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].query_name == "chain_integrity"
        assert results[0].status == "PASS"
        assert results[0].is_fatal is False
        # The per-query budget is handed to the client as a server-side timeout.
        assert client.timeouts_seen == [5.0]

    def test_wall_clock_timeout_downgrades_to_warning(self, tmp_path: Path) -> None:
        _write_query(tmp_path)
        # Client sleeps well past the budget and never returns on its own.
        client = _FakeClient(rows=[], sleep=30.0)
        start = time.monotonic()
        results = run_validation(client, tmp_path, timeout_seconds=0.2)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start
        client.release()  # let the daemon worker exit promptly
        assert elapsed < 5.0, "run_validation must not hang on a slow query"
        assert len(results) == 1
        assert results[0].warning is True
        assert results[0].is_fatal is False
        assert results[0].status == "WARN"
        assert "timed out" in results[0].details

    def test_server_side_timeout_downgrades_to_warning(self, tmp_path: Path) -> None:
        _write_query(tmp_path)
        client = _FakeClient(raise_exc=_FakeNeo4jTimeout())
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].warning is True
        assert results[0].is_fatal is False
        assert results[0].status == "WARN"

    def test_non_timeout_error_is_hard_failure(self, tmp_path: Path) -> None:
        _write_query(tmp_path)
        client = _FakeClient(raise_exc=RuntimeError("boom"))
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].warning is False
        assert results[0].is_fatal is True
        assert results[0].details == "query execution failed"

    def test_warning_result_keeps_load_non_fatal(self, tmp_path: Path) -> None:
        """The aggregation load_all uses: warnings must not trip validation_passed."""
        _write_query(tmp_path)
        client = _FakeClient(raise_exc=_FakeNeo4jTimeout())
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]
        validation_passed = not any(v.is_fatal for v in results)
        assert validation_passed is True
