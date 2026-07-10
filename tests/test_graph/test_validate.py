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
    _mask_line_comments,
    _split_statements,
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


def _census(self_loops: int, reciprocal_pairs: int) -> list[dict[str, object]]:
    """The single aggregate row ``chain_integrity.cypher`` returns."""
    return [{"self_loops": self_loops, "reciprocal_pairs": reciprocal_pairs}]


class TestChainIntegrityClassification:
    """da#250: the gate reports a true population, never a cap.

    ``self_loops`` is the gate (a narrator transmitting to himself is corrupt);
    ``reciprocal_pairs`` is a metric (over-merge thermometer, da#248) and never
    gates. Both are exact counts -- the query carries no ``LIMIT``.
    """

    def test_query_carries_no_limit_clause(self) -> None:
        """A LIMIT makes the reported count a cap, not a measurement.

        Comments are masked first: the file *discusses* the removed ``LIMIT
        100`` in prose, which must not be mistaken for a live clause.
        """
        code = _mask_line_comments((QUERIES_DIR / "chain_integrity.cypher").read_text())
        assert "LIMIT" not in code.upper(), "chain_integrity.cypher must not cap its own result"

    def test_clean_census_passes_and_reports_exact_counts(self) -> None:
        result = _classify("chain_integrity", _census(self_loops=0, reciprocal_pairs=23139))
        assert result.passed is True
        # The real population is reported verbatim -- not a cap, not a row count.
        assert "23139" in result.details
        assert result.row_count == 23139

    def test_self_loop_is_the_gate_and_is_fatal(self) -> None:
        result = _classify("chain_integrity", _census(self_loops=1, reciprocal_pairs=0))
        assert result.passed is False
        assert result.is_fatal is True

    def test_reciprocal_pairs_alone_never_gate(self) -> None:
        """da#248: reciprocal pairs are manufactured upstream (resolve), not by
        the loader. They are a metric to re-measure after da#356, not a defect
        for this gate to fail on."""
        result = _classify("chain_integrity", _census(self_loops=0, reciprocal_pairs=23139))
        assert result.passed is True

    def test_missing_row_is_fatal_not_a_silent_pass(self) -> None:
        """An empty result means the census did not run. A zero you never
        measured is not a zero -- the old classifier passed on ``[]``."""
        result = _classify("chain_integrity", [])
        assert result.passed is False
        assert result.is_fatal is True

    def test_malformed_row_is_fatal(self) -> None:
        result = _classify("chain_integrity", [{"narrator_id": "nar:y", "cycle_length": 2}])
        assert result.passed is False
        assert result.is_fatal is True

    def test_details_disclose_the_uncovered_cycle_lengths(self) -> None:
        """A PASS must not read as 'this graph is acyclic'. Cycles of length
        >= 3 are not measured, and the gate has to say so out loud."""
        result = _classify("chain_integrity", _census(self_loops=0, reciprocal_pairs=0))
        assert result.passed is True
        assert "not measured" in result.details.lower()


class TestTransmittedToHadithRefClassification:
    """da#325 regression guard: 0 dangling edge hadith_ids = pass."""

    def test_registered(self) -> None:
        assert "transmitted_to_hadith_ref" in _CLASSIFIER_REGISTRY

    def test_cypher_file_exists(self) -> None:
        path = QUERIES_DIR / "transmitted_to_hadith_ref.cypher"
        assert path.exists(), f"Missing {path}"

    def test_zero_results_is_pass(self) -> None:
        result = _classify("transmitted_to_hadith_ref", [])
        assert result.passed is True
        assert result.row_count == 0

    def test_dangling_edge_id_is_fail(self) -> None:
        # An edge hadith_id that matches no Hadith node — the da#325 mismatch.
        rows: list[dict[str, object]] = [
            {"hadith_id": "sanadset:sanadset:0:0:2326"},
        ]
        result = _classify("transmitted_to_hadith_ref", rows)
        assert result.passed is False
        assert result.row_count == 1
        assert "da#325" in result.details


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

    # ``chain_integrity`` is deliberately excluded: its query always returns
    # exactly one aggregate census row, so an empty result means the census did
    # not run. Passing on `[]` is the vacuous pass da#250 removed -- the
    # zero-rows-is-pass convention does not apply to an aggregate query.
    @pytest.mark.parametrize("name", [n for n in _CLASSIFIER_REGISTRY if n != "chain_integrity"])
    def test_registered_classifier_pass_on_empty(self, name: str) -> None:
        result = _classify(name, [])
        assert result.passed is True
        assert result.row_count == 0

    @pytest.mark.parametrize(
        ("name", "rows"),
        [
            ("orphan_narrators", [{"narrator_id": "nar:x"}]),
            ("chain_integrity", [{"self_loops": 1, "reciprocal_pairs": 0}]),
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
    """Write a single-statement validation query mirroring the shipped census.

    Kept in step with ``queries/validation/chain_integrity.cypher``: one
    statement, no ``LIMIT``, returning one aggregate row.
    """
    validation = queries_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (validation / f"{name}.cypher").write_text(
        "CALL { MATCH (n:Narrator)-[:TRANSMITTED_TO]->(n) "
        "RETURN count(DISTINCT n.id) AS self_loops }\n"
        "CALL { MATCH (a:Narrator)-[:TRANSMITTED_TO]->(b:Narrator) "
        "WHERE a.id < b.id AND EXISTS { MATCH (b)-[:TRANSMITTED_TO]->(a) } "
        "RETURN count(DISTINCT [a.id, b.id]) AS reciprocal_pairs }\n"
        "RETURN self_loops, reciprocal_pairs",
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
        # A clean census row -- 0 self-loops -> chain_integrity passes.
        client = _FakeClient(rows=[{"self_loops": 0, "reciprocal_pairs": 0}])
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


# --- da#319: multi-statement validation files ----------------------------------


class TestSplitStatements:
    """``_split_statements`` splits a file's text on top-level ``;`` boundaries.

    These files use only ``//`` line comments and never a semicolon inside a
    string, so a plain split is safe (see the docstring on the function under
    test). A single-statement file with no ``;`` must round-trip unchanged so
    the pre-da#319 behavior is preserved exactly.
    """

    def test_single_statement_no_semicolon_is_unchanged(self) -> None:
        text = "MATCH (n:Narrator) RETURN n.id AS narrator_id\nORDER BY n.id"
        assert _split_statements(text) == [text]

    def test_multi_statement_splits_on_semicolon(self) -> None:
        text = "MATCH (a) RETURN a;\nMATCH (b) RETURN b;\nMATCH (c) RETURN c"
        result = _split_statements(text)
        assert result == ["MATCH (a) RETURN a", "MATCH (b) RETURN b", "MATCH (c) RETURN c"]

    def test_trailing_semicolon_does_not_add_empty_statement(self) -> None:
        assert _split_statements("MATCH (a) RETURN a;\n") == ["MATCH (a) RETURN a"]

    def test_comment_only_segment_is_dropped(self) -> None:
        text = "// just a header\n// no code here\n;\nMATCH (a) RETURN a"
        assert _split_statements(text) == ["MATCH (a) RETURN a"]

    def test_leading_header_comment_stays_attached_to_first_statement(self) -> None:
        text = "// header line 1\n// header line 2\nMATCH (a) RETURN a;\nMATCH (b) RETURN b"
        result = _split_statements(text)
        assert len(result) == 2
        assert "header line 1" in result[0]
        assert result[1] == "MATCH (b) RETURN b"

    def test_empty_text_returns_no_statements(self) -> None:
        assert _split_statements("") == []
        assert _split_statements("   \n  ") == []

    def test_semicolon_inside_comment_prose_is_not_a_split_point(self) -> None:
        """Regression: this is the real da#319 shape that broke a naive split.

        ``graph_integrity_deferred_inventory.cypher`` has a comment reading
        "...decision must cover; it is contractual, not a defect." — a plain
        ``text.split(";")`` cuts that single sentence into two "statements",
        and the fragment ``it is contractual, not a defect.`` is not valid
        Cypher (caught by the real-Neo4j integration test, not a canned-row
        unit stub, since only a real driver rejects the fragment).
        """
        text = (
            "// the missing count is the population a future decision must\n"
            "// cover; it is contractual, not a defect.\n"
            "MATCH (a) RETURN a;\n"
            "MATCH (b) RETURN b"
        )
        result = _split_statements(text)
        assert len(result) == 2
        assert result[0].endswith("MATCH (a) RETURN a")
        assert result[1] == "MATCH (b) RETURN b"
        # The comment (with its embedded `;`) stays attached to the first
        # statement rather than being sliced into its own fragment.
        assert "it is contractual, not a defect." in result[0]

    def test_semicolon_in_comment_on_otherwise_single_statement_file(self) -> None:
        """A single-statement file whose only comment contains a `;` must
        still be recognized as ONE statement (not two)."""
        text = "// note: these; are fine in prose\nMATCH (a) RETURN a"
        assert _split_statements(text) == [text]


class TestInformationalInventoryClassification:
    """da#319: ADR-003/ADR-004 inventory reports are owner-accepted sizing

    reports, not pass/fail checks -- they must always PASS while still
    surfacing the counts, mirroring the ``hadith_number_coverage`` precedent
    in ``src/parse/validate.py``.
    """

    def test_graph_integrity_deferred_inventory_pass_regardless_of_counts(self) -> None:
        rows: list[dict[str, object]] = [
            {"block": 1, "row_count": 12},
            {"block": 2, "row_count": 0},
            {"block": 3, "row_count": 587},
        ]
        result = _classify("graph_integrity_deferred_inventory", rows)
        assert result.passed is True
        assert result.status == "PASS"
        assert result.row_count == 599
        assert "block 1: 12" in result.details
        assert "block 3: 587" in result.details

    def test_sanadset_orphan_inventory_pass_regardless_of_counts(self) -> None:
        rows: list[dict[str, object]] = [{"block": 1, "row_count": 7580}]
        result = _classify("sanadset_orphan_inventory", rows)
        assert result.passed is True
        assert result.status == "PASS"
        assert result.row_count == 7580

    def test_pass_on_no_blocks(self) -> None:
        result = _classify("graph_integrity_deferred_inventory", [])
        assert result.passed is True
        assert result.row_count == 0
        assert "no statement blocks" in result.details

    def test_both_names_registered(self) -> None:
        assert "graph_integrity_deferred_inventory" in _CLASSIFIER_REGISTRY
        assert "sanadset_orphan_inventory" in _CLASSIFIER_REGISTRY


class TestDefaultClassifierBlockSummaryAware:
    """da#319 follow-up: the fallback ``_classify_default`` must interpret the
    per-block summary shape (an unregistered multi-statement file) by SUMMING
    ``row_count`` across blocks, while keeping ``len(rows)`` semantics for the
    single-statement path's real flat rows byte-for-byte.
    """

    def test_block_summary_zero_total_is_pass(self) -> None:
        rows: list[dict[str, object]] = [
            {"block": 1, "row_count": 0},
            {"block": 2, "row_count": 0},
        ]
        result = _classify("unregistered_check", rows)
        assert result.passed is True
        assert result.row_count == 0

    def test_block_summary_nonzero_total_is_fail_on_sum_not_block_count(self) -> None:
        # 2 blocks (len==2) but real totals are 0 -> must PASS. If the default
        # classifier keyed on len(rows) this would wrongly FAIL.
        rows: list[dict[str, object]] = [
            {"block": 1, "row_count": 0},
            {"block": 2, "row_count": 0},
        ]
        assert _classify("unregistered_check", rows).passed is True
        # Now a genuine nonzero total -> FAIL, and row_count is the SUM (5), not 2.
        rows_nonzero: list[dict[str, object]] = [
            {"block": 1, "row_count": 5},
            {"block": 2, "row_count": 0},
        ]
        result = _classify("unregistered_check", rows_nonzero)
        assert result.passed is False
        assert result.row_count == 5

    def test_single_statement_real_rows_keep_len_semantics(self) -> None:
        # Real flat query rows (NOT the exact {block,row_count} 2-key shape) must
        # still classify on len(rows) exactly as before the fix.
        real_rows: list[dict[str, object]] = [
            {"narrator_id": "nar:1", "name": "A"},
            {"narrator_id": "nar:2", "name": "B"},
        ]
        result = _classify("unregistered_check", real_rows)
        assert result.passed is False
        assert result.row_count == 2
        assert _classify("unregistered_check", []).passed is True

    def test_real_rows_named_block_and_rowcount_with_extra_keys_not_summary(self) -> None:
        # Defensive: a real row that merely happens to include block/row_count
        # columns AMONG others is NOT the summary shape (exact 2-key match), so
        # it keeps len-based semantics rather than being summed.
        real_rows: list[dict[str, object]] = [
            {"block": "b", "row_count": 9, "extra": 1},
        ]
        result = _classify("unregistered_check", real_rows)
        assert result.row_count == 1  # len(rows), not summed 9


class _SequencedClient:
    """Stub client that returns one canned row-set per call and counts calls.

    Unlike ``_FakeClient`` (which returns the same canned rows/behavior for
    every call), this stub is for proving ``run_validation`` executes a
    multi-statement file as N separate ``execute_read`` calls -- each call
    gets its own entry from ``rows_per_call``.
    """

    def __init__(self, rows_per_call: list[list[dict[str, Any]]]) -> None:
        self._rows_per_call = rows_per_call
        self.queries_seen: list[str] = []

    def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        idx = len(self.queries_seen)
        self.queries_seen.append(query)
        return self._rows_per_call[idx]


def _write_multi_statement_file(queries_dir: Path, name: str, text: str) -> None:
    validation = queries_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (validation / f"{name}.cypher").write_text(text, encoding="utf-8")


class TestMultiStatementExecution:
    """da#319: a multi-statement ``.cypher`` file is split and run per-statement.

    Root cause: the Neo4j driver enforces one statement per ``tx.run()`` and
    raises ``CypherSyntaxError: Expected exactly one statement per query but
    got: N`` when handed a whole multi-block file as one string. These tests
    prove ``run_validation`` now issues one ``execute_read`` call per
    top-level statement instead of one call for the whole file, without
    erroring, and aggregates the result back into a single ``ValidationResult``
    per file (query_name unchanged).
    """

    def test_five_statement_file_executes_five_separate_calls(self, tmp_path: Path) -> None:
        # Mirrors the real da#319 shape: graph_integrity_deferred_inventory.cypher
        # and sanadset_orphan_inventory.cypher are both 5-statement scripts.
        text = (
            "// header comment for the whole file\n"
            "MATCH (a) RETURN count(a) AS n;\n\n"
            "// block 2\n"
            "MATCH (b) RETURN count(b) AS n;\n\n"
            "MATCH (c) RETURN count(c) AS n;\n\n"
            "MATCH (d) RETURN count(d) AS n;\n\n"
            "MATCH (e) RETURN count(e) AS n"
        )
        _write_multi_statement_file(tmp_path, "graph_integrity_deferred_inventory", text)
        client = _SequencedClient(
            rows_per_call=[[{"n": 4}], [{"n": 0}], [{"n": 12}], [{"n": 1}], [{"n": 587}]]
        )
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]

        assert len(client.queries_seen) == 5, (
            "each top-level statement must be its own tx.run() call, not one "
            "5-statement blob (da#319 CypherSyntaxError root cause)"
        )
        assert len(results) == 1, "aggregated into a single ValidationResult per file"
        result = results[0]
        assert result.query_name == "graph_integrity_deferred_inventory"
        # Registered informational classifier -> always PASS even with nonzero
        # counts in every block (ADR-004: owner-accepted tracked inventory).
        assert result.passed is True
        assert result.status == "PASS"
        assert result.row_count == 5  # sum of per-block row counts (1 each)

    def test_sanadset_orphan_inventory_five_statements(self, tmp_path: Path) -> None:
        text = ";\n".join(f"MATCH (n{i}) RETURN count(n{i}) AS n" for i in range(5))
        _write_multi_statement_file(tmp_path, "sanadset_orphan_inventory", text)
        client = _SequencedClient(rows_per_call=[[{"n": 1}] for _ in range(5)])
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]

        assert len(client.queries_seen) == 5
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].status == "PASS"

    def test_single_statement_file_still_executes_exactly_once(self, tmp_path: Path) -> None:
        """Backward-compat: pre-da#319 single-statement files are untouched."""
        _write_query(tmp_path, name="chain_integrity")
        client = _SequencedClient(rows_per_call=[[{"self_loops": 0, "reciprocal_pairs": 0}]])
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]

        assert len(client.queries_seen) == 1
        assert results[0].query_name == "chain_integrity"
        assert results[0].status == "PASS"

    def test_unregistered_multi_statement_all_zero_rows_passes(self, tmp_path: Path) -> None:
        """Generic split-and-run: works for ANY multi-statement file, not just
        the two named in da#319 -- a future third file gets the same
        statement-splitting treatment even without a dedicated classifier.

        Outcome assertion (not just call-count): with every block returning 0
        rows, the fallback ``_classify_default`` must PASS. Before the block-
        summary-aware fix it FAILed unconditionally, because ``len(rows)`` was
        the block count (2), never 0 -- the exact da#319 bug relocated to the
        default path."""
        text = "MATCH (a) RETURN a.id AS id;\nMATCH (b) RETURN b.id AS id"
        _write_multi_statement_file(tmp_path, "some_future_check", text)
        client = _SequencedClient(rows_per_call=[[], []])
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]

        assert len(client.queries_seen) == 2
        assert len(results) == 1
        assert results[0].query_name == "some_future_check"
        # Summed row_count across blocks is 0 -> conservative zero-is-pass.
        assert results[0].passed is True
        assert results[0].status == "PASS"
        assert results[0].row_count == 0

    def test_unregistered_multi_statement_nonzero_rows_fails(self, tmp_path: Path) -> None:
        """Companion to the all-zero case: an unregistered multi-statement file
        whose statements return rows must FAIL under the default classifier's
        conservative zero-is-pass semantics -- and it must fail on the SUMMED
        row total, not the block count. Here block 1 returns 3 rows and block 2
        returns 0 (summed total 3 > 0 -> FAIL), which also proves the count is
        the real row total, not len(blocks)=2."""
        text = "MATCH (a) RETURN a.id AS id;\nMATCH (b) RETURN b.id AS id"
        _write_multi_statement_file(tmp_path, "some_future_check", text)
        client = _SequencedClient(rows_per_call=[[{"id": "x"}, {"id": "y"}, {"id": "z"}], []])
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].status == "FAIL"
        assert results[0].is_fatal is True
        assert results[0].row_count == 3

    def test_error_in_second_statement_is_hard_failure_for_whole_file(self, tmp_path: Path) -> None:
        """A real Cypher error partway through a multi-statement file still
        surfaces as a fatal failure for the file (no silent partial pass)."""
        validation = tmp_path / "validation"
        validation.mkdir()
        (validation / "graph_integrity_deferred_inventory.cypher").write_text(
            "MATCH (a) RETURN a;\nMATCH (b) RETURN b", encoding="utf-8"
        )

        class _FailingSecondCall:
            def __init__(self) -> None:
                self.calls = 0

            def execute_read(
                self,
                query: str,
                parameters: dict[str, Any] | None = None,
                *,
                timeout: float | None = None,
            ) -> list[dict[str, Any]]:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("boom")
                return []

        client = _FailingSecondCall()
        results = run_validation(client, tmp_path, timeout_seconds=5.0)  # type: ignore[arg-type]
        assert client.calls == 2
        assert len(results) == 1
        assert results[0].is_fatal is True
        assert results[0].details == "query execution failed"
