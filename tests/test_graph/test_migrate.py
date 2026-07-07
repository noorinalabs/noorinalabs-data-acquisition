"""Tests for src.graph.migrate — in-place TRANSMITTED_TO.hadith_id remediation (da#325)."""

from __future__ import annotations

from src.graph.migrate import (
    MigrationResult,
    compute_rewrites,
    migrate_transmitted_hadith_ids,
)
from tests.test_graph.conftest import MockNeo4jClient


class TestComputeRewrites:
    def test_raw_double_prefixed_maps_to_canonical(self) -> None:
        # The da#325 shape: raw double-prefixed id -> canonical hdt: id with the
        # main#139 double-corpus collapsed.
        rewrites = compute_rewrites([{"hadith_id": "sanadset:sanadset:0:0:2326"}])
        assert rewrites == [{"raw": "sanadset:sanadset:0:0:2326", "canon": "hdt:sanadset:0:0:2326"}]

    def test_single_prefix_raw_gets_hdt(self) -> None:
        rewrites = compute_rewrites([{"hadith_id": "sanadset:0:0:2326"}])
        assert rewrites == [{"raw": "sanadset:0:0:2326", "canon": "hdt:sanadset:0:0:2326"}]

    def test_already_canonical_is_noop(self) -> None:
        # Idempotency: an id that is already canonical yields no rewrite.
        assert compute_rewrites([{"hadith_id": "hdt:sanadset:0:0:2326"}]) == []

    def test_empty_and_null_ids_skipped(self) -> None:
        rewrites = compute_rewrites(
            [{"hadith_id": ""}, {"hadith_id": None}, {}, {"hadith_id": "sanadset:0:0:1"}]
        )
        assert rewrites == [{"raw": "sanadset:0:0:1", "canon": "hdt:sanadset:0:0:1"}]


class TestMigrateTransmittedHadithIds:
    def test_counts_seen_and_rewritten(self) -> None:
        client = MockNeo4jClient()
        client.set_read_results(
            [
                {"hadith_id": "sanadset:sanadset:0:0:1"},  # needs rewrite
                {"hadith_id": "hdt:sanadset:0:0:2"},  # already canonical
                {"hadith_id": ""},  # skipped (not counted as seen)
            ]
        )
        result = migrate_transmitted_hadith_ids(client)
        assert isinstance(result, MigrationResult)
        assert result.distinct_ids_seen == 2
        assert result.distinct_ids_rewritten == 1

    def test_write_issued_with_canonical_mapping(self) -> None:
        client = MockNeo4jClient()
        client.set_read_results([{"hadith_id": "sanadset:sanadset:0:0:9"}])
        migrate_transmitted_hadith_ids(client)
        # The write call carries the raw->canon mapping in its $batch parameter.
        write_calls = [
            params
            for query, params in client.calls
            if isinstance(params, dict) and "batch" in params
        ]
        assert write_calls, "expected a batched write call"
        assert write_calls[0]["batch"] == [
            {"raw": "sanadset:sanadset:0:0:9", "canon": "hdt:sanadset:0:0:9"}
        ]

    def test_idempotent_second_run_writes_nothing(self) -> None:
        # A fully-canonical graph: nothing to rewrite, no write call issued.
        client = MockNeo4jClient()
        client.set_read_results([{"hadith_id": "hdt:sanadset:0:0:1"}])
        result = migrate_transmitted_hadith_ids(client)
        assert result.distinct_ids_rewritten == 0
        write_calls = [
            params
            for query, params in client.calls
            if isinstance(params, dict) and "batch" in params
        ]
        assert write_calls == []
