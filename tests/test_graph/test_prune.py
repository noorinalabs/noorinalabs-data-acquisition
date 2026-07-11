"""Unit tests for src.graph.prune — canonical-set SHRINK of Narrator nodes (da#413).

These drive the orphan-computation and the guard against a :class:`MockNeo4jClient`,
so they run with no Docker. The graph-truth acceptance (an edge-bearing orphan is
deleted with its edges, a zero-degree canonical narrator survives) needs a real graph
and lives in ``tests/integration/test_prune_narrators.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.prune import (
    EmptyCanonicalSetError,
    PruneResult,
    prune_narrators,
    read_canonical_ids,
)
from tests.test_graph.conftest import MockNeo4jClient, write_narrators_canonical


def _graph_rows(ids: list[str]) -> list[dict[str, str]]:
    """Shape of ``MATCH (n:Narrator) RETURN n.id AS id`` records."""
    return [{"id": nid} for nid in ids]


def _write_calls(client: MockNeo4jClient) -> list[dict[str, object]]:
    return [
        params for _query, params in client.calls if isinstance(params, dict) and "batch" in params
    ]


class TestReadCanonicalIds:
    def test_reads_the_canonical_id_set(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:001"}, {"canonical_id": "nar:002"}]
        )
        assert read_canonical_ids(path) == {"nar:001", "nar:002"}

    def test_missing_parquet_refuses(self, curated_dir: Path) -> None:
        missing = curated_dir / "does_not_exist.parquet"
        with pytest.raises(EmptyCanonicalSetError, match="missing"):
            read_canonical_ids(missing)

    def test_unreadable_parquet_refuses(self, curated_dir: Path) -> None:
        # A non-parquet file with the right extension: pq.read_table raises, and the
        # guard folds that into a refusal rather than letting it crash mid-prune.
        bogus = curated_dir / "narrators_canonical.parquet"
        bogus.write_text("this is not a parquet file")
        with pytest.raises(EmptyCanonicalSetError, match="unreadable"):
            read_canonical_ids(bogus)

    def test_empty_parquet_refuses(self, curated_dir: Path) -> None:
        # Present, readable, zero canonical_id values: an empty keep-set would delete
        # every narrator, so it is exactly the catastrophic case the guard exists for.
        path = write_narrators_canonical(curated_dir, [])
        with pytest.raises(EmptyCanonicalSetError, match="zero canonical_id"):
            read_canonical_ids(path)

    def test_blank_ids_do_not_count_as_a_keep_set(self, curated_dir: Path) -> None:
        # Rows present but every canonical_id blank → no usable ids → refuse. (The
        # schema is non-nullable, so blank is the reachable "no id here" value.)
        path = write_narrators_canonical(curated_dir, [{"canonical_id": ""}, {"canonical_id": ""}])
        with pytest.raises(EmptyCanonicalSetError):
            read_canonical_ids(path)


class TestGuardTouchesNothing:
    """The load-bearing safety property: a bad read deletes nothing, reads no graph."""

    @pytest.mark.parametrize("kind", ["missing", "empty", "unreadable"])
    def test_guard_refuses_with_zero_client_calls(self, curated_dir: Path, kind: str) -> None:
        if kind == "missing":
            path = curated_dir / "nope.parquet"
        elif kind == "empty":
            path = write_narrators_canonical(curated_dir, [])
        else:
            path = curated_dir / "narrators_canonical.parquet"
            path.write_text("garbage")

        client = MockNeo4jClient()
        with pytest.raises(EmptyCanonicalSetError):
            prune_narrators(client, path)

        # Neither the read of graph ids nor any DETACH DELETE was issued. The guard
        # fired before the graph was touched at all.
        assert client.calls == [], f"guard ({kind}) reached the graph: {client.calls}"


class TestPruneComputesTheComplement:
    def test_orphan_is_deleted_zero_degree_canonical_survives(self, curated_dir: Path) -> None:
        """The whole reason pure-cypher failed, at the set level.

        ``nar:keep_zero_degree`` is a canonical narrator with no edges — the 57,993
        class that a degree prune wrongly deletes. It is in the keep-set, so it is
        NOT in the delete batch. ``nar:orphan`` is absent from the keep-set, so it IS
        — regardless of how many edges it has (DETACH removes them).
        """
        path = write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:keep_zero_degree"}, {"canonical_id": "nar:keep_busy"}],
        )
        client = MockNeo4jClient()
        client.set_read_results(
            _graph_rows(["nar:keep_zero_degree", "nar:keep_busy", "nar:orphan"])
        )

        result = prune_narrators(client, path)

        writes = _write_calls(client)
        assert writes, "expected a batched DETACH DELETE call"
        deleted_batch = writes[0]["batch"]
        assert deleted_batch == ["nar:orphan"]
        assert "nar:keep_zero_degree" not in deleted_batch
        assert result.orphans_identified == 1

    def test_no_orphans_issues_no_write(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:001"}, {"canonical_id": "nar:002"}]
        )
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:001", "nar:002"]))

        result = prune_narrators(client, path)

        assert result.orphans_identified == 0
        assert _write_calls(client) == []

    def test_batches_respect_batch_size(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        orphans = [f"nar:orphan{i}" for i in range(5)]
        client.set_read_results(_graph_rows(["nar:keep", *orphans]))

        prune_narrators(client, path, batch_size=2)

        batches = [w["batch"] for w in _write_calls(client)]
        assert batches == [orphans[0:2], orphans[2:4], orphans[4:5]]

    def test_result_fields(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:001"}, {"canonical_id": "nar:002"}]
        )
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:001", "nar:orphan"]))

        result = prune_narrators(client, path)

        assert isinstance(result, PruneResult)
        assert result.canonical_ids_seen == 2
        assert result.graph_narrators_seen == 2
        assert result.orphans_identified == 1
        assert result.sample == ["nar:orphan"]
        assert result.dry_run is False


class TestDryRun:
    def test_dry_run_issues_no_write(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:keep", "nar:orphan_a", "nar:orphan_b"]))

        result = prune_narrators(client, path, dry_run=True)

        assert result.dry_run is True
        assert result.orphans_identified == 2  # counted
        assert result.deleted == 0  # but nothing deleted
        assert result.sample == ["nar:orphan_a", "nar:orphan_b"]
        assert _write_calls(client) == [], "dry-run must not issue a DETACH DELETE"

    def test_dry_run_still_reads_the_graph(self, curated_dir: Path) -> None:
        # It reports a real count, so it does read the graph — it just writes nothing.
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:keep", "nar:orphan"]))

        prune_narrators(client, path, dry_run=True)

        read_calls = [q for q, _p in client.calls if "MATCH (n:Narrator)" in q]
        assert read_calls, "dry-run should have read the graph's narrator ids"
