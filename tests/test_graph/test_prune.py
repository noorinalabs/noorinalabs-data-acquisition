"""Unit tests for src.graph.prune — canonical-set SHRINK of Narrator nodes (da#413).

These drive the orphan-computation and the guard against a :class:`MockNeo4jClient`,
so they run with no Docker. The graph-truth acceptance (an edge-bearing orphan is
deleted with its edges, a zero-degree canonical narrator survives) needs a real graph
and lives in ``tests/integration/test_prune_narrators.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.graph.prune import (
    EmptyCanonicalSetError,
    PruneResult,
    prune_narrators,
    read_canonical_ids,
    summary_line,
)
from tests.test_graph.conftest import MockNeo4jClient, write_narrators_canonical


def _graph_rows(ids: list[str]) -> list[dict[str, str]]:
    """Shape of ``MATCH (n:Narrator) RETURN n.id AS id`` records."""
    return [{"id": nid} for nid in ids]


def _write_calls(client: MockNeo4jClient) -> list[dict[str, object]]:
    return [
        params for _query, params in client.calls if isinstance(params, dict) and "batch" in params
    ]


class StatefulFakeNeo4j:
    """A minimal Narrator-set fake that actually applies the DETACH DELETE.

    The shared :class:`MockNeo4jClient` is stateless — its ``execute_read`` always
    returns the same rows — so it cannot exercise the post-delete readback that
    ``deleted`` and the summary ``orphans=`` (post-count) are measured from. This fake
    holds a live id set: reads return the current set, and a batched delete removes
    those ids. Enough to prove the readback semantics without Docker; the real graph
    confirms them in the integration suite.
    """

    def __init__(self, narrator_ids: list[str]) -> None:
        self.narrators: list[str] = list(narrator_ids)
        self.deleted_batches: list[list[str]] = []

    def execute_read(
        self, query: str, parameters: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> list[dict[str, str]]:
        return [{"id": nid} for nid in self.narrators]

    def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        batch = list((parameters or {}).get("batch", []))
        self.deleted_batches.append(batch)
        drop = set(batch)
        self.narrators = [n for n in self.narrators if n not in drop]
        return []


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
    def test_delete_batch_targets_orphan_not_zero_degree_canonical(self, curated_dir: Path) -> None:
        """The whole reason pure-cypher failed, at the set level.

        ``nar:keep_zero_degree`` is a canonical narrator with no edges — the 57,993
        class that a degree prune wrongly deletes. It is in the keep-set, so it is
        NOT in the delete batch. ``nar:orphan`` is absent from the keep-set, so it IS
        — regardless of how many edges it has (DETACH removes them). This asserts the
        batch TARGETING; the real deletion + survival is proven against the stateful
        fake below and on a live graph in the integration suite.
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


class TestReadbackSemantics:
    """``deleted`` and the summary ``orphans=`` are measured by reading the graph BACK.

    Driven against the stateful fake so the post-delete state is real, not the pre-read
    echoed. This is where the by-mode ``orphans=`` contract and the pre/post ``deleted``
    are pinned without Docker.
    """

    def test_real_run_deletes_and_post_orphans_is_zero(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:keep_zero"}, {"canonical_id": "nar:keep_busy"}]
        )
        client = StatefulFakeNeo4j(
            ["nar:keep_zero", "nar:keep_busy", "nar:orphan_a", "nar:orphan_b"]
        )

        result = prune_narrators(client, path)

        assert isinstance(result, PruneResult)
        assert result.canonical_ids_seen == 2
        assert result.deleted == 2  # pre(4) - post(2), read back
        assert result.graph_total == 2  # survivors, post
        assert result.orphans == 0  # post-count MUST be 0
        assert result.orphans_identified == 2  # would-delete, pre
        assert result.dry_run is False
        # The graph actually shrank to exactly the canonical survivors.
        assert set(client.narrators) == {"nar:keep_zero", "nar:keep_busy"}

    def test_dry_run_fields_are_pre_state(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan_a", "nar:orphan_b"])

        result = prune_narrators(client, path, dry_run=True)

        assert result.dry_run is True
        assert result.graph_total == 3  # current total, nothing removed
        assert result.orphans == 2  # would-delete count
        assert result.deleted == 0
        assert client.deleted_batches == [], "dry-run issued a delete"
        assert set(client.narrators) == {"nar:keep", "nar:orphan_a", "nar:orphan_b"}


class TestDryRun:
    def test_dry_run_issues_no_write(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:keep", "nar:orphan_a", "nar:orphan_b"]))

        result = prune_narrators(client, path, dry_run=True)

        assert result.dry_run is True
        assert result.orphans_identified == 2  # counted
        assert result.orphans == 2  # summary field: would-delete on a dry run
        assert result.graph_total == 3  # current total (nothing removed)
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


class TestSummaryLine:
    """The one machine-readable line deploy#557's verify greps. Format is a contract."""

    def test_dry_run_line_reports_would_delete(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan_a", "nar:orphan_b"])
        result = prune_narrators(client, path, dry_run=True)
        assert (
            summary_line(result)
            == "PRUNE_NARRATORS_SUMMARY canonical=1 graph_total=3 orphans=2 deleted=0"
        )

    def test_real_run_line_reports_zero_orphans_remaining(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:keep_a"}, {"canonical_id": "nar:keep_b"}]
        )
        client = StatefulFakeNeo4j(["nar:keep_a", "nar:keep_b", "nar:orphan_a"])
        result = prune_narrators(client, path)
        # canonical=2, graph_total (post survivors)=2, orphans (post)=0, deleted=1.
        assert (
            summary_line(result)
            == "PRUNE_NARRATORS_SUMMARY canonical=2 graph_total=2 orphans=0 deleted=1"
        )


class TestNoWriteIntoCanonicalDir:
    """deploy#557 mounts the parquet dir read-only: the subcommand must not write there.

    Unlike ``load`` (which writes a manifest + audit entry into the data dir and trips
    the ``:ro`` mount, da#348), ``prune-narrators`` reads the parquet and writes ONLY to
    the graph. Proven structurally: no new file appears beside the canonical parquet on
    either a dry run or a real run.
    """

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_no_side_file_written_next_to_canonical(self, curated_dir: Path, dry_run: bool) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        before = sorted(p.name for p in curated_dir.iterdir())
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan"])

        prune_narrators(client, path, dry_run=dry_run)

        after = sorted(p.name for p in curated_dir.iterdir())
        assert after == before == ["narrators_canonical.parquet"]


class TestCliEmitsSummaryLine:
    """deploy#557's verify HARD-FAILS if the summary line is absent from stdout, so the
    CLI must always print it. Stub the DB layer and assert the line reaches stdout."""

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_cmd_prints_summary_line(
        self,
        curated_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        dry_run: bool,
    ) -> None:
        import src.cli as cli

        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        canned = PruneResult(
            canonical_ids_seen=1,
            graph_total=1,
            orphans=0,
            deleted=3,
            dry_run=dry_run,
            orphans_identified=3,
            sample=["nar:orphan"],
        )

        class _DummyClient:
            def __enter__(self) -> _DummyClient:
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", lambda *_a, **_k: _DummyClient())
        monkeypatch.setattr(
            "src.graph.prune.prune_narrators",
            lambda _client, _p, *, dry_run=False: canned,
        )

        cli._cmd_prune_narrators(canonical=str(path), dry_run=dry_run)

        out = capsys.readouterr().out
        assert "PRUNE_NARRATORS_SUMMARY canonical=1 graph_total=1 orphans=0 deleted=3" in out
