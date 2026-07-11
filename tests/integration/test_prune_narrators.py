"""Integration tests for the canonical-set SHRINK against a real Neo4j container (da#413).

These are the load-bearing acceptance tests: the whole reason ``prune-narrators``
exists rather than a cypher one-liner is that the orphan predicate is canonical-set
membership, NOT node degree. Only a real graph can prove that an orphan WITH edges is
deleted (edges and all) while a legitimate zero-degree canonical narrator survives —
the two facts a degree-based prune gets backwards (deploy#557's measure-first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.prune import EmptyCanonicalSetError, prune_narrators, summary_line
from src.utils.neo4j_client import Neo4jClient
from tests.test_graph.conftest import write_narrators_canonical

pytestmark = pytest.mark.integration


# The seeded graph, chosen to put each of deploy#557's measured classes on the board:
#
#   nar:keep_zero    canonical, ZERO edges  -> the 57,993 class a degree prune wipes.
#   nar:keep_busy    canonical, HAS an edge -> a normal surviving narrator.
#   nar:orphan_edged NOT canonical, HAS an edge -> the 18,518 class a degree prune misses.
#   nar:orphan_lonely NOT canonical, zero edges -> a plain orphan.
_KEEP_ZERO = "nar:keep_zero"
_KEEP_BUSY = "nar:keep_busy"
_ORPHAN_EDGED = "nar:orphan_edged"
_ORPHAN_LONELY = "nar:orphan_lonely"

_SEED = """\
CREATE (kz:Narrator {id: $keep_zero})
CREATE (kb:Narrator {id: $keep_busy})
CREATE (oe:Narrator {id: $orphan_edged})
CREATE (ol:Narrator {id: $orphan_lonely})
// The orphan's edge lands on a surviving canonical narrator: DETACH must remove it
// when the orphan goes, and the survivor must stay.
CREATE (oe)-[:TRANSMITTED_TO {hadith_id: 'hdt:x:1'}]->(kb)
"""


def _seed_graph(client: Neo4jClient) -> None:
    client.execute_write(
        _SEED,
        {
            "keep_zero": _KEEP_ZERO,
            "keep_busy": _KEEP_BUSY,
            "orphan_edged": _ORPHAN_EDGED,
            "orphan_lonely": _ORPHAN_LONELY,
        },
    )


def _narrator_ids(client: Neo4jClient) -> set[str]:
    rows = client.execute_read("MATCH (n:Narrator) RETURN n.id AS id")
    return {r["id"] for r in rows}


def _transmitted_edge_count(client: Neo4jClient) -> int:
    rows = client.execute_read("MATCH ()-[r:TRANSMITTED_TO]->() RETURN count(r) AS c")
    return int(rows[0]["c"])


def _canonical(tmp_path: Path, ids: list[str]) -> Path:
    return write_narrators_canonical(tmp_path, [{"canonical_id": i} for i in ids])


class TestPruneAgainstRealGraph:
    def test_orphan_with_edges_deleted_zero_degree_canonical_survives(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        _seed_graph(neo4j_client)
        assert _narrator_ids(neo4j_client) == {
            _KEEP_ZERO,
            _KEEP_BUSY,
            _ORPHAN_EDGED,
            _ORPHAN_LONELY,
        }
        assert _transmitted_edge_count(neo4j_client) == 1

        canonical = _canonical(tmp_path, [_KEEP_ZERO, _KEEP_BUSY])
        result = prune_narrators(neo4j_client, canonical)

        # Read the graph back — do not trust the returned count (count>=0 memory).
        survivors = _narrator_ids(neo4j_client)
        # (1) the edge-bearing orphan is gone; (2) the zero-degree canonical survives.
        assert survivors == {_KEEP_ZERO, _KEEP_BUSY}
        assert _ORPHAN_EDGED not in survivors
        assert _KEEP_ZERO in survivors, "a zero-degree canonical narrator was wrongly deleted"
        # DETACH removed the orphan's edge; the canonical endpoint stayed.
        assert _transmitted_edge_count(neo4j_client) == 0
        # Readback-measured summary fields — the deploy#557 contract on a real graph.
        assert result.deleted == 2  # pre(4) - post(2)
        assert result.graph_total == 2  # survivors, re-read
        assert result.orphans == 0  # post-count MUST be 0 (exactly the orphans went)
        assert result.orphans_identified == 2  # would-delete, pre
        assert (
            summary_line(result)
            == "PRUNE_NARRATORS_SUMMARY canonical=2 graph_total=2 orphans=0 deleted=2"
        )

    def test_dry_run_deletes_nothing(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        _seed_graph(neo4j_client)
        before = _narrator_ids(neo4j_client)

        canonical = _canonical(tmp_path, [_KEEP_ZERO, _KEEP_BUSY])
        result = prune_narrators(neo4j_client, canonical, dry_run=True)

        assert result.dry_run is True
        assert result.orphans_identified == 2  # it counts
        assert result.deleted == 0
        assert _narrator_ids(neo4j_client) == before, "dry-run mutated the graph"
        assert _transmitted_edge_count(neo4j_client) == 1

    def test_guard_refuses_empty_canonical_without_touching_graph(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        _seed_graph(neo4j_client)
        before = _narrator_ids(neo4j_client)

        empty = _canonical(tmp_path, [])  # readable, zero canonical_id values
        with pytest.raises(EmptyCanonicalSetError):
            prune_narrators(neo4j_client, empty)

        # The graph is EXACTLY as it was: a bad read wiped nothing.
        assert _narrator_ids(neo4j_client) == before
        assert _transmitted_edge_count(neo4j_client) == 1

    def test_guard_refuses_missing_canonical_without_touching_graph(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        _seed_graph(neo4j_client)
        before = _narrator_ids(neo4j_client)

        with pytest.raises(EmptyCanonicalSetError):
            prune_narrators(neo4j_client, tmp_path / "does_not_exist.parquet")

        assert _narrator_ids(neo4j_client) == before
