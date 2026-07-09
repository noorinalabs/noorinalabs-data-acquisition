"""da#351: a load must not leave stale enrich metrics readable as current.

`load_all` is MERGE-only, so `betweenness_centrality` written by an earlier
enrich survives a reload and reads as current -- Cypher has no strict-property
mode, so a reader cannot tell a stale value from a fresh one.

Measured on stg 2026-07-09: all five metrics sat on exactly 150,187 Narrator
nodes while the graph held 160,614. Those 150,187 values were computed against
the pre-reload topology.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.graph import LoadSummary
from src.graph.invalidate import (
    _INVALIDATE_BATCH_QUERY,
    TOPOLOGY_DERIVED_NARRATOR_PROPERTIES,
    invalidate_topology_derived_properties,
)


class _FakeClient:
    """Returns a scripted sequence of per-batch removal counts."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = list(counts)
        self.queries: list[str] = []
        self.params: list[dict[str, Any]] = []

    def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        self.queries.append(query)
        self.params.append(parameters or {})
        removed = self._counts.pop(0) if self._counts else 0
        return [{"invalidated": removed}]


class TestPropertyScope:
    def test_covers_exactly_the_gds_written_metrics(self) -> None:
        """Kept in step with `writeProperty:` in src/enrich/metrics.py."""
        assert set(TOPOLOGY_DERIVED_NARRATOR_PROPERTIES) == {
            "betweenness_centrality",
            "community_id",
            "in_degree",
            "out_degree",
            "pagerank",
        }

    def test_topic_labels_are_not_invalidated(self) -> None:
        """Topics come from the matn text, not the graph structure.

        A reload changes topology; it does not change what a hadith is about.
        Invalidating them would throw away a correct, expensive classification.
        """
        for prop in ("topic_1", "topic_1_score", "topic_2", "topic_3"):
            assert prop not in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES
            assert prop not in _INVALIDATE_BATCH_QUERY

    def test_source_derived_properties_are_not_invalidated(self) -> None:
        """`trustworthiness`/`generation` come from the rijal sources, not GDS."""
        for prop in ("trustworthiness", "generation", "death_year_ah", "mention_count"):
            assert prop not in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES

    def test_predicate_and_remove_clause_cannot_drift(self) -> None:
        """Both are generated from one tuple.

        If the WHERE matched a property the REMOVE did not clear, the batch loop
        would select the same rows forever. Assert every property appears in
        both halves of the query.
        """
        where, _, remove = _INVALIDATE_BATCH_QUERY.partition("REMOVE")
        for prop in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES:
            assert f"n.{prop} IS NOT NULL" in where
            assert f"n.{prop}" in remove

    def test_property_names_are_safe_to_interpolate(self) -> None:
        for prop in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES:
            assert prop.isidentifier()


class TestBatchLoop:
    def test_sums_batches_and_stops_on_zero(self) -> None:
        client = _FakeClient([10_000, 10_000, 187])
        assert invalidate_topology_derived_properties(client) == 20_187  # type: ignore[arg-type]
        # 3 productive batches + 1 terminating probe that returns 0.
        assert len(client.queries) == 4

    def test_unenriched_graph_removes_nothing_in_one_query(self) -> None:
        client = _FakeClient([0])
        assert invalidate_topology_derived_properties(client) == 0  # type: ignore[arg-type]
        assert len(client.queries) == 1

    def test_is_idempotent(self) -> None:
        client = _FakeClient([5])
        first = invalidate_topology_derived_properties(client)  # type: ignore[arg-type]
        second = invalidate_topology_derived_properties(client)  # type: ignore[arg-type]
        assert (first, second) == (5, 0)

    def test_batch_size_is_passed_through(self) -> None:
        client = _FakeClient([0])
        invalidate_topology_derived_properties(client, batch_size=250)  # type: ignore[arg-type]
        assert client.params[0] == {"batch_size": 250}

    def test_empty_result_set_terminates(self) -> None:
        """A driver that returns no rows must not spin forever."""

        class _NoRows:
            def execute_write(self, query: str, parameters: Any = None) -> list[Any]:
                return []

        assert invalidate_topology_derived_properties(_NoRows()) == 0  # type: ignore[arg-type]


class TestLoadSummaryReportsInvalidation:
    def test_summary_carries_the_count(self) -> None:
        summary = LoadSummary(node_results=[], edge_results=[], invalidated_narrators=150_187)
        assert summary.invalidated_narrators == 150_187

    def test_defaults_to_zero(self) -> None:
        assert LoadSummary(node_results=[], edge_results=[]).invalidated_narrators == 0


@pytest.mark.parametrize("prop", TOPOLOGY_DERIVED_NARRATOR_PROPERTIES)
def test_every_property_is_in_the_query(prop: str) -> None:
    assert prop in _INVALIDATE_BATCH_QUERY
