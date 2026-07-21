"""Tests for src.graph.load_edges — Neo4j edge loading with mock client."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import structlog

from src.graph.load_edges import (
    _APPEARS_IN_QUERY,
    EdgeLoadResult,
    _build_chain_pairs,
    _load_studied_under,
    _studied_under_endpoint,
    load_all_edges,
)
from src.parse.identity import DoubledCorpusPrefixError, make_canonical_id, narrator_node_id
from src.parse.schemas import EDGE_RELATION_TRANSMITTED_TO, NETWORK_EDGE_SCHEMA
from src.utils.arabic import normalize_arabic
from tests.test_graph.conftest import (
    MockNeo4jClient,
    write_hadiths,
    write_narrator_mentions_resolved,
    write_parallel_links,
)


def _write_network_edges(staging: Path, suffix: str, rows: list[dict[str, object]]) -> Path:
    """Write a network_edges_<suffix>.parquet."""
    full = []
    for r in rows:
        base: dict[str, object] = {f.name: None for f in NETWORK_EDGE_SCHEMA}
        base.update(r)
        full.append(base)
    arrays = {f.name: [r[f.name] for r in full] for f in NETWORK_EDGE_SCHEMA}
    path = staging / f"network_edges_{suffix}.parquet"
    pq.write_table(pa.table(arrays, schema=NETWORK_EDGE_SCHEMA), path)
    return path


class TestEdgeLoadResult:
    def test_frozen(self) -> None:
        r = EdgeLoadResult("NARRATED", 5, 1, 0)
        with pytest.raises(AttributeError):
            r.edge_type = "other"  # type: ignore[misc]


class TestBuildChainPairs:
    def test_three_narrators_two_pairs(self) -> None:
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:2", "position_in_chain": 1, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:3", "position_in_chain": 2, "hadith_id": "h1"},
        ]
        pairs = _build_chain_pairs(mentions)
        assert len(pairs) == 2
        # hadith_id is canonicalized to match Hadith.id (da#325): "h1" -> "hdt:h1".
        assert pairs[0] == {
            "from_id": "nar:1",
            "to_id": "nar:2",
            "position": 0,
            "hadith_id": "hdt:h1",
        }
        assert pairs[1] == {
            "from_id": "nar:2",
            "to_id": "nar:3",
            "position": 1,
            "hadith_id": "hdt:h1",
        }

    def test_hadith_id_is_canonicalized(self) -> None:
        # da#325: the edge hadith_id must match Hadith.id — canonical (hdt: prefix),
        # not the raw staging id. Post-da#353 the sanadset id carries a book-name
        # digest as its collection segment, never a doubled corpus.
        mentions = [
            {
                "canonical_narrator_id": "nar:1",
                "position_in_chain": 0,
                "hadith_id": "sanadset:e10f73d3eede9edc:2326",
            },
            {
                "canonical_narrator_id": "nar:2",
                "position_in_chain": 1,
                "hadith_id": "sanadset:e10f73d3eede9edc:2326",
            },
        ]
        pairs = _build_chain_pairs(mentions)
        assert len(pairs) == 1
        assert pairs[0]["hadith_id"] == "hdt:sanadset:e10f73d3eede9edc:2326"

    def test_double_prefixed_mention_hadith_id_raises(self) -> None:
        # da#355: a doubled corpus on a mention row is a producer defect. The edge
        # builder must not silently rewrite it onto a different hadith's id.
        mentions = [
            {
                "canonical_narrator_id": "nar:1",
                "position_in_chain": 0,
                "hadith_id": "sanadset:sanadset:0:0:2326",
            },
            {
                "canonical_narrator_id": "nar:2",
                "position_in_chain": 1,
                "hadith_id": "sanadset:sanadset:0:0:2326",
            },
        ]
        with pytest.raises(DoubledCorpusPrefixError):
            _build_chain_pairs(mentions)

    def test_hadith_id_already_canonical_unchanged(self) -> None:
        # Idempotent: a canonical id passes through untouched (hadith_node_id no-op).
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "hdt:s:0:0:1"},
            {"canonical_narrator_id": "nar:2", "position_in_chain": 1, "hadith_id": "hdt:s:0:0:1"},
        ]
        pairs = _build_chain_pairs(mentions)
        assert pairs[0]["hadith_id"] == "hdt:s:0:0:1"

    def test_missing_hadith_id_preserved_as_empty(self) -> None:
        # A missing/empty raw id stays "" — never a bare "hdt:" (da#325 guard).
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0},
            {"canonical_narrator_id": "nar:2", "position_in_chain": 1},
        ]
        pairs = _build_chain_pairs(mentions)
        assert pairs[0]["hadith_id"] == ""

    def test_single_narrator_no_pairs(self) -> None:
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "h1"},
        ]
        assert _build_chain_pairs(mentions) == []

    def test_empty_list(self) -> None:
        assert _build_chain_pairs([]) == []

    def test_unresolved_narrators_filtered(self) -> None:
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "h1"},
            {"canonical_narrator_id": None, "position_in_chain": 1, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:3", "position_in_chain": 2, "hadith_id": "h1"},
        ]
        pairs = _build_chain_pairs(mentions)
        assert len(pairs) == 1
        assert pairs[0]["from_id"] == "nar:1"
        assert pairs[0]["to_id"] == "nar:3"

    def test_sorts_by_position(self) -> None:
        mentions = [
            {"canonical_narrator_id": "nar:3", "position_in_chain": 2, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:2", "position_in_chain": 1, "hadith_id": "h1"},
        ]
        pairs = _build_chain_pairs(mentions)
        assert pairs[0]["from_id"] == "nar:1"
        assert pairs[0]["to_id"] == "nar:2"
        assert pairs[1]["from_id"] == "nar:2"
        assert pairs[1]["to_id"] == "nar:3"

    def test_adjacent_duplicate_narrator_skipped(self) -> None:
        # Two adjacent mentions resolving to the same narrator must not produce a
        # self-loop TRANSMITTED_TO edge (da#148 — 8 live self-loops).
        mentions = [
            {"canonical_narrator_id": "nar:1", "position_in_chain": 0, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:1", "position_in_chain": 1, "hadith_id": "h1"},
            {"canonical_narrator_id": "nar:2", "position_in_chain": 2, "hadith_id": "h1"},
        ]
        pairs = _build_chain_pairs(mentions)
        # nar:1->nar:1 dropped; only nar:1->nar:2 survives.
        assert len(pairs) == 1
        assert pairs[0] == {
            "from_id": "nar:1",
            "to_id": "nar:2",
            "position": 1,
            "hadith_id": "hdt:h1",
        }
        assert all(p["from_id"] != p["to_id"] for p in pairs)


class TestLoadTransmittedTo:
    def test_creates_edges_from_resolved_mentions(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Mock all endpoint checks as existing
        mock_client.set_read_results(
            [
                {"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True},
            ]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt_result = results[0]
        assert tt_result.edge_type == "TRANSMITTED_TO"
        assert tt_result.created == 1

    def test_missing_endpoints_counted(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [
                {"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": False},
            ]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt_result = results[0]
        assert tt_result.missing_endpoints == 1


class TestChainIdentityNoCrossChainFabrication:
    """da#282: TRANSMITTED_TO adjacencies are built WITHIN an isnad-chain, never
    across a hadith's multiple chains.

    A single ``hadith_id`` may carry more than one transmission sequence — today
    ``lk`` emits both an Arabic and an English isnad for one hadith, each numbered
    from ``position_in_chain = 0``. Grouping mentions by hadith and sorting by
    position interleaves the chains and MERGEs narrator pairs that appear in no
    actual chain. ``chain_index`` distinguishes the chains so the loader groups by
    ``(hadith_id, chain_index)``.
    """

    @staticmethod
    def _merged_pairs(mock_client: MockNeo4jClient) -> set[tuple[str, str]]:
        """The (from_id, to_id) pairs MERGEd as TRANSMITTED_TO edges."""
        pairs: set[tuple[str, str]] = set()
        for query, payload in mock_client.calls:
            if "MERGE (n1)-[:TRANSMITTED_TO" in query and isinstance(payload, list):
                for row in payload:
                    pairs.add((row["from_id"], row["to_id"]))
        return pairs

    def test_two_chains_one_hadith_no_cross_chain_edge(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Every endpoint exists — enough True rows to cover any pairing the loader
        # builds (2 within-chain with the fix; 3 interleaved without it).
        mock_client.set_read_results([{"from_exists": True, "to_exists": True} for _ in range(6)])
        # One hadith, TWO isnad-chains. chain 0: A -> B. chain 1: C -> D. Both
        # chains start at position 0, exactly the lk ar/en shape.
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m0",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:B",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "chain_index": 1,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:C",
                },
                {
                    "mention_id": "m3",
                    "hadith_id": "h1",
                    "chain_index": 1,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:D",
                },
            ],
        )
        load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        pairs = self._merged_pairs(mock_client)

        # Within-chain adjacencies are present …
        assert ("nar:A", "nar:B") in pairs
        assert ("nar:C", "nar:D") in pairs
        # … and NO cross-chain adjacency was fabricated. Under the pre-da#282
        # per-hadith flatten these are exactly the edges that appeared:
        # sorted [A(0), C(0), B(1), D(1)] -> (A,C), (C,B), (B,D).
        assert ("nar:A", "nar:C") not in pairs
        assert ("nar:C", "nar:B") not in pairs
        assert ("nar:B", "nar:D") not in pairs
        assert pairs == {("nar:A", "nar:B"), ("nar:C", "nar:D")}

    def test_single_isnad_hadith_unaffected(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A single-chain hadith (all chain_index 0) builds the same edges as before."""
        mock_client.set_read_results([{"from_exists": True, "to_exists": True} for _ in range(4)])
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m0",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:B",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 2,
                    "canonical_narrator_id": "nar:C",
                },
            ],
        )
        load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        assert self._merged_pairs(mock_client) == {("nar:A", "nar:B"), ("nar:B", "nar:C")}

    def test_absent_chain_index_defaults_to_single_chain(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A legacy row that omits chain_index (→ null) coalesces to chain 0, so a
        pre-da#282 file still builds one contiguous chain rather than fragmenting."""
        mock_client.set_read_results([{"from_exists": True, "to_exists": True} for _ in range(4)])
        # Explicit None → a genuine null chain_index column in the parquet, the
        # legacy pre-da#282 shape; the loader must coalesce it to chain 0.
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m0",
                    "hadith_id": "h1",
                    "chain_index": None,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "chain_index": None,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:B",
                },
            ],
        )
        load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        assert self._merged_pairs(mock_client) == {("nar:A", "nar:B")}

    def test_two_chains_narrated_dedupes_to_one_when_openers_match(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Both chains of a hadith emit a NARRATED edge from their position-0
        narrator; when the openers resolve to the same canonical (the lk ar/en
        case) they MERGE to one edge, so per-chain NARRATED never inflates."""
        mock_client.set_read_results(
            [{"narrator_exists": True, "hadith_exists": True} for _ in range(4)]
        )
        write_narrator_mentions_resolved(
            curated_dir,
            [
                # chain 0 opener and chain 1 opener are the SAME canonical narrator.
                {
                    "mention_id": "m0",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "chain_index": 0,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:B",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "chain_index": 1,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m3",
                    "hadith_id": "h1",
                    "chain_index": 1,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:C",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated = next(r for r in results if r.edge_type == "NARRATED")
        # Two chain-openers, both nar:A -> hdt:h1 → one MERGEd edge after dedup.
        narrated_targets = {
            (row["narrator_id"], row["hadith_id"])
            for query, payload in mock_client.calls
            if "NARRATED" in query and isinstance(payload, list)
            for row in payload
        }
        assert ("nar:A", "hdt:h1") in narrated_targets
        assert narrated.created >= 1


class TestLoadNarrated:
    def test_first_narrator_per_hadith(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Position 0 should be chosen as the "first narrator"
        mock_client.set_read_results(
            [
                {
                    "narrator_id": "nar:1",
                    "hadith_id": "hdt:h1",
                    "narrator_exists": True,
                    "hadith_exists": True,
                },
            ]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated_result = results[1]
        assert narrated_result.edge_type == "NARRATED"
        assert narrated_result.created == 1

    def test_unresolved_chain_emits_no_narrated_no_fabrication(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#153 item #2 / ADR-004: a hadith with no resolved position-0 narrator
        gets ZERO NARRATED edges — the load layer never fabricates one.

        The 51 NARRATED-less hadiths are an upstream NER/mention-coverage gap: a
        hadith whose mentions carry no ``canonical_narrator_id`` has no resolved
        narrator to attribute narration to. The contract is that ``_load_narrated``
        skips such rows (no fabricated/synthetic narrator) rather than inventing a
        position-0 endpoint — closing the gap is producer-side NER work, not a load
        guard. This pins the no-fabrication behaviour.
        """
        write_narrator_mentions_resolved(
            curated_dir,
            [
                # Mentions exist, but none resolved to a canonical narrator id.
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": None,
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h1",
                    "position_in_chain": 1,
                    "canonical_narrator_id": None,
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated_result = next(r for r in results if r.edge_type == "NARRATED")
        # No fabricated narrator: zero edges created, and nothing counted as a
        # missing-endpoint either (the row never enters the batch at all).
        assert narrated_result.created == 0
        assert narrated_result.missing_endpoints == 0
        # And no NARRATED write was ever issued to the client.
        assert not any("NARRATED" in str(q) for q, _ in mock_client.calls)


class TestNarratedProvenance:
    """da#228 / ADR-004 item #3 — curated muhaddithat orphan-links land as NARRATED
    edges carrying first-class ``provenance``.

    The curated mention-link producer (``src.resolve.muhaddithat_links``) writes a
    superset resolved-mentions file the existing NARRATED loader reads unchanged;
    the loader propagates the link's ``provenance`` onto the edge via a property-
    less MERGE + coalesce-preserve SET (null-safe, idempotent).
    """

    def test_orphan_links_load_with_provenance(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        from src.graph.load_edges import _load_narrated
        from src.resolve.muhaddithat_links import (
            MUHADDITHAT_ORPHAN_LINKS,
            build_muhaddithat_mention_links,
        )

        # Producer writes the curated links into the curated dir (no canonical
        # guard here — all 8 emitted), which the loader globs alongside any main
        # resolved-mentions file.
        build_muhaddithat_mention_links(curated_dir)

        n = len(MUHADDITHAT_ORPHAN_LINKS)
        mock_client.set_read_results(
            [{"narrator_exists": True, "hadith_exists": True} for _ in range(n)]
        )

        result = _load_narrated(mock_client, staging_dir, curated_dir=curated_dir, strict=False)
        assert result.edge_type == "NARRATED"
        assert result.created == n  # one NARRATED edge per orphan link

        # The single write call carries provenance on every row, and the query SETs
        # it onto the edge.
        write_calls = [
            (q, b) for q, b in mock_client.calls if isinstance(b, list) and "[r:NARRATED]" in str(q)
        ]
        assert len(write_calls) == 1
        query, batch = write_calls[0]
        assert "SET r.provenance = coalesce(row.provenance, r.provenance)" in query
        assert len(batch) == n
        assert all(item["provenance"] for item in batch)
        assert {item["provenance"] for item in batch} == {
            link.provenance for link in MUHADDITHAT_ORPHAN_LINKS
        }

    def test_chain_mention_without_provenance_is_null_safe(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # An ordinary resolved chain mention carries no provenance column → the
        # batch row's provenance is None and the coalesce-preserve SET leaves the
        # edge property absent rather than failing.
        from src.graph.load_edges import _load_narrated

        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
            ],
        )
        mock_client.set_read_results([{"narrator_exists": True, "hadith_exists": True}])
        result = _load_narrated(mock_client, staging_dir, curated_dir=curated_dir, strict=False)
        assert result.created == 1
        write_calls = [
            (q, b) for q, b in mock_client.calls if isinstance(b, list) and "[r:NARRATED]" in str(q)
        ]
        assert len(write_calls) == 1
        _query, batch = write_calls[0]
        assert batch[0]["provenance"] is None

    def test_curated_link_never_fabricates_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#228 integrity bug (reviewer Nikolaos): a curated NARRATED-only link
        co-located on the SAME hadith as a real NER chain mention must NOT produce a
        TRANSMITTED_TO pair between the orphan narrator and the chain's Companion.

        Reproduces the fabrication path: the curated link's ``hadith_id`` is a real
        ``sunnah`` hadith that ALSO carries its own resolved chain mentions (the main
        resolved-mentions file). Pre-fix, ``_load_transmitted_to`` grouped both by
        hadith and paired the orphan with the Companion into a wrong-attribution
        edge. The provenance filter must exclude the curated link from chain pairing
        while the legitimate Companion→Companion pair and the NARRATED edge survive.
        """
        from src.resolve.muhaddithat_links import (
            MUHADDITHAT_ORPHAN_LINKS,
            build_muhaddithat_mention_links,
            canonical_id_for,
        )

        link = MUHADDITHAT_ORPHAN_LINKS[0]
        curated_nid = canonical_id_for(link.name_ar)

        # Curated NARRATED-only link for the orphan on its hadith.
        build_muhaddithat_mention_links(curated_dir, links=(link,))
        # A real resolved chain on the SAME hadith id (two Companion mentions).
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "c0",
                    "hadith_id": link.hadith_id,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:companion-a",
                },
                {
                    "mention_id": "c1",
                    "hadith_id": link.hadith_id,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:companion-b",
                },
            ],
        )

        mock_client.set_read_results(
            [
                {
                    "from_exists": True,
                    "to_exists": True,
                    "narrator_exists": True,
                    "hadith_exists": True,
                }
                for _ in range(4)
            ]
        )

        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt_result = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        narrated_result = next(r for r in results if r.edge_type == "NARRATED")

        # The legitimate Companion→Companion pair survives; the orphan link does not
        # add a second (fabricated) pair.
        assert tt_result.created == 1
        # NARRATED narration still lands for the hadith.
        assert narrated_result.created >= 1

        # No TRANSMITTED_TO pair involves the curated orphan narrator.
        tt_writes = [
            b for q, b in mock_client.calls if isinstance(b, list) and "[:TRANSMITTED_TO" in str(q)
        ]
        endpoints = {
            nid for batch in tt_writes for pair in batch for nid in (pair["from_id"], pair["to_id"])
        }
        assert curated_nid not in endpoints, "orphan narrator must not enter a TRANSMITTED_TO pair"


class TestChainEdgesReadResolvedFromCurated:
    """Regression for da#141 / main#601 criterion #1 — NARRATED/TRANSMITTED_TO = 0.

    ``resolve.run_all`` writes ``narrator_mentions_resolved.parquet`` (the only
    mentions carrying ``canonical_narrator_id``, which both chain edges key on)
    to the **curated** dir, not staging. ``load_all_edges`` used to glob staging
    only, so on a real orchestrated load the chain edges silently fell back to
    the raw, canonical-id-less staging mentions and created ZERO edges — exactly
    the staging gap #601 found. These assert the loaders read the resolved file
    from ``curated_dir`` even when staging holds no resolved mentions at all.
    """

    def _resolved_chain_rows(self) -> list[dict[str, object]]:
        return [
            {
                "mention_id": "m1",
                "hadith_id": "h1",
                "position_in_chain": 0,
                "canonical_narrator_id": "nar:1",
            },
            {
                "mention_id": "m2",
                "hadith_id": "h1",
                "position_in_chain": 1,
                "canonical_narrator_id": "nar:2",
            },
        ]

    def test_transmitted_to_reads_resolved_from_curated(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [{"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True}]
        )
        # Resolved mentions live in CURATED (the resolve-output dir); staging has none.
        write_narrator_mentions_resolved(curated_dir, self._resolved_chain_rows())

        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt_result = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        assert tt_result.created == 1, "TRANSMITTED_TO must read resolved mentions from curated_dir"

    def test_narrated_reads_resolved_from_curated(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [
                {
                    "narrator_id": "nar:1",
                    "hadith_id": "hdt:h1",
                    "narrator_exists": True,
                    "hadith_exists": True,
                }
            ]
        )
        write_narrator_mentions_resolved(curated_dir, self._resolved_chain_rows())

        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated_result = next(r for r in results if r.edge_type == "NARRATED")
        assert narrated_result.created == 1, "NARRATED must read resolved mentions from curated_dir"


class TestLoadAppearsIn:
    def test_hadith_to_collection_edges(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [
                {
                    "hadith_id": "hdt:h-1",
                    "collection_id": "col:bukhari",
                    "hadith_exists": True,
                    "collection_exists": True,
                },
            ]
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "h-1",
                    "collection_name": "bukhari",
                    "book_number": 1,
                    "chapter_number": 1,
                    "hadith_number": 1,
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        ai_result = results[2]
        assert ai_result.edge_type == "APPEARS_IN"
        assert ai_result.created == 1

    def test_appears_in_edge_uses_canonical_property_key(self) -> None:
        # The APPEARS_IN edge property MUST be the ig#935-canonical
        # ``hadith_number_in_book`` mapped from ``row.hadith_number`` (matches
        # AppearsIn model + isnad-graph src/models/edges.py), NOT the legacy bare
        # ``hadith_number`` (da#65). Post-da#77 it is SET after the MERGE with the
        # streaming path's coalesce-preserve contract, so assert that SET form.
        assert (
            "r.hadith_number_in_book = coalesce(row.hadith_number, r.hadith_number_in_book)"
            in _APPEARS_IN_QUERY
        )
        # The legacy bare ``hadith_number`` edge key must never appear.
        assert "hadith_number:" not in _APPEARS_IN_QUERY

    def test_appears_in_merge_has_no_property_key_null_unsafe(self) -> None:
        # da#77: positional props MUST be SET after the MERGE, never inside the
        # MERGE relationship pattern — Neo4j aborts a MERGE on a null property,
        # and scraped hadiths carry null ``hadith_number`` until da#72. Guard the
        # MERGE line so the null-unsafe keyed form can't be reintroduced.
        merge_line = next(line for line in _APPEARS_IN_QUERY.splitlines() if "MERGE (" in line)
        assert merge_line.strip() == "MERGE (h)-[r:APPEARS_IN]->(c)"
        assert "{" not in merge_line  # no inline property map on the MERGE pattern

    def test_missing_collection_name_skipped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1", "collection_name": ""},  # empty
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        ai_result = results[2]
        assert ai_result.skipped >= 1


class TestLoadParallelOf:
    def test_direction_enforcement(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # id_b < id_a alphabetically, should be swapped
        mock_client.set_read_results(
            [
                {"id_a": "hdt:aaa", "id_b": "hdt:zzz", "a_exists": True, "b_exists": True},
            ]
        )
        write_parallel_links(
            staging_dir,
            [
                {"hadith_id_a": "zzz", "hadith_id_b": "aaa", "similarity_score": 0.95},
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        po_result = results[3]
        assert po_result.edge_type == "PARALLEL_OF"
        # Check that the batch was direction-corrected
        write_calls = [
            (q, b) for q, b in mock_client.calls if isinstance(b, list) and b and "id_a" in b[0]
        ]
        if write_calls:
            batch = write_calls[-1][1]
            assert batch[0]["id_a"] < batch[0]["id_b"]

    def test_graceful_skip_missing_file(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        po_result = results[3]
        assert po_result.created == 0
        assert po_result.missing_endpoints == 0

    def test_strict_raises_on_missing(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Need to also provide narrator_mentions and hadiths for earlier edge types
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
            ],
        )
        write_hadiths(staging_dir, [{"source_id": "h-1"}])
        # Set read results for TRANSMITTED_TO and NARRATED checks
        mock_client.set_read_results([])
        with pytest.raises(FileNotFoundError, match="parallel_links"):
            load_all_edges(mock_client, staging_dir, curated_dir, strict=True)


class TestLoadStudiedUnder:
    def test_graceful_skip_when_missing(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        su_result = results[4]
        assert su_result.edge_type == "STUDIED_UNDER"
        assert su_result.created == 0


class TestLoadAllEdges:
    def test_returns_six_results(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        assert len(results) == 6
        expected_types = [
            "TRANSMITTED_TO",
            "NARRATED",
            "APPEARS_IN",
            "PARALLEL_OF",
            "STUDIED_UNDER",
            "GRADED_BY",
        ]
        assert [r.edge_type for r in results] == expected_types

    def test_edge_load_result_counting(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        for r in results:
            assert isinstance(r, EdgeLoadResult)
            assert r.created >= 0
            assert r.skipped >= 0
            assert r.missing_endpoints >= 0


class TestStudiedUnderEndpoint:
    def test_resolves_by_name_to_canonical_id(self) -> None:
        # Canonical Narrator nodes are keyed by name, not by the source id.
        assert _studied_under_endpoint("مالك", "42") == make_canonical_id(normalize_arabic("مالك"))

    def test_falls_back_to_external_id_when_name_missing(self) -> None:
        assert _studied_under_endpoint(None, "42") == narrator_node_id("42")
        assert _studied_under_endpoint("   ", "42") == narrator_node_id("42")

    def test_none_when_both_absent(self) -> None:
        assert _studied_under_endpoint(None, None) is None


class TestLoadStudiedUnderGlob:
    def test_globs_all_sources_and_resolves_by_name(self, staging_dir: Path) -> None:
        # Two source files: muhaddithat (A->B) and itqan (B->C).
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "STUDIED_UNDER",
                }
            ],
        )
        _write_network_edges(
            staging_dir,
            "itqan",
            [
                {
                    "from_narrator_name": "ب",
                    "to_narrator_name": "ج",
                    "source": "itqan",
                    "from_external_id": "2",
                    "to_external_id": "3",
                    "relation": "STUDIED_UNDER",
                }
            ],
        )
        client = MockNeo4jClient()
        client.set_read_results([{"from_exists": True, "to_exists": True} for _ in range(2)])
        result = _load_studied_under(client, staging_dir)
        assert result.created == 2  # both files loaded
        # The written batch keys endpoints by canonical name id, not nar:<source-id>.
        write_calls = [b for q, b in client.calls if "MERGE" in q and isinstance(b, list)]
        batch = write_calls[-1]
        assert {
            "from_id": make_canonical_id(normalize_arabic("أ")),
            "to_id": make_canonical_id(normalize_arabic("ب")),
        } in batch
        assert {
            "from_id": make_canonical_id(normalize_arabic("ب")),
            "to_id": make_canonical_id(normalize_arabic("ج")),
        } in batch

    def test_dedups_same_pair_across_files(self, staging_dir: Path) -> None:
        edge = {
            "from_narrator_name": "أ",
            "to_narrator_name": "ب",
            "source": "x",
            "from_external_id": "1",
            "to_external_id": "2",
            "relation": "STUDIED_UNDER",
        }
        _write_network_edges(staging_dir, "muhaddithat", [edge])
        _write_network_edges(staging_dir, "itqan", [dict(edge, source="itqan")])
        client = MockNeo4jClient()
        client.set_read_results([{"from_exists": True, "to_exists": True}])
        result = _load_studied_under(client, staging_dir)
        assert result.created == 1  # the duplicate pair is collapsed

    def test_skips_when_no_edge_files(self, staging_dir: Path) -> None:
        result = _load_studied_under(MockNeo4jClient(), staging_dir)
        assert result.edge_type == "STUDIED_UNDER"
        assert result.created == 0

    def test_excludes_non_studentship_source(self, staging_dir: Path) -> None:
        """A non-studentship NETWORK_EDGE producer (mis = isnad transmission, declaring
        TRANSMITTED_TO) is NOT globbed into STUDIED_UNDER — only the muhaddithat edge
        loads (da#133)."""
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "STUDIED_UNDER",
                }
            ],
        )
        _write_network_edges(
            staging_dir,
            "mis",
            [
                {
                    "from_narrator_name": "X",
                    "to_narrator_name": "Y",
                    "source": "mis",
                    "hadith_id": "mis:sahih_muslim:1:1",
                    "relation": "TRANSMITTED_TO",
                }
            ],
        )
        client = MockNeo4jClient()
        client.set_read_results([{"from_exists": True, "to_exists": True}])
        result = _load_studied_under(client, staging_dir)
        assert result.created == 1  # only the muhaddithat studentship edge

        batch = [b for q, b in client.calls if "MERGE" in q and isinstance(b, list)][-1]
        # The mis isnad pair never reaches the STUDIED_UNDER batch.
        assert {
            "from_id": make_canonical_id(normalize_arabic("X")),
            "to_id": make_canonical_id(normalize_arabic("Y")),
        } not in batch
        assert {
            "from_id": make_canonical_id(normalize_arabic("أ")),
            "to_id": make_canonical_id(normalize_arabic("ب")),
        } in batch


class TestStudiedUnderRelationRouting:
    """da#133: the loader routes by the declared ``relation`` field, not the
    filename. da#157 removed the relation-less default — every row MUST declare a
    relation (fail-fast covered by :class:`TestStudiedUnderRelationRequired`)."""

    def test_relation_field_includes_non_allowlisted_source(self, staging_dir: Path) -> None:
        """A NETWORK_EDGE file whose slug is NOT in the studentship allowlist still
        loads when its rows DECLARE STUDIED_UNDER — proving routing keys on the
        field, not the filename, so a new studentship source needs no allowlist
        edit."""
        _write_network_edges(
            staging_dir,
            "newsource",  # deliberately not muhaddithat/itqan
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "newsource",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "STUDIED_UNDER",
                }
            ],
        )
        client = MockNeo4jClient()
        client.set_read_results([{"from_exists": True, "to_exists": True}])
        result = _load_studied_under(client, staging_dir)
        assert result.created == 1

    def test_transmitted_to_relation_excluded_despite_studentship_filename(
        self, staging_dir: Path
    ) -> None:
        """A row that DECLARES TRANSMITTED_TO is kept off STUDIED_UNDER even when it
        lives in an allowlisted (``muhaddithat``) file — the explicit relation
        overrides the filename so an isnad-transmission row can never be
        mislabeled."""
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "X",
                    "to_narrator_name": "Y",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "TRANSMITTED_TO",
                }
            ],
        )
        client = MockNeo4jClient()
        result = _load_studied_under(client, staging_dir)
        assert result.created == 0  # the transmission row is not loaded as STUDIED_UNDER

    def test_explicit_relation_routes_mixed_file(self, staging_dir: Path) -> None:
        """Within one file, only the STUDIED_UNDER-declaring row loads; the
        TRANSMITTED_TO-declaring row is skipped."""
        _write_network_edges(
            staging_dir,
            "mis",  # not allowlisted; both rows carry explicit relations
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "mis",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "STUDIED_UNDER",
                },
                {
                    "from_narrator_name": "X",
                    "to_narrator_name": "Y",
                    "source": "mis",
                    "from_external_id": "3",
                    "to_external_id": "4",
                    "relation": "TRANSMITTED_TO",
                },
            ],
        )
        client = MockNeo4jClient()
        client.set_read_results([{"from_exists": True, "to_exists": True}])
        result = _load_studied_under(client, staging_dir)
        assert result.created == 1
        batch = [b for q, b in client.calls if "MERGE" in q and isinstance(b, list)][-1]
        assert {
            "from_id": make_canonical_id(normalize_arabic("أ")),
            "to_id": make_canonical_id(normalize_arabic("ب")),
        } in batch
        assert {
            "from_id": make_canonical_id(normalize_arabic("X")),
            "to_id": make_canonical_id(normalize_arabic("Y")),
        } not in batch


class TestStudiedUnderRelationRequired:
    """da#157: a NETWORK_EDGE that declares NO relation is REFUSED (raises), never
    silently defaulted to STUDIED_UNDER. Removes the footgun where the next
    isnad-transmission producer that forgot to set ``relation`` would have its edges
    mis-routed onto the studentship relation."""

    def test_relationless_row_raises_for_allowlisted_filename(self, staging_dir: Path) -> None:
        """A relation-less row in a ``muhaddithat`` (formerly allowlisted) file no
        longer silently becomes STUDIED_UNDER — it raises. This is the exact trap
        da#157 closes."""
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": None,  # producer forgot to declare the relation
                }
            ],
        )
        with pytest.raises(ValueError, match=r"declares no `relation`"):
            _load_studied_under(MockNeo4jClient(), staging_dir)

    def test_relationless_row_raises_for_unknown_source(self, staging_dir: Path) -> None:
        """A relation-less row in a non-allowlisted file used to be silently dropped;
        now it raises loudly so the producer defect cannot pass unnoticed."""
        _write_network_edges(
            staging_dir,
            "newsource",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "newsource",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": None,
                }
            ],
        )
        with pytest.raises(ValueError, match=r"declares no `relation`"):
            _load_studied_under(MockNeo4jClient(), staging_dir)

    def test_blank_relation_raises(self, staging_dir: Path) -> None:
        """A whitespace-only relation is treated as undeclared and refused."""
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "   ",
                }
            ],
        )
        with pytest.raises(ValueError, match=r"declares no `relation`"):
            _load_studied_under(MockNeo4jClient(), staging_dir)

    def test_unknown_relation_value_raises(self, staging_dir: Path) -> None:
        """A relation value outside the recognized set is refused rather than
        silently skipped — guards against typos like ``STUDENT_OF``."""
        _write_network_edges(
            staging_dir,
            "muhaddithat",
            [
                {
                    "from_narrator_name": "أ",
                    "to_narrator_name": "ب",
                    "source": "muhaddithat",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "STUDENT_OF",
                }
            ],
        )
        with pytest.raises(ValueError, match=r"unknown relation 'STUDENT_OF'"):
            _load_studied_under(MockNeo4jClient(), staging_dir)

    def test_transmission_producer_never_mislabeled_studied_under(self, staging_dir: Path) -> None:
        """A transmission producer (mis) whose every row declares TRANSMITTED_TO
        contributes ZERO STUDIED_UNDER edges — its isnad edges are never mislabeled
        as studentship."""
        _write_network_edges(
            staging_dir,
            "mis",
            [
                {
                    "from_narrator_name": "X",
                    "to_narrator_name": "Y",
                    "source": "mis",
                    "from_external_id": "1",
                    "to_external_id": "2",
                    "relation": "TRANSMITTED_TO",
                },
                {
                    "from_narrator_name": "Y",
                    "to_narrator_name": "Z",
                    "source": "mis",
                    "from_external_id": "2",
                    "to_external_id": "3",
                    "relation": "TRANSMITTED_TO",
                },
            ],
        )
        result = _load_studied_under(MockNeo4jClient(), staging_dir)
        assert result.created == 0


class TestParallelOfConformance:
    """PARALLEL_OF load conformance — empty / zero-edge loads must be surfaced,
    and a production-shaped links file must actually load edges (da#160)."""

    def test_production_shaped_links_load_edges(self, staging_dir: Path, curated_dir: Path) -> None:
        # A production-shaped batch of detected parallels (many pairs) loads one
        # PARALLEL_OF edge each when both endpoints exist.
        links = [
            {"hadith_id_a": f"sunnah:c{c}:h0", "hadith_id_b": f"sunnah:c{c}:h1"} for c in range(25)
        ]
        write_parallel_links(staging_dir, links)
        client = MockNeo4jClient()
        client.set_read_results([{"a_exists": True, "b_exists": True}] * len(links))

        results = load_all_edges(client, staging_dir, curated_dir, strict=False)
        po = next(r for r in results if r.edge_type == "PARALLEL_OF")
        assert po.created == 25
        assert po.missing_endpoints == 0

    def test_empty_links_file_warns(self, staging_dir: Path, curated_dir: Path) -> None:
        # An empty parallel_links.parquet (the da#160 state) must surface a warning,
        # not pass silently, and the conformance summary must flag PARALLEL_OF.
        write_parallel_links(staging_dir, [])
        client = MockNeo4jClient()

        with structlog.testing.capture_logs() as logs:
            results = load_all_edges(client, staging_dir, curated_dir, strict=False)

        po = next(r for r in results if r.edge_type == "PARALLEL_OF")
        assert po.created == 0
        events = {(e["event"], e.get("log_level")) for e in logs}
        assert ("parallel_of_no_edges", "warning") in events
        conformance = next(e for e in logs if e["event"] == "edge_load_conformance")
        assert conformance["log_level"] == "warning"
        assert "PARALLEL_OF" in conformance["zero_count_edge_types"]

    def test_all_endpoints_missing_warns(self, staging_dir: Path, curated_dir: Path) -> None:
        # Links present but every endpoint missing (e.g. an id-scheme mismatch) is
        # the silent-failure mode the counter must catch.
        write_parallel_links(
            staging_dir, [{"hadith_id_a": "sunnah:x:1", "hadith_id_b": "sunnah:y:1"}]
        )
        client = MockNeo4jClient()
        client.set_read_results([{"a_exists": False, "b_exists": False}])

        with structlog.testing.capture_logs() as logs:
            load_all_edges(client, staging_dir, curated_dir, strict=False)

        events = {(e["event"], e.get("log_level")) for e in logs}
        assert ("parallel_of_loaded_zero", "warning") in events


class TestCompositionGateOnChainEdges:
    """da#333 — the canonical-hadith composition gate applied to the chain-edge path.

    The node loader excludes fawaz's six-books Hadith nodes (deduped to the lk
    spine) but the narrator-mention chain path did not, so fawaz's NER-derived
    six-books chains loaded as ~196k orphaned TRANSMITTED_TO edges keyed
    ``fawaz:<book>:<n>`` with no Hadith node. The gate drops a mention whose
    hadith would not be a canonical node — mirroring the node dedup — while
    leaving lk / other canonical chains untouched.

    Carve-out: ``mis`` is chains-only for Sahih Muslim and its transmission edges
    are produced as ``network_edges_mis.parquet`` (keyed ``mis:sahih_muslim:...``),
    NOT as narrator mentions — NER never runs over mis. They therefore never reach
    this mention path, so the gate cannot drop them; ``test_mis_network_edges_...``
    pins that the co-present mis edge file is handled by its own loader unchanged.
    """

    def test_fawaz_six_books_mention_drops_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        # Dropped at the source: no pair built, no missing-endpoint counted, and no
        # TRANSMITTED_TO write ever issued to the client.
        assert tt.created == 0
        assert tt.missing_endpoints == 0
        assert not any(
            "TRANSMITTED_TO" in str(q) and "MERGE" in str(q) for q, _ in mock_client.calls
        )

    def test_fawaz_six_books_mention_drops_narrated(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated = next(r for r in results if r.edge_type == "NARRATED")
        assert narrated.created == 0
        assert narrated.missing_endpoints == 0
        assert not any("NARRATED" in str(q) for q, _ in mock_client.calls)

    def test_fawaz_unique_collection_mention_kept_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # fawaz's UNIQUE collections (nawawi/dehlawi/qudsi) DO load nodes, so their
        # chains must survive the gate — only the six-books duplicates are dropped.
        mock_client.set_read_results(
            [{"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True}]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "fawaz:nawawi:5",
                    "source_corpus": "fawaz",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "fawaz:nawawi:5",
                    "source_corpus": "fawaz",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        assert tt.created == 1

    def test_lk_mention_kept_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [{"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True}]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "lk:bukhari:1",
                    "source_corpus": "lk",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "lk:bukhari:1",
                    "source_corpus": "lk",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        assert tt.created == 1

    def test_lk_mention_kept_narrated(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        mock_client.set_read_results(
            [
                {
                    "narrator_id": "nar:1",
                    "hadith_id": "hdt:lk:bukhari:1",
                    "narrator_exists": True,
                    "hadith_exists": True,
                }
            ]
        )
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "lk:bukhari:1",
                    "source_corpus": "lk",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                }
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated = next(r for r in results if r.edge_type == "NARRATED")
        assert narrated.created == 1

    def test_mis_network_edges_unaffected_by_mention_gate(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """The mis carve-out. mis transmission edges live in network_edges_mis.parquet
        (declared relation TRANSMITTED_TO) and are keyed ``mis:sahih_muslim:...`` — a
        NON-canonical id (mis loads no Hadith nodes). They never enter the
        narrator-mention path, so the da#333 gate cannot touch them: the mention-path
        gate drops the co-present fawaz six-books chain, while the mis network-edge
        file is handled by its own loader exactly as before (routed off STUDIED_UNDER
        by its declared relation — the da#133 behaviour, unchanged, no error).
        """
        # fawaz six-books mention — must be dropped by the gate.
        write_narrator_mentions_resolved(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        # mis's real transmission edges — network_edges, keyed to a mis hadith id.
        _write_network_edges(
            staging_dir,
            "mis",
            [
                {
                    "from_narrator_name": "Yahya",
                    "to_narrator_name": "Malik",
                    "hadith_id": "mis:sahih_muslim:1:5",
                    "source": "mis",
                    "relation": EDGE_RELATION_TRANSMITTED_TO,
                }
            ],
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        studied = next(r for r in results if r.edge_type == "STUDIED_UNDER")
        # fawaz six-books dropped from the mention path ...
        assert tt.created == 0
        # ... and the mis network-edge file is routed by relation (skipped off
        # STUDIED_UNDER, da#133) without error — the gate did not perturb it.
        assert studied.created == 0


class TestCrossEditionDedupVisibleToEdges:
    """da#373 — the chain-edge loader gates on the node loader's REAL kept set, so a
    cross-edition-deduped sanadset hadith yields no TRANSMITTED_TO / NARRATED edge.

    Without this the edge loader re-derived only the composition gate from the id
    string (``is_canonical_hadith_id`` keeps every sanadset id — the matn dedup is
    invisible to it), so a repaired matn key would emit ~196k chain edges against
    Hadith nodes the node loader dropped — the da#333 orphan bug on the axis nobody
    had closed. Endpoints are set to exist, so the ONLY reason an edge is not
    created is the da#373 drop: that is what makes these tests bite.
    """

    _SHARED_MATN = "انما الاعمال بالنيات"
    _UNIQUE_MATN = "لا ضرر ولا ضرار"

    def _write_curated_plus_dup(self, staging_dir: Path) -> None:
        # lk curated edition occupies the identity; the sanadset copy shares its matn
        # and is therefore cross-edition-deduped away by the node loader.
        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": self._SHARED_MATN}],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "matn_ar": self._SHARED_MATN,
                }
            ],
            suffix="sanadset_dup",
        )

    def test_deduped_sanadset_mention_drops_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        self._write_curated_plus_dup(staging_dir)
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        # Both narrator endpoints exist — so a pair, if built, WOULD create an edge.
        mock_client.set_read_results(
            [{"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True}]
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        assert tt.created == 0
        assert not any(
            "[:TRANSMITTED_TO" in str(q) and "MERGE" in str(q) for q, _ in mock_client.calls
        )

    def test_deduped_sanadset_mention_drops_narrated(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        self._write_curated_plus_dup(staging_dir)
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
            ],
        )
        mock_client.set_read_results([{"narrator_exists": True, "hadith_exists": True}])
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        narrated = next(r for r in results if r.edge_type == "NARRATED")
        assert narrated.created == 0
        assert not any("[r:NARRATED]" in str(q) for q, _ in mock_client.calls)

    def test_non_deduped_sanadset_mention_keeps_transmitted_to(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # A sanadset hadith with a UNIQUE matn (no curated twin) IS loaded, so its
        # chain MUST survive — the fix must not over-drop legitimately-loaded chains.
        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": self._SHARED_MATN}],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "matn_ar": self._UNIQUE_MATN,
                }
            ],
            suffix="sanadset_uniq",
        )
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:1",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:2",
                },
            ],
        )
        mock_client.set_read_results(
            [{"from_id": "nar:1", "to_id": "nar:2", "from_exists": True, "to_exists": True}]
        )
        results = load_all_edges(mock_client, staging_dir, curated_dir, strict=False)
        tt = next(r for r in results if r.edge_type == "TRANSMITTED_TO")
        assert tt.created == 1
