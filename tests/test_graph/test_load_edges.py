"""Tests for src.graph.load_edges — Neo4j edge loading with mock client."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.graph.load_edges import (
    _APPEARS_IN_QUERY,
    EdgeLoadResult,
    _build_chain_pairs,
    _load_studied_under,
    _studied_under_endpoint,
    load_all_edges,
)
from src.parse.identity import make_canonical_id, narrator_node_id
from src.parse.schemas import NETWORK_EDGE_SCHEMA
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
        assert pairs[0] == {"from_id": "nar:1", "to_id": "nar:2", "position": 0, "hadith_id": "h1"}
        assert pairs[1] == {"from_id": "nar:2", "to_id": "nar:3", "position": 1, "hadith_id": "h1"}

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
        """A non-studentship NETWORK_EDGE producer (mis = isnad transmission) is
        NOT globbed into STUDIED_UNDER — only the muhaddithat edge loads (da#133)."""
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
    filename. The filename allowlist only survives as the default for legacy
    relation-less rows (covered by :class:`TestLoadStudiedUnderGlob`)."""

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
