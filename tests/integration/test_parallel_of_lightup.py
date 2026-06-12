"""Cross-sect PARALLEL_OF light-up against a live Neo4j container (da#100).

Proves the deterministic detector + graph loader materialize PARALLEL_OF edges of
all three kinds the Browse-Parallels view (isnad-graph#964) must span —
intra-sunni, intra-shia, and cross-sect — and that the edge is queryable the way
that view reads it. Also proves the da#77 / main#139 fix: the PARALLEL_OF edge is
keyed on the hadith PAIR (bare MERGE + SET), so a re-run with a changed score
updates the one edge rather than minting a duplicate.

Requires Docker; ``neo4j_client`` ``pytest.skip``s when Docker is unavailable
(see ``tests/integration/conftest.py``), so this degrades to a clean SKIP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.resolve.parallels import detect_parallels
from src.utils.neo4j_client import Neo4jClient
from tests.test_graph.conftest import write_hadiths, write_parallel_links

pytestmark = pytest.mark.integration


def _seed(staging: Path) -> None:
    """Two near-duplicate Sunni hadiths (intra) + a Shia/Sunni pair (cross-sect)."""
    write_hadiths(
        staging,
        [
            # intra-sunni near-duplicate pair
            {
                "source_id": "sunnah:bukhari:1",
                "matn_en": "actions are judged by their intentions",
                "sect": "sunni",
                "source_corpus": "sunnah",
                "collection_name": "bukhari",
            },
            {
                "source_id": "lk:bukhari:1",
                "matn_en": "actions are judged by the intentions",
                "sect": "sunni",
                "source_corpus": "lk",
                "collection_name": "bukhari",
            },
            # cross-sect pair (Shia <-> Sunni)
            {
                "source_id": "thaqalayn:al-kafi:1",
                "matn_en": "purification is half of faith",
                "sect": "shia",
                "source_corpus": "thaqalayn",
                "collection_name": "al-kafi",
            },
            {
                "source_id": "sunnah:muslim:1",
                "matn_en": "purification is half of the faith",
                "sect": "sunni",
                "source_corpus": "sunnah",
                "collection_name": "muslim",
            },
        ],
        suffix="seed",
    )


class TestParallelOfLightup:
    def test_intra_and_cross_sect_edges_materialize(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()
        _seed(staging)

        detect_parallels(staging, threshold=0.5)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        edge_results = load_all_edges(neo4j_client, staging, curated, strict=False)

        parallel = next(r for r in edge_results if r.edge_type == "PARALLEL_OF")
        assert parallel.created == 2
        assert parallel.missing_endpoints == 0

        # Both an intra-sect (cross_sect=false) AND a cross-sect (true) edge exist.
        rows = neo4j_client.execute_read(
            "MATCH (h1:Hadith)-[r:PARALLEL_OF]->(h2:Hadith) "
            "RETURN r.cross_sect AS cross_sect, h1.sect AS s1, h2.sect AS s2"
        )
        assert len(rows) == 2
        cross_flags = sorted(r["cross_sect"] for r in rows)
        assert cross_flags == [False, True]

        # The cross-sect edge genuinely spans the two sects.
        cross = next(r for r in rows if r["cross_sect"])
        assert {cross["s1"], cross["s2"]} == {"sunni", "shia"}

        # The isnad-graph#964 read: parallels regardless of sect — must return both
        # the intra-sunni and the cross-sect relationship (not exclusively cross).
        view = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:PARALLEL_OF]->(:Hadith) "
            "RETURN count(r) AS total, "
            "sum(CASE WHEN r.cross_sect THEN 1 ELSE 0 END) AS cross, "
            "sum(CASE WHEN r.cross_sect THEN 0 ELSE 1 END) AS intra"
        )
        assert view[0]["total"] == 2
        assert view[0]["cross"] == 1
        assert view[0]["intra"] == 1

    def test_reload_with_changed_score_is_idempotent(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        """Re-running with a different score updates the one edge, not a duplicate.

        This is the da#77 / main#139 regression guard: with the old
        property-in-MERGE pattern a changed score minted a SECOND PARALLEL_OF edge
        between the same pair.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()
        _seed(staging)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        # First load: a close-paraphrase score.
        write_parallel_links(
            staging,
            [
                {
                    "hadith_id_a": "lk:bukhari:1",
                    "hadith_id_b": "sunnah:bukhari:1",
                    "similarity_score": 0.70,
                    "variant_type": "close_paraphrase",
                    "cross_sect": False,
                }
            ],
        )
        load_all_edges(neo4j_client, staging, curated, strict=False)

        # Second load of the SAME pair with a higher score.
        write_parallel_links(
            staging,
            [
                {
                    "hadith_id_a": "lk:bukhari:1",
                    "hadith_id_b": "sunnah:bukhari:1",
                    "similarity_score": 0.95,
                    "variant_type": "verbatim",
                    "cross_sect": False,
                }
            ],
        )
        load_all_edges(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read(
            "MATCH ()-[r:PARALLEL_OF]->() "
            "RETURN count(r) AS cnt, collect(r.similarity_score) AS scores, "
            "collect(r.variant_type) AS variants"
        )
        assert rows[0]["cnt"] == 1, "changed score minted a duplicate edge (MERGE-key bug)"
        assert rows[0]["scores"][0] == pytest.approx(0.95)
        assert rows[0]["variants"][0] == "verbatim"
