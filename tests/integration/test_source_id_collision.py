"""Live-Neo4j collision-safety proof for the canonical source_id scheme (da#82).

These exercise the REAL batch loaders against a live ``neo4j:5`` container and
assert the two identity guarantees the keystone owes:

1. **One node per hadith** — the same logical hadith arriving bare
   (``sunnah:bukhari:...``) and ``hdt:``-prefixed converges on exactly one Hadith
   node, not two. The main#139 streaming-bug shape (``sunnah:sunnah:bukhari:...``)
   is NOT converged onto it: since da#355 it aborts the load, because collapsing it
   cannot be told apart from dropping a valid collection segment (``lk:lk:1``).
2. **In-book ordinal on the edge** — ``APPEARS_IN.hadith_number_in_book`` carries
   the staging ``hadith_number`` (the in-book ordinal), per da#77.

Requires Docker; the ``neo4j_client`` fixture ``pytest.skip``s when Docker is
unreachable (see ``tests/integration/conftest.py``), so this degrades to a clean
SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse.identity import DoubledCorpusPrefixError, hadith_node_id
from tests.test_graph.conftest import write_collections, write_hadiths


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    d = tmp_path / "staging"
    d.mkdir()
    return d


@pytest.fixture
def curated(tmp_path: Path) -> Path:
    d = tmp_path / "curated"
    d.mkdir()
    return d


@pytest.mark.integration
class TestSourceIdCollisionLive:
    def test_prefixed_and_bare_shapes_of_one_hadith_load_one_node(
        self, neo4j_client, staging: Path, curated: Path
    ) -> None:
        """The SAME hadith arriving bare and ``hdt:``-prefixed -> one Hadith node.

        This is the genuine convergence guarantee: ``hdt:`` stripping is
        unambiguous, so both shapes canonicalize to one id. (It is NOT achieved by
        repairing a doubled corpus — see the next test.)
        """
        coords = {
            "collection_name": "bukhari",
            "source_corpus": "sunnah",
            "book_number": 1,
            "chapter_number": 1,
            "hadith_number": 1,
            "matn_en": "the prophet said",
        }
        write_hadiths(staging, [{"source_id": "sunnah:bukhari:1:1:1", **coords}], suffix="bare")
        write_hadiths(
            staging, [{"source_id": "hdt:sunnah:bukhari:1:1:1", **coords}], suffix="prefixed"
        )

        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN count(h) AS n")
        assert rows[0]["n"] == 1, "bare + hdt: shapes produced two nodes — collision NOT prevented"

        id_rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN h.id AS id")
        assert (
            id_rows[0]["id"] == hadith_node_id("sunnah:bukhari:1:1:1") == "hdt:sunnah:bukhari:1:1:1"
        )

    def test_double_prefixed_staging_row_fails_the_load_loudly(
        self, neo4j_client, staging: Path, curated: Path
    ) -> None:
        """A main#139 double-prefixed staging id aborts the load (da#355).

        It used to be silently collapsed onto the correct node — which also meant a
        corpus whose collection is named after itself (``lk:lk:1``) silently lost
        its collection segment. No producer emits this form since da#353, so it is a
        producer defect and must not load at all.
        """
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:sunnah:bukhari:1:1:1",
                    "collection_name": "bukhari",
                    "source_corpus": "sunnah",
                }
            ],
            suffix="streaming",
        )

        with pytest.raises(DoubledCorpusPrefixError, match="doubled leading corpus"):
            load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN count(h) AS n")
        assert rows[0]["n"] == 0, "a rejected id must not have been written"

    def test_idempotent_reload(self, neo4j_client, staging: Path, curated: Path) -> None:
        """Re-running the load does not create a second node (MERGE idempotency)."""
        write_hadiths(
            staging,
            [{"source_id": "lk:bukhari:1:1", "collection_name": "bukhari", "source_corpus": "lk"}],
            suffix="batch",
        )
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN count(h) AS n")
        assert rows[0]["n"] == 1

    def test_in_book_ordinal_on_appears_in_edge(
        self, neo4j_client, staging: Path, curated: Path
    ) -> None:
        """APPEARS_IN.hadith_number_in_book carries the in-book ordinal (da#77)."""
        # in-book ordinal (7) is distinct from book_number (1) so a conflation is
        # observable on the edge property.
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1:2:7",
                    "collection_name": "bukhari",
                    "source_corpus": "sunnah",
                    "book_number": 1,
                    "chapter_number": 2,
                    "hadith_number": 7,
                }
            ],
            suffix="batch",
        )
        write_collections(
            staging,
            [
                {
                    "collection_id": "sunnah:bukhari",
                    "name_en": "Sahih al-Bukhari",
                    "source_corpus": "sunnah",
                }
            ],
            suffix="batch",
        )

        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_edges(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) "
            "RETURN r.hadith_number_in_book AS ordinal, r.book_number AS book"
        )
        assert len(rows) == 1
        assert rows[0]["ordinal"] == 7
        assert rows[0]["book"] == 1
