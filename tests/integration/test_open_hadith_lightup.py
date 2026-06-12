"""Live-Neo4j light-up for the Open-Hadith adapter (da#87).

Parses small REAL Open-Hadith-Data samples (Musnad Ahmad + Sunan al-Nasai,
mhashim6/Open-Hadith-Data) through the real batch loaders into a live ``neo4j:5``
container and asserts the corpus lands as a usable graph: Hadith **and**
Collection nodes (one Collection per book, the gap this light-up closes), all
``sect=sunni`` / ``source_corpus=open_hadith`` tagged, with ``APPEARS_IN`` edges
resolving each hadith to its own book — no dangling edges.

Skips cleanly when Docker is unreachable (see ``tests/integration/conftest.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import open_hadith

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "open_hadith"

# Real committed slices: 3 hadiths from each of two books.
EXPECTED_PER_COLLECTION = {
    "col:open_hadith:musnad_ahmad_ibn-hanbal": 3,
    "col:open_hadith:sunan_al-nasai": 3,
}
EXPECTED_HADITHS = sum(EXPECTED_PER_COLLECTION.values())


@pytest.fixture
def staged_open_hadith(tmp_path: Path) -> Path:
    """Parse the committed Open-Hadith samples into a staging dir."""
    raw = tmp_path / "raw" / "open_hadith"
    raw.mkdir(parents=True)
    assert _FIXTURE_DIR.exists(), f"committed fixtures missing: {_FIXTURE_DIR}"
    shutil.copytree(_FIXTURE_DIR, raw, dirs_exist_ok=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    open_hadith.run(tmp_path / "raw", staging)
    return staging


@pytest.mark.integration
class TestOpenHadithLightUpLive:
    def test_hadith_and_collection_nodes_load_tagged(
        self, neo4j_client, staged_open_hadith: Path, tmp_path: Path
    ) -> None:
        """Both books' Hadith + Collection nodes land, sunni / open_hadith tagged."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_open_hadith, curated, strict=False)

        collections = neo4j_client.execute_read(
            "MATCH (c:Collection) "
            "RETURN c.id AS id, c.sect AS sect, c.source_corpus AS corpus ORDER BY c.id"
        )
        assert [c["id"] for c in collections] == sorted(EXPECTED_PER_COLLECTION)
        assert all(c["sect"] == "sunni" and c["corpus"] == "open_hadith" for c in collections)

        tally = neo4j_client.execute_read(
            "MATCH (h:Hadith) RETURN count(h) AS total, "
            "sum(CASE WHEN h.sect = 'sunni' AND h.source_corpus = 'open_hadith' THEN 1 ELSE 0 END) "
            "AS tagged"
        )[0]
        assert tally["total"] == EXPECTED_HADITHS
        assert tally["tagged"] == EXPECTED_HADITHS

    def test_appears_in_resolves_each_hadith_to_its_book(
        self, neo4j_client, staged_open_hadith: Path, tmp_path: Path
    ) -> None:
        """APPEARS_IN edges land (closing the no-Collection gap) and route each
        hadith to its own book — no dangling edges."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_open_hadith, curated, strict=False)
        load_all_edges(neo4j_client, staged_open_hadith, curated, strict=False)

        per_collection = neo4j_client.execute_read(
            "MATCH (:Hadith)-[:APPEARS_IN]->(c:Collection) "
            "RETURN c.id AS coll, count(*) AS n ORDER BY c.id"
        )
        assert {r["coll"]: r["n"] for r in per_collection} == EXPECTED_PER_COLLECTION

        total_edges = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS n"
        )[0]["n"]
        assert total_edges == EXPECTED_HADITHS
