"""Live-Neo4j light-up for the Fawaz hadith-api adapter (da#88).

Parses small REAL Fawaz editions (An-Nawawi's Forty + Forty Hadith Qudsi, from
fawazahmed0/hadith-api via jsDelivr) through the real batch loaders into a live
``neo4j:5`` container and asserts the corpus lands as a usable graph: Hadith +
Collection nodes (with the human-readable names the adapter merges from edition
metadata), all ``sect=sunni`` / ``source_corpus=fawaz`` tagged, with
``APPEARS_IN`` edges resolving each hadith to its own collection — no dangling.
The adapter already emits collections and merges eng+ara editions, so this is a
pure load-through light-up.

Skips cleanly when Docker is unreachable (see ``tests/integration/conftest.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import fawaz

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "fawaz"

# Real committed slices: 5 hadiths from each of two small Sunni collections.
EXPECTED_PER_COLLECTION = {"col:fawaz:nawawi": 5, "col:fawaz:qudsi": 5}
EXPECTED_HADITHS = sum(EXPECTED_PER_COLLECTION.values())


@pytest.fixture
def staged_fawaz(tmp_path: Path) -> Path:
    """Parse the committed Fawaz editions into a staging dir."""
    raw = tmp_path / "raw" / "fawaz"
    raw.mkdir(parents=True)
    assert _FIXTURE_DIR.exists(), f"committed fixtures missing: {_FIXTURE_DIR}"
    shutil.copytree(_FIXTURE_DIR, raw, dirs_exist_ok=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    fawaz.run(tmp_path / "raw", staging)
    return staging


@pytest.mark.integration
class TestFawazLightUpLive:
    def test_hadith_and_collection_nodes_load_tagged(
        self, neo4j_client, staged_fawaz: Path, tmp_path: Path
    ) -> None:
        """Both editions' Hadith + Collection nodes land, sunni / fawaz tagged."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_fawaz, curated, strict=False)

        collections = neo4j_client.execute_read(
            "MATCH (c:Collection) RETURN c.id AS id, c.name_en AS name, "
            "c.sect AS sect, c.source_corpus AS corpus ORDER BY c.id"
        )
        assert [c["id"] for c in collections] == sorted(EXPECTED_PER_COLLECTION)
        assert all(c["sect"] == "sunni" and c["corpus"] == "fawaz" for c in collections)
        names = {c["id"]: c["name"] for c in collections}
        assert names["col:fawaz:nawawi"] == "Forty Hadith of an-Nawawi"
        assert names["col:fawaz:qudsi"] == "Forty Hadith Qudsi"

        tally = neo4j_client.execute_read(
            "MATCH (h:Hadith) RETURN count(h) AS total, "
            "sum(CASE WHEN h.sect = 'sunni' AND h.source_corpus = 'fawaz' THEN 1 ELSE 0 END) "
            "AS tagged"
        )[0]
        assert tally["total"] == EXPECTED_HADITHS
        assert tally["tagged"] == EXPECTED_HADITHS

    def test_appears_in_resolves_each_hadith_to_its_collection(
        self, neo4j_client, staged_fawaz: Path, tmp_path: Path
    ) -> None:
        """APPEARS_IN edges route each hadith to its own edition — no dangling."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_fawaz, curated, strict=False)
        load_all_edges(neo4j_client, staged_fawaz, curated, strict=False)

        per_collection = neo4j_client.execute_read(
            "MATCH (:Hadith)-[:APPEARS_IN]->(c:Collection) "
            "RETURN c.id AS coll, count(*) AS n ORDER BY c.id"
        )
        assert {r["coll"]: r["n"] for r in per_collection} == EXPECTED_PER_COLLECTION

        total_edges = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS n"
        )[0]["n"]
        assert total_edges == EXPECTED_HADITHS
