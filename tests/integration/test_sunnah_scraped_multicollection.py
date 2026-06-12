"""Live-Neo4j multi-collection light-up for the sunnah.com scraper (da#84).

Extends the riyadussalihin first-light (da#73) *beyond a single collection*:
parses two REAL committed sunnah.com samples — ``riyadussalihin`` (47) and
``adab`` (46) — through the real batch loaders into a live ``neo4j:5`` container
and asserts both collections' Hadith and Collection nodes land, correctly
``sect`` / ``source_corpus`` tagged, with ``APPEARS_IN`` edges pointing at the
right collection. Real data, real graph — the owner's live-local acceptance bar.

Skips cleanly when Docker is unreachable (see ``tests/integration/conftest.py``),
so it degrades to a SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import sunnah_scraped

_SAMPLE_DIR = (
    Path(__file__).resolve().parents[2] / "scripts" / "first_light" / "sample" / "sunnah_scraped"
)

# Real committed samples: book 1 of each collection. Counts are locked so a
# parse/keying regression that drops or duplicates rows is caught.
EXPECTED_PER_COLLECTION = {"riyadussalihin": 47, "adab": 46}
EXPECTED_HADITHS = sum(EXPECTED_PER_COLLECTION.values())


@pytest.fixture
def staged_multi(tmp_path: Path) -> Path:
    """Parse both real samples into one staging dir and return it."""
    raw = tmp_path / "raw" / "sunnah_scraped"
    raw.mkdir(parents=True)
    for collection in EXPECTED_PER_COLLECTION:
        sample = _SAMPLE_DIR / f"{collection}.json"
        assert sample.exists(), f"committed sample missing: {sample}"
        shutil.copyfile(sample, raw / f"{collection}.json")
    staging = tmp_path / "staging"
    staging.mkdir()
    sunnah_scraped.run(tmp_path / "raw", staging)
    return staging


@pytest.mark.integration
class TestSunnahScrapedMultiCollectionLive:
    def test_both_collections_load_sect_tagged(
        self, neo4j_client, staged_multi: Path, tmp_path: Path
    ) -> None:
        """Two Collection nodes + every Hadith node tagged sunni / sunnah."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_multi, curated, strict=False)

        collections = neo4j_client.execute_read(
            "MATCH (c:Collection) "
            "RETURN c.id AS id, c.sect AS sect, c.source_corpus AS corpus ORDER BY c.id"
        )
        assert [c["id"] for c in collections] == ["col:sunnah:adab", "col:sunnah:riyadussalihin"]
        assert all(c["sect"] == "sunni" and c["corpus"] == "sunnah" for c in collections)

        tally = neo4j_client.execute_read(
            "MATCH (h:Hadith) RETURN count(h) AS total, "
            "sum(CASE WHEN h.sect = 'sunni' AND h.source_corpus = 'sunnah' THEN 1 ELSE 0 END) "
            "AS tagged"
        )[0]
        assert tally["total"] == EXPECTED_HADITHS
        assert tally["tagged"] == EXPECTED_HADITHS, "some Hadith nodes missing sunni/sunnah tags"

    def test_appears_in_routes_each_hadith_to_its_collection(
        self, neo4j_client, staged_multi: Path, tmp_path: Path
    ) -> None:
        """Every hadith gets an APPEARS_IN edge to ITS OWN collection — adab
        hadiths to col:sunnah:adab, riyad hadiths to col:sunnah:riyadussalihin,
        with no cross-collection leakage and no dangling edges."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_multi, curated, strict=False)
        load_all_edges(neo4j_client, staged_multi, curated, strict=False)

        per_collection = neo4j_client.execute_read(
            "MATCH (:Hadith)-[:APPEARS_IN]->(c:Collection) "
            "RETURN c.id AS coll, count(*) AS n ORDER BY c.id"
        )
        assert {r["coll"]: r["n"] for r in per_collection} == {
            "col:sunnah:adab": EXPECTED_PER_COLLECTION["adab"],
            "col:sunnah:riyadussalihin": EXPECTED_PER_COLLECTION["riyadussalihin"],
        }

        total_edges = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS n"
        )[0]["n"]
        assert total_edges == EXPECTED_HADITHS, "an APPEARS_IN edge is dangling or missing"
