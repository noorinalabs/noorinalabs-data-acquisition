"""First-light vertical slice (da#73) over the committed real sample.

Drives parse -> load-nodes -> load-edges across the *real* scraped
riyadussalihin sample (``scripts/first_light/sample/``) using an in-process
mock Neo4j client. Locks in the node/edge counts and the collection-id keying
that the staging load depends on, so a regression in the parse schema, the
``source_id`` shape, or the APPEARS_IN collection-id derivation is caught in CI
without needing a live Neo4j.

The actual load into the staging Neo4j (and the frontend render) is an
operational step run from inside the cluster — see ``scripts/first_light/`` and
``docs/first-light-slice.md``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import sunnah_scraped
from src.utils.neo4j_client import Neo4jClient

SAMPLE_JSON = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "first_light"
    / "sample"
    / "sunnah_scraped"
    / "riyadussalihin.json"
)

# Sample is riyadussalihin book 1: 47 hadiths, 1 collection. Each hadith sits in
# its own synthetic chapter (101..147), so source_ids are distinct despite
# hadith_number being unextracted (da#72 latent, not active on this sample).
EXPECTED_HADITHS = 47
EXPECTED_COLLECTIONS = 1


class _AllEndpointsExistMock:
    """Mock Neo4j client: write batches return ``len(batch)``; endpoint CHECK
    reads echo the batch back with every ``*_exists`` flag set true."""

    _EXISTS_FLAGS = (
        "hadith_exists",
        "collection_exists",
        "from_exists",
        "to_exists",
        "narrator_exists",
        "grading_exists",
        "a_exists",
        "b_exists",
    )

    def __init__(self) -> None:
        self.write_queries: list[str] = []

    def ensure_constraints(self) -> None:  # pragma: no cover - trivial
        pass

    def ensure_fulltext_indexes(self) -> None:  # pragma: no cover - trivial
        pass

    def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[Any]:
        return []

    def execute_read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        batch = (parameters or {}).get("batch", [])
        out: list[dict[str, Any]] = []
        for row in batch:
            rec = dict(row)
            for flag in self._EXISTS_FLAGS:
                rec[flag] = True
            out.append(rec)
        return out

    def execute_write_batch(
        self, query: str, batch: list[dict[str, Any]], batch_size: int = 1000
    ) -> int:
        self.write_queries.append(query)
        return len(batch)


@pytest.fixture
def staged_sample(tmp_path: Path) -> Path:
    """Parse the committed real sample into a staging dir and return it."""
    assert SAMPLE_JSON.exists(), f"committed sample missing: {SAMPLE_JSON}"
    raw = tmp_path / "raw" / "sunnah_scraped"
    raw.mkdir(parents=True)
    shutil.copyfile(SAMPLE_JSON, raw / "riyadussalihin.json")

    staging = tmp_path / "staging"
    staging.mkdir()
    sunnah_scraped.run(tmp_path / "raw", staging)
    return staging


def test_sample_parses_to_distinct_source_ids(staged_sample: Path) -> None:
    """Real sample parses to 47 hadiths with no source_id collisions."""
    rows = pq.read_table(staged_sample / "hadiths_sunnah_scraped.parquet").to_pylist()
    assert len(rows) == EXPECTED_HADITHS
    source_ids = [r["source_id"] for r in rows]
    assert len(set(source_ids)) == EXPECTED_HADITHS, "source_id collisions (regression of da#72?)"
    # Bilingual real content is present.
    assert all(r["matn_ar"] for r in rows)
    assert all(r["matn_en"] for r in rows)
    # source_id shape: sunnah:<collection>:<book>:<chapter>:<hadith_no|0>
    assert all(r["source_id"].startswith("sunnah:riyadussalihin:1:") for r in rows)


def test_node_load_counts(staged_sample: Path, tmp_path: Path) -> None:
    """load_all_nodes loads 47 Hadith + 1 Collection from the real sample."""
    curated = tmp_path / "curated"
    curated.mkdir()
    client = _AllEndpointsExistMock()
    results = load_all_nodes(cast(Neo4jClient, client), staged_sample, curated, strict=False)
    by_type = {r.node_type: r for r in results}
    assert by_type["Hadith"].created == EXPECTED_HADITHS
    assert by_type["Collection"].created == EXPECTED_COLLECTIONS
    # The thin slice has no narrator/chain/event source files.
    assert by_type["Narrator"].created == 0
    assert by_type["Chain"].created == 0


def test_appears_in_edges_match_collection(staged_sample: Path, tmp_path: Path) -> None:
    """Every hadith gets an APPEARS_IN edge whose target id matches the loaded
    Collection node id (col:sunnah:riyadussalihin) — no dangling edges."""
    curated = tmp_path / "curated"
    curated.mkdir()
    client = _AllEndpointsExistMock()
    results = load_all_edges(cast(Neo4jClient, client), staged_sample, curated, strict=False)
    by_type = {r.edge_type: r for r in results}
    assert by_type["APPEARS_IN"].created == EXPECTED_HADITHS
    assert by_type["APPEARS_IN"].missing_endpoints == 0

    # The collection-id the edge loader derives must equal the node loader's id.
    hadith_rows = pq.read_table(staged_sample / "hadiths_sunnah_scraped.parquet").to_pylist()
    coll_rows = pq.read_table(staged_sample / "collections_sunnah_scraped.parquet").to_pylist()
    h0 = hadith_rows[0]
    edge_collection_id = f"col:{h0['source_corpus']}:{h0['collection_name']}"
    node_collection_id = f"col:{coll_rows[0]['collection_id']}"
    assert edge_collection_id == node_collection_id == "col:sunnah:riyadussalihin"
