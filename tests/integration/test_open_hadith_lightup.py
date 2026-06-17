"""Live-Neo4j proof that the Open-Hadith corpus is EXCLUDED by the canonical
composition (da#191).

Open-Hadith was a confirmed 100% duplicate of ``halimbahae`` (pre-cutover audit).
It is dropped from the production graph: marked ``active=False`` in the registry
so ``run_all`` never acquires/parses it, and — defence in depth — the node loader
filters it out via :data:`src.parse.composition.HADITH_COMPOSITION`. This test
parses the committed Open-Hadith samples directly and asserts that loading their
parquet yields ZERO graph nodes, so a stray ``hadiths_open_hadith.parquet`` in a
staging dir can never re-introduce the duplicate.

Originally (da#87) this light-up asserted Open-Hadith loaded as a usable graph;
that behaviour was intentionally reversed by the da#191 dedup.

Skips cleanly when Docker is unreachable (see ``tests/integration/conftest.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.graph.load_nodes import load_all_nodes
from src.parse import open_hadith

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "open_hadith"


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
class TestOpenHadithExcludedByComposition:
    def test_open_hadith_loads_no_nodes(
        self, neo4j_client, staged_open_hadith: Path, tmp_path: Path
    ) -> None:
        """Loading Open-Hadith parquet yields no Hadith or Collection nodes —
        the corpus is dropped as a duplicate of halimbahae (da#191)."""
        curated = tmp_path / "curated"
        curated.mkdir()
        load_all_nodes(neo4j_client, staged_open_hadith, curated, strict=False)

        open_hadith_hadiths = neo4j_client.execute_read(
            "MATCH (h:Hadith {source_corpus: 'open_hadith'}) RETURN count(h) AS n"
        )[0]["n"]
        open_hadith_collections = neo4j_client.execute_read(
            "MATCH (c:Collection {source_corpus: 'open_hadith'}) RETURN count(c) AS n"
        )[0]["n"]
        assert open_hadith_hadiths == 0
        assert open_hadith_collections == 0
