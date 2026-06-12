"""Live-Neo4j cross-source narrator-identity resolution (da#99).

Proves the X1 keystone: the SAME narrator appearing across ≥2 sources collapses
to ONE canonical ``Narrator`` node, instead of one node per source.

Pipeline exercised: a consolidated ``narrator_mentions_resolved.parquet`` (the
NER output) spanning two corpora (``sanadset`` + ``thaqalayn``) → the REAL
``disambiguate.run`` (bio-matched AND bio-less exact-name canonicalization) →
the REAL ``load_all_nodes`` into a live ``neo4j:5``. Asserts the collapsed
canonical narrator count is strictly less than the naive per-source union, that a
shared narrator is a single node, and reports pairwise precision/recall of the
collapse against a labelled gold sample.

Requires Docker; ``neo4j_client`` ``pytest.skip``s when Docker is unreachable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.graph.load_nodes import load_all_nodes
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve import disambiguate
from src.resolve.disambiguate import _make_canonical_id
from src.resolve.quality import pairwise_quality
from src.utils.arabic import normalize_arabic
from tests.test_graph.conftest import write_narrator_mentions_resolved

# Narrator names (gold entity label → the surface name used in mentions).
_MALIK = "مالك بن أنس"  # has a bio → bio-matched, appears in both sources
_JAFAR = "جعفر بن محمد"  # no bio → fallback-canonicalized, appears in both sources
_ZAYD = "زيد بن ثابت"  # sanadset only
_AMMAR = "عمار بن ياسر"  # thaqalayn only

# (mention_id, source_corpus, name, gold_entity)
_MENTIONS = [
    ("m1", "sanadset", _MALIK, "malik"),
    ("m2", "thaqalayn", _MALIK, "malik"),
    ("m3", "sanadset", _JAFAR, "jafar"),
    ("m4", "thaqalayn", _JAFAR, "jafar"),
    ("m5", "sanadset", _ZAYD, "zayd"),
    ("m6", "thaqalayn", _AMMAR, "ammar"),
]
_EXPECTED_CANONICAL = 4  # malik, jafar, zayd, ammar
_NAIVE_UNION = 6  # one node per (source, narrator)


def _write_bios(staging: Path, rows: list[dict[str, object]]) -> Path:
    """Write a narrators_bio_*.parquet conforming to NARRATOR_BIO_SCHEMA."""
    arrays = {
        f.name: pa.array([r.get(f.name) for r in rows], type=f.type) for f in NARRATOR_BIO_SCHEMA
    }
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    path = staging / "narrators_bio_test.parquet"
    pq.write_table(table, path)
    return path


@pytest.mark.integration
class TestCrossSourceResolutionLive:
    def test_same_narrator_across_sources_collapses_to_one_node(
        self, neo4j_client, tmp_path: Path
    ) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()
        resolve_out = tmp_path / "resolve"
        resolve_out.mkdir()

        # Consolidated NER output spanning two corpora.
        write_narrator_mentions_resolved(
            resolve_out,
            [
                {
                    "mention_id": mid,
                    "hadith_id": f"hdt:{corpus}:c:1:{i}",
                    "source_corpus": corpus,
                    "position_in_chain": 0,
                    "name_raw": name,
                    "name_normalized": normalize_arabic(name),
                }
                for i, (mid, corpus, name, _gold) in enumerate(_MENTIONS)
            ],
        )

        # A single rijal bio for Malik (the others resolve bio-less by name).
        _write_bios(
            staging,
            [{"bio_id": "bio-malik", "source": "test", "name_ar": _MALIK, "death_year_ah": 179}],
        )

        # REAL disambiguation across both sources.
        disambiguate.run(staging, resolve_out)
        canonical_path = resolve_out / "narrators_canonical.parquet"
        assert canonical_path.exists()

        canonical_tbl = pq.read_table(canonical_path)
        # Cross-source collapse: distinct canonicals < naive per-source union.
        assert canonical_tbl.num_rows == _EXPECTED_CANONICAL
        assert canonical_tbl.num_rows < _NAIVE_UNION

        # Load the canonical narrators into the live graph.
        shutil.copy(canonical_path, staging / "narrators_canonical.parquet")
        results = load_all_nodes(neo4j_client, staging, curated, strict=False)
        narrator_result = next(r for r in results if r.node_type == "Narrator")
        assert narrator_result.created == _EXPECTED_CANONICAL

        rows = neo4j_client.execute_read("MATCH (n:Narrator) RETURN n.id AS id")
        ids = {r["id"] for r in rows}
        assert len(ids) == _EXPECTED_CANONICAL
        assert all(i.startswith("nar:") for i in ids)

        # The shared narrators (Malik, Jafar) are each ONE node spanning both
        # sources, not one-per-source.
        malik_id = _make_canonical_id(normalize_arabic(_MALIK))
        jafar_id = _make_canonical_id(normalize_arabic(_JAFAR))
        assert malik_id in ids
        assert jafar_id in ids
        # Malik carries the bio metadata; the bio-less Jafar does not.
        malik = neo4j_client.execute_read(
            "MATCH (n:Narrator {id: $id}) RETURN n.death_year_ah AS dy", {"id": malik_id}
        )
        assert malik[0]["dy"] == 179

        # Reported dedup quality on the labelled sample: the resolver's identity
        # function (canonical id keyed on normalized name) vs the gold entities.
        predicted = {
            mid: _make_canonical_id(normalize_arabic(name)) for mid, _c, name, _g in _MENTIONS
        }
        gold = {mid: gold_entity for mid, _c, _name, gold_entity in _MENTIONS}
        quality = pairwise_quality(predicted, gold)
        # Clean exact-match sample → perfect collapse.
        assert quality.precision == 1.0
        assert quality.recall == 1.0
