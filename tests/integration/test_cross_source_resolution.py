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
from src.resolve.bio_promote import promote_bios_to_canonical
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


def _write_bios(staging: Path, rows: list[dict[str, object]], *, suffix: str = "test") -> Path:
    """Write a narrators_bio_<suffix>.parquet conforming to NARRATOR_BIO_SCHEMA."""
    arrays = {
        f.name: pa.array([r.get(f.name) for r in rows], type=f.type) for f in NARRATOR_BIO_SCHEMA
    }
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    path = staging / f"narrators_bio_{suffix}.parquet"
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

    def test_three_sources_including_itqan_converge_on_one_canonical(
        self, neo4j_client, tmp_path: Path
    ) -> None:
        """Three sources — two mention-driven + the bio-direct Itqan rijal source.

        da#110 added Itqan as a profile-only narrator corpus: it contributes
        biographies but no isnad mentions, so its narrators reach the graph via
        ``bio_promote`` (the bio-direct path), not the mention-driven
        disambiguator. This proves all THREE present sources resolve into one
        identity space — the same narrator seen in ``sanadset`` + ``thaqalayn``
        mentions AND in an Itqan bio collapses to ONE ``nar:`` node, while
        distinct narrators (incl. an Itqan-only one) stay distinct.

        It also locks the producer-ordering invariant: ``disambiguate``
        (OVERWRITES ``narrators_canonical.parquet``) must run BEFORE
        ``bio_promote`` (MERGES), or the Itqan-only narrator would be clobbered.
        Both producers key on the same ``identity.make_canonical_id``, so they
        converge instead of duplicating.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()
        resolve_out = tmp_path / "resolve"
        resolve_out.mkdir()

        shuba = "شعبة بن الحجاج"  # Itqan bio ONLY — no mention in any corpus.

        # Mention-driven sources: Malik spans sanadset + thaqalayn; Zayd is
        # sanadset-only; Ammar is thaqalayn-only.
        mentions = [
            ("m1", "sanadset", _MALIK, 0),
            ("m2", "thaqalayn", _MALIK, 1),
            ("m3", "sanadset", _ZAYD, 0),
            ("m4", "thaqalayn", _AMMAR, 0),
        ]
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
                for i, (mid, corpus, name, _pos) in enumerate(mentions)
            ],
        )

        # The Itqan rijal slice (the THIRD source): a bio for Malik (who is also
        # mentioned in the two isnad corpora) and a bio for Shuba (mentioned
        # nowhere — a profile-only narrator). Present at disambiguation time so
        # Malik's mentions bio-match the Itqan record; Shuba, lacking any
        # mention, is materialised only later by bio_promote.
        _write_bios(
            staging,
            [
                {
                    "bio_id": "itqan:malik",
                    "source": "itqan",
                    "name_ar": _MALIK,
                    "name_ar_normalized": normalize_arabic(_MALIK),
                    "death_year_ah": 179,
                    "trustworthiness": "thiqa",
                    "external_id": "malik",
                },
                {
                    "bio_id": "itqan:shuba",
                    "source": "itqan",
                    "name_ar": shuba,
                    "name_ar_normalized": normalize_arabic(shuba),
                    "death_year_ah": 160,
                    "trustworthiness": "thiqa",
                    "external_id": "shuba",
                },
            ],
            suffix="itqan",
        )

        # Step 1 — REAL disambiguation over the two mention corpora. Emits a
        # canonical node per mentioned narrator only (Shuba, mention-less, is
        # absent here): Malik (collapsed across both corpora), Zayd, Ammar.
        disambiguate.run(staging, resolve_out)
        canonical_path = resolve_out / "narrators_canonical.parquet"
        assert canonical_path.exists()
        after_disambig = {r["canonical_id"] for r in pq.read_table(canonical_path).to_pylist()}
        assert len(after_disambig) == 3  # malik, zayd, ammar — NOT yet shuba.

        # Step 2 — bio-direct promotion of the Itqan source. MERGES into the
        # disambiguator output (must run second): Malik's record gains the Itqan
        # provenance, Shuba becomes a new canonical node. Source-filtered to
        # itqan so the ordering invariant is what is under test.
        promoted_path = promote_bios_to_canonical(staging, resolve_out, sources={"itqan"})
        assert promoted_path == canonical_path

        records = {r["canonical_id"]: r for r in pq.read_table(canonical_path).to_pylist()}
        malik_id = _make_canonical_id(normalize_arabic(_MALIK))
        zayd_id = _make_canonical_id(normalize_arabic(_ZAYD))
        ammar_id = _make_canonical_id(normalize_arabic(_AMMAR))
        shuba_id = _make_canonical_id(normalize_arabic(shuba))

        # Four distinct canonical narrators across the three sources.
        assert set(records) == {malik_id, zayd_id, ammar_id, shuba_id}
        assert all(cid.startswith("nar:") for cid in records)

        # Malik is ONE node spanning sanadset + thaqalayn mentions AND the Itqan
        # bio: the two mention contributions survive the merge (mention_count
        # preserved, NOT reset by bio_promote) and the Itqan provenance is on it.
        malik = records[malik_id]
        assert malik["mention_count"] == 2
        assert "itqan:malik" in malik["source_ids"]
        assert malik["death_year_ah"] == 179

        # The Itqan-only narrator survived the disambiguate→bio_promote ordering
        # (would have been clobbered had bio_promote run first) and stays
        # distinct: a bio-direct node with no mentions.
        shuba_rec = records[shuba_id]
        assert shuba_rec["mention_count"] == 0
        assert shuba_rec["source_ids"] == ["itqan:shuba"]

        # The mention-only narrators stay distinct and carry no Itqan provenance.
        for cid in (zayd_id, ammar_id):
            assert "itqan:shuba" not in records[cid]["source_ids"]
            assert "itqan:malik" not in records[cid]["source_ids"]

        # End-to-end: the merged canonical loads as exactly four Narrator nodes,
        # all nar:-prefixed, into the live graph.
        shutil.copy(canonical_path, staging / "narrators_canonical.parquet")
        results = load_all_nodes(neo4j_client, staging, curated, strict=False)
        narrator_result = next(r for r in results if r.node_type == "Narrator")
        assert narrator_result.created == 4

        rows = neo4j_client.execute_read("MATCH (n:Narrator) RETURN n.id AS id")
        ids = {r["id"] for r in rows}
        assert ids == {malik_id, zayd_id, ammar_id, shuba_id}

        # The Itqan provenance is queryable on the graph (owner's removability
        # guarantee for the unlicensed Itqan corpus rides on source_ids).
        malik_node = neo4j_client.execute_read(
            "MATCH (n:Narrator {id: $id}) RETURN n.source_ids AS sids, n.death_year_ah AS dy",
            {"id": malik_id},
        )
        assert "itqan:malik" in malik_node[0]["sids"]
        assert malik_node[0]["dy"] == 179
