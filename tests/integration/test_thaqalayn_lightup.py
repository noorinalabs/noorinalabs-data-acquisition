"""Thaqalayn (Shia) end-to-end light-up against a real Neo4j container (da#85).

The ``thaqalayn`` adapter + parser already existed and were parse-validated, but
the corpus had **never been loaded into the graph**. This is the first Shia
corpus to land — the keystone for cross-sect parallels (isnad-graph#964, whose
Browse-Parallels view needs real ``sect=shia`` hadith in the graph).

These tests exercise the real ``parse -> load_nodes -> load_edges`` path against
a live ``neo4j:5-community`` container (via the shared ``neo4j_client`` fixture in
``conftest.py``). They ``pytest.skip`` cleanly when Docker is unavailable, so a
no-Docker environment yields a SKIP rather than a red ERROR (#69). The parser's
own conformance is covered by ``tests/test_parse/test_thaqalayn_parser.py``; what
is asserted *here* and nowhere else is that real Shia nodes/edges actually
**persist in Neo4j** with the correct ``sect`` + ``source_corpus`` and
collision-safe, single-prefixed ids (the da#82 / main#139 identity contract).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import thaqalayn
from src.parse.identity import is_double_prefixed, validate_source_id
from src.utils.neo4j_client import Neo4jClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Sample Thaqalayn data — the REAL upstream record shape (da#175): a JSON array
# of records per book under ``github_clone/V2/ThaqalaynData/<n>.json``, with
# ``id``/``bookId``/``arabicText``/``thaqalaynSanad``/``majlisiGrading`` fields.
# Two of the Four Books (al-Kafi, Man Lā Yaḥḍuruhu al-Faqīh).
# ---------------------------------------------------------------------------

_AL_KAFI = [
    {
        "id": 1,
        "bookId": "Al-Kafi-Volume-1-Kulayni",
        "book": "Al-Kāfi",
        "volume": 1,
        "category": "The Book of Intellect and Ignorance",
        "categoryId": 1,
        "chapter": "Chapter 1",
        "chapterInCategoryId": 1,
        "author": "al-Kulayni",
        "arabicText": "إنما الأعمال بالنيات ولكل امرئ ما نوى",
        "englishText": "Actions are but by intentions; everyone shall have what they intended.",
        "majlisiGrading": "Sahih",
        "thaqalaynSanad": "A number of our companions narrated",
        "thaqalaynMatn": "Actions are but by intentions.",
    },
    {
        "id": 2,
        "bookId": "Al-Kafi-Volume-1-Kulayni",
        "book": "Al-Kāfi",
        "volume": 1,
        "category": "The Book of the Excellence of Knowledge",
        "categoryId": 2,
        "chapter": "Chapter 2",
        "chapterInCategoryId": 1,
        "author": "al-Kulayni",
        "arabicText": "العلم نور يقذفه الله في قلب من يشاء",
        "englishText": "Knowledge is a light that Allah casts into the heart of whom He wills.",
        "majlisiGrading": "Hasan",
        "thaqalaynSanad": "",
        "thaqalaynMatn": "",
    },
    {
        "id": 3,
        "bookId": "Al-Kafi-Volume-1-Kulayni",
        "book": "Al-Kāfi",
        "volume": 1,
        "category": "The Book of the Excellence of Knowledge",
        "categoryId": 2,
        "chapter": "Chapter 2",
        "chapterInCategoryId": 2,
        "author": "al-Kulayni",
        "arabicText": "من سلك طريقا يطلب فيه علما سلك الله به طريقا إلى الجنة",
        "englishText": "Whoever travels a path seeking knowledge, Allah sets them toward Paradise.",
        "majlisiGrading": "Sahih",
        "thaqalaynSanad": "",
        "thaqalaynMatn": "",
    },
]

_MAN_LA_YAHDURUHU = [
    {
        "id": 1,
        "bookId": "Man-La-Yahduruh-al-Faqih-Volume-1-Saduq",
        "book": "Man Lā Yaḥḍuruhu al-Faqīh",
        "volume": 1,
        "category": "The Book of Purification",
        "categoryId": 1,
        "chapter": "Chapter 1",
        "chapterInCategoryId": 1,
        "author": "al-Shaykh al-Saduq",
        "arabicText": "الطهور نصف الإيمان",
        "englishText": "Purification is half of faith.",
        "majlisiGrading": "Sahih",
        "thaqalaynSanad": "",
        "thaqalaynMatn": "",
    },
    {
        "id": 2,
        "bookId": "Man-La-Yahduruh-al-Faqih-Volume-1-Saduq",
        "book": "Man Lā Yaḥḍuruhu al-Faqīh",
        "volume": 1,
        "category": "The Book of Prayer",
        "categoryId": 2,
        "chapter": "Chapter 2",
        "chapterInCategoryId": 1,
        "author": "al-Shaykh al-Saduq",
        "arabicText": "الصلاة عمود الدين",
        "englishText": "Prayer is the pillar of the religion.",
        "majlisiGrading": "Sahih",
        "thaqalaynSanad": "",
        "thaqalaynMatn": "",
    },
]

# Total hadith / collection counts the fixture is expected to produce.
_EXPECTED_HADITHS = len(_AL_KAFI) + len(_MAN_LA_YAHDURUHU)  # 5
_EXPECTED_COLLECTIONS = 2


def _write_thaqalayn_raw(raw_dir: Path) -> None:
    """Write the two-book Thaqalayn fixture under the real V2 clone layout."""
    data_dir = raw_dir / "thaqalayn" / "github_clone" / "V2" / "ThaqalaynData"
    data_dir.mkdir(parents=True)
    (data_dir / "1.json").write_text(json.dumps(_AL_KAFI, ensure_ascii=False), encoding="utf-8")
    (data_dir / "2.json").write_text(
        json.dumps(_MAN_LA_YAHDURUHU, ensure_ascii=False), encoding="utf-8"
    )


def _parse_to_staging(tmp_path: Path) -> Path:
    """Run the real thaqalayn parser; return the staging dir with parquet output."""
    raw_dir = tmp_path / "raw"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True)
    _write_thaqalayn_raw(raw_dir)

    hadiths_path, collections_path = thaqalayn.run(raw_dir, staging_dir)
    assert hadiths_path.exists()
    assert collections_path.exists()
    return staging_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThaqalaynLightup:
    """parse -> load -> read-back the first Shia corpus into a live graph."""

    def test_source_ids_are_canonical(self, tmp_path: Path) -> None:
        """Every staged source_id is corpus-namespaced + collision-safe (da#82).

        Pure parser/identity check (no Neo4j) — guards that the parser builds ids
        through ``generate_source_id`` so they satisfy the canonical grammar and
        never re-prepend the corpus (the main#139 double-prefix hazard).
        """
        staging_dir = _parse_to_staging(tmp_path)
        rows = pq.read_table(staging_dir / "hadiths_thaqalayn.parquet").to_pylist()

        assert len(rows) == _EXPECTED_HADITHS
        for row in rows:
            sid = row["source_id"]
            assert validate_source_id(sid) == [], f"non-canonical source_id: {sid!r}"
            assert sid.startswith("thaqalayn:"), sid
            assert not is_double_prefixed(sid), f"double-prefixed source_id: {sid!r}"

    def test_nodes_land_with_shia_tagging(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        """Real Hadith/Collection nodes persist, tagged sect=shia + corpus=thaqalayn."""
        staging_dir = _parse_to_staging(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        results = load_all_nodes(neo4j_client, staging_dir, curated_dir, strict=False)

        hadith_result = next(r for r in results if r.node_type == "Hadith")
        assert hadith_result.created == _EXPECTED_HADITHS
        collection_result = next(r for r in results if r.node_type == "Collection")
        assert collection_result.created == _EXPECTED_COLLECTIONS

        # Read the nodes back from the live graph and assert the Shia tagging that
        # isnad-graph#964 keys on — not just that *some* nodes exist.
        hadith_rows = neo4j_client.execute_read(
            "MATCH (h:Hadith) RETURN count(h) AS cnt, "
            "collect(DISTINCT h.sect) AS sects, "
            "collect(DISTINCT h.source_corpus) AS corpora"
        )
        assert hadith_rows[0]["cnt"] == _EXPECTED_HADITHS
        assert hadith_rows[0]["sects"] == ["shia"]
        assert hadith_rows[0]["corpora"] == ["thaqalayn"]

        coll_rows = neo4j_client.execute_read(
            "MATCH (c:Collection) RETURN collect(DISTINCT c.sect) AS sects, "
            "collect(DISTINCT c.source_corpus) AS corpora"
        )
        assert coll_rows[0]["sects"] == ["shia"]
        assert coll_rows[0]["corpora"] == ["thaqalayn"]

    def test_node_ids_single_prefixed_in_graph(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        """Persisted Hadith node ids carry exactly one ``hdt:thaqalayn:`` prefix.

        The same hadith must not double-prefix to two nodes (main#139). Asserts on
        the ids *as stored in Neo4j*, the layer the mock-client unit suite cannot
        reach.
        """
        staging_dir = _parse_to_staging(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        load_all_nodes(neo4j_client, staging_dir, curated_dir, strict=False)

        rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN h.id AS id")
        ids = [r["id"] for r in rows]
        assert len(ids) == _EXPECTED_HADITHS
        for hid in ids:
            assert hid.startswith("hdt:thaqalayn:"), hid
            assert not is_double_prefixed(hid), f"double-prefixed node id: {hid!r}"
        # Ids are unique — no hadith collapsed/duplicated across the two books.
        assert len(set(ids)) == _EXPECTED_HADITHS

    def test_appears_in_edges_link_hadith_to_collection(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        """Each Shia hadith gets an APPEARS_IN edge to its (Shia) collection."""
        staging_dir = _parse_to_staging(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        load_all_nodes(neo4j_client, staging_dir, curated_dir, strict=False)

        edge_results = load_all_edges(neo4j_client, staging_dir, curated_dir, strict=False)
        appears_in = next(r for r in edge_results if r.edge_type == "APPEARS_IN")
        assert appears_in.created == _EXPECTED_HADITHS
        assert appears_in.missing_endpoints == 0

        # Read the edges back: every Shia hadith resolves to a Shia collection.
        rows = neo4j_client.execute_read(
            "MATCH (h:Hadith)-[r:APPEARS_IN]->(c:Collection) "
            "RETURN count(r) AS cnt, collect(DISTINCT c.sect) AS coll_sects"
        )
        assert rows[0]["cnt"] == _EXPECTED_HADITHS
        assert rows[0]["coll_sects"] == ["shia"]

    def test_shia_corpus_is_queryable(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        """The Browse-Parallels Shia slice (isnad-graph#964) is now answerable.

        A ``sect=shia`` lookup against the live graph returns the loaded Thaqalayn
        hadith — the concrete capability this keystone unblocks.
        """
        staging_dir = _parse_to_staging(tmp_path)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        load_all_nodes(neo4j_client, staging_dir, curated_dir, strict=False)

        rows = neo4j_client.execute_read("MATCH (h:Hadith {sect: 'shia'}) RETURN count(h) AS cnt")
        assert rows[0]["cnt"] == _EXPECTED_HADITHS
