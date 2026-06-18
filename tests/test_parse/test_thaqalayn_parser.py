"""Tests for the Thaqalayn parser against the REAL upstream repo schema.

The previous fixtures here used invented field names (``hadithNumber``,
``textAr``, ``arabicText`` with a ``data`` wrapper) that do not exist in the
``MohammedArab1/ThaqalaynAPI`` repo. They passed while the parser produced
unloadable garbage on real data — 0% Arabic text and one ``source_id`` per book
(da#175). This suite seeds the **actual** record shape (``id``, ``bookId``,
``arabicText``, ``thaqalaynSanad``/``thaqalaynMatn``, ``majlisiGrading`` …) under
the real ``github_clone/V2/ThaqalaynData/<n>.json`` layout, plus the aggregate
and repo-config decoys the parser must ignore. This is main#671's fixture-masks-
bug rule applied in miniature: the fixture mirrors production, not the parser's
assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from src.parse.identity import is_double_prefixed, validate_source_id
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.parse.thaqalayn import book_json_files, run

# Two real-shaped records from al-Kafi Volume 1: an introduction entry (no
# sanad/matn split, ungraded) and a graded narration with the isnad/matn split.
_BOOK_1 = [
    {
        "id": 1,
        "bookId": "Al-Kafi-Volume-1-Kulayni",
        "book": "Al-Kāfi",
        "volume": 1,
        "category": "Introduction",
        "categoryId": 1,
        "chapter": "Introduction",
        "chapterInCategoryId": 1,
        "author": "Shaykh Muḥammad b. Yaʿqūb al-Kulaynī",
        "translator": "Muhammad Sarwar",
        "englishText": "In the praise of Allah...",
        "arabicText": "الحمد لله الذي حمده",
        "majlisiGrading": "",
        "behbudiGrading": "",
        "mohseniGrading": "",
        "thaqalaynSanad": "",
        "thaqalaynMatn": "",
    },
    {
        "id": 2,
        "bookId": "Al-Kafi-Volume-1-Kulayni",
        "book": "Al-Kāfi",
        "volume": 1,
        "category": "The Book of Reason and Ignorance",
        "categoryId": 2,
        "chapter": "Chapter 1",
        "chapterInCategoryId": 1,
        "author": "Shaykh Muḥammad b. Yaʿqūb al-Kulaynī",
        "translator": "Muhammad Sarwar",
        "englishText": "Reason is that by which the Merciful is worshipped.",
        "arabicText": "عدة من أصحابنا عن أحمد العقل ما عبد به الرحمن",
        "majlisiGrading": "Ṣaḥīḥ",
        "behbudiGrading": "",
        "mohseniGrading": "Muʿtabar",
        # Thaqalayn's sanad/matn split is on the ENGLISH translation, not Arabic.
        "thaqalaynSanad": "A number of our companions narrated from Ahmad",
        "thaqalaynMatn": "Reason is that by which the Merciful is worshipped.",
    },
]

# A second book (different bookId, same multi-volume title) — exercises the
# per-volume Collection split and the volume-suffixed display name.
_BOOK_2 = [
    {
        "id": 1,
        "bookId": "Al-Kafi-Volume-2-Kulayni",
        "book": "Al-Kāfi",
        "volume": 2,
        "category": "The Book of Faith and Disbelief",
        "categoryId": 1,
        "chapter": "Chapter 1",
        "chapterInCategoryId": 1,
        "author": "Shaykh Muḥammad b. Yaʿqūb al-Kulaynī",
        "translator": "Muhammad Sarwar",
        "englishText": "When Allah created reason...",
        "arabicText": "نص عربي للحديث",
        "majlisiGrading": "Ḥasan",
        "behbudiGrading": "Ḥasan",
        "mohseniGrading": "",
        "thaqalaynSanad": "Muhammad b. Yahya narrated",
        "thaqalaynMatn": "When Allah created reason.",
    },
]


def _seed_real_clone(raw_dir: Path) -> None:
    """Materialise the real ``V2/ThaqalaynData`` layout plus must-ignore decoys."""
    clone = raw_dir / "thaqalayn" / "github_clone"
    data_dir = clone / "V2" / "ThaqalaynData"
    data_dir.mkdir(parents=True)

    (data_dir / "1.json").write_text(json.dumps(_BOOK_1, ensure_ascii=False), encoding="utf-8")
    (data_dir / "2.json").write_text(json.dumps(_BOOK_2, ensure_ascii=False), encoding="utf-8")

    # Decoys the selector MUST exclude (da#175): the giant aggregate that
    # re-lists every hadith, the index file, the Ingredients subdir, and repo
    # config. If any leaked in, row counts would inflate / collections corrupt.
    (data_dir / "allBooks.json").write_text(
        json.dumps(_BOOK_1 + _BOOK_2, ensure_ascii=False), encoding="utf-8"
    )
    (data_dir / "BookNames.json").write_text(
        json.dumps([{"bookId": "Al-Kafi-Volume-1-Kulayni", "name": "Al-Kāfi"}]), encoding="utf-8"
    )
    (data_dir / "Ingredients").mkdir()
    (data_dir / "Ingredients" / "ingredients.json").write_text("[]", encoding="utf-8")
    (clone / "package-lock.json").write_text('{"name": "thaqalayn"}', encoding="utf-8")
    (clone / "API").mkdir()
    (clone / "API" / "tsconfig.json").write_text("{}", encoding="utf-8")


_TOTAL_HADITHS = len(_BOOK_1) + len(_BOOK_2)


class TestBookFileSelection:
    def test_selects_only_numeric_book_files(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        files = book_json_files(tmp_path / "thaqalayn")
        names = sorted(f.name for f in files)
        assert names == ["1.json", "2.json"]  # aggregates / index / config excluded

    def test_empty_when_no_clone(self, tmp_path: Path) -> None:
        assert book_json_files(tmp_path / "thaqalayn") == []


class TestThaqalaynParser:
    def test_produces_parquet_files(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        h, c = run(tmp_path, tmp_path / "staging")
        assert h.exists() and c.exists()

    def test_hadith_schema_and_row_count(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        table = pq.read_table(h)
        assert table.schema == HADITH_SCHEMA
        # Only the two numeric book files — the allBooks aggregate is NOT counted.
        assert table.num_rows == _TOTAL_HADITHS

    def test_source_ids_unique_and_canonical(self, tmp_path: Path) -> None:
        """One node per hadith — the da#175 collision regression guard."""
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        sids = pq.read_table(h).column("source_id").to_pylist()
        assert len(set(sids)) == _TOTAL_HADITHS  # no collapse onto one id per book
        assert "thaqalayn:Al-Kafi-Volume-1-Kulayni:1" in sids
        for sid in sids:
            assert validate_source_id(sid) == [], f"non-canonical source_id: {sid!r}"
            assert not is_double_prefixed(sid), f"double-prefixed source_id: {sid!r}"

    def test_matn_always_populated(self, tmp_path: Path) -> None:
        """Every row carries Arabic text (the 100%-empty-matn regression guard)."""
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        matn = pq.read_table(h).column("matn_ar").to_pylist()
        assert all(m for m in matn)

    def test_isnad_matn_split_is_english_arabic_intact(self, tmp_path: Path) -> None:
        """The (English) sanad/matn split feeds *_en; matn_ar stays full Arabic."""
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        rows = {r["source_id"]: r for r in pq.read_table(h).to_pylist()}
        graded = rows["thaqalayn:Al-Kafi-Volume-1-Kulayni:2"]
        # thaqalaynSanad/thaqalaynMatn are English -> isnad_raw_en / matn_en.
        assert graded["isnad_raw_en"] == "A number of our companions narrated from Ahmad"
        assert graded["matn_en"] == "Reason is that by which the Merciful is worshipped."
        # Arabic is never split upstream: matn_ar == full_text_ar == arabicText,
        # and no Arabic isnad is fabricated.
        assert graded["isnad_raw_ar"] is None
        assert graded["matn_ar"] == "عدة من أصحابنا عن أحمد العقل ما عبد به الرحمن"
        assert graded["full_text_ar"] == graded["matn_ar"]
        # Introduction entry has no English split — matn_en falls back to englishText.
        intro = rows["thaqalayn:Al-Kafi-Volume-1-Kulayni:1"]
        assert intro["isnad_raw_en"] is None
        assert intro["matn_ar"] == "الحمد لله الذي حمده"
        assert intro["matn_en"] == "In the praise of Allah..."

    def test_primary_grade_prefers_majlisi(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        rows = {r["source_id"]: r for r in pq.read_table(h).to_pylist()}
        # Majlisi present -> chosen over Mohseni; ungraded intro -> None.
        assert rows["thaqalayn:Al-Kafi-Volume-1-Kulayni:2"]["grade"] == "Ṣaḥīḥ"
        assert rows["thaqalayn:Al-Kafi-Volume-1-Kulayni:1"]["grade"] is None

    def test_provenance_is_shia_thaqalayn(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        table = pq.read_table(h)
        assert all(s == "shia" for s in table.column("sect").to_pylist())
        assert all(c == "thaqalayn" for c in table.column("source_corpus").to_pylist())

    def test_collection_identity_and_appears_in_key(self, tmp_path: Path) -> None:
        """collection_name == bookId so APPEARS_IN ({corpus}:{name}) links cleanly."""
        _seed_real_clone(tmp_path)
        h, c = run(tmp_path, tmp_path / "staging")
        hrows = pq.read_table(h).to_pylist()
        assert {r["collection_name"] for r in hrows} == {
            "Al-Kafi-Volume-1-Kulayni",
            "Al-Kafi-Volume-2-Kulayni",
        }
        crows = {r["collection_id"]: r for r in pq.read_table(c).to_pylist()}
        # collection_id = thaqalayn:<bookId> == f"{corpus}:{hadith.collection_name}"
        assert "thaqalayn:Al-Kafi-Volume-1-Kulayni" in crows
        assert "thaqalayn:Al-Kafi-Volume-2-Kulayni" in crows

    def test_collection_schema_and_volume_display(self, tmp_path: Path) -> None:
        _seed_real_clone(tmp_path)
        _, c = run(tmp_path, tmp_path / "staging")
        table = pq.read_table(c)
        assert table.schema == COLLECTION_SCHEMA
        assert table.num_rows == 2  # one Collection per volume
        names = {r["collection_id"]: r for r in table.to_pylist()}
        v1 = names["thaqalayn:Al-Kafi-Volume-1-Kulayni"]
        assert v1["name_en"] == "Al-Kāfi (Volume 1)"  # volume disambiguates the title
        assert v1["sect"] == "shia"
        assert v1["source_corpus"] == "thaqalayn"
        assert v1["total_hadiths"] == len(_BOOK_1)

    def test_no_duplication_from_aggregate(self, tmp_path: Path) -> None:
        """The allBooks.json aggregate must NOT double the row count (da#175)."""
        _seed_real_clone(tmp_path)
        h, _ = run(tmp_path, tmp_path / "staging")
        assert pq.read_table(h).num_rows == _TOTAL_HADITHS
