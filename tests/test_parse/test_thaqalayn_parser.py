"""End-to-end tests for the Thaqalayn parser."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from src.parse.identity import is_double_prefixed, validate_source_id
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.parse.thaqalayn import run


def _make_thaqalayn_json(raw_dir: Path) -> None:
    """Create minimal Thaqalayn API-format JSON test data."""
    thaq_dir = raw_dir / "thaqalayn"
    thaq_dir.mkdir(parents=True)

    data = {
        "bookName": "Al-Kafi",
        "bookNameAr": "الكافي",
        "data": [
            {
                "hadithNumber": 1,
                "textAr": "نص الحديث الأول",
                "textEn": "First hadith text",
                "grade": "Sahih",
                "chapterEn": "Chapter One",
                "chapterNumber": 1,
            },
            {
                "hadithNumber": 2,
                "textAr": "نص الحديث الثاني",
                "textEn": "Second hadith text",
                "grade": "Hasan",
                "chapterEn": "Chapter One",
                "chapterNumber": 1,
            },
        ],
    }
    (thaq_dir / "book_1.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestThaqalaynParser:
    def test_produces_parquet_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_json(raw_dir)

        hadiths_path, collections_path = run(raw_dir, staging_dir)

        assert hadiths_path.exists()
        assert collections_path.exists()

    def test_hadith_schema_conforms(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows == 2

    def test_collection_schema_conforms(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_json(raw_dir)

        _, collections_path = run(raw_dir, staging_dir)

        table = pq.read_table(collections_path)
        assert table.schema == COLLECTION_SCHEMA
        assert table.num_rows >= 1

    def test_sect_is_shia(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        table = pq.read_table(hadiths_path)
        sects = table.column("sect").to_pylist()
        assert all(s == "shia" for s in sects)


def _make_thaqalayn_github_json(raw_dir: Path) -> None:
    """Create GitHub-format Thaqalayn test data under ``github_clone/``.

    Mirrors the production acquisition path: ``src/acquire/thaqalayn.py`` clones
    the ThaqalaynAPI repo into ``thaqalayn/github_clone/`` and the parser walks it
    with ``rglob("*.json")`` (no top-level ``book_*.json``), so detection takes the
    ``_extract_hadiths_github`` branch instead of the API ``data``-wrapper branch.

    Two shapes are seeded to cover both GitHub-format variants the extractor
    handles:

    * a **bare array** of hadith objects (with one non-hadith metadata dict that
      ``_looks_like_hadith`` must filter out), and
    * a **wrapper dict** whose hadiths live under a ``"hadiths"`` key.
    """
    gh_dir = raw_dir / "thaqalayn" / "github_clone" / "dataset"
    gh_dir.mkdir(parents=True)

    # Bare-array form — the most common ThaqalaynAPI dataset shape. The trailing
    # metadata dict has no hadith-indicative field and must be dropped.
    al_kafi = [
        {
            "hadithNumber": 1,
            "arabicText": "نص الحديث الأول",
            "englishText": "First hadith text",
            "grade": "Sahih",
            "chapter": "Book of Reason",
            "chapterNumber": 1,
        },
        {
            "hadithNumber": 2,
            "arabicText": "نص الحديث الثاني",
            "englishText": "Second hadith text",
            "grade": ["Sahih", "Hasan"],
            "chapter": "Book of Reason",
            "chapterNumber": 1,
        },
        {"_meta": "license", "version": "2.0"},
    ]
    (gh_dir / "al_kafi.json").write_text(json.dumps(al_kafi, ensure_ascii=False), encoding="utf-8")

    # Wrapper-dict form — hadiths nested under a "hadiths" key, no top-level
    # "data" list (so detection stays on the GitHub branch).
    faqih = {
        "bookName": "Man La Yahduruhu al-Faqih",
        "bookNameAr": "من لا يحضره الفقيه",
        "author": "al-Shaykh al-Saduq",
        "hadiths": [
            {
                "hadith_number": 1,
                "textAr": "نص الحديث الثالث",
                "textEn": "Third hadith text",
                "grading": "Muwaththaq",
                "chapterEn": "Book of Purity",
            },
        ],
    }
    (gh_dir / "faqih.json").write_text(json.dumps(faqih, ensure_ascii=False), encoding="utf-8")


class TestThaqalaynGithubFormat:
    """Cover the production GitHub-format parse path (``_extract_hadiths_github``).

    Production acquisition is a ``git clone`` of the ThaqalaynAPI repo, so the
    GitHub branch — not the API ``data``-wrapper branch the other tests exercise —
    is what runs against real data (da#108, follow-up from da#105 / PR #105).
    """

    def test_produces_parquet_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, collections_path = run(raw_dir, staging_dir)

        assert hadiths_path.exists()
        assert collections_path.exists()

    def test_hadith_schema_conforms_and_filters_non_hadith(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        # 2 from the bare array (the non-hadith metadata dict is filtered) +
        # 1 from the wrapper dict's "hadiths" key == 3.
        assert table.num_rows == 3

    def test_both_github_shapes_extracted(self, tmp_path: Path) -> None:
        """Both the bare-array and wrapper-dict hadiths reach staging."""
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        rows = pq.read_table(hadiths_path).to_pylist()
        english = {r["matn_en"] for r in rows}
        # bare-array hadiths + the wrapper-dict ("hadiths" key) hadith.
        assert english == {"First hadith text", "Second hadith text", "Third hadith text"}

    def test_provenance_matches_api_path(self, tmp_path: Path) -> None:
        """sect=shia + source_corpus=thaqalayn on every GitHub-format row."""
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        table = pq.read_table(hadiths_path)
        assert all(s == "shia" for s in table.column("sect").to_pylist())
        assert all(c == "thaqalayn" for c in table.column("source_corpus").to_pylist())

    def test_source_ids_satisfy_identity_contract(self, tmp_path: Path) -> None:
        """Every GitHub-path source_id is canonical and not double-prefixed.

        Same contract the API path and the live light-up assert (da#82 / main#139):
        a known leading corpus, no empty segments, no doubled corpus prefix.
        """
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        source_ids = pq.read_table(hadiths_path).column("source_id").to_pylist()
        assert source_ids  # non-empty
        for sid in source_ids:
            assert validate_source_id(sid) == [], f"non-canonical source_id: {sid!r}"
            assert not is_double_prefixed(sid), f"double-prefixed source_id: {sid!r}"

    def test_grade_array_serialized_to_json(self, tmp_path: Path) -> None:
        """A multi-value grade array is preserved as a JSON string, not dropped."""
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        hadiths_path, _ = run(raw_dir, staging_dir)

        rows = pq.read_table(hadiths_path).to_pylist()
        grades = {r["matn_en"]: r["grade"] for r in rows}
        assert grades["First hadith text"] == "Sahih"
        assert json.loads(grades["Second hadith text"]) == ["Sahih", "Hasan"]

    def test_collections_carry_shia_provenance(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_thaqalayn_github_json(raw_dir)

        _, collections_path = run(raw_dir, staging_dir)

        table = pq.read_table(collections_path)
        assert table.schema == COLLECTION_SCHEMA
        assert all(s == "shia" for s in table.column("sect").to_pylist())
        assert all(c == "thaqalayn" for c in table.column("source_corpus").to_pylist())
