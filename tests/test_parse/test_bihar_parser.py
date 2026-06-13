"""End-to-end tests for the Bihar al-Anwar (hubeali) parser.

The fixture ``fixtures/bihar_volume-1-part-1_sample.html`` is a trimmed but
otherwise verbatim capture of a real hubeali.com Bihar al-Anwar page (Volume 1,
Part 1 — the start of Kitab al-Aql, chapter 1, hadiths 1-8), so the parser is
exercised against genuine source markup, not a synthetic toy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.models.enums import Sect, SourceCorpus
from src.parse.bihar import run
from src.parse.identity import is_double_prefixed, validate_source_id
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "bihar_volume-1-part-1_sample.html"


def _stage(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the real hubeali fixture into a raw/bihar dir and run the parser."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "bihar").mkdir(parents=True)
    shutil.copy(FIXTURE, raw_dir / "bihar" / "volume-1-part-1.html")
    return run(raw_dir, tmp_path / "staging")


class TestBiharParser:
    def test_produces_both_parquet_files(self, tmp_path: Path) -> None:
        hadiths_path, collections_path = _stage(tmp_path)
        assert hadiths_path.exists()
        assert collections_path.exists()

    def test_hadith_schema_conforms_and_nonempty(self, tmp_path: Path) -> None:
        hadiths_path, _ = _stage(tmp_path)
        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        # The real Volume-1 Part-1 chapter-1 sample holds 8 numbered hadiths.
        assert table.num_rows == 8

    def test_collection_schema_conforms(self, tmp_path: Path) -> None:
        _, collections_path = _stage(tmp_path)
        table = pq.read_table(collections_path)
        assert table.schema == COLLECTION_SCHEMA
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["name_en"] == "Bihar al-Anwar"
        assert row["source_corpus"] == SourceCorpus.BIHAR.value
        assert row["sect"] == Sect.SHIA.value
        assert row["total_hadiths"] == 8

    def test_provenance_tagging(self, tmp_path: Path) -> None:
        hadiths_path, _ = _stage(tmp_path)
        rows = pq.read_table(hadiths_path).to_pylist()
        assert all(r["sect"] == Sect.SHIA.value for r in rows)
        assert all(r["source_corpus"] == SourceCorpus.BIHAR.value for r in rows)
        assert all(r["collection_name"] == "Bihar al-Anwar" for r in rows)

    def test_real_bilingual_text_extracted(self, tmp_path: Path) -> None:
        hadiths_path, _ = _stage(tmp_path)
        rows = pq.read_table(hadiths_path).to_pylist()
        # Every hadith carries both Arabic and English from the alternating <p> run.
        assert all(r["full_text_ar"] for r in rows)
        assert all(r["full_text_en"] for r in rows)
        # The "<N>- " ordinal marker is stripped from the Arabic body.
        assert not rows[0]["full_text_ar"].lstrip().startswith("1-")

    def test_in_chapter_ordinals_are_sequential(self, tmp_path: Path) -> None:
        hadiths_path, _ = _stage(tmp_path)
        rows = pq.read_table(hadiths_path).to_pylist()
        assert [r["hadith_number"] for r in rows] == list(range(1, 9))
        assert all(r["chapter_number"] == 1 for r in rows)
        assert all(r["book_number"] == 1 for r in rows)  # volume 1

    def test_source_ids_valid_and_not_double_prefixed(self, tmp_path: Path) -> None:
        hadiths_path, _ = _stage(tmp_path)
        rows = pq.read_table(hadiths_path).to_pylist()
        ids = [r["source_id"] for r in rows]
        # Collision-safe grammar: "bihar:bihar-al-anwar:<vol>:<chapter>:<ordinal>".
        for sid in ids:
            assert validate_source_id(sid) == [], f"{sid}: {validate_source_id(sid)}"
            assert not is_double_prefixed(sid), sid
            assert sid.startswith("bihar:bihar-al-anwar:1:1:")
        assert len(set(ids)) == len(ids)  # unique

    def test_missing_raw_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "raw" / "bihar").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            run(tmp_path / "raw", tmp_path / "staging")
