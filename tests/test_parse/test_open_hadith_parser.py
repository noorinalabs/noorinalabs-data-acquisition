"""End-to-end tests for the Open Hadith parser."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from src.parse.open_hadith import run
from src.parse.schemas import HADITH_SCHEMA


def _make_open_hadith_csv(raw_dir: Path) -> None:
    """Create minimal Open Hadith diacritics CSV test data.

    Real Open Hadith CSVs have no header row — just number,text per line.
    """
    oh_dir = raw_dir / "open_hadith"
    oh_dir.mkdir(parents=True)

    rows = [
        '"1","إنما الأعمال بالنيات"',
        '"2","لا ضرر ولا ضرار"',
        '"3","من حسن إسلام المرء تركه ما لا يعنيه"',
    ]
    csv_path = oh_dir / "bukhari_tashkeel.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")


class TestOpenHadithParser:
    def test_produces_hadith_and_collection_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_open_hadith_csv(raw_dir)

        out_paths = run(raw_dir, staging_dir)

        names = {p.name for p in out_paths}
        assert names == {"hadiths_open_hadith.parquet", "collections_open_hadith.parquet"}
        assert all(p.exists() for p in out_paths)

    def test_hadith_schema_conforms(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_open_hadith_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_open_hadith.parquet")
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows == 3

    def test_source_corpus_field(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_open_hadith_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_open_hadith.parquet")
        corpora = table.column("source_corpus").to_pylist()
        assert all(c == "open_hadith" for c in corpora)

    def test_emits_one_collection_per_book_sect_tagged(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_open_hadith_csv(raw_dir)

        run(raw_dir, staging_dir)

        collections = pq.read_table(staging_dir / "collections_open_hadith.parquet").to_pylist()
        assert len(collections) == 1
        coll = collections[0]
        # Clean slug — the "_tashkeel" diacritics marker is stripped from the name.
        assert coll["name_en"] == "bukhari"
        assert coll["collection_id"] == "open_hadith:bukhari"
        assert coll["sect"] == "sunni"
        assert coll["source_corpus"] == "open_hadith"
        assert coll["total_hadiths"] == 3

    def test_collection_name_strips_descriptor_tail(self, tmp_path: Path) -> None:
        # The real Open-Hadith filename shape: <book>_ahadith_mushakkala_mufassala.utf8.
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        oh_dir = raw_dir / "open_hadith" / "Sahih_Al-Bukhari"
        oh_dir.mkdir(parents=True)
        (oh_dir / "sahih_al-bukhari_ahadith_mushakkala_mufassala.utf8.csv").write_text(
            '"1","إنما الأعمال بالنيات"\n', encoding="utf-8"
        )

        run(raw_dir, staging_dir)

        collections = pq.read_table(staging_dir / "collections_open_hadith.parquet").to_pylist()
        assert collections[0]["name_en"] == "sahih_al-bukhari"

    def test_no_source_dir_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        import pytest

        with pytest.raises(FileNotFoundError):
            run(raw_dir, staging_dir)
