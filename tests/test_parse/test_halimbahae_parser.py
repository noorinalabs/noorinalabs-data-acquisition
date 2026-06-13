"""End-to-end tests for the halimbahae/Hadith parser (da#96)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parse.halimbahae import run
from src.parse.schemas import HADITH_SCHEMA
from src.utils.arabic import strip_diacritics

# A diacritised matn with stray RTL marks (U+200F) between words — the exact
# shape halimbahae ships. The tashkeel (e.g. the shadda/fatha on حَدَّثَنَا) must
# survive; the bidi marks must not.
_RLM = "\u200f"  # RIGHT-TO-LEFT MARK
_DIACRITISED = f"{_RLM} {_RLM}حَدَّثَنَا {_RLM} {_RLM}الْحُمَيْدِيُّ عَبْدُ اللَّهِ"


def _make_halimbahae_csv(raw_dir: Path) -> None:
    """Create minimal halimbahae diacritised CSV test data (headerless num,text)."""
    book_dir = raw_dir / "halimbahae" / "Sahih_Al-Bukhari"
    book_dir.mkdir(parents=True)
    rows = [
        f'"1","{_DIACRITISED}"',
        '"2","لَا ضَرَرَ وَلَا ضِرَارَ"',
        '"3","مَنْ حَسُنَ إِسْلَامُ الْمَرْءِ"',
    ]
    csv_path = book_dir / "sahih_al-bukhari_ahadith_mushakkala_mufassala.utf8.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")


class TestHalimbahaeParser:
    def test_produces_hadith_and_collection_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        out_paths = run(raw_dir, staging_dir)

        names = {p.name for p in out_paths}
        assert names == {"hadiths_halimbahae.parquet", "collections_halimbahae.parquet"}
        assert all(p.exists() for p in out_paths)

    def test_hadith_schema_conforms(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_halimbahae.parquet")
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows == 3

    def test_provenance_tagging(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_halimbahae.parquet")
        assert all(c == "halimbahae" for c in table.column("source_corpus").to_pylist())
        assert all(s == "sunni" for s in table.column("sect").to_pylist())

    def test_source_id_is_corpus_namespaced(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_halimbahae.parquet")
        ids = table.column("source_id").to_pylist()
        assert ids[0] == "halimbahae:sahih_al-bukhari:1"
        assert all(i.startswith("halimbahae:sahih_al-bukhari:") for i in ids)

    def test_diacritics_preserved_bidi_marks_stripped(self, tmp_path: Path) -> None:
        """The diacritic-handling contract: tashkeel survives, bidi marks do not."""
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        run(raw_dir, staging_dir)

        table = pq.read_table(staging_dir / "hadiths_halimbahae.parquet")
        first_text = table.column("full_text_ar").to_pylist()[0]
        # Bidi control marks removed.
        assert "\u200f" not in first_text
        assert "\u200e" not in first_text
        # Tashkeel preserved: the stored text still differs from its
        # diacritics-stripped form (i.e. real diacritics remain in the matn).
        assert first_text != strip_diacritics(first_text)
        assert "حَدَّثَنَا" in first_text
        # Whitespace collapsed — no doubled spaces left by the removed marks.
        assert "  " not in first_text

    def test_emits_one_collection_per_book_sect_tagged(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        _make_halimbahae_csv(raw_dir)

        run(raw_dir, staging_dir)

        collections = pq.read_table(staging_dir / "collections_halimbahae.parquet").to_pylist()
        assert len(collections) == 1
        coll = collections[0]
        assert coll["name_en"] == "sahih_al-bukhari"
        assert coll["collection_id"] == "halimbahae:sahih_al-bukhari"
        assert coll["sect"] == "sunni"
        assert coll["source_corpus"] == "halimbahae"
        assert coll["total_hadiths"] == 3

    def test_no_source_dir_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        with pytest.raises(FileNotFoundError):
            run(raw_dir, staging_dir)

    def test_no_diacritised_csv_raises(self, tmp_path: Path) -> None:
        # A book dir holding only the plain (non-mushakkala) CSV is not parsed.
        raw_dir = tmp_path / "raw"
        book_dir = raw_dir / "halimbahae" / "Sahih_Al-Bukhari"
        book_dir.mkdir(parents=True)
        (book_dir / "sahih_al-bukhari_ahadith.utf8.csv").write_text(
            '"1","حدثنا"\n', encoding="utf-8"
        )
        staging_dir = tmp_path / "staging"

        with pytest.raises(FileNotFoundError):
            run(raw_dir, staging_dir)
