"""Tests for the Sanadset parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from src.parse.identity import collection_node_id
from src.parse.sanadset import parse_sanadset
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA, NARRATOR_MENTION_SCHEMA

# books.csv mapping the two book_ids used by ``_make_sanadset_csv`` to collections.
_BOOKS_CSV = "book_id,name,author\n1,صحيح البخاري,البخاري\n2,صحيح مسلم,مسلم\n"


def _appears_in_collection_id(hadith_row: dict[str, object]) -> str:
    """Rebuild the Collection node id ``load_edges._load_appears_in`` MATCHes on.

    Mirrors ``f"{source_corpus}:{collection_name}"`` → ``collection_node_id`` so a
    test can assert the parser's emitted Collection ids cover every hadith's
    APPEARS_IN endpoint without standing up Neo4j.
    """
    corpus = hadith_row["source_corpus"]
    cname = hadith_row["collection_name"]
    return collection_node_id(f"{corpus}:{cname}")


def _make_sanadset_csv(path: Path) -> Path:
    """Write a mock Sanadset CSV with NAR-tagged rows and a 'No SANAD' row."""
    header = "hadith_id,book_id,hadith,grade"
    rows = [
        (
            "1",
            "1",
            "<SANAD><NAR>محمد بن عبدالله</NAR> عن <NAR>علي بن أبي طالب</NAR></SANAD>"
            "<MATN>إنما الأعمال بالنيات</MATN>",
            "Sahih",
        ),
        (
            "2",
            "1",
            "<SANAD><NAR>أبو هريرة</NAR> عن <NAR>أنس بن مالك</NAR> عن "
            "<NAR>مالك بن أنس</NAR></SANAD><MATN>لا ضرر ولا ضرار</MATN>",
            "Hasan",
        ),
        (
            "3",
            "2",
            "<SANAD>No SANAD</SANAD><MATN>بعض المتن هنا</MATN>",
            "",
        ),
    ]
    lines = [header]
    for r in rows:
        # Quote the hadith field to handle embedded commas
        lines.append(f'{r[0]},{r[1]},"{r[2]}",{r[3]}')
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestSanadsetParser:
    def test_produces_parquet_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        assert "hadiths" in outputs
        assert (staging_dir / "hadiths_sanadset.parquet").exists()

    def test_hadith_row_count(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "hadiths_sanadset.parquet")
        assert table.num_rows == 3

    def test_hadith_schema_conforms(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "hadiths_sanadset.parquet")
        assert table.schema == HADITH_SCHEMA

    def test_narrator_mentions_extracted(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        assert "narrator_mentions" in outputs
        table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
        assert table.schema == NARRATOR_MENTION_SCHEMA
        # Row 1 has 2 narrators, row 2 has 3, row 3 (No SANAD) has 0 => 5 total
        assert table.num_rows == 5

    def test_position_in_chain_sequential(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
        positions = table.column("position_in_chain").to_pylist()
        # First hadith: [0, 1], second: [0, 1, 2]
        assert positions == [0, 1, 0, 1, 2]

    def test_no_sanad_row_has_null_isnad(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "hadiths_sanadset.parquet")
        isnads = table.column("isnad_raw_ar").to_pylist()
        # Third row should be None (No SANAD)
        assert isnads[2] is None
        # First two should have values
        assert isnads[0] is not None
        assert isnads[1] is not None

    def test_no_csv_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"

        import pytest

        with pytest.raises(FileNotFoundError):
            with patch("src.parse.sanadset.get_settings") as mock_settings:
                mock_settings.return_value.data_raw_dir = tmp_path / "raw"
                mock_settings.return_value.data_staging_dir = staging_dir
                parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

    def test_narrator_bios_parsed_without_corpus_failfast(self, tmp_path: Path) -> None:
        """Bios parse cleanly even though their id namespace is not a SourceCorpus.

        Regression for da#89: the bio_id is namespaced by the bio *source*
        (``kaggle_narrators``), which is not a hadith ``SourceCorpus``. Building
        it through ``generate_source_id`` would trip that helper's da#82
        fail-fast on an unknown corpus and abort the whole parse the moment a
        ``narrators/`` directory is present (the real acquisition path). Here a
        bio CSV IS present, so a regression resurfaces as a raised ValueError.
        """
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        narrators_dir = raw_dir / "narrators"
        narrators_dir.mkdir()
        staging_dir = tmp_path / "staging"

        _make_sanadset_csv(raw_dir / "hadiths.csv")
        (narrators_dir / "bios.csv").write_text(
            "name,death_year\nمالك بن أنس,179\nأبو هريرة,59\n",
            encoding="utf-8",
        )

        with patch("src.parse.sanadset.get_settings") as mock_settings:
            mock_settings.return_value.data_raw_dir = tmp_path / "raw"
            mock_settings.return_value.data_staging_dir = staging_dir
            outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        assert "narrators_bio" in outputs
        bio_table = pq.read_table(outputs["narrators_bio"])
        assert bio_table.num_rows == 2
        # bio_id is the bio-source-namespaced provenance key, never corpus-gated.
        bio_ids = bio_table.column("bio_id").to_pylist()
        assert all(bid.startswith("kaggle_narrators:bios:") for bid in bio_ids)


class TestSanadsetCollections:
    """Collection emission for APPEARS_IN linkage (da#219 / Path B-B1)."""

    def test_collections_emitted_without_books_csv(self, tmp_path: Path) -> None:
        """Absent books.csv → one Collection per CSV stem (pre-B1 fallback)."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "hadiths.csv")

        outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        assert "collections" in outputs
        table = pq.read_table(staging_dir / "collections_sanadset.parquet")
        assert table.schema == COLLECTION_SCHEMA
        # All three hadiths fall under the single CSV-stem collection "hadiths".
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["collection_id"] == "sanadset:hadiths"
        assert row["source_corpus"] == "sanadset"
        assert row["sect"] == "sunni"
        assert row["total_hadiths"] == 3

    def test_collections_from_books_csv(self, tmp_path: Path) -> None:
        """books.csv present → one Collection per book_id, named from books.csv."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "hadiths.csv")
        (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")

        outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        # books.csv must NOT be parsed as hadiths — still exactly 3 hadiths.
        hadiths = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
        assert len(hadiths) == 3
        # collection_name is now the book_id (per-book breadth), not the CSV stem.
        assert sorted(h["collection_name"] for h in hadiths) == ["1", "1", "2"]

        assert "collections" in outputs
        collections = pq.read_table(staging_dir / "collections_sanadset.parquet").to_pylist()
        by_id = {c["collection_id"]: c for c in collections}
        assert set(by_id) == {"sanadset:1", "sanadset:2"}
        # Book metadata flows from books.csv onto the Collection node.
        assert by_id["sanadset:1"]["name_ar"] == "صحيح البخاري"
        assert by_id["sanadset:1"]["compiler_name"] == "البخاري"
        assert by_id["sanadset:1"]["total_hadiths"] == 2  # book_id 1 has 2 hadiths
        assert by_id["sanadset:2"]["total_hadiths"] == 1
        assert all(c["sect"] == "sunni" and c["source_corpus"] == "sanadset" for c in collections)

    def test_emitted_collections_cover_every_appears_in_endpoint(self, tmp_path: Path) -> None:
        """Every hadith's APPEARS_IN collection endpoint has a matching Collection.

        This is the linkage contract the orphan defect (ADR-003) broke: the
        Collection node ids the edge loader MATCHes on (rebuilt here from each
        hadith) must all exist among the emitted collections, or APPEARS_IN drops.
        """
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "hadiths.csv")
        (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        hadiths = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
        collections = pq.read_table(staging_dir / "collections_sanadset.parquet").to_pylist()
        emitted_node_ids = {collection_node_id(c["collection_id"]) for c in collections}
        needed_node_ids = {_appears_in_collection_id(h) for h in hadiths}
        assert needed_node_ids <= emitted_node_ids
        # And no hadith is left without a collection endpoint.
        assert needed_node_ids


class TestNarratorResegmentation:
    """``<NAR>`` re-segmentation + pollution filtering (da#221 / Path B-B3)."""

    def test_is_narrator_like_accepts_genuine_names(self) -> None:
        from src.parse.sanadset import _is_narrator_like

        for name in ("محمد بن عبدالله", "علي بن أبي طالب", "أبو هريرة", "أنس بن مالك"):
            assert _is_narrator_like(name), name

    def test_is_narrator_like_rejects_pollution(self) -> None:
        from src.parse.sanadset import _is_narrator_like

        # Bare transmission verbs, honorifics, English / Latin transliteration
        # fragments, and dangling particles are all rejected.
        for junk in (
            "قال",  # bare transmission verb
            "عن",  # bare particle / marker
            "رضي الله عنه",  # honorific phrase, names no narrator
            "صلى الله عليه وسلم",
            "He said",  # English fragment
            "that",
            "Muhammad ibn Abdullah",  # vowel-stripped Latin transliteration
            "بن",  # bare nasab particle
            "عبد",  # incomplete theophoric particle
            "",
            "   ",
        ):
            assert not _is_narrator_like(junk), junk

    def test_segment_single_clean_name_preserves_raw(self) -> None:
        from src.parse.sanadset import _segment_nar_content

        out = _segment_nar_content("مَالِك بن أَنَس")
        assert len(out) == 1
        name_ar, name_norm, method = out[0]
        # Raw voweled form is preserved for the common single-name tag…
        assert name_ar == "مَالِك بن أَنَس"
        # …while the normalized key strips the diacritics.
        assert name_norm == "مالك بن انس"
        assert method is None

    def test_segment_drops_pollution_tags(self) -> None:
        from src.parse.sanadset import _segment_nar_content

        assert _segment_nar_content("قال") == []
        assert _segment_nar_content("رضي الله عنه") == []
        assert _segment_nar_content("He said") == []
        assert _segment_nar_content("   ") == []

    def test_segment_resegments_mistagged_subchain(self) -> None:
        from src.parse.sanadset import _segment_nar_content

        # A single <NAR> tag mistakenly holding a whole sub-chain with an internal
        # transmission marker is split into its constituent narrators.
        out = _segment_nar_content("محمد بن عبدالله حدثنا علي بن أبي طالب")
        names = [name_norm for _raw, name_norm, _m in out]
        assert names == ["محمد بن عبدالله", "علي بن ابي طالب"]

    def test_clean_corpus_mentions_unchanged(self, tmp_path: Path) -> None:
        """Genuine narrators flow through B3 unchanged (5 for the clean fixture)."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "hadiths.csv")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
        assert table.num_rows == 5
        assert table.column("position_in_chain").to_pylist() == [0, 1, 0, 1, 2]

    def test_polluted_corpus_is_filtered(self, tmp_path: Path) -> None:
        """A polluted firehose drops fragments and keeps only genuine narrators."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        # Two genuine narrators, interleaved with a transmission verb, an
        # honorific, an English fragment, and a Latin transliteration — all <NAR>.
        polluted = (
            "hadith_id,book_id,hadith,grade\n"
            '1,1,"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>قال</NAR> '
            "<NAR>رضي الله عنه</NAR> <NAR>He said</NAR> "
            "<NAR>Malik ibn Anas</NAR> عن <NAR>أبو هريرة</NAR></SANAD>"
            '<MATN>متن</MATN>",Sahih\n'
        )
        (raw_dir / "hadiths.csv").write_text(polluted, encoding="utf-8")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
        names = table.column("name_ar").to_pylist()
        # Only the two genuine narrators survive; positions stay gap-free.
        assert names == ["مالك بن أنس", "أبو هريرة"]
        assert table.column("position_in_chain").to_pylist() == [0, 1]
        # Schema is still conformant after filtering.
        assert table.schema == NARRATOR_MENTION_SCHEMA
