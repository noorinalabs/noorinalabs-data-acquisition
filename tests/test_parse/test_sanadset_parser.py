"""Tests for the Sanadset parser."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from src.parse.identity import collection_node_id, is_double_prefixed
from src.parse.sanadset import _book_key, parse_sanadset
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA, NARRATOR_MENTION_SCHEMA

# Book names shared by the mock hadith rows and ``_BOOKS_CSV`` so the name→book_key
# join resolves — mirrors the real corpus, where books.csv and the hadith CSV both
# cite a book by its Arabic NAME, not an id (da#353).
_BOOK_BUKHARI = "صحيح البخاري"
_BOOK_MUSLIM = "صحيح مسلم"

# books.csv is a NAME-ONLY list in production (a single ``Book`` column, no id).
_BOOKS_CSV = f"Book\n{_BOOK_BUKHARI}\n{_BOOK_MUSLIM}\n"


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
    """Write a mock Sanadset CSV mirroring the real header (da#353).

    Real header: ``Hadith,Book,Num_hadith,Matn,Sanad,Sanad_Length`` — ``Book`` is
    the book's Arabic NAME (not an int id) and ``Num_hadith`` the in-book citation
    ordinal. A ``grade`` column is appended for the grade path. Rows 1-2 share the
    البخاري book, row 3 (No SANAD) is in مسلم.
    """
    header = "Hadith,Book,Num_hadith,grade"
    rows = [
        (
            "<SANAD><NAR>محمد بن عبدالله</NAR> عن <NAR>علي بن أبي طالب</NAR></SANAD>"
            "<MATN>إنما الأعمال بالنيات</MATN>",
            _BOOK_BUKHARI,
            "1",
            "Sahih",
        ),
        (
            "<SANAD><NAR>أبو هريرة</NAR> عن <NAR>أنس بن مالك</NAR> عن "
            "<NAR>مالك بن أنس</NAR></SANAD><MATN>لا ضرر ولا ضرار</MATN>",
            _BOOK_BUKHARI,
            "2",
            "Hasan",
        ),
        (
            "<SANAD>No SANAD</SANAD><MATN>بعض المتن هنا</MATN>",
            _BOOK_MUSLIM,
            "3",
            "",
        ),
    ]
    lines = [header]
    for r in rows:
        # Quote the hadith field to handle embedded commas.
        lines.append(f'"{r[0]}",{r[1]},{r[2]},{r[3]}')
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
        """Absent books.csv → still one Collection per BOOK, keyed on the hadith
        row's own ``Book`` digest. books.csv supplies metadata, never the join key.

        The fixture is named ``sanadset.csv`` — the production filename — because
        the collapse this guards against only reproduces when the CSV stem equals
        the corpus name (see ``TestBooksCsvAbsentIdentity``).
        """
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "sanadset.csv")

        outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        assert "collections" in outputs
        table = pq.read_table(staging_dir / "collections_sanadset.parquet")
        assert table.schema == COLLECTION_SCHEMA
        key_b, key_m = _book_key(_BOOK_BUKHARI), _book_key(_BOOK_MUSLIM)
        assert table.num_rows == 2
        by_id = {c["collection_id"]: c for c in table.to_pylist()}
        assert set(by_id) == {f"sanadset:{key_b}", f"sanadset:{key_m}"}
        # No books.csv → no Arabic name, but a deterministic human label keyed on
        # the digest — never the bare digest as name_en.
        assert by_id[f"sanadset:{key_b}"]["name_ar"] is None
        assert by_id[f"sanadset:{key_b}"]["name_en"] == f"Sanadset book {key_b}"
        assert by_id[f"sanadset:{key_m}"]["name_en"] == f"Sanadset book {key_m}"
        assert by_id[f"sanadset:{key_b}"]["total_hadiths"] == 2
        assert by_id[f"sanadset:{key_m}"]["total_hadiths"] == 1
        assert all(
            c["source_corpus"] == "sanadset" and c["sect"] == "sunni" for c in by_id.values()
        )

    def test_collections_from_books_csv(self, tmp_path: Path) -> None:
        """books.csv present → one Collection per book NAME (da#353 name-keyed join)."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "hadiths.csv")
        (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")

        outputs = parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        # books.csv must NOT be parsed as hadiths — still exactly 3 hadiths.
        hadiths = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
        assert len(hadiths) == 3
        # collection_name is now the book NAME digest (per-book breadth), not the
        # CSV stem and not a positional book_id.
        key_b, key_m = _book_key(_BOOK_BUKHARI), _book_key(_BOOK_MUSLIM)
        assert sorted(h["collection_name"] for h in hadiths) == sorted([key_b, key_b, key_m])

        assert "collections" in outputs
        collections = pq.read_table(staging_dir / "collections_sanadset.parquet").to_pylist()
        by_id = {c["collection_id"]: c for c in collections}
        assert set(by_id) == {f"sanadset:{key_b}", f"sanadset:{key_m}"}
        # The Arabic book name flows from books.csv onto the Collection node.
        assert by_id[f"sanadset:{key_b}"]["name_ar"] == _BOOK_BUKHARI
        assert by_id[f"sanadset:{key_m}"]["name_ar"] == _BOOK_MUSLIM
        # total_hadiths reflects what was parsed (البخاري has 2 rows, مسلم has 1).
        assert by_id[f"sanadset:{key_b}"]["total_hadiths"] == 2
        assert by_id[f"sanadset:{key_m}"]["total_hadiths"] == 1
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
            "Hadith,Book,Num_hadith,grade\n"
            '"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>قال</NAR> '
            "<NAR>رضي الله عنه</NAR> <NAR>He said</NAR> "
            "<NAR>Malik ibn Anas</NAR> عن <NAR>أبو هريرة</NAR></SANAD>"
            f'<MATN>متن</MATN>",{_BOOK_BUKHARI},1,Sahih\n'
        )
        (raw_dir / "sanadset.csv").write_text(polluted, encoding="utf-8")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
        names = table.column("name_ar").to_pylist()
        # Only the two genuine narrators survive; positions stay gap-free.
        assert names == ["مالك بن أنس", "أبو هريرة"]
        assert table.column("position_in_chain").to_pylist() == [0, 1]
        # Schema is still conformant after filtering.
        assert table.schema == NARRATOR_MENTION_SCHEMA


# Regex catching any residual <NAR>/<SANAD>/<MATN> structural markup in a string,
# used by the da#318 tests to assert the stored matn is markup-free.
_MARKUP_PRESENT = re.compile(r"<\s*/?\s*(?:NAR|SANAD|MATN)\s*/?\s*>")


class TestMatnMarkupStrip:
    """Structural-markup leak into ``matn_ar`` (da#318).

    A malformed upstream row carries a doubled ``<SANAD>…<MATN>`` segment closed by
    a single ``</MATN>``; ``_MATN_RE``'s first-open→first-close capture then swallows
    the intervening ``</SANAD>``, second ``<SANAD>``, ``<NAR>…</NAR>`` and second
    ``<MATN>`` tags verbatim into ``matn_ar``. The parser now strips them before
    store (``_strip_structural_tags``) — WITHOUT touching ``isnad_raw_ar``, which
    legitimately carries ``<NAR>`` tags for the chain extractor.
    """

    # A doubled-SANAD row à la ``sanadset:0:0:107270``: two ``<SANAD>…<MATN>``
    # segments, but only ONE closing ``</MATN>`` at the very end.
    _MALFORMED_ROW = (
        "<SANAD><NAR>محمد بن عبدالله</NAR></SANAD>"
        "<MATN>إنما الأعمال بالنيات"
        "<SANAD><NAR>علي بن أبي طالب</NAR></SANAD>"
        "<MATN>وإنما لكل امرئ ما نوى</MATN>"
    )

    def _parse_single_row(self, tmp_path: Path, hadith_field: str) -> dict[str, object]:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        csv = f'Hadith,Book,Num_hadith,grade\n"{hadith_field}",{_BOOK_BUKHARI},1,Sahih\n'
        (raw_dir / "sanadset.csv").write_text(csv, encoding="utf-8")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        rows = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
        assert len(rows) == 1
        return rows[0]

    def test_malformed_row_matn_has_no_residual_markup(self, tmp_path: Path) -> None:
        row = self._parse_single_row(tmp_path, self._MALFORMED_ROW)
        matn = row["matn_ar"]
        assert matn is not None
        # No literal structural tag survives into the stored matn.
        assert _MARKUP_PRESENT.search(matn) is None, matn
        # The genuine Arabic matn text from both swallowed segments is retained.
        assert "إنما الأعمال بالنيات" in matn
        assert "وإنما لكل امرئ ما نوى" in matn

    def test_malformed_row_isnad_raw_ar_still_carries_nar_tags(self, tmp_path: Path) -> None:
        """Regression guard: the fix must NOT touch ``isnad_raw_ar`` (da#318 hazard)."""
        row = self._parse_single_row(tmp_path, self._MALFORMED_ROW)
        isnad = row["isnad_raw_ar"]
        assert isnad is not None
        # isnad_raw_ar is the first <SANAD>…</SANAD> content and MUST still carry its
        # <NAR> tags verbatim — the chain extractor (_extract_narrator_mentions)
        # depends on them.
        assert "<NAR>" in isnad and "</NAR>" in isnad
        assert isnad == "<NAR>محمد بن عبدالله</NAR>"

    def test_well_formed_row_matn_untouched(self, tmp_path: Path) -> None:
        row = self._parse_single_row(
            tmp_path,
            "<SANAD><NAR>أبو هريرة</NAR></SANAD><MATN>لا ضرر ولا ضرار</MATN>",
        )
        # A clean matn carries no structural tags and is stored exactly as-is.
        assert row["matn_ar"] == "لا ضرر ولا ضرار"

    def test_strip_is_idempotent(self) -> None:
        from src.parse.sanadset import _strip_structural_tags

        raw = "إنما الأعمال بالنيات<SANAD><NAR>علي</NAR></SANAD><MATN>وإنما لكل امرئ"
        once = _strip_structural_tags(raw)
        twice = _strip_structural_tags(once)
        assert once == twice
        assert once is not None
        assert _MARKUP_PRESENT.search(once) is None

    def test_strip_covers_self_closing_tag(self) -> None:
        from src.parse.sanadset import _strip_structural_tags

        out = _strip_structural_tags("متن <NAR/> نص")
        assert out == "متن نص"

    def test_strip_passes_through_clean_and_none(self) -> None:
        from src.parse.sanadset import _strip_structural_tags

        # Clean text is returned unchanged (no whitespace normalization applied).
        assert _strip_structural_tags("متن\nنظيف") == "متن\nنظيف"
        assert _strip_structural_tags(None) is None
        # An all-markup string collapses to None (mirrors _extract_tag's empty rule).
        assert _strip_structural_tags("<MATN></MATN>") is None


class TestSanadsetColumnMapping:
    """da#353: name-keyed books join, populated hadith_number, stable content id.

    The Sanadset parser previously looked for ``hadith_id``/``book_id`` columns that
    do not exist in the real CSV (whose header is ``Hadith,Book,Num_hadith,Matn,
    Sanad,Sanad_Length``), nulling ``hadith_number`` and collapsing all 650,986 rows
    into one collection with a positional, re-acquisition-unstable source_id.
    """

    def _parse(self, tmp_path: Path) -> Path:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "sanadset.csv")
        (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")
        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)
        return staging_dir

    def test_name_keyed_join_one_collection_per_distinct_book(self, tmp_path: Path) -> None:
        """N distinct book names → N Collection nodes (the da#219 breadth, now real)."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        names = ["كتاب الأول", "كتاب الثاني", "كتاب الثالث", "كتاب الرابع", "كتاب الخامس"]
        lines = ["Hadith,Book,Num_hadith"]
        for i, nm in enumerate(names, start=1):
            lines.append(f'"<SANAD><NAR>راوي {i}</NAR></SANAD><MATN>متن {i}</MATN>",{nm},{i}')
        (raw_dir / "sanadset.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (raw_dir / "books.csv").write_text("Book\n" + "\n".join(names) + "\n", encoding="utf-8")

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

        collections = pq.read_table(staging_dir / "collections_sanadset.parquet").to_pylist()
        assert len(collections) == len(names)
        by_id = {c["collection_id"]: c for c in collections}
        assert set(by_id) == {f"sanadset:{_book_key(nm)}" for nm in names}
        assert {c["name_ar"] for c in collections} == set(names)

    def test_hadith_number_populated(self, tmp_path: Path) -> None:
        table = pq.read_table(self._parse(tmp_path) / "hadiths_sanadset.parquet")
        hnums = table.column("hadith_number").to_pylist()
        # Every row carries its Num_hadith — none null (the 76%-null defect gone).
        assert hnums == [1, 2, 3]
        assert all(n is not None for n in hnums)

    def test_no_double_corpus_source_ids(self, tmp_path: Path) -> None:
        sids = (
            pq.read_table(self._parse(tmp_path) / "hadiths_sanadset.parquet")
            .column("source_id")
            .to_pylist()
        )
        for sid in sids:
            # The collection segment is a name-digest, never "sanadset", so nothing
            # looks double-prefixed and _collapse_double_corpus never fires.
            assert not sid.startswith("sanadset:sanadset:"), sid
            assert not is_double_prefixed(sid), sid

    def _sid_for(self, tmp_path: Path, hadith_csv: str, book_name: str, num: int) -> str:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        (raw_dir / "sanadset.csv").write_text(hadith_csv, encoding="utf-8")
        (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")
        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)
        rows = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
        (row,) = [
            r
            for r in rows
            if r["collection_name"] == _book_key(book_name) and r["hadith_number"] == num
        ]
        return str(row["source_id"])

    def test_source_id_stable_under_row_insertion(self, tmp_path: Path) -> None:
        """A fixed (Book, Num_hadith) keeps a byte-identical source_id when an
        unrelated row is inserted ahead of it — the determinism contract da#353
        restores (the old row-offset id re-keyed every following hadith)."""
        header = "Hadith,Book,Num_hadith"
        target = f'"<SANAD><NAR>مالك</NAR></SANAD><MATN>متن الهدف</MATN>",{_BOOK_MUSLIM},7'
        inserted = f'"<SANAD><NAR>أنس</NAR></SANAD><MATN>متن مقحم</MATN>",{_BOOK_BUKHARI},99'
        csv_a = header + "\n" + target + "\n"
        csv_b = header + "\n" + inserted + "\n" + target + "\n"

        sid_a = self._sid_for(tmp_path / "a", csv_a, _BOOK_MUSLIM, 7)
        sid_b = self._sid_for(tmp_path / "b", csv_b, _BOOK_MUSLIM, 7)
        assert sid_a == sid_b
        assert sid_a == f"sanadset:{_book_key(_BOOK_MUSLIM)}:7"

    def test_no_book_column_raises(self, tmp_path: Path) -> None:
        """A hadith CSV with no ``Book`` column raises. That column is the collection
        join key; falling back to the CSV stem would re-key every row onto
        ``{corpus}:{corpus}:{Num_hadith}`` and collapse the corpus."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        (raw_dir / "sanadset.csv").write_text(
            'Hadith,Num_hadith\n"<SANAD><NAR>مالك</NAR></SANAD><MATN>متن</MATN>",1\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="no 'Book' column"):
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)

    def test_present_but_unmappable_books_raises(self, tmp_path: Path) -> None:
        """A present books.csv with no resolvable name column raises (da#353 item 4):
        a dead join key must fail loudly, not silently disable per-book breadth."""
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        _make_sanadset_csv(raw_dir / "sanadset.csv")
        # An id-only books.csv — no name column at all.
        (raw_dir / "books.csv").write_text("serial\n1\n2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="book-name column"):
            parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)


class TestBooksCsvAbsentIdentity:
    """A missing books.csv must degrade METADATA, never IDENTITY.

    ``src/acquire/sanadset.py`` skips the whole Mendeley download when the raw dir
    already holds any ``*.csv`` and never checks that ``books.csv`` landed, and
    ``_download_mendeley`` has no rollback. A transient failure fetching the 48 KB
    books.csv *after* the 1.4 GB sanadset.csv is written poisons every later run —
    so ``books.csv``-absent is a reachable, sticky production state, not a
    hypothetical.

    Two properties must hold in that state, and both are load-bearing:

    * **Row-name realism.** The hadith CSV is named ``sanadset.csv``, exactly as
      production has it, so its stem equals ``_SOURCE_CORPUS``. Any fixture named
      otherwise (the old ``hadiths.csv``) makes ``default_name != corpus`` and is
      structurally incapable of reproducing the doubled-prefix collapse.
    * **Ordinal reuse.** The two books share ``Num_hadith`` 1 and 2. ``Num_hadith``
      is an in-book ordinal (37,239 distinct values across 650,986 real rows), so a
      fixture with globally unique ordinals would keep ids distinct even under the
      collapsing code and prove nothing.
    """

    _CSV = (
        "Hadith,Book,Num_hadith\n"
        f'"<SANAD><NAR>مالك</NAR></SANAD><MATN>متن أ</MATN>",{_BOOK_BUKHARI},1\n'
        f'"<SANAD><NAR>أنس</NAR></SANAD><MATN>متن ب</MATN>",{_BOOK_BUKHARI},2\n'
        f'"<SANAD><NAR>شعبة</NAR></SANAD><MATN>متن ج</MATN>",{_BOOK_MUSLIM},1\n'
        f'"<SANAD><NAR>قتادة</NAR></SANAD><MATN>متن د</MATN>",{_BOOK_MUSLIM},2\n'
    )
    _ROW_COUNT = 4

    def _parse_without_books(self, tmp_path: Path) -> list[dict[str, object]]:
        raw_dir = tmp_path / "raw" / "sanadset"
        raw_dir.mkdir(parents=True)
        staging_dir = tmp_path / "staging"
        (raw_dir / "sanadset.csv").write_text(self._CSV, encoding="utf-8")
        assert not (raw_dir / "books.csv").exists()

        parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)
        rows: list[dict[str, object]] = pq.read_table(
            staging_dir / "hadiths_sanadset.parquet"
        ).to_pylist()
        return rows

    def test_every_hadith_keeps_a_distinct_source_id(self, tmp_path: Path) -> None:
        """650,986 rows → 650,986 ids. Keyed on Num_hadith alone they would collapse
        to 37,239, silently destroying 613,747 hadiths (94.3%) at MERGE time."""
        rows = self._parse_without_books(tmp_path)
        assert len(rows) == self._ROW_COUNT

        source_ids = [str(r["source_id"]) for r in rows]
        assert len(set(source_ids)) == self._ROW_COUNT, (
            f"source_id collision without books.csv: {self._ROW_COUNT} rows -> "
            f"{len(set(source_ids))} distinct ids ({sorted(source_ids)})"
        )

    def test_no_source_id_is_double_prefixed(self, tmp_path: Path) -> None:
        """No id carries the doubled corpus segment, so ``hadith_node_id``'s
        ``_collapse_double_corpus`` never fires and never merges two hadiths."""
        rows = self._parse_without_books(tmp_path)
        for r in rows:
            sid = str(r["source_id"])
            assert not sid.startswith("sanadset:sanadset:"), sid
            assert not is_double_prefixed(sid), sid

    def test_collection_key_is_the_book_not_the_csv_stem(self, tmp_path: Path) -> None:
        """The join key comes from the row's own ``Book``, so per-book breadth
        survives a missing books.csv — degraded metadata, intact identity."""
        rows = self._parse_without_books(tmp_path)
        key_b, key_m = _book_key(_BOOK_BUKHARI), _book_key(_BOOK_MUSLIM)
        assert {str(r["collection_name"]) for r in rows} == {key_b, key_m}
        assert sorted(str(r["source_id"]) for r in rows) == sorted(
            [
                f"sanadset:{key_b}:1",
                f"sanadset:{key_b}:2",
                f"sanadset:{key_m}:1",
                f"sanadset:{key_m}:2",
            ]
        )
