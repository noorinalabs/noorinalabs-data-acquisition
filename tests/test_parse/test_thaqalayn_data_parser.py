"""End-to-end tests for the ThaqalaynData (Tahdhib + al-Istibsar) parser.

These run against REAL upstream verse files vendored under
``fixtures/thaqalayn_data_raw/`` (CC0), not a hand-built mock — the main#671
guard: a schema drift that silently zeroed the Arabic body would fail
``test_every_hadith_has_arabic`` here, instead of shipping 0%-text undetected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

from src.parse.identity import is_double_prefixed, validate_source_id
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.parse.thaqalayn_data import run

_ARABIC = re.compile(r"[؀-ۿ]")
_FIXTURE_RAW = Path(__file__).parent / "fixtures" / "thaqalayn_data_raw"
_BOOK_SLUGS = {"al-istibsar", "tahdhib-al-ahkam"}


def _parse(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    """Run the parser against the real fixtures; return (hadith rows, collection rows)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    hadiths_path, collections_path = run(_FIXTURE_RAW, staging)
    hadiths = pq.read_table(hadiths_path).to_pylist()
    collections = pq.read_table(collections_path).to_pylist()
    return hadiths, collections


class TestThaqalaynDataParser:
    def test_produces_both_schemas(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        hadiths_path, collections_path = run(_FIXTURE_RAW, staging)
        assert pq.read_schema(hadiths_path).equals(HADITH_SCHEMA)
        assert pq.read_schema(collections_path).equals(COLLECTION_SCHEMA)

    def test_verse_detail_count_skips_non_hadith_nodes(self, tmp_path: Path) -> None:
        # 2 verse_detail per book are vendored; the verse_list TOC files must be skipped.
        hadiths, _ = _parse(tmp_path)
        assert len(hadiths) == 4

    def test_every_hadith_has_arabic(self, tmp_path: Path) -> None:
        # The anti-fixture-masking assertion (main#671): real Arabic, never empty.
        # matn_ar is the field the graph loader surfaces on the Hadith node, so it
        # MUST carry the Arabic (full_text_ar mirrors it for Phase-2).
        hadiths, _ = _parse(tmp_path)
        for row in hadiths:
            for col in ("matn_ar", "full_text_ar"):
                assert row[col], f"empty {col} for {row['source_id']}"
                assert _ARABIC.search(row[col]), f"no Arabic in {col} for {row['source_id']}"

    def test_isnad_lead_in_surfaced_as_arabic(self, tmp_path: Path) -> None:
        # The clean narrator_chain lead-in is surfaced as isnad_raw_ar (present for
        # ~98% of verses); when present it must be genuine Arabic.
        hadiths, _ = _parse(tmp_path)
        with_isnad = [r for r in hadiths if r["isnad_raw_ar"]]
        assert with_isnad, "no isnad_raw_ar populated at all"
        for row in with_isnad:
            assert _ARABIC.search(row["isnad_raw_ar"]), f"no Arabic isnad for {row['source_id']}"

    def test_english_and_grade_are_null(self, tmp_path: Path) -> None:
        # AI translations and (absent) grading are deliberately not loaded.
        hadiths, _ = _parse(tmp_path)
        for row in hadiths:
            for col in ("matn_en", "isnad_raw_en", "full_text_en", "grade"):
                assert row[col] is None, f"{col} unexpectedly populated for {row['source_id']}"
            assert row["sect"] == "shia"
            assert row["source_corpus"] == "thaqalayn_data"

    def test_source_ids_valid_and_not_double_prefixed(self, tmp_path: Path) -> None:
        hadiths, _ = _parse(tmp_path)
        for row in hadiths:
            sid = row["source_id"]
            validate_source_id(sid)  # raises on a malformed id
            assert not is_double_prefixed(sid), sid
            # grammar: thaqalayn_data:<slug>:<part>:<chapter>:<hadith>
            segments = sid.split(":")
            assert segments[0] == "thaqalayn_data"
            assert segments[1] in _BOOK_SLUGS
            assert len(segments) == 5

    def test_positional_fields_parsed_from_path(self, tmp_path: Path) -> None:
        hadiths, _ = _parse(tmp_path)
        istibsar = sorted(
            (r for r in hadiths if r["collection_name"] == "al-istibsar"),
            key=lambda r: r["hadith_number"],
        )
        assert [r["hadith_number"] for r in istibsar] == [1, 2]
        assert all(r["book_number"] == 1 and r["chapter_number"] == 1 for r in istibsar)
        assert all(r["chapter_name_ar"] for r in istibsar)

    def test_collections_one_row_per_book(self, tmp_path: Path) -> None:
        _, collections = _parse(tmp_path)
        by_id = {c["collection_id"]: c for c in collections}
        assert set(by_id) == {"thaqalayn_data:al-istibsar", "thaqalayn_data:tahdhib-al-ahkam"}
        istibsar = by_id["thaqalayn_data:al-istibsar"]
        assert istibsar["name_en"] == "al-Istibsar"
        assert istibsar["name_ar"] == "الاستبصار"
        assert istibsar["compiler_name"] == "Muhammad ibn al-Hasan al-Tusi"
        assert istibsar["sect"] == "shia"
        assert istibsar["total_hadiths"] == 2

    def test_appears_in_linkage_invariant(self, tmp_path: Path) -> None:
        # APPEARS_IN keys a hadith to its collection via "{corpus}:{collection_name}"
        # (src/graph/load_edges.py); that MUST equal a real collection_id.
        hadiths, collections = _parse(tmp_path)
        collection_ids = {c["collection_id"] for c in collections}
        for row in hadiths:
            key = f"{row['source_corpus']}:{row['collection_name']}"
            assert key in collection_ids, f"{key} has no Collection node"
