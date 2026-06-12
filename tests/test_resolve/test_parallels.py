"""Unit tests for the deterministic lexical PARALLEL_OF detector (da#100)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.models.enums import VariantType
from src.resolve.parallels import (
    _classify_pair,
    _jaccard,
    _tokenize,
    detect_parallels,
)
from tests.test_graph.conftest import write_hadiths


class TestClassifyPair:
    def test_verbatim(self) -> None:
        assert _classify_pair(0.95) == VariantType.VERBATIM
        assert _classify_pair(0.90) == VariantType.VERBATIM

    def test_close_paraphrase(self) -> None:
        assert _classify_pair(0.85) == VariantType.CLOSE_PARAPHRASE
        assert _classify_pair(0.80) == VariantType.CLOSE_PARAPHRASE

    def test_thematic(self) -> None:
        assert _classify_pair(0.79) == VariantType.THEMATIC
        assert _classify_pair(0.50) == VariantType.THEMATIC


class TestJaccard:
    def test_identical(self) -> None:
        s = frozenset({"a", "b", "c"})
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_partial(self) -> None:
        # {a,b,c} vs {b,c,d}: intersection 2, union 4 -> 0.5
        assert _jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == 0.5

    def test_empty(self) -> None:
        assert _jaccard(frozenset(), frozenset({"a"})) == 0.0


class TestTokenize:
    def test_arabic_preferred_and_normalized(self) -> None:
        # Diacritics differ but normalize_arabic collapses them -> same tokens.
        a = _tokenize("إنَّمَا الأعمال بالنيات", None)
        b = _tokenize("إنما الاعمال بالنيات", None)
        assert a == b
        assert a  # non-empty

    def test_english_fallback_when_no_arabic(self) -> None:
        assert _tokenize(None, "Actions Are By Intentions") == frozenset(
            {"actions", "are", "by", "intentions"}
        )

    def test_empty_when_neither(self) -> None:
        assert _tokenize(None, None) == frozenset()
        assert _tokenize("", "   ") == frozenset()


def _coords(**over: object) -> dict[str, object]:
    base: dict[str, object] = {"collection_name": "bukhari", "source_corpus": "sunnah"}
    base.update(over)
    return base


class TestDetectParallels:
    def test_writes_empty_when_no_pairs(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {"source_id": "sunnah:bukhari:1", "matn_en": "the sky is blue", **_coords()},
                {"source_id": "sunnah:bukhari:2", "matn_en": "fish swim in water", **_coords()},
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        assert out.exists()
        assert pq.read_table(out).num_rows == 0

    def test_intra_sect_pair_detected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "actions are judged by their intentions",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "lk:bukhari:1",
                    "matn_en": "actions are judged by the intentions",
                    "sect": "sunni",
                    "source_corpus": "lk",
                    "collection_name": "bukhari",
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        assert len(rows) == 1
        assert rows[0]["cross_sect"] is False
        # Canonical ordering: lk: < sunnah:
        assert rows[0]["hadith_id_a"] == "lk:bukhari:1"
        assert rows[0]["hadith_id_b"] == "sunnah:bukhari:1"

    def test_cross_sect_pair_flagged(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "purification is half of faith",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "thaqalayn:al-kafi:1",
                    "matn_en": "purification is half of the faith",
                    "sect": "shia",
                    "source_corpus": "thaqalayn",
                    "collection_name": "al-kafi",
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        assert len(rows) == 1
        assert rows[0]["cross_sect"] is True

    def test_symmetric_dedup_single_row(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "a b c d e",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "sunnah:bukhari:2",
                    "matn_en": "a b c d e",
                    "sect": "sunni",
                    **_coords(),
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        # One identical pair -> exactly one (canonically ordered) row, score 1.0.
        assert len(rows) == 1
        assert rows[0]["similarity_score"] == pytest.approx(1.0)
        assert rows[0]["variant_type"] == VariantType.VERBATIM.value

    def test_threshold_filters_low_similarity(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "alpha beta gamma delta",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "sunnah:bukhari:2",
                    "matn_en": "alpha epsilon zeta eta",
                    "sect": "sunni",
                    **_coords(),
                },
            ],
            suffix="a",
        )
        # Jaccard here = 1/7 ~= 0.14 -> below 0.5, no rows.
        assert pq.read_table(detect_parallels(staging, threshold=0.5)).num_rows == 0
        # Lower threshold catches the thematic link.
        rows = pq.read_table(detect_parallels(staging, threshold=0.1)).to_pylist()
        assert len(rows) == 1
        assert rows[0]["variant_type"] == VariantType.THEMATIC.value
