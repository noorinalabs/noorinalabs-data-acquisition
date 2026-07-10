"""Tests for src.resolve.dedup — hadith deduplication and parallel detection.

ML-dependent tests are marked with @pytest.mark.ml and skip gracefully when
sentence-transformers or faiss-cpu are not installed.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.models.enums import VariantType
from src.resolve.dedup import (
    _classify_pair,
    _is_cross_sect,
    _load_hadith_texts,
    _write_empty_output,
    run_dedup,
)
from src.resolve.schemas import PARALLEL_LINKS_SCHEMA
from tests.test_resolve.conftest import write_hadiths

# Check ML availability at module level for marker.
_ml_available = True
try:
    import faiss  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
except ImportError:
    _ml_available = False

ml = pytest.mark.skipif(not _ml_available, reason="ML deps not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_hadith(
    source_id: str,
    matn_en: str,
    source_corpus: str = "sunnah",
    sect: str = "sunni",
) -> dict:
    return {
        "source_id": source_id,
        "source_corpus": source_corpus,
        "collection_name": "test",
        "matn_en": matn_en,
        "sect": sect,
        "book_number": None,
        "chapter_number": None,
        "hadith_number": None,
        "matn_ar": None,
        "isnad_raw_ar": None,
        "isnad_raw_en": None,
        "full_text_ar": None,
        "full_text_en": None,
        "grade": None,
        "chapter_name_ar": None,
        "chapter_name_en": None,
    }


# ---------------------------------------------------------------------------
# Tests: _classify_pair
# ---------------------------------------------------------------------------
class TestClassifyPair:
    def test_verbatim(self) -> None:
        assert _classify_pair(0.95) == VariantType.VERBATIM

    def test_verbatim_boundary(self) -> None:
        assert _classify_pair(0.90) == VariantType.VERBATIM

    def test_close_paraphrase(self) -> None:
        assert _classify_pair(0.85) == VariantType.CLOSE_PARAPHRASE

    def test_close_paraphrase_boundary(self) -> None:
        assert _classify_pair(0.80) == VariantType.CLOSE_PARAPHRASE

    def test_thematic(self) -> None:
        assert _classify_pair(0.75) == VariantType.THEMATIC

    def test_thematic_low(self) -> None:
        assert _classify_pair(0.50) == VariantType.THEMATIC


# ---------------------------------------------------------------------------
# Tests: _is_cross_sect
# ---------------------------------------------------------------------------
class TestIsCrossSect:
    # da#321: cross-sect is derived from the authoritative `sect` column, not the
    # source-corpus name — matching parallels.py so the composed link is stable.
    def test_sunni_shia_is_cross(self) -> None:
        assert _is_cross_sect("sunni", "shia") is True

    def test_shia_sunni_is_cross(self) -> None:
        assert _is_cross_sect("shia", "sunni") is True

    def test_same_sect_not_cross(self) -> None:
        assert _is_cross_sect("sunni", "sunni") is False

    def test_missing_sect_not_cross(self) -> None:
        # A null/empty sect on either side is not enough to call it cross-sect.
        assert _is_cross_sect(None, "sunni") is False
        assert _is_cross_sect("sunni", "") is False


# ---------------------------------------------------------------------------
# Tests: _load_hadith_texts
# ---------------------------------------------------------------------------
class TestLoadHadithTexts:
    def test_loads_valid_texts(self, tmp_path: Path) -> None:
        rows = [
            _make_hadith("h-1", "Actions are judged by intentions"),
            _make_hadith("h-2", "The best of you is the one who learns the Quran"),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)
        ids, texts, sects = _load_hadith_texts(tmp_path)
        assert len(ids) == 2
        assert len(texts) == 2
        # Third element is now the authoritative `sect` column (da#321).
        assert all(s == "sunni" for s in sects)

    def test_skips_null_matn(self, tmp_path: Path) -> None:
        rows = [
            _make_hadith("h-1", "Valid text"),
            _make_hadith("h-2", None),  # type: ignore[arg-type]
            _make_hadith("h-3", "   "),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)
        ids, texts, sects = _load_hadith_texts(tmp_path)
        assert len(ids) == 1
        assert ids[0] == "h-1"

    def test_no_files_returns_empty(self, tmp_path: Path) -> None:
        ids, texts, sects = _load_hadith_texts(tmp_path)
        assert ids == []


# ---------------------------------------------------------------------------
# Tests: _write_empty_output
# ---------------------------------------------------------------------------
class TestWriteEmptyOutput:
    def test_creates_empty_parquet(self, tmp_path: Path) -> None:
        path = _write_empty_output(tmp_path)
        assert path.exists()
        table = pq.read_table(path)
        assert table.num_rows == 0
        assert table.schema.equals(PARALLEL_LINKS_SCHEMA)


# ---------------------------------------------------------------------------
# Tests: canonical pair ordering (integration)
# ---------------------------------------------------------------------------
@ml
class TestCanonicalPairOrdering:
    def test_output_pairs_are_canonically_ordered(self, tmp_path: Path) -> None:
        """Verify that run_dedup produces output with hadith_id_a < hadith_id_b
        and that the output conforms to PARALLEL_LINKS_SCHEMA.

        Uses intentionally reversed IDs (z-id before a-id) so the ordering logic
        in run_dedup is exercised.

        This test previously wrote ``hadith_test.parquet`` (singular), which the
        stage's ``**/hadiths_*.parquet`` glob never matched: the run found zero
        input files, returned an empty table, and the ordering assertion below —
        guarded by ``if table.num_rows > 0`` — never executed. It asserted nothing.
        Fixed to the plural filename and marked ``@ml``, because real pairs require
        the embedder (da#309).
        """
        rows = [
            _make_hadith("z-id", "Actions are judged by intentions"),
            _make_hadith("a-id", "Actions are judged by intentions"),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)

        output_path = run_dedup(tmp_path, threshold=0.70)
        assert output_path.exists()

        table = pq.read_table(output_path)
        assert table.schema.equals(PARALLEL_LINKS_SCHEMA)

        # Identical matn -> the pair must be found, so the ordering below is reached.
        assert table.num_rows >= 1
        ids_a = table.column("hadith_id_a").to_pylist()
        ids_b = table.column("hadith_id_b").to_pylist()
        for a, b in zip(ids_a, ids_b):
            assert a < b, f"Expected {a!r} < {b!r} (canonical ordering)"


# ---------------------------------------------------------------------------
# Tests: ML-dependent (embedding + FAISS)
# ---------------------------------------------------------------------------
@ml
class TestEmbeddingPipeline:
    def test_run_dedup_with_tiny_sample(self, tmp_path: Path) -> None:
        rows = [
            _make_hadith(
                "h-1",
                "Actions are judged by intentions and every person will get what they intended",
            ),
            _make_hadith(
                "h-2",
                "Actions are judged by intentions and every man shall have what he intended",
            ),
            _make_hadith("h-3", "The best of you is the one who learns the Quran and teaches it"),
            _make_hadith(
                "h-4",
                "Whoever believes in Allah and the Last Day should speak good or keep silent",
            ),
            _make_hadith(
                "h-5",
                "Actions are judged according to intentions",
                source_corpus="thaqalayn",
                sect="shia",
            ),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)

        output_path = run_dedup(tmp_path, threshold=0.70, top_k=5)
        assert output_path.exists()

        table = pq.read_table(output_path)
        assert table.schema.equals(PARALLEL_LINKS_SCHEMA)
        # With near-duplicate texts, we should get at least one pair.
        assert table.num_rows >= 1

    def test_faiss_index_created(self, tmp_path: Path) -> None:
        rows = [
            _make_hadith("h-1", "Test text one"),
            _make_hadith("h-2", "Test text two"),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)

        run_dedup(tmp_path, threshold=0.70)
        assert (tmp_path / "hadith_embeddings.faiss").exists()
        assert (tmp_path / "hadith_embeddings.npy").exists()
        assert (tmp_path / "hadith_id_mapping.json").exists()

    def test_cross_sect_flagging(self, tmp_path: Path) -> None:
        rows = [
            _make_hadith("h-1", "Actions are judged by intentions", source_corpus="sunnah"),
            _make_hadith(
                "h-2",
                "Actions are judged by intentions",
                source_corpus="thaqalayn",
                sect="shia",
            ),
        ]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)

        output_path = run_dedup(tmp_path, threshold=0.70)
        table = pq.read_table(output_path)
        # Identical matn across corpora: the pair must exist, so the cross-sect
        # assertion below is reached rather than vacuously skipped (da#309).
        assert table.num_rows >= 1
        cross_flags = table.column("cross_sect").to_pylist()
        assert any(cross_flags), "Expected at least one cross-sect pair"


# ---------------------------------------------------------------------------
# Tests: empty inputs
#
# There is no longer a "graceful fallback when ML libs are missing": an enabled
# stage whose declared dep is absent raises (da#309, see test_dedup_fail_loud.py).
# What remains here are the genuinely-empty-input paths, kept distinct because
# they are NOT the same condition:
#   Case A -- no hadith_*.parquet files at all (upstream defect, see below)
#   Case B -- files present, but no row carries a non-empty English matn
# Both are @ml: without the embedder the stage exits at the dependency guard
# before it ever reads its input.
# ---------------------------------------------------------------------------
@ml
class TestEmptyInput:
    def test_empty_output_on_no_hadith_files(self, tmp_path: Path) -> None:
        """Case A. Still non-fatal, but no longer conflated with Case B.

        Zero input files means the upstream parse stage produced nothing — an
        upstream defect rather than a true negative. Making it fatal is a
        behaviour change tracked as a follow-up, not folded into da#309.
        """
        path = run_dedup(tmp_path)
        assert path.exists()
        assert pq.read_table(path).num_rows == 0

    def test_empty_output_when_no_row_has_english_matn(self, tmp_path: Path) -> None:
        """Case B: a legitimate empty input for an English-matn semantic detector."""
        rows = [_make_hadith("h-1", ""), _make_hadith("h-2", "")]
        write_hadiths(tmp_path / "hadiths_test.parquet", rows)

        path = run_dedup(tmp_path)
        assert path.exists()
        assert pq.read_table(path).num_rows == 0
