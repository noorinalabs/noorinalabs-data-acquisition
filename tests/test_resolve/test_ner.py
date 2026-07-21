"""Tests for src.resolve.ner — narrator NER extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.narrator_extraction import IsnadSegmentationError, extract_narrator_mentions
from src.parse.schemas import NARRATOR_MENTION_SCHEMA
from src.resolve.ner import (
    UnroutedCorpusError,
    _CleanRate,
    _extract_from_hadiths,
    _load_phase1_mentions,
    run,
)
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA
from tests.test_resolve.conftest import write_hadiths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_narrator_mentions_parquet(path: Path, rows: list[dict]) -> Path:
    """Write a narrator_mentions Parquet with NARRATOR_MENTION_SCHEMA."""
    arrays = {
        "mention_id": pa.array([r["mention_id"] for r in rows], type=pa.string()),
        "source_hadith_id": pa.array([r["source_hadith_id"] for r in rows], type=pa.string()),
        "source_corpus": pa.array([r["source_corpus"] for r in rows], type=pa.string()),
        "position_in_chain": pa.array([r["position_in_chain"] for r in rows], type=pa.int32()),
        "chain_index": pa.array([r.get("chain_index", 0) for r in rows], type=pa.int32()),
        "name_ar": pa.array([r.get("name_ar") for r in rows], type=pa.string()),
        "name_en": pa.array([r.get("name_en") for r in rows], type=pa.string()),
        "name_ar_normalized": pa.array(
            [r.get("name_ar_normalized") for r in rows], type=pa.string()
        ),
        "transmission_method": pa.array(
            [r.get("transmission_method") for r in rows], type=pa.string()
        ),
    }
    table = pa.table(arrays, schema=NARRATOR_MENTION_SCHEMA)
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# Tests: Phase 1 mention loading
# ---------------------------------------------------------------------------
class TestLoadPhase1Mentions:
    def test_loads_existing_mentions(self, tmp_path: Path) -> None:
        mentions = [
            {
                "mention_id": "m-1",
                "source_hadith_id": "h-1",
                "source_corpus": "sanadset",
                "position_in_chain": 0,
                "name_ar": "\u0623\u0628\u0648 \u0647\u0631\u064a\u0631\u0629",
                "name_en": "Abu Hurayra",
                "name_ar_normalized": "\u0627\u0628\u0648 \u0647\u0631\u064a\u0631\u0647",
                "transmission_method": "haddathana",
            },
        ]
        _write_narrator_mentions_parquet(tmp_path / "narrator_mentions_sanadset.parquet", mentions)
        rows = _load_phase1_mentions(tmp_path, "sanadset", "narrator_mentions_sanadset.parquet")
        assert len(rows) == 1
        assert rows[0]["source_corpus"] == "sanadset"
        assert rows[0]["name_raw"] is not None
        assert rows[0]["mention_id"] is not None

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        rows = _load_phase1_mentions(tmp_path, "sanadset", "missing.parquet")
        assert rows == []

    def test_prefers_arabic_name(self, tmp_path: Path) -> None:
        mentions = [
            {
                "mention_id": "m-2",
                "source_hadith_id": "h-2",
                "source_corpus": "lk",
                "position_in_chain": 1,
                "name_ar": "\u0639\u0644\u064a",
                "name_en": "Ali",
                "name_ar_normalized": "\u0639\u0644\u064a",
                "transmission_method": None,
            },
        ]
        _write_narrator_mentions_parquet(tmp_path / "narrator_mentions_lk.parquet", mentions)
        rows = _load_phase1_mentions(tmp_path, "lk", "narrator_mentions_lk.parquet")
        assert rows[0]["name_raw"] == "\u0639\u0644\u064a"

    def test_colon_join_display_and_key_both_cleaned(self, tmp_path: Path) -> None:
        """da#253 coupling: an English-only mention whose name is a "<name>:<matn>"
        colon-join (the reported prod node nar:00063b2c… shape) must have BOTH its
        clustering key (name_normalized, via clean_narrator_name) AND its DISPLAY
        field (name_raw, via strip_markup) cleaned. Exercises _load_phase1_mentions
        end-to-end — the stage coupling clean_narrator_name in isolation can't cover.
        Pre-fix, strip_markup left name_raw = the full matn (display still polluted)."""
        polluted = "Thawban:The Messenger of Allah (\ufdfa) sacrificed during a journey and then"
        mentions = [
            {
                "mention_id": "m-253",
                "source_hadith_id": "h-253",
                "source_corpus": "sunnah",
                "position_in_chain": 0,
                "name_ar": None,
                "name_en": polluted,
                "name_ar_normalized": None,
                "transmission_method": None,
            },
        ]
        _write_narrator_mentions_parquet(tmp_path / "narrator_mentions_sunnah.parquet", mentions)
        rows = _load_phase1_mentions(tmp_path, "sunnah", "narrator_mentions_sunnah.parquet")
        assert len(rows) == 1
        assert rows[0]["name_raw"] == "Thawban"  # DISPLAY field cleaned (the CR gap)
        assert rows[0]["name_normalized"] == "Thawban"  # clustering key cleaned

    def test_falls_back_to_english_name(self, tmp_path: Path) -> None:
        mentions = [
            {
                "mention_id": "m-3",
                "source_hadith_id": "h-3",
                "source_corpus": "lk",
                "position_in_chain": 0,
                "name_ar": None,
                "name_en": "Malik",
                "name_ar_normalized": None,
                "transmission_method": None,
            },
        ]
        _write_narrator_mentions_parquet(tmp_path / "narrator_mentions_lk.parquet", mentions)
        rows = _load_phase1_mentions(tmp_path, "lk", "narrator_mentions_lk.parquet")
        assert rows[0]["name_raw"] == "Malik"
        assert rows[0]["name_normalized"] == "Malik"


# ---------------------------------------------------------------------------
# Tests: Arabic extraction
# ---------------------------------------------------------------------------
class TestArabicExtraction:
    def test_extracts_from_arabic_isnad(self, tmp_path: Path) -> None:
        hadiths = [
            {
                "source_id": "th-1",
                "source_corpus": "thaqalayn",
                "collection_name": "al-kafi",
                "isnad_raw_ar": (
                    "\u062d\u062f\u062b\u0646\u0627 \u0645\u062d\u0645\u062f"
                    " \u0639\u0646 \u0639\u0644\u064a"
                ),
                "isnad_raw_en": None,
                "full_text_ar": None,
                "full_text_en": None,
                "matn_ar": "text",
                "matn_en": None,
                "grade": None,
                "sect": "shia",
                "book_number": 1,
                "chapter_number": 1,
                "hadith_number": 1,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(tmp_path / "hadiths_thaqalayn.parquet", hadiths)
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar")
        assert len(rows) > 0
        assert all(r["source_corpus"] == "thaqalayn" for r in rows)

    def test_does_not_mine_full_text_when_isnad_null(self, tmp_path: Path) -> None:
        """da#369 (red-first): a row whose ``isnad_raw_ar`` is null but whose
        ``full_text_ar`` is populated must yield NO mentions. The old code fell
        back to mining ``full_text_ar`` (isnad+matn) \u2014 the path that produced
        380,771 matn-derived pseudo-narrator mentions (11.7%) and 38,262 narrators
        that existed ONLY in matn text. That fallback is removed; NER reads only
        the dedicated isnad column. Pre-fix this returned mentions; post-fix [].
        """
        hadiths = [
            {
                "source_id": "th-2",
                "source_corpus": "thaqalayn",
                "collection_name": "al-kafi",
                "isnad_raw_ar": None,
                "isnad_raw_en": None,
                # A voweled isnad+matn blob: pre-da#369 the segmenter mined this.
                "full_text_ar": "حدثنا أنس عن النبي",
                "full_text_en": None,
                "matn_ar": None,
                "matn_en": None,
                "grade": None,
                "sect": "shia",
                "book_number": 1,
                "chapter_number": 1,
                "hadith_number": 1,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(tmp_path / "hadiths_thaqalayn.parquet", hadiths)
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar")
        assert rows == []

    def test_unsegmentable_hadith_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """da#244: a single hadith whose isnad cannot be segmented (e.g. matn in
        the isnad field — thaqalayn :807) must be SKIPPED, not abort the whole NER
        pass. Previously one IsnadSegmentationError propagated out of
        ``_extract_from_hadiths`` and cascaded to skip disambiguate, stranding the
        entire resolve. The good hadith's mentions must still be extracted."""

        def _base(source_id: str, isnad: str) -> dict:
            return {
                "source_id": source_id,
                "source_corpus": "thaqalayn",
                "collection_name": "al-kafi",
                "isnad_raw_ar": isnad,
                "isnad_raw_en": None,
                "full_text_ar": None,
                "full_text_en": None,
                "matn_ar": "text",
                "matn_en": None,
                "grade": None,
                "sect": "shia",
                "book_number": 1,
                "chapter_number": 1,
                "hadith_number": 1,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            }

        good = "حدثنا محمد عن علي"
        bad = "UNSEGMENTABLE_MATN_BLOB"
        write_hadiths(
            tmp_path / "hadiths_thaqalayn.parquet",
            [_base("th-good", good), _base("th-bad", bad)],
        )

        real = extract_narrator_mentions

        def _maybe_raise(isnad_text: str, language: str):  # type: ignore[no-untyped-def]
            if isnad_text == bad:
                raise IsnadSegmentationError("partial chain blob (simulated :807)")
            return real(isnad_text, language)

        monkeypatch.setattr("src.resolve.ner.extract_narrator_mentions", _maybe_raise)

        # Must NOT raise — the bad hadith is skipped, the good one still extracted.
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar")
        assert len(rows) > 0
        assert all(r["hadith_id"] == "th-good" for r in rows)
        assert not any(r["hadith_id"] == "th-bad" for r in rows)


# ---------------------------------------------------------------------------
# Tests: English extraction
# ---------------------------------------------------------------------------
class TestEnglishExtraction:
    def test_extracts_from_english_isnad(self, tmp_path: Path) -> None:
        hadiths = [
            {
                "source_id": "fw-1",
                "source_corpus": "fawaz",
                "collection_name": "fawaz-collection",
                "isnad_raw_ar": None,
                "isnad_raw_en": "Narrated Abu Hurayra from the Prophet",
                "full_text_ar": None,
                "full_text_en": None,
                "matn_ar": None,
                "matn_en": "The Prophet said...",
                "grade": None,
                "sect": "sunni",
                "book_number": 1,
                "chapter_number": 1,
                "hadith_number": 1,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(tmp_path / "hadiths_fawaz.parquet", hadiths)
        rows = _extract_from_hadiths(tmp_path, "fawaz", "en")
        assert len(rows) > 0
        assert all(r["source_corpus"] == "fawaz" for r in rows)

    def test_no_hadith_files_returns_empty(self, tmp_path: Path) -> None:
        rows = _extract_from_hadiths(tmp_path, "fawaz", "en")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: Null/empty isnad handling
# ---------------------------------------------------------------------------
class TestNullHandling:
    def test_null_isnad_and_full_text_skips_row(self, tmp_path: Path) -> None:
        hadiths = [
            {
                "source_id": "fw-null",
                "source_corpus": "fawaz",
                "collection_name": "fawaz-collection",
                "isnad_raw_ar": None,
                "isnad_raw_en": None,
                "full_text_ar": None,
                "full_text_en": None,
                "matn_ar": None,
                "matn_en": None,
                "grade": None,
                "sect": "sunni",
                "book_number": None,
                "chapter_number": None,
                "hadith_number": None,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(tmp_path / "hadiths_fawaz.parquet", hadiths)
        rows = _extract_from_hadiths(tmp_path, "fawaz", "en")
        assert rows == []

    def test_empty_string_isnad_skips_row(self, tmp_path: Path) -> None:
        hadiths = [
            {
                "source_id": "fw-empty",
                "source_corpus": "fawaz",
                "collection_name": "fawaz-collection",
                "isnad_raw_ar": None,
                "isnad_raw_en": "",
                "full_text_ar": None,
                "full_text_en": "",
                "matn_ar": None,
                "matn_en": None,
                "grade": None,
                "sect": "sunni",
                "book_number": None,
                "chapter_number": None,
                "hadith_number": None,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(tmp_path / "hadiths_fawaz.parquet", hadiths)
        rows = _extract_from_hadiths(tmp_path, "fawaz", "en")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: Output schema conformance
# ---------------------------------------------------------------------------
class TestOutputSchema:
    def test_output_matches_resolved_schema(self, tmp_path: Path) -> None:
        """Full run should produce a Parquet matching NARRATOR_MENTIONS_RESOLVED_SCHEMA."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        # Create a minimal English source (sunnah — fawaz now routes to Arabic, da#271).
        hadiths = [
            {
                "source_id": "sn-schema",
                "source_corpus": "sunnah",
                "collection_name": "sunnah-collection",
                "isnad_raw_ar": None,
                "isnad_raw_en": "Narrated Abu Bakr from Umar",
                "full_text_ar": None,
                "full_text_en": None,
                "matn_ar": None,
                "matn_en": "Text",
                "grade": None,
                "sect": "sunni",
                "book_number": 1,
                "chapter_number": 1,
                "hadith_number": 1,
                "chapter_name_ar": None,
                "chapter_name_en": None,
            },
        ]
        write_hadiths(staging / "hadiths_sunnah.parquet", hadiths)

        paths = run(staging, output)
        assert len(paths) >= 1

        resolved_path = output / "narrator_mentions_resolved.parquet"
        assert resolved_path.exists()

        table = pq.read_table(resolved_path)
        assert table.schema.equals(NARRATOR_MENTIONS_RESOLVED_SCHEMA)

    def test_run_with_no_data_produces_no_output(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        paths = run(staging, output)
        assert paths == []


# ---------------------------------------------------------------------------
# Tests: per-source NER clean-rate metric (da#300)
# ---------------------------------------------------------------------------
class TestCleanRateDataclass:
    """da#300 — the _CleanRate accumulator: kept / dropped / considered / rate."""

    def test_rate_and_considered(self) -> None:
        cr = _CleanRate(kept=84, dropped=16)
        assert cr.considered == 100
        assert cr.clean_rate == pytest.approx(0.84)

    def test_all_kept_is_one(self) -> None:
        assert _CleanRate(kept=5, dropped=0).clean_rate == pytest.approx(1.0)

    def test_all_dropped_is_zero(self) -> None:
        assert _CleanRate(kept=0, dropped=5).clean_rate == pytest.approx(0.0)

    def test_nothing_considered_is_zero_not_division_error(self) -> None:
        cr = _CleanRate()
        assert cr.considered == 0
        assert cr.clean_rate == 0.0


class TestCleanRateMetric:
    """da#300 — the per-source clean-rate is recorded and persisted, never gated."""

    # An isnad that segments into one real name (kept) + one Prophet reference the
    # cleaner drops (dropped) → clean_rate 0.5 for that source.
    _MIXED_ISNAD = "حدثنا محمد بن يعقوب عن النبي صلى الله عليه وسلم"

    def _thaqalayn_hadith(self, source_id: str, isnad: str) -> dict:
        return {
            "source_id": source_id,
            "source_corpus": "thaqalayn",
            "collection_name": "al-kafi",
            "isnad_raw_ar": isnad,
            "isnad_raw_en": None,
            "full_text_ar": None,
            "full_text_en": None,
            "matn_ar": "text",
            "matn_en": None,
            "grade": None,
            "sect": "shia",
            "book_number": 1,
            "chapter_number": 1,
            "hadith_number": 1,
            "chapter_name_ar": None,
            "chapter_name_en": None,
        }

    def test_extractor_records_kept_and_dropped(self, tmp_path: Path) -> None:
        write_hadiths(
            tmp_path / "hadiths_thaqalayn.parquet",
            [self._thaqalayn_hadith("th-cr", self._MIXED_ISNAD)],
        )
        clean_stats: dict[str, _CleanRate] = {}
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar", clean_stats=clean_stats)

        assert len(rows) == 1  # only محمد بن يعقوب survived
        cr = clean_stats["thaqalayn"]
        assert (cr.kept, cr.dropped, cr.considered) == (1, 1, 2)
        assert cr.clean_rate == pytest.approx(0.5)

    def test_all_null_isnad_source_not_recorded(self, tmp_path: Path) -> None:
        """A source that considered zero spans (all null isnad) has no clean-rate —
        it must NOT create a misleading 0/0 entry (guards the no-output contract)."""
        write_hadiths(
            tmp_path / "hadiths_thaqalayn.parquet",
            [self._thaqalayn_hadith("th-null", None)],
        )
        clean_stats: dict[str, _CleanRate] = {}
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar", clean_stats=clean_stats)
        assert rows == []
        assert clean_stats == {}

    def test_omitting_clean_stats_leaves_behaviour_unchanged(self, tmp_path: Path) -> None:
        """The accumulator is opt-in — the legacy call signature still works."""
        write_hadiths(
            tmp_path / "hadiths_thaqalayn.parquet",
            [self._thaqalayn_hadith("th-cr2", self._MIXED_ISNAD)],
        )
        rows = _extract_from_hadiths(tmp_path, "thaqalayn", "ar")
        assert len(rows) == 1

    def test_run_writes_clean_rate_csv(self, tmp_path: Path) -> None:
        import csv

        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_thaqalayn.parquet",
            [self._thaqalayn_hadith("th-cr3", self._MIXED_ISNAD)],
        )

        paths = run(staging, output)

        csv_path = output / "ner_clean_rate.csv"
        assert csv_path.exists()
        assert csv_path in paths
        with open(csv_path, encoding="utf-8") as f:
            by_source = {row["source_corpus"]: row for row in csv.DictReader(f)}
        assert by_source["thaqalayn"]["kept"] == "1"
        assert by_source["thaqalayn"]["dropped"] == "1"
        assert by_source["thaqalayn"]["considered"] == "2"
        assert float(by_source["thaqalayn"]["clean_rate"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: fawaz Arabic extraction (da#271 cross-script fold)
# ---------------------------------------------------------------------------
class TestFawazArabicExtraction:
    """da#271 routed fawaz to Arabic extraction and relied on the NER full_text
    fallback to mine fawaz's voweled ``full_text_ar`` (isnad+matn) blob, since
    fawaz ships an empty isnad column. da#369 REMOVES that fallback (it was the
    dominant matn-mining pollution source), so fawaz's ``full_text_ar`` is no
    longer mined. fawaz stays *routed* to Arabic (routing is da#365's domain, not
    touched here) but now contributes 0 mentions until da#365 gives it a surgical
    per-corpus route that splits isnad from matn before extraction.
    """

    def test_fawaz_still_routed_to_arabic(self) -> None:
        # Routing is unchanged by da#369 — fawaz remains opted in to the Arabic
        # route (so it is not an unrouted-corpus hard-fail). da#365 owns routing.
        from src.resolve.ner import _ARABIC_SOURCES, _ENGLISH_SOURCES

        assert "fawaz" in _ARABIC_SOURCES
        assert "fawaz" not in _ENGLISH_SOURCES

    def test_fawaz_full_text_no_longer_mined(self, tmp_path: Path) -> None:
        """da#369: fawaz (empty isnad column, only a voweled ``full_text_ar``
        isnad+matn blob) now yields NO mentions — the full_text fallback da#271
        depended on is removed. da#365 will re-enable fawaz via a surgical route
        that isolates the isnad before extraction, rather than mining the blob.
        """
        isnad = "حَدَّثَنَا أَنَسُ بْنُ مَالِكٍ عَنْ جَابِرِ بْنِ عَبْدِ اللَّهِ قَالَ نَهَى النَّبِيُّ"
        write_hadiths(
            tmp_path / "hadiths_fawaz.parquet",
            [
                {
                    "source_id": "fw-1",
                    "source_corpus": "fawaz",
                    "collection_name": "fawaz-collection",
                    "isnad_raw_ar": None,
                    "isnad_raw_en": None,
                    "full_text_ar": isnad,
                    "full_text_en": "Narrated Anas bin Malik: ...",
                    "matn_ar": None,
                    "matn_en": None,
                    "grade": None,
                    "sect": "sunni",
                    "book_number": 1,
                    "chapter_number": 1,
                    "hadith_number": 1,
                    "chapter_name_ar": None,
                    "chapter_name_en": None,
                },
            ],
        )
        rows = _extract_from_hadiths(tmp_path, "fawaz", "ar")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: unrouted-corpus hard fail (da#369)
# ---------------------------------------------------------------------------
class TestUnroutedCorpusHardFail:
    """da#369: a staged corpus with no NER extraction route must raise loudly
    (``UnroutedCorpusError``), replacing the removed matn fallback that would
    otherwise silently drop or mine it. da#365 layers the correct routing on top.
    """

    @staticmethod
    def _hadith(source_corpus: str, isnad: str | None) -> dict:
        return {
            "source_id": f"{source_corpus}-1",
            "source_corpus": source_corpus,
            "collection_name": f"{source_corpus}-collection",
            "isnad_raw_ar": isnad,
            "isnad_raw_en": None,
            "full_text_ar": None,
            "full_text_en": None,
            "matn_ar": None,
            "matn_en": None,
            "grade": None,
            "sect": "shia",
            "book_number": 1,
            "chapter_number": 1,
            "hadith_number": 1,
            "chapter_name_ar": None,
            "chapter_name_en": None,
        }

    def test_unrouted_staged_corpus_raises(self, tmp_path: Path) -> None:
        """RED-FIRST: a staged corpus in NO route bucket must raise rather than be
        silently dropped. Pre-da#369 run() only iterated the fixed route sets, so an
        unrouted hadiths file was ignored with no error (and, for a routed-but-isnad-
        null corpus, silently matn-mined). da#365 routed the four real offenders
        (tusi / halimbahae / bihar / mis), so this uses a synthetic corpus name that
        is in no bucket -- the contract must hold for any future unclassified corpus."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_uncharted.parquet",
            [self._hadith("uncharted_corpus", "حدثنا محمد عن علي")],
        )
        with pytest.raises(UnroutedCorpusError) as excinfo:
            run(staging, output)
        assert "uncharted_corpus" in excinfo.value.unrouted
        # The hard-fail fires before any mention output is written.
        assert not (output / "narrator_mentions_resolved.parquet").exists()

    def test_unrouted_error_is_base_exception(self) -> None:
        """It must escape ``run_all``'s per-stage ``except Exception`` — so, like
        StopAfterReached / MissingDependencyError, it subclasses BaseException and
        is NOT an Exception (which would be swallowed and reported as success)."""
        assert issubclass(UnroutedCorpusError, BaseException)
        assert not issubclass(UnroutedCorpusError, Exception)

    def test_routed_corpus_does_not_raise(self, tmp_path: Path) -> None:
        """A staged corpus that IS routed (thaqalayn → Arabic) must not raise."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_thaqalayn.parquet",
            [self._hadith("thaqalayn", "حدثنا محمد عن علي")],
        )
        # Must not raise; a routed corpus with a real isnad produces mentions.
        paths = run(staging, output)
        assert (output / "narrator_mentions_resolved.parquet") in paths


# ---------------------------------------------------------------------------
# Tests: per-corpus routing (da#365)
# ---------------------------------------------------------------------------
class TestPerCorpusRouting:
    """da#365: every staged corpus that produces a globbed NER input declares an
    explicit route, layered on top of da#369's unrouted-corpus hard-fail. The four
    corpora that were staged-but-routed-nowhere (tusi / halimbahae / bihar / mis)
    each get their correct destination: tusi -> Arabic extraction (real isnad
    column); halimbahae + bihar -> splitter-deferred (isnad-in-matn, held for
    da#366); mis -> skip (chains come from network_edges_mis.parquet, da#364)."""

    @staticmethod
    def _hadith(
        source_corpus: str,
        *,
        isnad_ar: str | None = None,
        full_text_ar: str | None = None,
    ) -> dict:
        return {
            "source_id": f"{source_corpus}-1",
            "source_corpus": source_corpus,
            "collection_name": f"{source_corpus}-collection",
            "isnad_raw_ar": isnad_ar,
            "isnad_raw_en": None,
            "full_text_ar": full_text_ar,
            "full_text_en": None,
            "matn_ar": None,
            "matn_en": None,
            "grade": None,
            "sect": "shia",
            "book_number": 1,
            "chapter_number": 1,
            "hadith_number": 1,
            "chapter_name_ar": None,
            "chapter_name_en": None,
        }

    # -- tusi: pure routing miss, real isnad column -> Arabic extraction ------
    def test_tusi_routed_to_arabic(self) -> None:
        from src.resolve.ner import _ARABIC_SOURCES

        assert "tusi" in _ARABIC_SOURCES

    def test_tusi_isnad_extracted_end_to_end(self, tmp_path: Path) -> None:
        """RED-FIRST: before da#365, tusi was in no route set, so run() raised
        UnroutedCorpusError and tusi's 17,089 real isnads yielded 0 mentions. Now a
        staged tusi hadith with a populated isnad_raw_ar extracts through run()."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_tusi.parquet",
            [self._hadith("tusi", isnad_ar="حدثنا محمد عن علي")],
        )
        paths = run(staging, output)
        resolved = output / "narrator_mentions_resolved.parquet"
        assert resolved in paths
        corpora = set(pq.read_table(resolved).column("source_corpus").to_pylist())
        assert "tusi" in corpora

    # -- halimbahae / bihar: isnad-in-matn, deferred pending da#366 -----------
    def test_halimbahae_and_bihar_deferred_not_force_routed(self) -> None:
        from src.resolve.ner import (
            _ARABIC_SOURCES,
            _ENGLISH_SOURCES,
            _SPLITTER_DEFERRED_SOURCES,
        )

        assert "halimbahae" in _SPLITTER_DEFERRED_SOURCES
        assert "bihar" in _SPLITTER_DEFERRED_SOURCES
        # NOT force-routed into extraction on their null-isnad rows.
        for corpus in ("halimbahae", "bihar"):
            assert corpus not in _ARABIC_SOURCES
            assert corpus not in _ENGLISH_SOURCES

    def test_deferred_corpus_not_matn_mined_and_does_not_raise(self, tmp_path: Path) -> None:
        """RED-FIRST: before da#365 a staged halimbahae hadiths file raised
        UnroutedCorpusError. A splitter-dependent corpus staged with a null isnad but
        an isnad-BEARING full_text_ar must (a) not raise and (b) yield 0 mentions --
        the full_text (matn) is NOT mined (better silent than polluted, #928). Held
        until the da#366 splitter lands."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_halimbahae.parquet",
            [self._hadith("halimbahae", isnad_ar=None, full_text_ar="حدثنا محمد عن علي")],
        )
        # Must not raise -- halimbahae is explicitly routed (deferred).
        paths = run(staging, output)
        resolved = output / "narrator_mentions_resolved.parquet"
        if resolved.exists():
            corpora = set(pq.read_table(resolved).column("source_corpus").to_pylist())
            assert "halimbahae" not in corpora
        else:
            assert paths == []

    # -- mis: skip; chains come from network_edges_mis.parquet (da#364) -------
    def test_mis_routed_to_skip(self) -> None:
        from src.resolve.ner import _SKIP_SOURCES

        assert "mis" in _SKIP_SOURCES

    def test_mis_staged_hadiths_yield_no_mentions_and_do_not_raise(self, tmp_path: Path) -> None:
        """mis's hadiths_mis.parquet carries a null isnad; its transmission chains
        are the separate network_edges_mis.parquet (da#364), not NER over hadith
        rows. Staged, it must not raise and must contribute 0 NER mentions."""
        staging = tmp_path / "staging"
        staging.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        write_hadiths(
            staging / "hadiths_mis.parquet",
            [self._hadith("mis", isnad_ar=None)],
        )
        paths = run(staging, output)
        assert paths == []

    # -- the whole staged offender set is routed -----------------------------
    def test_all_issue_staged_corpora_are_routed(self) -> None:
        from src.resolve.ner import _ROUTED_CORPORA

        for corpus in ("tusi", "halimbahae", "bihar", "mis"):
            assert corpus in _ROUTED_CORPORA
