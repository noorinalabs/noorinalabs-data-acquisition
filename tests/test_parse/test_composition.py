"""Tests for the canonical corpus composition (da#191, da#220)."""

from __future__ import annotations

import pytest

from src.parse.composition import (
    CROSS_EDITION_DEDUP_SOURCES,
    HADITH_COMPOSITION,
    canonical_matn_identity,
    is_canonical_hadith,
    is_cross_edition_dedup_source,
)


class TestIsCanonicalHadith:
    def test_unlisted_source_loads_all_collections(self) -> None:
        # lk is the six-books spine — not in the composition map, so all kept.
        assert is_canonical_hadith("lk", "bukhari")
        assert is_canonical_hadith("lk", "muslim")
        # sanadset / thaqalayn / tusi / sunnah likewise load everything.
        assert is_canonical_hadith("sanadset", "sanadset")
        assert is_canonical_hadith("thaqalayn", "Al-Kafi-Volume-1-Kulayni")

    def test_halimbahae_keeps_only_unique_books(self) -> None:
        for keep in ("musnad_ahmad_ibn-hanbal", "sunan_al-darimi", "maliks_muwataa"):
            assert is_canonical_hadith("halimbahae", keep)
        for drop in (
            "sahih_al-bukhari",
            "sahih_muslim",
            "sunan_al-nasai",
            "sunan_abu-dawud",
            "sunan_ibn-maja",
            "sunan_al-tirmidhi",
        ):
            assert not is_canonical_hadith("halimbahae", drop)

    def test_fawaz_keeps_only_unique_collections(self) -> None:
        for keep in ("nawawi", "dehlawi", "qudsi"):
            assert is_canonical_hadith("fawaz", keep)
        for drop in ("bukhari", "muslim", "nasai", "abudawud", "ibnmajah", "tirmidhi", "malik"):
            assert not is_canonical_hadith("fawaz", drop)

    def test_mis_loads_no_hadith_nodes(self) -> None:
        # mis contributes chains/edges only — its Sahih Muslim matn duplicates lk.
        assert not is_canonical_hadith("mis", "Sahih Muslim")

    def test_open_hadith_dropped_entirely(self) -> None:
        # Defence-in-depth: dropped at the registry (active=False), and any leftover
        # parquet is a no-op here.
        assert not is_canonical_hadith("open_hadith", "sahih_al-bukhari")
        assert HADITH_COMPOSITION["open_hadith"] == frozenset()

    def test_explicit_none_value_loads_all_collections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # da#196: an explicit ``None`` value means "load all collections" — the same
        # as an absent key — matching the documented map semantics. Guards against a
        # regression to the old drop-all behaviour (``allowed is None`` returning
        # ``source_corpus not in HADITH_COMPOSITION``, i.e. False, when the key was
        # present with value None).
        monkeypatch.setitem(HADITH_COMPOSITION, "explicit_none_source", None)
        assert is_canonical_hadith("explicit_none_source", "any_collection")
        assert is_canonical_hadith("explicit_none_source", "another_collection")


class TestCrossEditionDedupSource:
    """The per-hadith dedup-source membership gate (da#220 / Path B-B2)."""

    def test_sanadset_is_a_dedup_source(self) -> None:
        assert is_cross_edition_dedup_source("sanadset")
        assert "sanadset" in CROSS_EDITION_DEDUP_SOURCES

    def test_curated_sources_are_not_dedup_sources(self) -> None:
        # The curated spine is what sanadset is deduped *against* — never itself
        # a dedup source, else it would dedup against nothing / itself.
        for curated in ("lk", "halimbahae", "fawaz", "thaqalayn", "tusi", "sunnah"):
            assert not is_cross_edition_dedup_source(curated)

    def test_sanadset_per_source_gate_still_admits_all_collections(self) -> None:
        # da#220 does NOT tighten the per-source map — sanadset keeps its full
        # collection breadth; dedup is per-hadith, not per-collection.
        assert is_canonical_hadith("sanadset", "1")
        assert is_canonical_hadith("sanadset", "any-book-id")


class TestCanonicalMatnIdentity:
    """Normalized-matn cross-edition identity (da#220 / Path B-B2)."""

    # Three-token Arabic matn (meets _MIN_IDENTITY_TOKENS); the second copy adds
    # diacritics + tatweel that normalize_arabic folds away.
    _MATN_PLAIN = "انما الاعمال بالنيات"
    _MATN_DIACRITIZED = "إِنَّمــا الأعمالُ بِالنِّيَّاتِ"

    def test_same_matn_yields_same_identity(self) -> None:
        a = canonical_matn_identity(self._MATN_PLAIN, None)
        b = canonical_matn_identity(self._MATN_PLAIN, None)
        assert a is not None
        assert a == b

    def test_normalization_collapses_spelling_variants(self) -> None:
        # Two editions of the same tradition differing only in diacritics / alif /
        # hamza / tatweel must converge to ONE identity — that is what lets a
        # sanadset copy dedup against the curated edition.
        assert canonical_matn_identity(self._MATN_PLAIN, None) == canonical_matn_identity(
            self._MATN_DIACRITIZED, None
        )

    def test_distinct_matn_yields_distinct_identity(self) -> None:
        other = canonical_matn_identity("لا ضرر ولا ضرار", None)
        assert other is not None
        assert other != canonical_matn_identity(self._MATN_PLAIN, None)

    def test_empty_matn_has_no_identity(self) -> None:
        assert canonical_matn_identity(None, None) is None
        assert canonical_matn_identity("", "") is None
        assert canonical_matn_identity("   ", None) is None

    def test_too_short_matn_has_no_identity(self) -> None:
        # Below _MIN_IDENTITY_TOKENS a matn is too generic to identify a
        # tradition, so it is never deduped (None == keep).
        assert canonical_matn_identity("الاعمال", None) is None
        assert canonical_matn_identity("انما الاعمال", None) is None

    def test_english_matn_fallback_when_no_arabic(self) -> None:
        # No Arabic present -> lowercased English matn is the identity basis, and
        # case differences fold together.
        lower = canonical_matn_identity(None, "actions are by intentions")
        upper = canonical_matn_identity(None, "Actions Are By Intentions")
        assert lower is not None
        assert lower == upper

    def test_arabic_preferred_over_english(self) -> None:
        # When both are present the Arabic drives identity, so the same Arabic
        # with differing English still collides (editions share matn, not gloss).
        with_en_a = canonical_matn_identity(self._MATN_PLAIN, "gloss one")
        with_en_b = canonical_matn_identity(self._MATN_PLAIN, "a different gloss")
        assert with_en_a == with_en_b
