"""Tests for the canonical corpus composition (da#191)."""

from __future__ import annotations

import pytest

from src.parse.composition import HADITH_COMPOSITION, is_canonical_hadith


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
