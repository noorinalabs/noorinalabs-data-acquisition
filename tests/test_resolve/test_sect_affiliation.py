"""Tests for narrator-level sect derivation (da#103)."""

from __future__ import annotations

import pytest

from src.resolve.sect_affiliation import (
    derive_sect_affiliation,
    normalize_corpus,
    primary_corpus,
)


class TestNormalizeCorpus:
    def test_passthrough_valid(self) -> None:
        assert normalize_corpus("sunnah") == "sunnah"
        assert normalize_corpus("thaqalayn") == "thaqalayn"

    def test_alias_mapped(self) -> None:
        assert normalize_corpus("kaggle_narrators") == "sanadset"

    def test_unknown_and_empty_to_none(self) -> None:
        assert normalize_corpus("not_a_corpus") is None
        assert normalize_corpus("") is None
        assert normalize_corpus(None) is None


class TestDeriveSectAffiliation:
    @pytest.mark.parametrize(
        "corpora,expected",
        [
            (["sunnah"], "sunni"),
            (["sunnah", "open_hadith"], "sunni"),
            (["thaqalayn"], "shia"),
            (["sunnah", "thaqalayn"], "neutral"),  # transmits in both traditions
            (["muhaddithat", "thaqalayn"], "neutral"),
            (["kaggle_narrators"], "sunni"),  # sanadset bio provenance alias
            ([], "unknown"),
            (["itqan"], "unknown"),  # mixed rijal DB — intentionally unmapped
            (["fawaz"], "unknown"),  # per-collection sect, not per-corpus
            (["itqan", "thaqalayn"], "shia"),  # unmapped corpus drops out
        ],
    )
    def test_derivation(self, corpora: list[str], expected: str) -> None:
        assert derive_sect_affiliation(corpora) == expected

    def test_ignores_empty_strings(self) -> None:
        assert derive_sect_affiliation(["", "sunnah"]) == "sunni"


class TestPrimaryCorpus:
    def test_lexicographically_first(self) -> None:
        assert primary_corpus(["thaqalayn", "sunnah"]) == "sunnah"

    def test_dedups(self) -> None:
        assert primary_corpus(["sunnah", "sunnah"]) == "sunnah"

    def test_none_when_empty(self) -> None:
        assert primary_corpus([]) is None
        assert primary_corpus(["", None]) is None  # type: ignore[list-item]
