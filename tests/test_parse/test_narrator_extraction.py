"""Tests for narrator mention extraction from isnad text."""

from __future__ import annotations

from src.parse.narrator_extraction import NarratorSpan, extract_narrator_mentions


class TestNarratorSpanDataclass:
    def test_fields(self) -> None:
        span = NarratorSpan(name="Abu Hurayra", position=0, transmission_method="haddathana")
        assert span.name == "Abu Hurayra"
        assert span.position == 0
        assert span.transmission_method == "haddathana"

    def test_defaults(self) -> None:
        span = NarratorSpan(name="Anas", position=1)
        assert span.transmission_method is None

    def test_frozen(self) -> None:
        import pytest

        span = NarratorSpan(name="Anas", position=0)
        with pytest.raises(AttributeError):
            span.name = "Other"  # type: ignore[misc]


class TestEnglishExtraction:
    def test_narrated_prefix(self) -> None:
        text = "Narrated Abu Hurayra"
        spans = extract_narrator_mentions(text, "en")
        names = [s.name for s in spans]
        assert "Abu Hurayra" in names

    def test_multi_narrator(self) -> None:
        text = "on the authority of Anas who heard from Malik"
        spans = extract_narrator_mentions(text, "en")
        names = [s.name for s in spans]
        assert "Anas" in names
        assert "Malik" in names

    def test_positions_sequential(self) -> None:
        text = "Narrated Abu Hurayra: from Anas"
        spans = extract_narrator_mentions(text, "en")
        positions = [s.position for s in spans]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)


class TestArabicExtraction:
    def test_transmission_phrase(self) -> None:
        text = "حدثنا محمد بن عبدالله عن علي بن أبي طالب"
        spans = extract_narrator_mentions(text, "ar")
        assert len(spans) >= 2

    def test_names_normalized(self) -> None:
        text = "حدثنا محمد"
        spans = extract_narrator_mentions(text, "ar")
        # At least one span should be extracted
        assert len(spans) >= 1


class TestDa146VoweledSegmentation:
    """Regression tests for da#146: fully-voweled (diacritized) Arabic isnads.

    Real lk-corpus chains are fully voweled (``حَدَّثَنَا ...``). Before da#146 the
    diacritic-free transmission patterns never matched, so the whole chain
    collapsed into a single un-segmented "narrator" mention. These tests pin the
    fixed behaviour: a voweled chain splits into one mention per narrator.
    """

    # Sahih al-Bukhari hadith 1 isnad — five named narrators in the chain.
    _BUKHARI_H1_ISNAD = (
        "حَدَّثَنَا الْحُمَيْدِيُّ عَبْدُاللَّهِ بْنُ الزُّبَيْرِ، قَالَ حَدَّثَنَا "
        "سُفْيَانُ، قَالَ حَدَّثَنَا يَحْيَى بْنُ سَعِيدٍ الأَنْصَارِيُّ، قَالَ "
        "أَخْبَرَنِي مُحَمَّدُ بْنُ إِبْرَاهِيمَ التَّيْمِيُّ، أَنَّهُ سَمِعَ "
        "عَلْقَمَةَ بْنَ وَقَّاصٍ اللَّيْثِيَّ"
    )

    def test_voweled_chain_does_not_collapse_to_blob(self) -> None:
        """The whole voweled chain must not become one giant mention."""
        spans = extract_narrator_mentions(self._BUKHARI_H1_ISNAD, "ar")
        assert len(spans) >= 4
        # No single span should swallow the entire chain.
        assert all(len(s.name) < len(self._BUKHARI_H1_ISNAD) // 2 for s in spans)

    def test_positions_sequential_and_unique(self) -> None:
        spans = extract_narrator_mentions(self._BUKHARI_H1_ISNAD, "ar")
        positions = [s.position for s in spans]
        assert positions == list(range(len(spans)))

    def test_first_narrator_segmented(self) -> None:
        """The first narrator after ``حدثنا`` is isolated (al-Humaydi)."""
        spans = extract_narrator_mentions(self._BUKHARI_H1_ISNAD, "ar")
        # Normalized form of الحميدي عبدالله بن الزبير, comma stripped.
        assert spans[0].name.startswith("الحميدي")
        assert "،" not in spans[0].name
        assert spans[0].transmission_method == "haddathana"

    def test_singular_variant_segments(self) -> None:
        """``أخبرني`` / ``سمع`` (singular/3rd-person) also split the chain."""
        spans = extract_narrator_mentions(self._BUKHARI_H1_ISNAD, "ar")
        methods = {s.transmission_method for s in spans}
        assert "akhbarani" in methods
        assert "samia" in methods


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert extract_narrator_mentions("", "en") == []
        assert extract_narrator_mentions("", "ar") == []

    def test_whitespace_only(self) -> None:
        assert extract_narrator_mentions("   ", "en") == []
        assert extract_narrator_mentions("   ", "ar") == []

    def test_no_narrator_keywords(self) -> None:
        text = "The Prophet said something important"
        spans = extract_narrator_mentions(text, "en")
        # Even without keywords, the full text may be returned as a single span
        # depending on splitting behavior — just verify no crash
        assert isinstance(spans, list)
