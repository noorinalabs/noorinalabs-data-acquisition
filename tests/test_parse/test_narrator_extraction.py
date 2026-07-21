"""Tests for narrator mention extraction from isnad text."""

from __future__ import annotations

import pytest

from src.parse.narrator_extraction import (
    IsnadSegmentationError,
    NarratorSpan,
    extract_narrator_mentions,
)
from src.utils.arabic import contains_transmission_marker, normalize_arabic


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


class TestDa158ProductionBlobSegmentation:
    """Regression tests for da#158: ~80% of loaded Narrator nodes were raw isnad
    blobs because the segmenter failed on production-realistic voweled Arabic.

    The keystone failure was orthographic: real chains mix hamza-alif (أخبرنا)
    and bare-alif (اخبرنا) transmission verbs, and the bare-alif forms were not
    matched — so a sub-chain starting with one stayed merged into the previous
    span as a blob. These fixtures are real-shaped (mixed orthography, voweled),
    not toy h-1 data (production-fixture standards, main#671).
    """

    # The exact staging blob cited in da#158 (note bare-alif اخبرنا, U+0627).
    _STAGING_BLOB = (
        "حدثنا علي بن عبد الله، اخبرنا سفيان، حدثنا شبيب بن غرقده، قال سمعت الحى، يحدثون عن عروه،"
    )

    # Fully-voweled Bukhari-style chain mixing hamza-alif and bare-alif verbs.
    _VOWELED_MIXED = "حَدَّثَنَا عَلِيُّ بْنُ عَبْدِ اللَّهِ، اخْبَرَنَا سُفْيَانُ، عَنْ عُرْوَةَ بْنِ الزُّبَيْرِ"

    def test_bare_alif_chain_fully_segmented(self) -> None:
        """The bare-alif اخبرنا must split the chain, not merge into a blob."""
        spans = extract_narrator_mentions(self._STAGING_BLOB, "ar")
        # علي / سفيان / شبيب / ... — at least four distinct narrators.
        assert len(spans) >= 4
        assert spans[1].name == "سفيان"
        assert spans[1].transmission_method == "akhbarana"

    def test_no_span_is_a_blob(self) -> None:
        """Corpus-level sanity invariant: NO produced mention may contain a
        transmission verb (da#158 asked for ~0%)."""
        for fixture in (self._STAGING_BLOB, self._VOWELED_MIXED):
            for span in extract_narrator_mentions(fixture, "ar"):
                assert not contains_transmission_marker(span.name), span.name

    def test_voweled_mixed_orthography_segments(self) -> None:
        spans = extract_narrator_mentions(self._VOWELED_MIXED, "ar")
        methods = {s.transmission_method for s in spans}
        assert {"haddathana", "akhbarana", "an"} <= methods

    def test_positions_sequential_zero_based(self) -> None:
        spans = extract_narrator_mentions(self._STAGING_BLOB, "ar")
        assert [s.position for s in spans] == list(range(len(spans)))


class TestDa158FailLoud:
    """The segmenter must FAIL LOUD rather than mint a whole-chain blob (da#158)."""

    _CHAIN = "حدثنا علي بن عبد الله، اخبرنا سفيان، عن عروه"

    def test_marker_without_phrases_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If phrase detection regresses to empty but a marker is present, the
        extractor raises instead of returning the whole chain as one span."""
        monkeypatch.setattr(
            "src.parse.narrator_extraction.extract_transmission_phrases",
            lambda _text: [],
        )
        with pytest.raises(IsnadSegmentationError):
            extract_narrator_mentions(self._CHAIN, "ar")

    def test_bare_name_without_marker_does_not_raise(self) -> None:
        """A genuine single narrator name (no transmission marker) is fine."""
        spans = extract_narrator_mentions("عبد الله بن مسلمة القعنبي", "ar")
        assert len(spans) == 1
        assert spans[0].transmission_method is None


class TestDa155ShortParticleBoundary:
    """da#155: عن/قال/سمع must not over-segment real names that contain them."""

    @pytest.mark.parametrize(
        "name",
        ["عنبسة", "معن", "يعني", "مقالة", "عبد الله بن عنبسة"],
    )
    def test_name_with_particle_substring_not_split(self, name: str) -> None:
        spans = extract_narrator_mentions(name, "ar")
        # Stays a single narrator — the particle substring does not split it.
        assert len(spans) == 1
        # No spurious fragment (e.g. "بسة" from عنبسة).
        assert "بسة" not in [s.name for s in spans]

    def test_real_particle_in_chain_still_splits(self) -> None:
        """A standalone عن between two names splits; the عن inside عنبسة does not."""
        spans = extract_narrator_mentions("حدثنا عنبسة عن عبد الله بن عنبسة", "ar")
        assert len(spans) == 2
        assert spans[1].transmission_method == "an"
        assert "عنبس" in spans[1].name  # the trailing عنبسة survived intact


class TestDa154NameBoundary:
    """da#154: deterministic name-boundary refinement — drop trailing matn /
    connectives; keep multi-token (kunya/laqab) names joined."""

    def test_english_trailing_report_verb_trimmed(self) -> None:
        spans = extract_narrator_mentions("Anas b. Malik reported", "en")
        assert [s.name for s in spans] == ["Anas b. Malik"]

    def test_english_trailing_matn_clause_trimmed(self) -> None:
        spans = extract_narrator_mentions("Aishah that the Prophet said", "en")
        assert [s.name for s in spans] == ["Aishah"]

    def test_english_kunya_stays_joined(self) -> None:
        spans = extract_narrator_mentions("Narrated Abu Hurayra", "en")
        assert [s.name for s in spans] == ["Abu Hurayra"]

    def test_arabic_trailing_connective_trimmed(self) -> None:
        # ...التيمي أنه سمع... — أنه must not ride along on the name span.
        chain = "أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة"
        spans = extract_narrator_mentions(chain, "ar")
        names = [s.name for s in spans]
        assert "محمد بن ابراهيم التيمي" in names
        assert all("انه" not in n.split() for n in names)

    def test_arabic_laqab_stays_joined(self) -> None:
        # Multi-token name with a laqab (al-Ansari) stays a single span.
        spans = extract_narrator_mentions("حدثنا يحيى بن سعيد الأنصاري", "ar")
        # compared in normalized space: da#427 folds alif-maqsura ى→ي (يحيى→يحيي)
        assert spans[0].name == normalize_arabic("يحيى بن سعيد الانصاري")


class TestDa244TrailingTransmissionMarker:
    """da#244: a transmission marker (قال/…) the phrase splitter fails to consume
    can survive into the *trailing* position of a name span. Such a residual
    trailing marker must be STRIPPED so the underlying name resolves — not aborted
    by the fail-loud guard, which would drop the whole chain. A marker in the
    *middle* of a span is a genuinely-unsegmented blob and must still fail loud.

    The asymmetry is real (arabic.py: the guard's normalizer-based detector
    catches orthographic variants the hand-built phrase patterns can miss). We
    reproduce it deterministically by simulating the splitter under-detecting,
    matching the existing TestDa158FailLoud monkeypatch style.
    """

    # The exact span that aborted the main#723 prod re-validation NER pass.
    _NAME = "ابو القاسم عبد الله بن احمد بن عامر الطاءي"

    def test_trailing_marker_stripped_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A trailing قال the splitter missed is stripped; the name resolves."""
        # Splitter detects only the leading حدثنا, leaving the trailing قال in the
        # tail segment — the production condition that tripped the guard.
        monkeypatch.setattr(
            "src.parse.narrator_extraction.extract_transmission_phrases",
            lambda _text: [(0, 5, "haddathana")],
        )
        spans = extract_narrator_mentions(f"حدثنا {self._NAME} قال", "ar")
        assert len(spans) == 1
        assert spans[0].name == self._NAME
        # No marker survives into the resolved name (the guard's invariant).
        assert not contains_transmission_marker(spans[0].name)

    def test_midspan_marker_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A marker BETWEEN two names is an unsegmented blob — still fail loud."""
        monkeypatch.setattr(
            "src.parse.narrator_extraction.extract_transmission_phrases",
            lambda _text: [(0, 5, "haddathana")],
        )
        with pytest.raises(IsnadSegmentationError):
            extract_narrator_mentions("حدثنا فلان قال علان", "ar")

    def test_real_name_with_marker_substring_unaffected(self) -> None:
        """Strip is whole-token + boundary-anchored: مقالة / معن keep their قال/عن
        (the marker substring is inside the token, not a standalone trailing one)."""
        for name in ["مقالة", "معن", "ابو القاسم الطاءي"]:
            spans = extract_narrator_mentions(name, "ar")
            assert [s.name for s in spans] == [normalize_arabic(name)]
