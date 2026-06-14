"""Tests for Arabic text processing utilities."""

from __future__ import annotations

import pytest

from src.utils.arabic import (
    clean_whitespace,
    contains_transmission_marker,
    extract_transmission_phrases,
    is_arabic,
    normalize_alif,
    normalize_arabic,
    normalize_hamza,
    normalize_taa_marbuta,
    strip_diacritics,
    transliterate,
)

# ---------------------------------------------------------------------------
# strip_diacritics
# ---------------------------------------------------------------------------


class TestStripDiacritics:
    def test_basmala(self) -> None:
        assert strip_diacritics("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ") == "بسم الله الرحمن الرحيم"

    def test_no_diacritics_unchanged(self) -> None:
        plain = "بسم الله"
        assert strip_diacritics(plain) == plain

    def test_empty_string(self) -> None:
        assert strip_diacritics("") == ""

    def test_diacritics_only(self) -> None:
        """A string of only diacritics should produce an empty string."""
        diacritics = "\u064b\u064c\u064d\u064e\u064f\u0650"
        assert strip_diacritics(diacritics) == ""

    def test_latin_text_unchanged(self) -> None:
        assert strip_diacritics("Hello World") == "Hello World"


# ---------------------------------------------------------------------------
# normalize_alif
# ---------------------------------------------------------------------------


class TestNormalizeAlif:
    @pytest.mark.parametrize(
        ("input_char", "expected"),
        [
            ("\u0623", "\u0627"),  # أ → ا
            ("\u0625", "\u0627"),  # إ → ا
            ("\u0622", "\u0627"),  # آ → ا
            ("\u0671", "\u0627"),  # ٱ → ا
        ],
        ids=["hamza-above", "hamza-below", "madda", "wasla"],
    )
    def test_individual_variants(self, input_char: str, expected: str) -> None:
        assert normalize_alif(input_char) == expected

    def test_within_word(self) -> None:
        assert normalize_alif("أحمد") == "احمد"

    def test_bare_alif_unchanged(self) -> None:
        assert normalize_alif("ا") == "ا"

    def test_empty_string(self) -> None:
        assert normalize_alif("") == ""


# ---------------------------------------------------------------------------
# normalize_hamza
# ---------------------------------------------------------------------------


class TestNormalizeHamza:
    def test_hamza_on_waw(self) -> None:
        assert normalize_hamza("ؤ") == "ء"

    def test_hamza_on_ya(self) -> None:
        assert normalize_hamza("ئ") == "ء"

    def test_in_word(self) -> None:
        assert normalize_hamza("مؤمن") == "مءمن"

    def test_empty_string(self) -> None:
        assert normalize_hamza("") == ""


# ---------------------------------------------------------------------------
# normalize_taa_marbuta
# ---------------------------------------------------------------------------


class TestNormalizeTaaMarbuta:
    def test_basic(self) -> None:
        assert normalize_taa_marbuta("ة") == "ه"

    def test_in_word(self) -> None:
        assert normalize_taa_marbuta("مدينة") == "مدينه"

    def test_empty_string(self) -> None:
        assert normalize_taa_marbuta("") == ""


# ---------------------------------------------------------------------------
# clean_whitespace
# ---------------------------------------------------------------------------


class TestCleanWhitespace:
    def test_multiple_spaces(self) -> None:
        assert clean_whitespace("أبو   هريرة") == "أبو هريرة"

    def test_leading_trailing(self) -> None:
        assert clean_whitespace("  بسم الله  ") == "بسم الله"

    def test_tabs_newlines(self) -> None:
        assert clean_whitespace("hello\t\nworld") == "hello world"

    def test_empty_string(self) -> None:
        assert clean_whitespace("") == ""


# ---------------------------------------------------------------------------
# normalize_arabic (full pipeline)
# ---------------------------------------------------------------------------


class TestNormalizeArabic:
    def test_full_pipeline(self) -> None:
        """Diacritics stripped, alif/hamza/taa normalized, tatweel removed, WS collapsed."""
        raw = "  بِسْمِ  اللَّهِ  الرَّحْمَـٰنِ  "
        result = normalize_arabic(raw)
        # Diacritics removed, tatweel removed, whitespace collapsed
        assert "ِ" not in result  # noqa: RUF001
        assert "ـ" not in result
        assert "  " not in result
        assert result == result.strip()

    def test_tatweel_removed(self) -> None:
        assert normalize_arabic("عـلـي") == "علي"

    def test_idempotent(self) -> None:
        """Normalizing already-normalized text returns the same result."""
        text = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(normalize_arabic(text)) == normalize_arabic(text)

    def test_empty_string(self) -> None:
        assert normalize_arabic("") == ""

    def test_mixed_arabic_latin(self) -> None:
        """Latin characters are preserved alongside normalized Arabic."""
        result = normalize_arabic("Hello أحمد World")
        assert "Hello" in result
        assert "World" in result


# ---------------------------------------------------------------------------
# is_arabic
# ---------------------------------------------------------------------------


class TestIsArabic:
    def test_arabic_text(self) -> None:
        assert is_arabic("بسم الله") is True

    def test_english_text(self) -> None:
        assert is_arabic("Hello World") is False

    def test_mixed_text(self) -> None:
        assert is_arabic("Hello أحمد") is True

    def test_empty_string(self) -> None:
        assert is_arabic("") is False

    def test_arabic_diacritics_only(self) -> None:
        """Diacritics (U+064B-U+065F) are within the Arabic block U+0600-U+06FF."""
        assert is_arabic("\u064b\u064c") is True

    def test_numbers_only(self) -> None:
        assert is_arabic("12345") is False


# ---------------------------------------------------------------------------
# extract_transmission_phrases
# ---------------------------------------------------------------------------


class TestExtractTransmissionPhrases:
    def test_haddathana_and_an(self) -> None:
        text = "حدثنا سفيان عن الزهري"
        results = extract_transmission_phrases(text)
        labels = [label for _, _, label in results]
        assert "haddathana" in labels
        assert "an" in labels

    def test_positions_ordered(self) -> None:
        text = "حدثنا سفيان عن الزهري"
        results = extract_transmission_phrases(text)
        starts = [start for start, _, _ in results]
        assert starts == sorted(starts)

    def test_multiple_patterns(self) -> None:
        text = "أخبرنا فلان قال سمعت فلانا"
        results = extract_transmission_phrases(text)
        labels = {label for _, _, label in results}
        assert "akhbarana" in labels
        assert "qala" in labels
        assert "samitu" in labels

    def test_no_matches(self) -> None:
        assert extract_transmission_phrases("Hello World") == []

    def test_empty_string(self) -> None:
        assert extract_transmission_phrases("") == []

    def test_return_type(self) -> None:
        results = extract_transmission_phrases("حدثنا")
        assert len(results) == 1
        start, end, label = results[0]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert label == "haddathana"

    def test_matches_diacritized_term(self) -> None:
        """Fully-voweled terms must match (da#146) — bare patterns alone do not."""
        # حَدَّثَنَا (voweled) vs the bare keyword حدثنا.
        results = extract_transmission_phrases("حَدَّثَنَا سُفْيَانُ")
        labels = [label for _, _, label in results]
        assert "haddathana" in labels

    def test_overlapping_variants_resolve_to_longer(self) -> None:
        """``سمعت`` must win over the bare ``سمع`` at a shared position."""
        results = extract_transmission_phrases("سمعت فلانا")
        # Exactly one transmission phrase at the start, labelled samitu.
        starts = [s for s, _, _ in results]
        assert starts == sorted(starts)
        assert results[0][2] == "samitu"
        # No duplicate/overlapping samia span at the same start.
        assert not any(label == "samia" and start == 0 for start, _, label in results)

    def test_singular_variants_recognized(self) -> None:
        results = extract_transmission_phrases("حدثني فلان أخبرني فلان")
        labels = {label for _, _, label in results}
        assert "haddathani" in labels
        assert "akhbarani" in labels


# ---------------------------------------------------------------------------
# transliterate
# ---------------------------------------------------------------------------


class TestTransliterate:
    @pytest.mark.parametrize(
        ("name_ar", "expected"),
        [
            ("محمد", "Muhammad"),
            ("أحمد", "Ahmad"),
            ("علي", "Ali"),
            ("عائشة", "Aisha"),
            ("فاطمة", "Fatima"),
            ("البخاري", "al-Bukhari"),
            ("الزهري", "al-Zuhri"),
        ],
        ids=[
            "muhammad",
            "ahmad",
            "ali",
            "aisha",
            "fatima",
            "al-bukhari",
            "al-zuhri",
        ],
    )
    def test_lexicon_single_token(self, name_ar: str, expected: str) -> None:
        assert transliterate(name_ar) == expected

    def test_theophoric_two_tokens(self) -> None:
        assert transliterate("عبد الله") == "Abd Allah"

    def test_theophoric_single_token(self) -> None:
        """A run-together ``عبدالله`` renders the same as the spaced form."""
        assert transliterate("عبدالله") == "Abd Allah"

    def test_kunya(self) -> None:
        assert transliterate("أبو هريرة") == "Abu Hurayra"

    def test_nasab_particle_lowercase_mid_name(self) -> None:
        """``بن``/``ابن`` read lowercase between names."""
        assert transliterate("محمد بن إسماعيل") == "Muhammad ibn Ismail"

    def test_nasab_particle_capitalized_when_leading(self) -> None:
        assert transliterate("ابن عباس") == "Ibn Abbas"

    def test_compound_full_name(self) -> None:
        assert transliterate("عبد الله بن عباس") == "Abd Allah ibn Abbas"

    def test_diacritics_are_ignored(self) -> None:
        """A voweled spelling normalizes to the same output as the bare form."""
        assert transliterate("مُحَمَّد") == transliterate("محمد") == "Muhammad"

    def test_diacritic_only_token_dropped(self) -> None:
        assert transliterate("ًٌٍ") == ""

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_input(self, blank: str) -> None:
        assert transliterate(blank) == ""

    def test_unknown_token_is_nonempty_ascii(self) -> None:
        """A token outside the lexicon still yields a non-empty ASCII skeleton."""
        out = transliterate("الخطاب")  # in lexicon -> al-Khattab
        assert out == "al-Khattab"
        # A genuinely rare token falls through to the consonant skeleton.
        rare = transliterate("بستيطير")
        assert rare
        assert rare[0].isupper()
        assert rare.isascii()

    def test_output_is_ascii_for_full_chain(self) -> None:
        out = transliterate("عبد الرحمن بن عوف الزهري")
        assert out
        assert out.isascii()
        assert out.startswith("Abd al-Rahman ibn Awf")

    def test_deterministic(self) -> None:
        name = "محمد بن إسماعيل البخاري"
        assert transliterate(name) == transliterate(name)

    def test_already_latin_passthrough(self) -> None:
        assert transliterate("Anas") == "Anas"


# ---------------------------------------------------------------------------
# Orthographic-variant tolerance (da#158) — patterns must match bare-alif forms
# ---------------------------------------------------------------------------


class TestTransmissionVariantTolerance:
    """The real corpus mixes hamza-alif (أخبرنا) and bare-alif (اخبرنا) forms.

    The transmission patterns are written with the classical hamza forms but
    must also match the bare-alif spellings, otherwise sub-chains starting with
    a bare-alif verb stay merged into the previous span as a blob (da#158).
    """

    def test_bare_alif_akhbarana_matches(self) -> None:
        # اخبرنا (bare alif U+0627), not أخبرنا (hamza-alif U+0623).
        labels = [label for _, _, label in extract_transmission_phrases("اخبرنا سفيان")]
        assert "akhbarana" in labels

    def test_bare_alif_anbaana_matches(self) -> None:
        labels = [label for _, _, label in extract_transmission_phrases("انبانا فلان")]
        assert "anba_ana" in labels

    def test_voweled_bare_alif_matches(self) -> None:
        # Voweled bare-alif form: اِخْبَرَنَا-style with diacritics on bare alif.
        text = "اخْبَرَنَا سُفْيَانُ"
        labels = [label for _, _, label in extract_transmission_phrases(text)]
        assert "akhbarana" in labels


# ---------------------------------------------------------------------------
# Short-particle word-boundary anchoring (da#155)
# ---------------------------------------------------------------------------


class TestShortParticleBoundary:
    """عن / قال / سمع must only match as standalone particles, never mid-word."""

    @pytest.mark.parametrize(
        "name",
        [
            "عنبسة",  # عن at offset 0, followed by a letter
            "معن",  # عن after a letter
            "يعني",  # عن inside the word
            "مقالة",  # قال inside the word
        ],
    )
    def test_no_mid_word_particle_match(self, name: str) -> None:
        assert extract_transmission_phrases(name) == []

    def test_standalone_particle_still_matches(self) -> None:
        labels = [label for _, _, label in extract_transmission_phrases("فلان عن فلان")]
        assert "an" in labels

    def test_only_real_particle_matches_in_mixed(self) -> None:
        # The real عن (between the two names) matches; the عن inside عنبسة does not.
        results = extract_transmission_phrases("عنبسة عن عبد الله")
        assert [label for _, _, label in results] == ["an"]


# ---------------------------------------------------------------------------
# contains_transmission_marker — fail-loud signal (da#158)
# ---------------------------------------------------------------------------


class TestContainsTransmissionMarker:
    def test_detects_voweled_long_form(self) -> None:
        assert contains_transmission_marker("حَدَّثَنَا سُفْيَانُ") is True

    def test_detects_bare_alif_variant(self) -> None:
        assert contains_transmission_marker("اخبرنا سفيان") is True

    def test_ignores_particle_inside_name(self) -> None:
        # عنبسة / مقالة contain عن / قال as substrings but carry no real marker.
        assert contains_transmission_marker("عنبسة") is False
        assert contains_transmission_marker("مقالة") is False

    def test_plain_name_has_no_marker(self) -> None:
        assert contains_transmission_marker("عبد الله بن مسلمة") is False

    def test_empty(self) -> None:
        assert contains_transmission_marker("") is False
