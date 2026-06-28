"""Tests for narrator name-quality cleaning + validation (da#247)."""

from __future__ import annotations

import pytest

from src.parse.name_quality import clean_narrator_name, strip_markup


class TestStripMarkup:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<NAR> أَبُو عُبَيْدَةَ", "أَبُو عُبَيْدَةَ"),
            ("<NAR> <NAR> عبد الرحمن", "عبد الرحمن"),
            ("اسماعيل <IDF> يعني</IDF> ابن عليه", "اسماعيل يعني ابن عليه"),
            ("شيخ من اهل المدينه ,", "شيخ من اهل المدينه"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_strips_markup_and_edge_punct(self, raw: str | None, expected: str) -> None:
        assert strip_markup(raw) == expected


class TestCleanNarratorName:
    # --- markup leakage: recover the real name (sanadset <NAR>) ---
    @pytest.mark.parametrize(
        "polluted,expected",
        [
            ("<NAR> ابو عبيده", "ابو عبيده"),
            ("<NAR> <NAR> عبد الرحمن", "عبد الرحمن"),
            ("<NAR> ابن عباس", "ابن عباس"),
            ("اسماعيل <IDF> يعني</IDF> ابن عليه", "اسماعيل ابن عليه"),
        ],
    )
    def test_strips_markup_recovers_name(self, polluted: str, expected: str) -> None:
        assert clean_narrator_name(polluted) == expected

    def test_no_angle_brackets_survive(self) -> None:
        for name in ("<NAR> فلان", "<x فلان", "فلان >", "<NAR><IDF> زيد"):
            cleaned = clean_narrator_name(name)
            assert cleaned is None or ("<" not in cleaned and ">" not in cleaned)

    # --- honorific phrases stripped, real name kept ---
    def test_strips_trailing_honorific(self) -> None:
        assert clean_narrator_name("عبد الله بن مسعود رضي الله عنه") == "عبد الله بن مسعود"

    def test_pure_honorific_rejected(self) -> None:
        assert clean_narrator_name("صلى الله عليه وسلم") is None

    # --- mubham / collective descriptors rejected ---
    @pytest.mark.parametrize(
        "mubham",
        [
            "رجل من اصحاب النبي",
            "جماعه من اصحاب",
            "رجال من قومه",
            "شيخ من اهل المدينه",
            "ناس من الصحابه",
            "نفر من اصحابه",
        ],
    )
    def test_mubham_descriptor_rejected(self, mubham: str) -> None:
        assert clean_narrator_name(mubham) is None

    # --- over-long span (thaqalayn whole-text) rejected ---
    def test_overlong_text_rejected(self) -> None:
        long_text = " ".join(["كلمه"] * 40)
        assert clean_narrator_name(long_text) is None

    # --- PRECISION: real names must survive (no false drops) ---
    @pytest.mark.parametrize(
        "real_name",
        [
            "محمد بن اسماعيل البخاري",
            "ابو هريره",
            "عبد الله بن عباس",
            "محمد بن اسماعيل بن ابراهيم بن المغيره الجعفي البخاري",
            # a real name that merely *contains* من mid-lineage (not mubham-leading)
            "عباد بن زياد من ولد المغيره بن شعبه",
            # a real name starting with a common token but no partitive من
            "شيخ الاسلام احمد بن تيميه",
        ],
    )
    def test_real_names_preserved(self, real_name: str) -> None:
        assert clean_narrator_name(real_name) == real_name

    # --- empties ---
    @pytest.mark.parametrize("empty", [None, "", "   ", "<NAR>", "يعني"])
    def test_empty_or_pure_noise_rejected(self, empty: str | None) -> None:
        assert clean_narrator_name(empty) is None
