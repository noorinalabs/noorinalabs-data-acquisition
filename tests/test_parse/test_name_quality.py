"""Tests for narrator name-quality cleaning + validation (da#247)."""

from __future__ import annotations

import pytest

from src.parse.name_quality import (
    clean_narrator_name,
    split_compound_narrators,
    strip_markup,
)


class TestStripMarkup:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<NAR> أَبُو عُبَيْدَةَ", "أَبُو عُبَيْدَةَ"),
            ("<NAR> <NAR> عبد الرحمن", "عبد الرحمن"),
            ("اسماعيل <IDF> يعني</IDF> ابن عليه", "اسماعيل يعني ابن عليه"),
            ("شيخ من اهل المدينه ,", "شيخ من اهل المدينه"),
            # da#253: colon-joined English matn truncated to the bare display name
            ("Thawban:The Messenger of Allah sacrificed and then", "Thawban"),
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

    # --- bare relational-pronoun (mubham) references rejected (da#247 residual) ---
    # ابيه ("his father") was the single most-mentioned "narrator" (65,755) on the
    # #723 reload; جده ("his grandfather"), ابي ("my father") followed.
    @pytest.mark.parametrize(
        "relational",
        [
            "ابيه",
            "ابي",
            "جده",
            "جدته",
            "امه",
            "ابنه",
            "اخيه",
            "عمه",
            "خاله",
            "عنه",
            # trailing-punctuation variants that shielded the bare token (da#247 residual)
            "ابيه،",
            "ابي ،",
            "ابيه :",
            "ابي . . .",
        ],
    )
    def test_relational_pronoun_rejected(self, relational: str) -> None:
        assert clean_narrator_name(relational) is None

    # --- PRECISION: a multi-token kunya that merely STARTS with a relational token
    # must survive — "ابي اسحاق" (Abū Isḥāq) is a real narrator, not "my father".
    @pytest.mark.parametrize(
        "real_kunya",
        ["ابي اسحاق", "ابي هريره", "ابي بكر الصديق", "عبد الله بن ابي بكر"],
    )
    def test_relational_token_in_real_name_preserved(self, real_kunya: str) -> None:
        assert clean_narrator_name(real_kunya) == real_kunya

    # --- English non-name fragments (translated isnad prose) rejected (da#247) ---
    @pytest.mark.parametrize(
        "fragment",
        [
            "It was",
            "It was narrated",
            "It has been narrated that",
            "This hadith has been",
            "The Prophet said",
            "his father",
            "He said",
            "They narrated from",
            "narrated by",
        ],
    )
    def test_english_fragment_rejected(self, fragment: str) -> None:
        assert clean_narrator_name(fragment) is None

    # --- PRECISION: romanized real narrators survive, incl. al-/an- article forms
    # and name-leaders abu/ibn/abd/umm that must NOT match the leader set ---
    @pytest.mark.parametrize(
        "romanized",
        [
            "Abu Huraira",
            "Ibn Umar",
            "Malik",
            "Anas bin Malik",
            "Al-Zuhri",
            "An-Nawawi",
            "Umm Salama",
            "Abd Allah ibn Abbas",
        ],
    )
    def test_romanized_name_preserved(self, romanized: str) -> None:
        assert clean_narrator_name(romanized) == romanized

    # --- English "<name>:<matn>" colon-join (da#253) ---
    # The reported prod node nar:00063b2c-… had name_en = a companion name
    # colon-joined to a hadith matn. The internal colon hides the "The" leader
    # from step 7's tokens[0] check and the ~11-token body sits under the token
    # cap, so the pre-fix filter passed it through. The fix truncates at the colon
    # whose tail begins with an English leader, recovering the bare name.
    def test_colon_joined_matn_recovers_bare_name(self) -> None:
        polluted = "Thawban:The Messenger of Allah (ﷺ) sacrificed during a journey and then"
        assert clean_narrator_name(polluted) == "Thawban"

    @pytest.mark.parametrize(
        "polluted,expected",
        [
            # short colon-join (escapes the >=2-leader guard; caught by colon-split)
            ("Anas:He said this", "Anas"),
            ("Jabir:It was narrated", "Jabir"),
            ("Aisha:The Prophet said", "Aisha"),
            # a pre-colon name that is itself prose still fails the leader guard
            ("The Messenger:said something", None),
        ],
    )
    def test_colon_join_variants(self, polluted: str, expected: str | None) -> None:
        assert clean_narrator_name(polluted) == expected

    # --- embedded English prose with no leading leader (da#253, step 8) ---
    # A name+matn run carrying >= 2 whole-token English function/stop words is a
    # sentence, not a name — even when it neither leads with a leader nor exceeds
    # the token cap.
    @pytest.mark.parametrize(
        "prose",
        [
            "Thawban sacrificed during a journey and then narrated",
            "Bilal called the people to prayer and stood",
        ],
    )
    def test_embedded_prose_rejected(self, prose: str) -> None:
        assert clean_narrator_name(prose) is None

    # --- PRECISION: a "name:name" colon-join (tail is a real name, not a leader)
    # must NOT be dropped — only prose tails are stripped. ---
    def test_colon_join_name_name_preserved(self) -> None:
        result = clean_narrator_name("Sufyan:Ibn Uyayna")
        assert result is not None
        assert "Uyayna" in result

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

    # =====================================================================
    # da#258 — Arabic-script isnad/matn residual (sibling of the da#253 fix)
    # =====================================================================

    # --- class 5: chain-connective fragments truncate to the leading name ---
    # "<real name> <isnad-connective> <next narrator>" — keep the lead, drop the
    # connective + tail. The Arabic mirror of da#253's colon-prose truncation.
    @pytest.mark.parametrize(
        "polluted,expected",
        [
            # عن ("from") — the task's canonical class-3 example
            ("ابو عبد الرحمن عن ابي هريره", "ابو عبد الرحمن"),
            ("ابو محمد بصري عن محمد بن علي", "ابو محمد بصري"),
            # ان ("that") — task example
            ("سالم بن عبد الله ان عبد الله بن عمر", "سالم بن عبد الله"),
            ("عبد الرحمن بن سلم ان", "عبد الرحمن بن سلم"),
            # الذي / التي (relative clause opening)
            ("ابو العنبس الذي", "ابو العنبس"),
            ("حمزه بن محمد الذي يقال", "حمزه بن محمد"),
            # transmission verb joining two narrators
            ("ابو الطيب الحربي انبانا احمد بن محمد", "ابو الطيب الحربي"),
            # editorial gloss "هو …" ("he is …") — recover the real lead
            ("بندار هو محمد بن بشار", "بندار"),
            ("مسلم هو البطين وهو ابن ابي عمران", "مسلم"),
            ("ابو تراب هو علي بن ابي طالب", "ابو تراب"),
            ("الصادق هو جعفر بن محمد بن علي بن الحسين", "الصادق"),
            ("ابو زرعه قيل هو ابن عمرو بن جرير", "ابو زرعه"),
            # matn verb after a real narrator name → keep the narrator
            ("عاءشه قالت كان", "عاءشه"),
        ],
    )
    def test_isnad_connective_truncates_to_lead(self, polluted: str, expected: str) -> None:
        assert clean_narrator_name(polluted) == expected

    # --- class 4: genuine matn body dropped (boundary token LEADS the span) ---
    @pytest.mark.parametrize(
        "matn",
        [
            # the task's canonical genuine-matn example
            "كان علي بن ابي طالب بالكوفه في الجامع اذ قام اليه رجل من اهل الشام فساله",
            "كان علي بن ابي طالب بالكوفه",
            "قالت عاءشه رضى الله عنها",
            "كان ابن عمر",
            # bare matn particle / verb minted as a name
            "قالوا",
            "كان",
            "ثم",
            "ثم ما ذا",
            "قالت",
        ],
    )
    def test_arabic_matn_body_rejected(self, matn: str) -> None:
        assert clean_narrator_name(matn) is None

    # --- PRECISION: long real ibn-lineage nasab names must survive UNCHANGED.
    # The 30-token cap alone cannot separate these from matn — the boundary rule
    # must not fire on any name component (بن / عبد / ال-nisba / given names). ---
    @pytest.mark.parametrize(
        "real_name",
        [
            "ابو الحسن علي بن احمد بن عبد العزيز الجرجاني",
            "محمد بن اسماعيل بن ابراهيم بن المغيره الجعفي البخاري",
            "سليمان بن مهران الاعمش",
            "عبد الله بن عمر",
            "عثمان بن عفان",
            "ابو اسحاق السبيعي",
            # contains من mid-lineage (not mubham-leading) — da#247 precision case
            "عباد بن زياد من ولد المغيره بن شعبه",
        ],
    )
    def test_long_real_nasab_preserved(self, real_name: str) -> None:
        assert clean_narrator_name(real_name) == real_name

    # --- class 6: a compound co-narrator join is PRESERVED, never dropped ---
    # (splitting into separate narrators is the resolve-stage follow-up; the hard
    # bar is that clean_narrator_name must not DELETE the real co-narrators).
    @pytest.mark.parametrize(
        "compound",
        [
            "يحيى بن يحيى و يحيى بن ايوب وقتيبه وابن حجر",
            "سعد بن عبد الله و عبد الله بن جعفر الحميري",
        ],
    )
    def test_compound_not_dropped(self, compound: str) -> None:
        assert clean_narrator_name(compound) is not None

    # --- da#253 / da#247 non-regression: the English + mubham/markup classes
    # still behave after the da#258 Arabic rules were added ---
    def test_da253_english_colon_join_still_truncates(self) -> None:
        assert clean_narrator_name("Thawban:The Messenger of Allah sacrificed") == "Thawban"

    def test_da247_relational_pronoun_still_dropped(self) -> None:
        assert clean_narrator_name("ابيه") is None
        assert clean_narrator_name("جده") is None

    def test_da247_markup_still_stripped(self) -> None:
        assert clean_narrator_name("<NAR> ابو عبيده") == "ابو عبيده"

    # --- empties ---
    @pytest.mark.parametrize("empty", [None, "", "   ", "<NAR>", "يعني"])
    def test_empty_or_pure_noise_rejected(self, empty: str | None) -> None:
        assert clean_narrator_name(empty) is None


class TestSplitCompoundNarrators:
    """da#258 class 6 — compound co-narrator join detection + split primitive."""

    # --- standalone-و join with trailing جميعا ("… together") ---
    def test_splits_standalone_waw_jamian(self) -> None:
        result = split_compound_narrators("سعد بن عبد الله و عبد الله بن جعفر الحميري جميعا")
        assert result == ["سعد بن عبد الله", "عبد الله بن جعفر الحميري"]

    # --- proclitic-و join with trailing قالوا ("… they said") ---
    def test_splits_proclitic_waw_qalu(self) -> None:
        result = split_compound_narrators("يحيى بن ايوب وقتيبه وابن حجر قالوا")
        assert result == ["يحيى بن ايوب", "قتيبه", "ابن حجر"]

    # --- mixed standalone + proclitic و, trailing marker stripped ---
    def test_splits_mixed_waw_forms(self) -> None:
        result = split_compound_narrators("يحيى بن يحيى و ابو بكر بن ابي شيبه وابو كريب")
        assert result == ["يحيى بن يحيى", "ابو بكر بن ابي شيبه", "ابو كريب"]

    # --- PRECISION: a lone و-initial REAL name is NOT a compound and is NOT split
    # (no standalone و and no trailing marker → proclitic split must not fire) ---
    @pytest.mark.parametrize("real_name", ["وكيع بن الجراح", "وهب بن منبه", "وائل بن حجر"])
    def test_waw_initial_real_name_not_split(self, real_name: str) -> None:
        assert split_compound_narrators(real_name) == [real_name]

    # --- a plain single narrator is returned as a one-element list ---
    def test_single_name_returned_as_singleton(self) -> None:
        assert split_compound_narrators("ابو هريره") == ["ابو هريره"]

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_returns_empty_list(self, empty: str | None) -> None:
        assert split_compound_narrators(empty) == []

    # --- each split member is a real name that survives clean_narrator_name
    # (the co-narrators are preserved, not dropped) ---
    def test_split_members_survive_cleaning(self) -> None:
        members = split_compound_narrators("سعد بن عبد الله و عبد الله بن جعفر الحميري جميعا")
        assert [clean_narrator_name(m) for m in members] == members
