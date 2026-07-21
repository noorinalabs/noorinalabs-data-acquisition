"""Tests for narrator name-quality cleaning + validation (da#247)."""

from __future__ import annotations

import pytest

from src.parse.identity import make_canonical_id
from src.parse.name_quality import (
    asserted_name_tokens,
    clean_narrator_name,
    clean_narrator_name_display,
    cleaner_removed_content,
    is_mubham_relational,
    split_compound_narrators,
    strip_markup,
)
from src.utils.arabic import extract_arabic_span, normalize_arabic


class TestIsMubhamRelational:
    """The da#391 alias-list scrub predicate — exposes the step-6 bare-relational
    drop so the resolve alias-union sites can apply the SAME lexicon to the
    ``aliases`` ``list<string>`` the cleaner never reaches."""

    # 3rd-person forms already in the lexicon + the da#391 1st-person/vocative set.
    @pytest.mark.parametrize(
        "deixis",
        [
            "ابيه",  # his father
            "خالته",  # his maternal aunt
            "عمته",  # his paternal aunt
            # da#391 1st-person possessive (-ي) and vocative (-اه) additions
            "عمتي",  # my paternal aunt (survived on the ʿĀʾisha chimera)
            "امتاه",  # O mother! (survived on the ʿĀʾisha chimera)
            "جدي",  # my grandfather
            "عمي",  # my paternal uncle
            "خالتي",  # my maternal aunt
            "اماه",  # O mother! (variant)
            "ابتاه",  # O father!
            # normalization is applied internally — a diacritized surface still matches
            "خَالَتُهُ",
        ],
    )
    def test_bare_relational_is_mubham(self, deixis: str) -> None:
        assert is_mubham_relational(deixis) is True

    # PRECISION: the single-token boundary — a real multi-token kunya that merely
    # CONTAINS a relational token, a real ism, and empty input are never mubham.
    @pytest.mark.parametrize(
        "not_deixis",
        [
            "ابي اسحاق",  # Abū Isḥāq — real narrator, not "my father"
            "عاءشه بنت سعد",  # a real nasab
            "عاءشه",  # a real ism
            "هشام",  # a real (male) ism
            "سعد",
            "",
            None,
        ],
    )
    def test_real_name_or_empty_is_not_mubham(self, not_deixis: str | None) -> None:
        assert is_mubham_relational(not_deixis) is False


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


class TestBenedictionAndProphetReference:
    """da#271 — bare Shia taṣliya + Prophet-title references dropped."""

    # --- the bare Shia taṣliya (no trailing وسلم) and its variants → drop ---
    @pytest.mark.parametrize(
        "name",
        [
            "رسول الله صلى الله عليه واله",  # the mc-365 run-3 false narrator
            "النبي صلى الله عليه واله",
            "رسول الله صلى الله عليه و اله",  # spaced و اله
            "رسول الله‏ صلى الله عليه واله",  # RLM-marked (normalized away)
            "النبي",
            "نبي الله",
            "الرسول",
        ],
    )
    def test_prophet_reference_dropped(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is None

    # --- PRECISION: real names that merely start with رسول/نبي (not + الله), or
    #     contain الله as a real theophoric, are KEPT ---
    @pytest.mark.parametrize(
        "name",
        [
            "عبد الله بن مسلمه بن قعنب القعنبي",
            "جابر بن عبد الله",
            "ابو بكر الصديق",
            "رسول بن جعفر",  # first token رسول but not followed by الله
            "عبد الله",
        ],
    )
    def test_real_names_kept(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)

    # --- the taṣliya is stripped even when it trails a REAL leading name (the name
    #     is recovered, not dropped) ---
    def test_tasliya_stripped_leaving_real_name(self) -> None:
        # a fabricated case: real narrator followed by the benediction
        assert clean_narrator_name(normalize_arabic("انس بن مالك صلى الله عليه واله")) == (
            "انس بن مالك"
        )


class TestUrduAndLatinResidue:
    """da#299 — Urdu-script + Latin benediction/title residue in kaggle rijāl bios.

    The kaggle bios store a mixed Latin + Urdu-script + Arabic blob. The Arabic
    span is pulled out (:func:`extract_arabic_span`), Urdu letterforms fold to
    Arabic (in :func:`normalize_arabic`), and the benediction — whether the Urdu
    yeh-form or the Arabic maksura-form — is stripped, so an Urdu-script name
    reaches the SAME benediction-free canonical form as its Arabic twin.
    """

    def _canonical(self, blob: str) -> str | None:
        cleaned = clean_narrator_name(normalize_arabic(extract_arabic_span(blob)))
        return make_canonical_id(cleaned) if cleaned else None

    def test_urdu_blob_folds_to_arabic_twin_canonical(self) -> None:
        urdu_blob = "Prophet Muhammad(saw) ( محمد صلی اللہ علیہ وسلم ( رضي الله عنه"
        arabic_twin = "محمد صلى الله عليه وسلم رضي الله عنه"
        assert self._canonical(urdu_blob) == self._canonical(arabic_twin)

    def test_urdu_blob_reduces_to_the_bare_name(self) -> None:
        urdu_blob = "Prophet Muhammad(saw) ( محمد صلی اللہ علیہ وسلم ( رضي الله عنه"
        assert clean_narrator_name(normalize_arabic(extract_arabic_span(urdu_blob))) == "محمد"

    def test_urdu_folded_tasliya_is_stripped(self) -> None:
        # the Urdu benediction folds to the yeh-spelled "صلي …" form, which the
        # da#299 honorific entries strip (mirrors the da#308 رضى/رضي pair).
        name = "محمد بن یعقوب صلی اللہ علیہ وسلم"
        assert clean_narrator_name(normalize_arabic(name)) == "محمد بن يعقوب"

    # --- Latin / English benediction abbreviations stripped (parenthesized only) ---
    @pytest.mark.parametrize(
        ("polluted", "expected"),
        [
            ("Prophet Muhammad(saw)", "Muhammad"),
            ("Abu Hurayra (r.a.)", "Abu Hurayra"),
            ("Ali (as)", "Ali"),
            ("Bilal (pbuh)", "Bilal"),
            ("Hazrat Bilal", "Bilal"),
            ("Sayyidina Umar", "Umar"),
        ],
    )
    def test_latin_benediction_and_title_stripped(self, polluted: str, expected: str) -> None:
        assert clean_narrator_name(normalize_arabic(polluted)) == expected

    # --- PRECISION: a bare (non-parenthesized) benediction substring inside a real
    #     romanized name is NOT stripped; real names pass through untouched ---
    @pytest.mark.parametrize(
        "name",
        [
            "Anas",
            "Malik",
            "Al-Zuhri",
            "Asma",  # leading "as" substring, no parens → untouched
            "Rabia",  # leading "ra" substring, no parens → untouched
        ],
    )
    def test_precision_real_romanized_names_kept(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == name


class TestMatnSentenceGate:
    """da#308 — matn / grading-commentary sentences dropped, real names spared.

    ``clean_narrator_name`` scrubbed honorifics but kept whole hadith-matn /
    grading-commentary spans as canonical narrators (~15% / 25,971 of run-4b). The
    drop-gate returns ``None`` for structurally-matn spans while a nasab-density
    precision guard spares genuine lineages, nisba tails, and kunya compounds.
    """

    # --- DROP: matn / grading-commentary spans (the issue's evidence set) ---
    @pytest.mark.parametrize(
        "matn",
        [
            # the two canonical examples from the issue body
            "قلت لابي عبد الله عليه السلام ذهبت ارمي فاذا في يدي ست حصيات",
            "صلى النبي صلى الله عليه وسلم الظهر خمسا. فقيل له: ازيد في الصلاه؟",
            # high-mention-count matn/commentary from the evidence bullets
            "نعم ، هو بالاراك بعرفه يرعى ابل القوم ، فركب عمر",  # mc=853 pure matn
            "ابو عيسى هذا حديث حسن صحيح",  # mc=552 al-Tirmidhī grading formula
            "نهى رسول الله صلى الله عليه وسلم ان نستقبل القبلتين ببول او غاءط",  # mc=319
            # residual verb-led openers that _ISNAD_BOUNDARY does not cut
            "كنا مع النبي في سفر فنزلنا منزلا",
            "قلنا يا رسول الله كيف نصلي عليك",
            "امر النبي بقتل الاسودين في الصلاه",
            "رايت النبي يمسح على الخفين",
            "صلى بنا رسول الله الظهر ثم انصرف",
            # grading verdicts
            "هذا حديث صحيح على شرط الشيخين",
            "حديث حسن صحيح لا باس به",
            # lone matn answer particle (post-truncation residue)
            "نعم",
        ],
    )
    def test_matn_sentence_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- KEEP: real names pass through unchanged (honorific-scrubbed only) ---
    # Covers the issue's precision-guard set: long nasab chain, nisba tail,
    # vocalized بْنُ, kunya compound, and an 8+-token real nasab lineage.
    @pytest.mark.parametrize(
        "name",
        [
            # long genuine nasab chain (15 tokens, 6× بن)
            "محمد بن مسلم بن عبيد الله بن عبد الله بن شهاب بن عبد الله بن الحارث بن زهره",
            # nisba tails (incl. the "…الليثي ثم الجندعي" tribal-transition ثم)
            "عبد الله بن عبد الرحمن المداءني",
            "عطاء بن يزيد الليثي ثم الجندعي",
            # kunya / laqab compounds
            "ابو عبد الله الشامي",
            "ابو بكر بن ابي شيبه",
            # 8+-token real nasab lineage
            "ابو القاسم عبد الرحمن بن عبد الله بن احمد بن محمد بن عبيد بن عبد الملك",
        ],
    )
    def test_real_name_kept_unchanged(self, name: str) -> None:
        normalized = normalize_arabic(name)
        assert clean_narrator_name(normalized) == normalized

    # --- PRECISION: a VOCALIZED nasab chain — the diacritized بْنُ must count as a
    # connector so the span is spared. Passed pre-normalization to exercise the
    # gate's own diacritics-insensitive folding. ---
    def test_vocalized_nasab_spared(self) -> None:
        # Fully diacritized بْنُ / أَبُو must count as nasab connectors so the span is
        # spared. clean_narrator_name preserves its input's diacritics (it folds them
        # only for internal matching), so a spared span comes back unchanged.
        vocalized = "وَأَبُو بَكْرٍ أَحْمَدُ بْنُ الْحَسَنِ الْقَاضِي"
        assert clean_narrator_name(vocalized) == vocalized

    # --- RECOVERY: "<real name> رضى الله عنه <matn tail>" recovers the leading name
    # rather than being dropped (precision-first: ابي هريره = Abū Hurayra is a real,
    # heavily-cited narrator — the mc-1215 evidence bullet). The alif-maqsura
    # honorific variant is scrubbed, then the ان matn tail is truncated. ---
    def test_real_lead_before_matn_recovered(self) -> None:
        assert clean_narrator_name(normalize_arabic("ابي هريره رضى الله عنه ان")) == "ابي هريره"

    # --- precision / recall over a hand-labelled sample (issue acceptance) ---
    # 24 matn/commentary + 34 real names. Precision MUST be 1.0 (never drop a real
    # name); recall is high. This is the guardrail against a future tweak that
    # over-drops. Kept in-test so it runs in CI, not just the PR body.
    _PR_SAMPLE: tuple[tuple[str, bool], ...] = (
        ("قلت لابي عبد الله عليه السلام ذهبت ارمي فاذا في يدي ست حصيات", True),
        ("صلى النبي صلى الله عليه وسلم الظهر خمسا فقيل له ازيد في الصلاه", True),
        ("نعم هو بالاراك بعرفه يرعى ابل القوم فركب عمر", True),
        ("ابو عيسى هذا حديث حسن صحيح", True),
        ("نهى رسول الله صلى الله عليه وسلم ان نستقبل القبلتين ببول او غاءط", True),
        ("كنا مع النبي في سفر فنزلنا منزلا", True),
        ("قال رسول الله من كذب علي متعمدا فليتبوا مقعده من النار", True),
        ("قلت يا رسول الله اي الاعمال افضل", True),
        ("هذا حديث غريب لا نعرفه الا من هذا الوجه", True),
        ("امر النبي بقتل الاسودين في الصلاه", True),
        ("رايت النبي يمسح على الخفين", True),
        ("بينما نحن جلوس عند النبي اذ جاء رجل", True),
        ("سمعت النبي يقول انما الاعمال بالنيات", True),
        ("كان اذا دخل العشر شد مءزره", True),
        ("حديث حسن صحيح لا باس به", True),
        ("فقال له عمر ما هذا يا رسول الله", True),
        ("لما نزلت هذه الايه دعا النبي", True),
        ("قلنا يا رسول الله كيف نصلي عليك", True),
        ("نعم", True),
        ("هذا حديث صحيح على شرط الشيخين", True),
        ("صلى بنا رسول الله الظهر ثم انصرف", True),
        ("والله لقد رايت رسول الله يفعل ذلك", True),
        ("بلى قد سمعنا ما قلت يا رسول الله", True),
        ("قالت عاءشه رضى الله عنها كان النبي", True),
        ("محمد بن مسلم بن عبيد الله بن عبد الله بن شهاب بن عبد الله بن الحارث بن زهره", False),
        ("عبد الله بن عبد الرحمن المداءني", False),
        ("عطاء بن يزيد الليثي ثم الجندعي", False),
        ("ابو عبد الله الشامي", False),
        ("ابو بكر بن ابي شيبه", False),
        ("محمد بن اسماعيل البخاري", False),
        ("ابو هريره", False),
        ("عبد الله بن عباس", False),
        ("سليمان بن مهران الاعمش", False),
        ("ابو اسحاق السبيعي", False),
        ("عثمان بن عفان", False),
        ("عبد الله بن عمر", False),
        ("يحيى بن معين", False),
        ("احمد بن حنبل", False),
        ("مالك بن انس", False),
        ("سفيان بن عيينه", False),
        ("ابو الحسن علي بن احمد بن عبد العزيز الجرجاني", False),
        ("شعبه بن الحجاج", False),
        ("عبد الرحمن بن مهدي", False),
        ("وكيع بن الجراح", False),
        ("حماد بن سلمه", False),
        ("عبد الله بن المبارك", False),
        ("ابو داود الطيالسي", False),
        ("قتيبه بن سعيد", False),
        ("اسماعيل بن ابراهيم بن مقسم الاسدي", False),
        ("عبد الرزاق بن همام الصنعاني", False),
        ("ابو بكر الصديق", False),
        ("جابر بن عبد الله", False),
        ("انس بن مالك", False),
        ("ابو صالح ذكوان السمان", False),
        ("معمر بن راشد", False),
        ("عبد الله بن عبد الرحمن الدارمي", False),
        ("محمد بن يحيى الذهلي", False),
        ("ابو القاسم عبد الرحمن بن عبد الله بن احمد بن محمد بن عبيد بن عبد الملك", False),
        # connector-less real names (nasab == 0) — mononyms + nisba/laqab-only
        ("قتاده", False),
        ("نافع", False),
        ("عكرمه", False),
        ("مجاهد", False),
        ("طاوس", False),
        ("الاعمش", False),
        ("الاوزاعي", False),
        ("الزهري", False),
        ("سفيان الثوري", False),
        ("حماد الكوفي", False),
        ("سعيد المقبري", False),
        # real name + truncated-off matn tail carrying ؟ / «» (Nikolaos #310) —
        # recovered as the real name, must not drop
        ("عبد الرحمن الاوزاعي قال سمعت النبي يقول كذا؟", False),
        ("عبد الله الاعمش قال كان النبي يصلي؟", False),
        ("سليمان الاعمش الكوفي قال كان هو يصلي؟", False),
        ("عبد الرحمن الاوزاعي الدمشقي؟", False),
        ("يحيى القطان قال «حدثنا»", False),
        ("مالك الاشتر قال «الحق»", False),
    )

    # --- GATING regression (Nikolaos, #310): a real name followed by a truncated-off
    # matn tail carrying ؟ / «» must NOT be dropped. The da#258 truncation recovers
    # the leading name; the matn-punctuation signal must be evaluated over the KEPT
    # span, not the pre-truncation text. Each must survive as its real name. ---
    @pytest.mark.parametrize(
        "polluted,expected",
        [
            ("عبد الرحمن الاوزاعي قال سمعت النبي يقول كذا؟", "عبد الرحمن الاوزاعي"),
            ("عبد الله الاعمش قال كان النبي يصلي؟", "عبد الله الاعمش"),
            ("سليمان الاعمش الكوفي قال كان هو يصلي؟", "سليمان الاعمش الكوفي"),
            ("عبد الله الاعمش قال هو؟", "عبد الله الاعمش"),
            # no isnad boundary at all — a stray trailing ؟ on a clean 4-token name
            ("عبد الرحمن الاوزاعي الدمشقي؟", "عبد الرحمن الاوزاعي الدمشقي"),
            # quoted-speech matn in the truncated-off tail
            ("يحيى القطان قال «حدثنا»", "يحيى القطان"),
            ("مالك الاشتر قال «الحق»", "مالك الاشتر"),
        ],
    )
    def test_truncated_matn_tail_punct_does_not_drop_name(
        self, polluted: str, expected: str
    ) -> None:
        assert clean_narrator_name(normalize_arabic(polluted)) == expected

    # --- PRECISION (Kavitha #2): connector-less real names (nasab == 0) — mononyms
    # and nisba/laqab-only forms — are NOT spared by the nasab guard, so they must
    # survive on their own (no weak matn signal fires). ---
    @pytest.mark.parametrize(
        "name",
        [
            "قتاده",
            "نافع",
            "عكرمه",
            "مجاهد",
            "طاوس",
            "الاعمش",
            "الاوزاعي",
            "الزهري",
            "سفيان الثوري",
            "حماد الكوفي",
            "سعيد المقبري",
        ],
    )
    def test_connectorless_real_name_kept(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)

    # --- PRECISION (Kavitha #4): grading formulae are token-anchored, so "حسن صحيح"
    # inside a real name token ("الحسن" + "صحيح") does NOT fire the grading signal. ---
    def test_grading_formula_token_anchored(self) -> None:
        # "الحسن صحيح النسب" — الحسن (al-Ḥasan) is a real name token, not the grading
        # word حسن; the substring حسن صحيح must not drop it.
        assert clean_narrator_name(normalize_arabic("علي بن الحسن الهمداني")) == normalize_arabic(
            "علي بن الحسن الهمداني"
        )

    def test_precision_recall_over_labelled_sample(self) -> None:
        tp = fp = fn = 0
        for span, is_matn in self._PR_SAMPLE:
            dropped = clean_narrator_name(normalize_arabic(span)) is None
            if is_matn and dropped:
                tp += 1
            elif is_matn and not dropped:
                fn += 1
            elif not is_matn and dropped:
                fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        assert fp == 0, f"dropped {fp} real name(s) — precision {precision:.3f}"
        assert recall >= 0.9, f"recall too low: {recall:.3f} ({fn} matn kept)"


class TestShiaDialogueMatnResidual:
    """da#311 — _MATN_OPENERS missed dominant Shia dialogue-hadith matn shapes.

    Run 5 left residual benediction/matn contamination in narrators_canonical,
    concentrated in thaqalayn/fawaz; itqan/lk were clean. Root causes: missing
    dialogue/isnad openers (``سألت``/``سُئل``/``روى``/``كتبت``, dual ``قالا``,
    و/ف-prefixed said, bare ``سمع``), a Prophet-*apposition* the gate left whole
    (``"عاءشه زوج النبي"`` — the mc-196 fawaz canonical, its trailing قالت +
    benediction already stripped at NER time), a vocative ``يا`` leader, and a
    short-residue interaction with the taṣliya ligature ``ﷺ``.

    Where the real narrator is RECOVERABLE (a leading name before an apposition /
    speech verb) the gate now CLEANS to that name rather than dropping the row —
    losing a real, heavily-cited companion (Aisha, mc-196) to a hard drop is worse
    than recovering her (coordinator directive: prefer clean over drop).
    """

    # --- DROP: issue examples where NO real narrator leads the matn (the speaker
    # is an elided "I", or the span is a bare Prophet action) → nothing to recover.
    @pytest.mark.parametrize(
        "matn",
        [
            # mc 149, fawaz — سألت "I asked" opener; the narrator is the elided
            # asker, Abu al-Hasan is the askee → no leading name to recover
            "سألت أبا الحسن (عليه السلام",
            # mc 49, fawaz — short residue: taṣliya ligature ﷺ inflated the
            # token count and masked the 2-token "نهى النبي" residue
            "نهى النبي ﷺ",
            # mc 82, thaqalayn — سُئل "was asked" opener leads → drop
            "سُئل أبو عبد الله (عليه السلام",
        ],
    )
    def test_issue_evidence_examples_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- RECOVER: the mc-196 subject-led / Prophet-apposition case cleans to the
    # real leading narrator (Aisha), NOT a hard drop — her mentions attach to the
    # real node instead of being lost.
    @pytest.mark.parametrize(
        "matn,expected",
        [
            # mc 196, fawaz — "Aisha, wife of the Prophet, said" → Aisha
            ("عائشة، زوج النبي ﷺ قالت", "عاءشه"),
            # the trimmed apposition as actually stored in the canonical (no verb)
            ("عاءشه زوج النبي", "عاءشه"),
            # daughter-of / wife-of the Prophet appositions + a trailing verb
            ("زينب بنت النبي حدثت", "زينب"),
            ("فاطمه بنت النبي روت", "فاطمه"),
            # رسول + الله (not just النبي/الرسول) as the apposition's Prophet ref
            ("هند زوجه رسول الله قالت", "هند"),
            # apposition with no trailing verb — still recovers the bare name
            ("ام سلمه زوج النبي", "ام سلمه"),
        ],
    )
    def test_subject_led_apposition_recovers_name(self, matn: str, expected: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) == normalize_arabic(expected)

    # --- DROP: a bare apposition with NO leading name ("wife of the Prophet") ---
    @pytest.mark.parametrize("matn", ["زوج النبي", "بنت رسول الله"])
    def test_bare_apposition_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- Recall shapes added in da#311's second pass (real-data survivors) ---
    @pytest.mark.parametrize(
        "matn,expected",
        [
            # compound co-narrator join + dual "they said" → preserve BOTH
            # co-narrators, strip only the trailing قالا (never drop the names)
            ("عمرو الناقد وزهير بن حرب قالا", "عمرو الناقد وزهير بن حرب"),
            ("ابو الطاهر وحرمله قالا", "ابو الطاهر وحرمله"),
            # bare سمع / trailing روى → recover the leading narrator
            ("ابو ضمره سمع عمر بن عبد العزيز روى", "ابو ضمره"),
            # "that he/she …" subordinator (ه/ها-suffixed ان the bare ان misses)
            ("عاءشه انها", "عاءشه"),
            ("ابي جعفر عليه السلام انه سءل", "ابي جعفر"),
            # trailing 3rd-person "narrated"
            ("ابو تمي حدث", "ابو تمي"),
        ],
    )
    def test_recall_shapes_recover_lead(self, matn: str, expected: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) == normalize_arabic(expected)

    # --- DROP: leading vocative / و-prefixed said / كتبت correspondence opener ---
    @pytest.mark.parametrize(
        "matn",
        [
            "يا رسول الله",  # vocative address, never a name
            "يا محمد",
            "وقال ابن رافع",  # "and Ibn Rafi said" — leading و+said
            "كتبت الى ابي الحسن اساله",  # "I wrote to Abu al-Hasan asking him"
        ],
    )
    def test_leading_vocative_and_prefixed_verbs_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- DROP: short residue — verb-opener + single noun, no ligature involved ---
    @pytest.mark.parametrize(
        "matn",
        [
            "نهى النبي",  # 2-token residue, no ﷺ ligature present at all
            "امر النبي",  # another _MATN_OPENERS verb + single-noun residue
        ],
    )
    def test_short_residue_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- PRECISION: a genuine 2-token kunya must survive the short-residue fix ---
    @pytest.mark.parametrize(
        "real_name",
        ["ابو هريره", "ابو بكر", "ابن عمر"],
    )
    def test_two_token_kunya_survives_short_residue_fix(self, real_name: str) -> None:
        assert clean_narrator_name(normalize_arabic(real_name)) == normalize_arabic(real_name)

    # --- KEEP: the al-Zuhri full nasab chain, plain kunyas, and an Imam-honorific
    # name whose leading TITLE strips but whose real name survives ---
    def test_long_nasab_chain_preserved(self) -> None:
        name = normalize_arabic(
            "محمد بن مسلم بن عبيد الله بن عبد الله بن شهاب بن عبد الله بن الحارث بن زهره"
        )
        assert clean_narrator_name(name) == name

    def test_abu_hurayra_preserved(self) -> None:
        assert clean_narrator_name(normalize_arabic("أبو هريرة")) == normalize_arabic("أبو هريرة")

    def test_anas_bin_malik_preserved(self) -> None:
        assert clean_narrator_name(normalize_arabic("أنس بن مالك")) == normalize_arabic(
            "أنس بن مالك"
        )

    def test_imam_honorific_title_stripped_name_kept(self) -> None:
        # "الشيخ الجليل" ("the venerable Sheikh") is the common Imami honorific
        # title for Ibn Babawayh (Shaykh al-Saduq) — only the TITLE strips, the
        # real name underneath must survive, not be dropped.
        polluted = normalize_arabic("الشيخ الجليل أبو جعفر محمد بن علي بن الحسين بن بابويه")
        expected = normalize_arabic("أبو جعفر محمد بن علي بن الحسين بن بابويه")
        assert clean_narrator_name(polluted) == expected

    # --- NON-REGRESSION (Nikolaos #310): a real name + قال + a matn tail
    # containing a Prophet reference (النبي) must still recover the leading
    # name — the subject-led signal only fires when the Prophet reference
    # survives INTO the retained span (i.e. appears BEFORE the truncation
    # boundary), never on content already truncated away. ---
    @pytest.mark.parametrize(
        "polluted,expected",
        [
            ("عبد الرحمن الاوزاعي قال سمعت النبي يقول كذا؟", "عبد الرحمن الاوزاعي"),
            ("عبد الله الاعمش قال كان النبي يصلي؟", "عبد الله الاعمش"),
        ],
    )
    def test_prophet_reference_in_truncated_tail_does_not_drop_name(
        self, polluted: str, expected: str
    ) -> None:
        assert clean_narrator_name(normalize_arabic(polluted)) == expected


class TestAskVerbAndDegenerate:
    """da#311 (third pass) — "asked" verb openers + degenerate-result floor."""

    # --- DROP: every "asked" conjugation (the mc-1035 سالته family) ---
    @pytest.mark.parametrize(
        "matn",
        [
            "سالته",  # I asked him (mc-1035)
            "وسالته",  # and I asked him (mc-96)
            "سالتها",
            "سالت ابا جعفر",  # I asked Abu Jaʿfar (mc-389)
            "سءلت",
            "سءل ابو عبد الله",
            "ساله عن كذا",  # he asked him about …
        ],
    )
    def test_ask_verb_forms_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- PRECISION: the name سالم (Salim) and its nasab/nisba forms MUST survive
    # (سالم = mc-4072; the ask-verb stem سال collides with it) ---
    @pytest.mark.parametrize(
        "name",
        [
            "سالم",
            "سالم بن عبد الله",
            "سالم بن عبد الله العمري",
            "سالم بن ابي حفصه",
            "سالم البراد",
            "سالم ابو حماد",
        ],
    )
    def test_salim_family_preserved(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)

    # --- DROP: a recovery that boils down to garbage (degenerate floor) ---
    @pytest.mark.parametrize("junk", ["ه", "و", "ف", "p", "‏", ".", "، ،"])
    def test_degenerate_result_dropped(self, junk: str) -> None:
        assert clean_narrator_name(normalize_arabic(junk)) is None

    # --- KEEP: a legit 2+-letter Arabic name and a romanized name clear the floor ---
    def test_short_real_name_and_romanized_kept(self) -> None:
        assert clean_narrator_name(normalize_arabic("علي")) == normalize_arabic("علي")
        assert clean_narrator_name("Malik") == "Malik"


class TestCleanNarratorNameDisplay:
    """da#311 (third pass) — clean the VOWELED, load-bearing name_ar field.

    The graph loader keys the node name on name_ar first (falling back to
    name_ar_normalized only when name_ar is empty), so the scrub must clean
    name_ar. name_ar also often carries the FULL matn body while
    name_ar_normalized was already truncated at NER time.
    """

    # --- clean names pass through with their DIACRITICS PRESERVED ---
    @pytest.mark.parametrize(
        "name_ar",
        [
            "أبو عائشة",
            "عائشة بنت سعد",
            "محمد بن اسماعيل البخاري",
            "سالم بن عبد الله",
        ],
    )
    def test_clean_voweled_name_preserved(self, name_ar: str) -> None:
        assert clean_narrator_name_display(name_ar) == name_ar

    # --- matn in the voweled field is recovered / dropped ---
    def test_apposition_recovered_from_voweled_matn(self) -> None:
        # the flagship mc-196 row: full voweled matn in name_ar → recover Aisha
        polluted = "عاءشه، زوج النبي صلى الله عليه وسلم قالت"
        assert clean_narrator_name_display(polluted) == "عاءشه"

    def test_full_matn_body_dropped(self) -> None:
        # name_ar holds the WHOLE hadith body (name_ar_normalized was pre-truncated)
        body = (
            "اثنى رجل على رجل عند النبي صلى الله عليه وسلم فقال ويلك قطعت عنق صاحبك "
            "قطعت عنق صاحبك مرارا ثم"
        )
        assert clean_narrator_name_display(body) is None

    def test_ask_verb_dropped_in_display(self) -> None:
        assert clean_narrator_name_display("سالته") is None
        assert clean_narrator_name_display("سالت ابا جعفر") is None

    def test_leading_title_stripped_preserving_voweling(self) -> None:
        polluted = "الشيخ الجليل أبو جعفر محمد بن علي بن الحسين بن بابويه"
        assert clean_narrator_name_display(polluted) == ("أبو جعفر محمد بن علي بن الحسين بن بابويه")

    def test_dual_benediction_leaves_no_dangling_token(self) -> None:
        # da#471: the dual-pronoun benediction "رضي الله عنهما" (after a PAIR of
        # Companions) was only prefix-matched by "رضي الله عنه", leaving a dangling
        # "ا"/"ما" fragment ("عبد الله بن عمر ا"). Now stripped whole — the display
        # is the clean voweled name, and the normalized cleaner drops it too.
        display = clean_narrator_name_display("عَبْدُ اللَّهِ بْنُ عُمَر رضي الله عنهما")
        assert display == "عَبْدُ اللَّهِ بْنُ عُمَر"
        assert clean_narrator_name(normalize_arabic("عبد الله بن عمر رضي الله عنهما")) == (
            "عبد الله بن عمر"
        )
        # alif-maqsura spelling of the dual form is stripped the same way.
        assert clean_narrator_name(normalize_arabic("عبد الله بن عمر رضى الله عنهما")) == (
            "عبد الله بن عمر"
        )

    @pytest.mark.parametrize("empty", [None, "", "   ", "<NAR>"])
    def test_empty_display_returns_none(self, empty: str | None) -> None:
        assert clean_narrator_name_display(empty) is None


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


class TestNumericPrefixRecovery:
    """da#311 (fourth pass) — recover real narrators hidden behind a leaked ordinal.

    A thaqalayn enumeration digit leaks in front of the name field ("1 علي بن
    ابراهيم", "10 علي بن ابراهيم", …). An earlier pass DROPPED any digit-leading
    span, which erased transmitters whose ONLY stored forms are numbered — Ali ibn
    Ibrahim al-Qummi (al-Kafi's most prolific isnad head) vanished from the graph
    entirely (0 rows). The gate now STRIPS the leading ordinal (+ a leaked waw) and
    re-gates the remainder: a real name is recovered, a mubham / isnad-formula /
    matn remainder still drops.
    """

    # --- RECOVER: a real narrator behind one or more leading ordinals ---
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1 علي بن ابراهيم", "علي بن ابراهيم"),  # Ali ibn Ibrahim al-Qummi (was 0 rows)
            ("10 علي بن ابراهيم", "علي بن ابراهيم"),
            ("2 محمد بن يحيى", "محمد بن يحيى"),
            ("1 - علي بن ابراهيم", "علي بن ابراهيم"),  # dash-separated ordinal variant
        ],
    )
    def test_leading_ordinal_recovered(self, raw: str, expected: str) -> None:
        assert clean_narrator_name(normalize_arabic(raw)) == normalize_arabic(expected)

    # --- DROP: an ordinal in front of a non-name remainder stays dropped ---
    @pytest.mark.parametrize(
        "junk",
        [
            "1 عده من اصحابنا",  # mubham collective ("a number of our companions")
            "5659 و روى محمد بن ابي عمير",  # leaked-waw isnad continuation
            "2 و عنه",  # "and from him"
            "3 و باسناده",  # "and by his chain"
            "3 باسناده",
            "16 و باسناده",
            "5659",  # bare ordinal, nothing behind it
        ],
    )
    def test_leading_ordinal_nonname_dropped(self, junk: str) -> None:
        assert clean_narrator_name(normalize_arabic(junk)) is None

    # --- PRECISION: a MEDIAL waw compound and real names that merely CONTAIN an
    # isnad-formula substring (معن ⊃ عنه, النعمان ⊃ نعم) are untouched ---
    @pytest.mark.parametrize(
        "name",
        ["محمد و علي جميعا", "معن بن عيسى", "النعمان بن الزبير", "ابو النعمان"],
    )
    def test_no_false_drop_from_substring_or_medial_waw(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)


class TestIsnadResidueFloor:
    """da#311 round-4b (Kavitha's review) — all-residue spans are not narrators.

    The round-4 numeric-strip newly admitted isnad/matn residue that the base gate
    never caught: the attached-waw continuation forms ("وعنه"/"وباسناده"), the
    passive narrate verb "روي" (distinct token from روى — normalize_arabic does not
    fold ى→ي), the particle "قد"/"قد روي", and bare mubham collectives WITHOUT the
    partitive "من" that guard 5 requires ("رجل" = "a man" was a mc-4920 pollution
    node). Rule 5b drops a span whose every token is such residue.
    """

    # --- DROP: attached-waw isnad fragments (numeric-derived and bare) ---
    @pytest.mark.parametrize("junk", ["2 وعنه", "1 وباسناده", "وعنه", "وباسناده"])
    def test_attached_waw_isnad_fragment_dropped(self, junk: str) -> None:
        assert clean_narrator_name(normalize_arabic(junk)) is None

    # --- DROP: the passive verb روي as a lone residue (was 541-mention junk) ---
    @pytest.mark.parametrize("junk", ["روي", "3 روي", "1 روي عن فلان", "قد روي", "1 قد روي"])
    def test_passive_narrate_verb_dropped(self, junk: str) -> None:
        assert clean_narrator_name(normalize_arabic(junk)) is None

    # --- DROP: bare mubham collectives without the partitive من (guard-5 gap) ---
    @pytest.mark.parametrize("junk", ["رجل", "بعض", "شيخ", "ناس", "نفر", "قوم", "جماعه", "1 جماعه"])
    def test_bare_mubham_collective_dropped(self, junk: str) -> None:
        assert clean_narrator_name(normalize_arabic(junk)) is None

    # --- PRECISION: real waw-initial names and mixed spans with a real token survive
    # (residue drop requires EVERY token to be residue) ---
    @pytest.mark.parametrize(
        "name",
        ["وهب بن منبه", "وكيع بن الجراح", "وائل بن حجر"],
    )
    def test_waw_initial_real_name_survives(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)


class TestMatnSentenceVerseGate:
    """da#317 — matn-sentence / Qur'anic-verse pollution in the zero-degree tail.

    The da#308/da#311 gate caught verb-led, grading-formula, apposition, and
    ask-verb matn but MISSED three classes that on the #723 reload made up ~26% of
    the 150,187 canonical "narrators" (all zero-degree orphans, id-ordered row 1 is
    Qur'an 22:32):

    (a) **conjunction-initial narrative** — a ``و…``/``ف…`` proclitic opener the
        opener check did not fold ("فقلت لو كنت …", "وتهيات للحج …");
    (b) **divine-name / Qur'anic-verse signature** — the bare divine name الله
        occurring twice+ ("الله ومن يعظم شعاءر الله فانها من تقوى القلوب");
    (c) **short function-word-only fragment** — a bare particle span ("وما" =
        "and not").

    The gate decides purely from the NAME TEXT (no graph / degree input — it runs at
    NER time). All new rules run AFTER the nasab-connector precision guard, so a
    genuine lineage / kunya is spared before any of them can fire.
    """

    # --- DROP: the four canonical issue examples (all normalized here) ---
    @pytest.mark.parametrize(
        "matn",
        [
            # (b) Qur'an 22:32 — the id-ordered /narrators row 1 (repeated الله)
            "الله ومن يعظم شعائر الله فإنها من تقوى القلوب",
            # (a) conjunction-initial narrative, dense in function words
            "وتهيأت للحج وودعت الناس وكنت على الخروج فورد نحن لذلك كارهون والأمر إليك",
            # (a) ف-proclitic matn opener ("so I said …")
            "فقلت لو كنت هناك ما قتلتهم",
            # (c) short function-word-only fragment
            "وما",
        ],
    )
    def test_issue_examples_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- DROP: further matn/verse shapes of the three classes ---
    @pytest.mark.parametrize(
        "matn",
        [
            # divine-name repetition (verse / oath phraseology)
            "ان الله يامر بالعدل والاحسان وايتاء ذي القربى",
            "الله لا اله الا هو الحي القيوم",
            # conjunction-initial narrative with ≥2 particles
            "فلما راى ذلك قال لهم انا معكم",
            "وكان ذلك في يوم من الايام على عهد النبي",
            # function-word-dense matn, no verb opener
            "هذا هو الذي كان على ذلك",
            # bare/short function-word fragments (proclitic folded)
            "على",
            "فلو",
            "ما",
        ],
    )
    def test_additional_matn_verse_classes_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- KEEP (recall guard): real transmitters MUST survive unchanged. The
    # coordinator's explicit recall set — long nasab, al-Zuhri, Abu Hurayra, the
    # al-Qummi numeric form, theophoric الله compounds, and mononyms/nisbas. ---
    @pytest.mark.parametrize(
        "name",
        [
            # long nasab chain (≥2 connectors → spared) — the issue's recall example
            "شاكر بن أبي بكر أحمد بن محمد الحريمي الخياط",
            "محمد بن مسلم بن شهاب الزهري",  # al-Zuhri
            "أبو هريرة",  # Abu Hurayra (kunya, nasab density spares it)
            # theophoric compounds carry الله exactly ONCE → not the verse signal
            "عبد الله بن عمر",
            "عبيد الله بن عبد الله بن عتبة",
            "هبة الله بن نصر",  # الله as second theophoric element, preceded by هبة
            "عطاء الله الشيرازي",
            # a two-theophoric nasab (الله twice) is spared by the nasab guard first
            "عبد الله بن عبيد الله العمري",
            # connector-less mononyms / nisba-only names (nasab == 0) survive on merit
            "قتاده",
            "مالك",
            "الزهري",
            "سفيان الثوري",
            "الاوزاعي",
        ],
    )
    def test_recall_guard_real_names_preserved(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)

    # --- RECALL (coordinator audit): a real DOUBLE-theophoric narrator must KEEP.
    # The divine-name rule (bare الله ≥2 → drop) must NOT count an الله that is the
    # second half of a theophoric compound (عبد الله, هبة الله, سعد الله). Adding
    # legitimate nisbas dilutes the nasab density below the 0.15 spare threshold, so
    # before this fix "عبد الله بن عبد الله القرشي الاموي" false-dropped while the
    # shorter "عبد الله بن عبد الله" (density 0.2) survived — a perverse regression
    # where more real name-material caused a drop. All of these must survive. ---
    @pytest.mark.parametrize(
        "name",
        [
            "عبد الله بن عبد الله",
            "عبد الله بن عبد الله القرشي الأموي",
            "عبد الله بن عبد الله الرازي قاضي الري",
            "عبد الله بن هبة الله الحلي البزاز",
            "عبد الله بن عبد الله الشنتريني الزاهد",
            "عبد الله بن عبد الله التجيبي القرطبي",
            "عبد الله بن سعد الله الشيخ ضياء الدين القرمي",
            "عبيد الله بن عبد الله بن عتبة بن مسعود",
            "أمة الله بنت عبد الرحمن",
        ],
    )
    def test_double_theophoric_narrator_preserved(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is not None

    # --- the verse detection MUST still fire after the theophoric exclusion: its two
    # الله are sentence-initial + شعائر الله (شعائر is NOT a theophoric head). ---
    def test_verse_still_dropped_after_theophoric_fix(self) -> None:
        verse = "الله ومن يعظم شعائر الله فإنها من تقوى القلوب"
        assert clean_narrator_name(normalize_arabic(verse)) is None

    # --- PRECISION: the al-Qummi numeric-prefixed form still recovers (da#311 3b
    # non-regression) — the leading ordinal strips and the real name survives. ---
    def test_qummi_numeric_form_recovered(self) -> None:
        assert clean_narrator_name(normalize_arabic("1 علي بن ابراهيم")) == normalize_arabic(
            "علي بن ابراهيم"
        )

    # --- PRECISION: على (preposition, alif-maqsura) must not be confused with علي
    # (the name ʿAlī, ya-final) — normalize_arabic does not fold ى→ي, so ʿAlī and
    # every nasab built on it survive while the preposition particle rule fires. ---
    @pytest.mark.parametrize(
        "name",
        ["علي", "علي بن ابي طالب", "علي بن الحسين", "علي بن ابراهيم"],
    )
    def test_ali_not_confused_with_preposition(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) == normalize_arabic(name)

    # --- PRECISION: a lone particle count of ONE does not drop an otherwise real
    # name (the density threshold is 2), and و/ف-initial real names are untouched. ---
    @pytest.mark.parametrize(
        "name",
        ["وهب بن منبه", "وكيع بن الجراح", "فروة بن مسيك", "فاطمه بنت سعد"],
    )
    def test_single_particle_and_conjunction_names_preserved(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is not None


class TestRlmShieldedEdgePunct:
    """da#316 — U+200F (RLM) interleaved with trailing punctuation on the RAW field.

    MECHANISM (verified at HEAD; the issue's stated one is STALE). The issue blamed
    ``normalize_arabic`` for not stripping U+200F and ``clean_narrator_name``'s
    ``token.strip(_EDGE_PUNCT)`` for being shielded by it. Both are FALSE at HEAD:
    ``normalize_arabic`` HAS stripped bidi/zero-width marks since da#271
    (``strip_format_marks``), so the NORMALIZED path is already clean — see
    :meth:`test_normalized_path_was_never_the_defect`.

    The live defect is in the **display** path. ``strip_markup`` /
    ``clean_narrator_name_display`` strip ``_EDGE_PUNCT`` from the *voweled* token —
    a string ``normalize_arabic`` deliberately never touches (it must keep the
    diacritics) — and ``_EDGE_PUNCT`` carried no format marks, so a trailing RLM
    standing BETWEEN the marks halted the peel before ``.`` and ``،`` were reached.
    ``name_ar`` is the load-bearing field the graph loader keys the node ``name`` on,
    so the cruft rode all the way into the graph.

    Fixtures are verbatim rows from ``data/curated.run5-scrubbed/narrators_canonical
    .parquet`` (``name_ar``), U+200F written as an explicit escape so the placement
    cannot be mangled by an editor.
    """

    # Real corpus rows: ism + "،" + RLM + "." + RLM.
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("قتاده،‏.‏", "قتاده"),  # mc-6
            ("يونس،‏.‏", "يونس"),  # mc-5
            ("الزهري،‏.‏", "الزهري"),  # mc-2
        ],
    )
    def test_rlm_shielded_punct_peeled_from_display(self, raw: str, expected: str) -> None:
        assert clean_narrator_name_display(raw) == expected

    def test_strip_markup_peels_rlm_shielded_punct(self) -> None:
        assert strip_markup("قتاده،‏.‏") == "قتاده"

    # The dangling-waw variant: the source stores a co-narrator join whose second
    # member was never captured ("مالك،‏.‏ و" — mc-4).
    def test_dangling_waw_after_rlm_punct_is_popped(self) -> None:
        assert clean_narrator_name_display("مالك،‏.‏ و") == "مالك"

    # Format marks appear INTERIOR to a span too, not only at token edges, so an
    # edge-strip alone cannot clear them — strip_markup removes them wholesale.
    def test_interior_format_marks_removed(self) -> None:
        assert "‏" not in strip_markup("جابر بن عبد الله في قوله تعالى‏:‏")

    # --- The issue's STATED mechanism, pinned as false: the normalized path was
    # already correct at HEAD (da#271). This test passes BEFORE the fix too; it
    # exists so a future reader does not "re-fix" normalize_arabic/_EDGE_PUNCT in
    # the normalized path and think that closed da#316. ---
    def test_normalized_path_was_never_the_defect(self) -> None:
        assert clean_narrator_name(normalize_arabic("قتاده،‏.‏")) == "قتاده"

    # --- PRECISION: the cleaner must be a FIXED POINT on the recovered display
    # name (no cruft left to re-peel on a second pass). ---
    @pytest.mark.parametrize("raw", ["قتاده،‏.‏", "مالك،‏.‏ و"])
    def test_display_clean_is_idempotent(self, raw: str) -> None:
        once = clean_narrator_name_display(raw)
        assert once is not None
        assert clean_narrator_name_display(once) == once

    # --- PRECISION: a clean voweled name keeps its diacritics (strip_format_marks
    # removes bidi/zero-width controls ONLY — it must not denature the display). ---
    def test_diacritics_preserved(self) -> None:
        assert clean_narrator_name_display("أَبُو هُرَيْرَة") == "أَبُو هُرَيْرَة"


class TestBiPronounBackreference:
    """da#315 — the bi+pronoun isnad back-reference (``وبه`` / ``به``).

    MECHANISM (verified at HEAD; the issue is HALF STALE). The issue reported two
    classes — ``ما`` (30 rows / 195 mentions) and ``وبه`` (2 rows / 18 mentions) —
    and proposed adding ``ما`` to the particle drop vocab. **``ما`` was already fixed**
    by da#317's all-function-word rule (2c), which landed after the issue was filed;
    only the ``وبه`` class is live. Measured at HEAD the surviving class is 13 rows /
    18 mentions, all ``وبه``/``به``.

    ``_is_isnad_residue_token`` already de-waws (``وعنه`` → ``عنه``), but ``وبه``
    de-waws to ``به``, which was not in ``_ISNAD_FORMULA_FRAGMENTS`` — so the
    all-residue floor (rule 5b) never fired on it.
    """

    # Verbatim corpus rows (name_ar == 'وبه', 13 rows on the run-5 scrubbed set).
    @pytest.mark.parametrize("residue", ["وبه", "به", "فيه", "منه", "نحوه", "مثله"])
    def test_bi_pronoun_backreference_dropped(self, residue: str) -> None:
        assert clean_narrator_name(normalize_arabic(residue)) is None

    # --- The issue's OTHER half, pinned as already-fixed: ``ما`` drops at HEAD via
    # da#317 rule (2c). Green before the fix; recorded so the stale half is not
    # "re-fixed" and so a regression in da#317 surfaces here. ---
    def test_ma_particle_already_dropped_by_da317(self) -> None:
        assert clean_narrator_name(normalize_arabic("ما")) is None

    # --- PRECISION DUAL: the de-waw fold must not eat a real waw-initial name, and
    # rule 5b only drops when EVERY token is residue, so a real multi-token name
    # containing one of these as a component survives. ---
    @pytest.mark.parametrize(
        "name",
        ["وهب", "وكيع", "وهب بن منبه", "وكيع بن الجراح", "به بن سالم"],
    )
    def test_real_names_not_eaten_by_residue_floor(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is not None


class TestMatnProvenanceAndOath:
    """da#314 — narration-provenance tails (``… من رسول الله``) and oath formulae.

    MECHANISM (verified at HEAD). Both existing Prophet-reference detectors miss this
    class: ``_is_prophet_reference`` (step 6b) is LEADER-anchored, but here the title
    sits in the TAIL behind a preposition; and signal (2p) of ``_is_matn_sentence`` is
    gated on ``was_truncated``, but these spans carry no isnad/matn verb, so nothing
    ever truncates and they are never re-scrutinized.

    The discriminator against a REAL name+title is the preposition: ``محمد رسول الله``
    (Muhammad, the Messenger of Allah — a real narrator, the muhaddithat fixture)
    carries ZERO matn particles, while a provenance fragment always carries the very
    preposition that makes it provenance (``من`` / ``كمن`` / a demonstrative).

    The oath formula (``والذي نفس محمد بيده``) escaped separately: ``الذي`` IS a
    boundary token, but the corpus stacks proclitics onto it (``والذي``, ``فوالذي``,
    ``فبالذي``) and the boundary check matched only the bare form.

    Fixtures are verbatim ``name_ar`` rows from the run-5 scrubbed canonical set.
    """

    @pytest.mark.parametrize(
        "provenance",
        [
            "من رسول الله",  # mc-38
            "ه من رسول الله",  # mc-55
            "هذا من رسول الله",  # mc-18
            "ه من النبي",  # mc-7
        ],
    )
    def test_prophet_provenance_dropped(self, provenance: str) -> None:
        assert clean_narrator_name(normalize_arabic(provenance)) is None

    # ACCEPTED RESIDUAL, deliberately asserted as KEPT (da#424).
    #
    # "كمن زار رسول الله" ("like one who visited the Messenger of Allah", mc-8) carries a
    # substantive verb, زار. The provenance rule is scoped all-residue precisely so it
    # can NEVER reach a span with substantive content — that scoping is what stops it
    # deleting rijal biographies ("وفد على رسول الله" = "he came as a delegate to the
    # Messenger of Allah" is the SAME shape, and it names a real companion). A rule that
    # drops this one would delete those, and we have already made that mistake twice.
    #
    # So this span stays, at no-worse-than-main (main keeps it too). Recovering it needs
    # truncate-and-re-gate, which is da#424's job. Asserted explicitly rather than left
    # silent so the trade is visible and a future edit cannot quietly "fix" it by
    # loosening the scoping.
    def test_verb_bearing_provenance_is_an_accepted_residual(self) -> None:
        assert clean_narrator_name(normalize_arabic("كمن زار رسول الله")) is not None

    @pytest.mark.parametrize(
        "oath",
        [
            "والذي نفس محمد بيده",  # mc-7
            "فوالذي نفس محمد بيده اني لارجو",  # mc-2, stacked ف+و proclitics
            "فبالذي ارسلك الله امرك بهذا",  # mc-11, stacked ف+ب proclitics
            "فانشدك بالله هل تعلم",  # mc-3, adjuration verb
        ],
    )
    def test_oath_formula_dropped(self, oath: str) -> None:
        assert clean_narrator_name(normalize_arabic(oath)) is None

    # --- PRECISION DUAL #1 (load-bearing): the Prophet's own name+title is a REAL
    # narrator (muhaddithat `variousnarrators.csv` row 1) and carries zero particles,
    # so the provenance rule must NOT fire on it. ---
    def test_prophet_own_name_and_title_preserved(self) -> None:
        assert clean_narrator_name(normalize_arabic("محمد رسول الله")) == normalize_arabic(
            "محمد رسول الله"
        )

    # --- PRECISION DUAL #2: the proclitic fold is restricted to the definite relative
    # pronouns precisely because folding it across the whole boundary set false-drops
    # real name material — "وان بن سالم" (mc-251, a بن lineage) would go via وان → ان.
    # This test is the guard on that restriction. ---
    @pytest.mark.parametrize("name", ["وان بن سالم", "فان", "فاذا"])
    def test_short_boundary_words_not_proclitic_folded(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is not None

    # --- PRECISION DUAL #3: a real nasab/kunya carrying a Prophet-adjacent component
    # or a single particle is untouched. ---
    @pytest.mark.parametrize(
        "name",
        ["عاءشه بنت سعد", "عبد الله بن عبد الله القرشي الاموي", "ابو هريره", "زكريا"],
    )
    def test_real_lineages_preserved(self, name: str) -> None:
        assert clean_narrator_name(normalize_arabic(name)) is not None


class TestScrubDoesNotDeleteRealNarrators:
    """The reverse direction of the drop-gate diff (Sofia Cardoso, PR #423 review).

    A drop-gate change has TWO failure directions, and a one-directional A/B — "which
    rows does this newly drop?" — is structurally blind to both of the ones that matter:

    1. it can **delete a real narrator**, and
    2. it can **stop dropping** a matn span, resurrecting pollution as a narrator node.

    Both happened. These tests pin both directions with verbatim corpus rows.
    """

    # === DIRECTION 1: deletion of a real transmitter ===
    #
    # "أم" (umm — the commonest female kunya connector) normalizes to "ام", which is
    # HOMOGRAPHIC with the disjunctive particle "أم" ("or") and is therefore a member of
    # _MATN_PARTICLES. It is the ONLY token that is both an apposition connector and a
    # matn particle (pinned below). The da#314 provenance rule's discriminator — "a real
    # narrator carries zero matn particles" — is simply FALSE in Arabic because of it, so
    # any narrator shaped "أم X <role> رسول الله" read as Prophet-reference + particle
    # and was deleted.
    #
    # Verbatim row, nar:d3937f2e-e077-598c-b139-9392d2c61359 (sanadset, mention_count 0):
    # Umm Kulthum bint Muhammad — the Prophet's own DAUGHTER. mention_count is 0 because
    # she is bio-promoted, which is exactly why a mention-weighted sweep could not see the
    # deletion: the loss is a NODE, not an edge.
    def test_umm_kulthum_bint_muhammad_survives(self) -> None:
        row = "Umm Kulthum bint Muhammad أم كلثوم بنت سيد البشر رسول الله"
        assert clean_narrator_name(normalize_arabic(row)) is not None

    def test_umm_homograph_is_the_reason(self) -> None:
        """`ام` is both an apposition connector and a matn particle — the sole such token.

        If a future edit adds another homograph to _MATN_PARTICLES that is also an
        apposition connector, the provenance rule silently regains a deletion surface.
        This pins the collision that makes the exclusion in rule (2d) necessary.
        """
        from src.parse.name_quality import _APPOSITION_CONNECTORS, _MATN_PARTICLES

        assert _APPOSITION_CONNECTORS & _MATN_PARTICLES == frozenset({"ام"})

    # PRECISION: the exclusion must not spare the mubham collectives the rule targets —
    # "بعض" is not an apposition connector, so these still drop. Verbatim voweled row
    # (nar:2bb390c0-..., mc-47).
    @pytest.mark.parametrize(
        "mubham",
        ["بَعْضِ أَصْحَابِ النَّبِيِّ", "بعض ازواج النبي", "من رسول الله", "هذا من رسول الله"],
    )
    def test_mubham_and_provenance_still_drop(self, mubham: str) -> None:
        assert clean_narrator_name(normalize_arabic(mubham)) is None

    # === DIRECTION 2: resurrecting matn as a narrator node ===
    #
    # _truncate_at_isnad_boundary has DUAL semantics: a boundary at index 0 DROPS the
    # span; a boundary at index > 0 TRUNCATES and keeps the head. The oath is a matn
    # SIGNAL, not a name TERMINATOR — and in the corpus it is overwhelmingly MID-matn.
    # Folding the proclitic at i > 0 therefore turned the oath into a truncation point
    # and handed the matn head back as a narrator name, minting 52 new zero-degree matn
    # nodes ("دعوني" = "leave me", "اعملوا وابشروا" = "act and rejoice").
    #
    # Every oath fixture in TestMatnProvenanceAndOath LEADS with the oath, so they only
    # ever exercised the drop path — which is exactly how this hid. These are verbatim
    # MID-matn corpus rows: they must DROP as the matn bodies they are, and in particular
    # must NOT come back as the truncated head.
    @pytest.mark.parametrize(
        "matn",
        [
            # nar:2324c241-..., mc-2 — the pushed fold returned "دعوني". Kept on ONE line
            # and noqa'd rather than wrapped: the fixture's value is that it is the corpus
            # row BYTE-FOR-BYTE, and implicit concatenation is an opportunity to mangle it.
            "دعوني فالذي انا فيه خير اوصيكم بثلاث اخرجوا المشركين من جزيره العرب واجيزوا الوفد بنحو ما كنت اجيزهم",  # noqa: E501
            # nar:1efadf02-..., mc-3 — the pushed fold returned "سريه تغزو في سبيل الله"
            "سريه تغزو في سبيل الله والذي نفسي بيده لوددت اني اقتل في سبيل الله",
        ],
    )
    def test_mid_matn_oath_drops_and_does_not_truncate_to_a_head(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # The oath-LEADING fixtures must still drop — the fold is retained at index 0.
    @pytest.mark.parametrize(
        "oath",
        ["والذي نفس محمد بيده", "فوالذي نفس محمد بيده اني لارجو", "فبالذي ارسلك الله امرك بهذا"],
    )
    def test_leading_oath_still_drops(self, oath: str) -> None:
        assert clean_narrator_name(normalize_arabic(oath)) is None


class TestBioPromotedNarratorsSurvive:
    """The provenance rule must not delete rijal BIOGRAPHY entries (2nd review round).

    A companion's biography mentions the Prophet BY NATURE — "شهد النبي في حجة الوداع"
    ("he witnessed the Prophet at the Farewell Pilgrimage"), "وفد على رسول الله" ("he
    came as a delegate to the Messenger of Allah"), "خليفه رسول الله" ("Successor of the
    Messenger of Allah"). Every one of those is a Prophet reference plus a preposition,
    so an `any()`-shaped provenance rule reads a real transmitter's bio as a matn
    fragment and deletes him.

    That is what happened: 44 real bio-promoted narrators were dropped, including **Abu
    Bakr al-Siddiq**, **al-Tayyib (the Prophet's son)** and **Dhu al-Yadayn**. Like Umm
    Kulthum, all are `mention_count = 0` — 45,613 of the 150,187 canonical rows are — so
    a mention-weighted A/B assigns them weight zero and CANNOT see the deletion. This is
    the same premise error as the `أم` homograph, one layer deeper: it is not only kunya
    connectors, it is that bio prose about a companion is indistinguishable from
    provenance under a "Prophet-ref + particle" discriminator.

    The rule is now scoped all-residue (every token must be a function word or part of
    the Prophet reference), mirroring rule 5b — the only shape in this module proven safe
    — so a span carrying ANY substantive token is structurally out of reach.

    Fixtures are verbatim `name_ar` rows (itqan / sanadset, all mention_count 0).
    """

    @pytest.mark.parametrize(
        "bio",
        [
            # Abu Bakr al-Siddiq, the first Caliph. Note the TRAILING DANGLING WAW: the
            # waw-pop must not set was_truncated, or signal (2p) fires on the "رسول الله"
            # in "خليفه رسول الله" and deletes him.
            "ابو بكر الصديق عبدالله بن ابي قحافه عثمان التيمي خليفه رسول الله و",
            "الطيب ولد رسول الله",  # al-Tayyib, the Prophet's son
            "ذواليدين صلي مع النبي",  # Dhu al-Yadayn (sanadset)
            "الحارث بن عمرو السهمي الباهلي شهد النبي في حجة الوداع عداده ف",
            "ضمرة بن سعد السلمي شهد مع النبي حنينا",
            "رغبان مولى حبيب بن مسلمة الفهري رأى أصحاب النبي يركعون ركعتين قبل المغرب",
            "أبو صفية رجل من المهاجرين من 3 أصحاب النبي",
            "زرارة بن عمرو النخعي وفد إلى رسول الله",
        ],
    )
    def test_bio_promoted_companion_survives(self, bio: str) -> None:
        assert clean_narrator_name(normalize_arabic(bio)) is not None

    # PRECISION: the all-residue scoping must still drop the pure fragments — every
    # token a function word or part of the Prophet reference, no naming content at all.
    @pytest.mark.parametrize(
        "fragment",
        [
            "من رسول الله",  # mc-38
            "ه من رسول الله",  # mc-55
            "ها من رسول الله",  # mc-7  (bare pronoun, same class as the 1-char "ه")
            "هن من رسول الله",  # mc-5
            "هذا من رسول الله",  # mc-18
            "ه من النبي",  # mc-7
            "بعض اصحاب النبي",  # mc-47 — mubham collective, must NOT be spared
            "بعض ازواج النبي",  # mc-23
        ],
    )
    def test_pure_provenance_fragment_still_drops(self, fragment: str) -> None:
        assert clean_narrator_name(normalize_arabic(fragment)) is None


class TestCleanerRemovedContentContract:
    """Pin `cleaner_removed_content`'s docstring CONTRACT (da#398).

    da#398 corrected the docstring headline: a ``True`` means the cleaner "removed
    more than an affix", NOT that it "cut into the name the source asserted." These
    tests hold that contract against regression — the boolean is deliberately COARSE
    and, in particular, returns ``True`` for BOTH a *cut-into-the-name* residue (a bare
    ism) AND a *tail-after-a-complete-name* residue (a full nasab that lost only an
    isnad tail). A caller keying identity on the cleaner's output must not read ``True``
    as "the residue is not the asserted name" — for the tail class, it IS.

    The finer split those two classes need is a SEPARATE predicate
    (`is_recoverable_gloss_tail`, da#397); this boolean answers only the affix/removal
    question, and this contract keeps `asserted_name_tokens` from quietly absorbing one
    of the cleaner's cuts (which would evaporate the refusals it carries against a green
    suite).
    """

    # Affix-only normalization → the asserted name is left WHOLE → False.
    @pytest.mark.parametrize(
        "raw",
        [
            "( 4875 ) عبد الله بن موسى",  # bracketed catalog entry number
            "[ 969 ] احمد بن يوسف الثعلبي",  # square-bracket entry number
            "مالك يعني بن انس",  # editorial connective يعني
            "مالك بن انس رضي الله عنه",  # honorific suffix
        ],
    )
    def test_affix_normalization_is_not_removed_content(self, raw: str) -> None:
        normalized = normalize_arabic(raw)
        cleaned = clean_narrator_name(normalized)
        assert cleaned is not None, raw
        assert not cleaner_removed_content(normalized, cleaned), (
            f"normalization read as a cut: {raw}"
        )
        # The affix/removal line is drawn against the cleaner's OWN pre-truncation
        # tokens, not a re-tokenization of the raw input.
        assert tuple(cleaned.split()) == asserted_name_tokens(normalized)

    def test_true_does_not_imply_cut_into_the_name(self) -> None:
        """The load-bearing da#398 contract: ``True`` covers a cut-into-name AND a tail-cut.

        Both a *cut-into-the-name* (bare-ism residue) and a *tail-after-a-complete-name*
        (full-nasab residue that lost only a teacher-key isnad tail) return ``True`` — so
        ``True`` alone cannot be read as "the residue is not the name the source
        asserted." The tail-class residue below is a complete nasab.
        """
        # cut INTO the name: `عبيده` is a bare ism, a fragment of the asserted span.
        cut_into = normalize_arabic("عبيدة مولى رسول الله ذكره بن شاهين واستدركه أبو موسى")
        cut_into_cleaned = clean_narrator_name(cut_into)
        assert cut_into_cleaned == "عبيده"
        assert cleaner_removed_content(cut_into, cut_into_cleaned)

        # tail AFTER a complete name: the full nasab is intact; only `عن انس` was cut.
        tail_cut = normalize_arabic("اسحاق بن مره عن انس")
        tail_cut_cleaned = clean_narrator_name(tail_cut)
        assert tail_cut_cleaned == "اسحاق بن مره"
        assert cleaner_removed_content(tail_cut, tail_cut_cleaned)

        # The contract: same ``True``, categorically different residues. The tail-class
        # residue is a COMPLETE nasab (>= 3 tokens carrying بن) — a real asserted name —
        # while the cut-into-name residue is a bare ism. `True` does not distinguish them.
        assert len(tail_cut_cleaned.split()) >= 3 and "بن" in tail_cut_cleaned.split()
        assert len(cut_into_cleaned.split()) == 1

    def test_removed_content_is_a_strict_shortening(self) -> None:
        """Whenever the predicate is True, the residue is strictly shorter than asserted.

        The dual framing of the contract: a cut removes tokens, so a ``True`` residue is
        a proper prefix-length-reduction of what the source asserted — never equal, never
        longer. This is what makes a truncation visible where a normalization is not.
        """
        for raw in ("اسحاق بن مره عن انس", "عبيدة مولى رسول الله ذكره بن شاهين واستدركه أبو موسى"):
            normalized = normalize_arabic(raw)
            cleaned = clean_narrator_name(normalized)
            assert cleaned is not None, raw
            assert cleaner_removed_content(normalized, cleaned), raw
            assert len(cleaned.split()) < len(asserted_name_tokens(normalized)), raw
