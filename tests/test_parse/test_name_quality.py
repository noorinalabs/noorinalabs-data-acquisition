"""Tests for narrator name-quality cleaning + validation (da#247)."""

from __future__ import annotations

import pytest

from src.parse.name_quality import (
    clean_narrator_name,
    split_compound_narrators,
    strip_markup,
)
from src.utils.arabic import normalize_arabic


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

    Run 5 left ~15% benediction/matn contamination in narrators_canonical,
    concentrated in thaqalayn (28.3%) and fawaz (18.8%); itqan/lk were clean. Root
    cause: three gaps in the da#308 gate — missing dialogue/isnad openers
    (``سألت``/``سُئل``/``روى``), a subject-led matn shape the gate only checked
    ``bare[0]`` for, and a short-residue interaction with the taṣliya ligature ``ﷺ``.
    """

    # --- DROP: the issue's 4 canonical mention-backed contamination examples ---
    @pytest.mark.parametrize(
        "matn",
        [
            # mc 196, fawaz — subject-led: name + appositive + trailing قالت,
            # recovered name itself carries a bare Prophet reference (النبي)
            "عائشة، زوج النبي ﷺ قالت",
            # mc 149, fawaz — سألت "I asked" opener (missing from _MATN_OPENERS)
            "سألت أبا الحسن (عليه السلام",
            # mc 49, fawaz — short residue: taṣliya ligature ﷺ inflated the
            # token count and masked the 2-token "نهى النبي" residue
            "نهى النبي ﷺ",
            # mc 82, thaqalayn — سُئل "was asked" opener (missing)
            "سُئل أبو عبد الله (عليه السلام",
        ],
    )
    def test_issue_evidence_examples_dropped(self, matn: str) -> None:
        assert clean_narrator_name(normalize_arabic(matn)) is None

    # --- DROP: additional subject-led matn (2nd-position speech verb) ---
    @pytest.mark.parametrize(
        "matn",
        [
            # بنت النبي ("daughter of the Prophet") appositive + حدثت, the
            # حدث/حدثت sibling of the قال/قالت subject-led pattern above
            "زينب بنت النبي حدثت",
            "فاطمه بنت النبي روت",
            # رسول + الله (not just النبي/الرسول) surviving into the retained span
            "هند زوجه رسول الله قالت",
        ],
    )
    def test_subject_led_matn_dropped(self, matn: str) -> None:
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
