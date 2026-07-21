"""Narrator name-quality cleaning + validation (da#247).

Removes the non-name pollution classes that leak into the canonical narrator
table — markup tags, honorific/eulogy phrases, mubham (anonymous) collective
descriptors, and over-long phrase/text spans — recovering the real name where
possible and rejecting the span otherwise. Applied at the NER stage over *every*
source (``src/resolve/ner.py``) so the pollution never reaches disambiguation.

Background (da#247): on the pre-fix resolve output 39.9% of canonical narrators
(115,405 / 289,385) had polluted names — three root causes:

1. ``<NAR>`` / ``<IDF>`` markup leakage (sanadset) — literal, often unclosed /
   nested tags left in the name (``"<NAR> ابو عبيده"``). The existing
   ``_is_narrator_like`` filter missed them because ``is_arabic`` accepts *mixed*
   Latin+Arabic. Stripped here (recovers ``"ابو عبيده"``), not dropped.
2. thaqalayn parser dumping whole hadith bodies into the name field — hundreds of
   tokens of Arabic + English text. Caught by the token-count cap.
2b. English-only sources putting translated isnad PROSE in the name field — a
   "name" that is really a sentence fragment (``"It was narrated…"``, ``"The
   Prophet said…"``, ``"his father"``). Caught by an English non-name leader-word
   guard; romanized real names (``"Abu Huraira"``, ``"Al-Zuhri"``) are kept.
2c. English ``<name>:<matn>`` colon-joins (da#253) — a companion name colon-joined
   to a hadith body (``"Thawban:The Messenger of Allah (ﷺ) sacrificed during a
   journey and then…"``). This defeats BOTH 2b guards: the leader guard only
   inspects ``tokens[0]`` (here ``"Thawban:The"`` — the colon is *internal*, so
   ``"thawban:the"`` is not a leader), and the ~11-token body sits under the token
   cap. Handled two ways: (i) truncate the span at the first colon whose tail
   begins with an English leader — recovering the pre-colon name (``"Thawban"``);
   (ii) a residual embedded-prose guard drops any span carrying two or more
   whole-token English function/stop words (real romanized names carry zero;
   Arabic script never matches an ASCII word).
3. mubham (unnamed) descriptors minted as named narrators — both *collective*
   phrases (``"رجل من اصحاب النبي"``, ``"جماعه من اصحاب"``) and bare *relational
   pronouns* (``"ابيه"`` "his father", ``"جده"`` "his grandfather"), plus bare
   honorific phrases. The relational-pronoun sub-class was the residual after the
   first da#247 pass: ``ابيه`` alone remained the single most-mentioned "narrator"
   (65,755) on the #723 stage reload because it has no partitive ``من`` for the
   collective guard to catch.

The Arabic-script residual (da#258) — the direct sibling of da#253's English
``<name>:<matn>`` fix — is three more classes that leak *Arabic* isnad/matn text
into the name field, none catchable by a token-count cap (real ``ibn``-lineage
names reach ~30 tokens):

4. **Genuine matn body** — a whole sentence dumped in the name field
   (``"كان علي بن ابي طالب بالكوفه في الجامع اذ قام اليه رجل …"`` "Ali ibn Abi
   Talib was in Kufa in the mosque when a man stood …"). Signalled by Arabic
   matn/verb function words (``كان``/``قال``/``اذ``/``الذي``/``لما``/…), NOT by
   length. Dropped — but only when no real leading name precedes the matn.
5. **Chain-connective fragments** — a real name followed by an isnad connective
   that opens the *next* link: ``"ابو عبد الرحمن عن ابي هريره"`` ("Abu Abd
   al-Rahman FROM Abu Huraira"), ``"سالم بن عبد الله ان عبد الله بن عمر"``,
   ``"ابو العنبس الذي …"``, ``"بندار هو محمد بن بشار"`` (an editorial gloss).
   Truncated at the connective, keeping the leading name (recovers ``"بندار"`` =
   Ibn Bashshar, ``"عاءشه"`` from ``"عاءشه قالت كان"``) — the Arabic mirror of
   da#253's :func:`_truncate_colon_prose`.
6. **Compound co-narrator joins** — multiple REAL narrators joined by ``و … جميعا``
   / ``X و Y و Z قالوا`` ("… together" / "… they said"): ``"سعد بن عبد الله و عبد
   الله بن جعفر الحميري جميعا"``. These must be **split into separate narrators,
   not dropped** — losing the co-narrators (and their mention edges) is a
   regression. :func:`split_compound_narrators` is the detection+split primitive;
   emitting one mention row per split member is a resolve-stage (``ner.py``) change
   tracked as a scoped follow-up, so :func:`clean_narrator_name` deliberately does
   **not** drop an un-split compound (no matn token fires on a name-only ``و``-join).

Classes 4+5 unify into a single **truncate-at-first-isnad-boundary** rule
(:data:`_ISNAD_BOUNDARY`): cut the span at the first matn-verb / connective /
relative-pronoun / gloss token and keep the preceding real name; if that boundary
token *leads* the span (no name before it) the span is dropped (the real narrator
sits after an elided-grammar verb and is unrecoverable here). A residual
matn-density backstop (:data:`_MATN_DENSITY`, ≥2) drops any sentence whose first
boundary is a non-verb preposition the truncation cannot anchor on.

Benediction / Prophet-reference residual (da#271) — two more classes surfaced by
the candidate-pair-inflation audit, both concentrated in thaqalayn/lk:

7. **Bare Shia taṣliya** — ``صلى الله عليه واله`` (without the trailing ``وسلم`` the
   existing honorific set required), which left "رسول الله صلى الله عليه واله" (an
   mc-365 false narrator) intact. Added to :data:`_HONORIFIC_PHRASES`.
8. **Prophet references** — after the taṣliya is stripped the residue is a bare
   title (``رسول الله``, ``النبي``, ``نبي الله``): the Prophet is the matn source,
   not an isnad narrator. Dropped by :func:`_is_prophet_reference` (leader-anchored).

Sentence / matn-body drop-gate (da#308) — the residual after da#258. The
boundary-truncation (classes 4+5) only fires on the specific verb / connective /
gloss tokens in :data:`_ISNAD_BOUNDARY`; a matn sentence whose opener is NOT one
of those (``قلت`` "I said", ``كنا`` "we were", ``نهى`` "forbade", ``صلى`` "prayed",
``امر`` "ordered"), a grading-commentary formula (``حديث حسن صحيح``, ``هذا حديث``),
or a quoted-speech / question span (``«…»`` / ``؟``) slips through and survives as
a canonical narrator — on run-4b ~15% (25,971) of canonical names were such matn.

9. **Matn / commentary sentence** — a whole hadith body or grading gloss kept as a
   name (``"قلت لابي عبد الله ذهبت ارمي فاذا في يدي ست حصيات"``,
   ``"ابو عيسى هذا حديث حسن صحيح"``). Dropped by :func:`_is_matn_sentence`, which
   combines matn signals (verb-led opener, grading formula, matn punctuation, and a
   verb/particle density backstop) — a bare token-count ceiling is deliberately
   NOT a signal on its own. The **precision guard** is nasab-connector density
   (:data:`_NASAB_CONNECTORS`, measured DIACRITICS-INSENSITIVELY so a vocalized
   ``بْنُ`` counts): a genuine ``ibn``-lineage / kunya-nasab string is spared from
   the length / punctuation / density signals regardless of how long it is, so real
   nasab chains, nisba tails, and kunya compounds pass through unchanged. A
   nisba-transition ``ثم`` between two ``ال``-nisbas (``"…الليثي ثم الجندعي"``) is
   likewise no longer truncated as a matn boundary.

Shia dialogue-hadith matn residual (da#311) — the da#308 gate was tuned on
Sunni/Tirmidhī matn shapes and missed the dominant Shia dialogue-hadith form
(thaqalayn 28.3%, fawaz 18.8% of canonical still matn on the run-5 audit, vs
0.3% for itqan/lk):

10. **Missing dialogue/isnad openers** — ``سألت``/``سُئل`` ("I asked" / "was
    asked", the ubiquitous Shia dialogue opener) and ``روى`` ("narrated") were
    absent from :data:`_MATN_OPENERS`. Added in normalized form (``سالت``,
    ``سءل`` — the alif/hamza normalization of the raw hamza forms).
11. **Subject-led matn** — ``"عاءشه زوج النبي قالت"`` ("Aisha, wife of the
    Prophet, said …") leads with a real name, so the da#258 boundary
    truncation at ``قالت`` recovers ``"عاءشه زوج النبي"`` as if it were a
    name — but the recovered span itself carries a bare Prophet/Messenger
    reference (``النبي``/``الرسول``/``رسول``+``الله``/``نبي``+``الله``), which
    a real person's name never embeds at any token position. :func:`_is_matn_sentence`
    now checks for this anywhere in the retained tokens (not just as the
    leader, unlike da#271's :func:`_is_prophet_reference`), before the
    nasab-density guard runs (an idafa appositive like ``زوج``/``بنت`` "wife
    of" / "daughter of" otherwise carries a nasab connector that would spare
    it). ``حدث``/``حدثت`` (3rd-person "narrated") were also added to
    :data:`_ISNAD_BOUNDARY` / :data:`_MATN_DENSITY` alongside ``قال``/``قالت``
    so the same class of construction truncates consistently.
12. **Short residue** — the taṣliya *ligature* ``ﷺ`` (U+FDFA, distinct from the
    spelled-out phrase already stripped) survived :func:`~src.utils.arabic.normalize_arabic`
    untouched and, left in place, could mask a short residue by inflating the
    token count; conversely once stripped a verb-opener + single-noun residue
    (``"نهى النبي"`` after the ligature is gone) fell under
    :data:`_MATN_SENTENCE_MIN_TOKENS` and escaped even though ``نهى`` IS a
    matn opener. The ligature is now stripped alongside the spelled-out
    honorific (:data:`_HONORIFIC_PHRASES`), and the opener check's 2-token
    case is handled explicitly (reusing the nasab guard so a genuine 2-token
    kunya, e.g. ``"ابو هريره"``, is never at risk — no kunya opener is ever a
    member of :data:`_MATN_OPENERS`).
13. **Honorific title prefix** — a leading scholarly epithet (e.g. the Imami
    title for Ibn Babawayh, ``"الشيخ الجليل ابو جعفر …"``) is now stripped by
    :data:`_HONORIFIC_TITLE_PHRASES`, recovering the real name rather than
    leaving it title-polluted (it was never dropped, since ``الشيخ`` with the
    definite article never matched the bare ``شيخ`` in :data:`_MUBHAM_LEADERS`).

Truncate-and-re-gate recovery (da#424) — the remedy for the da#314 residual. The
matn-sentence gate (class 9) has two failure modes on a ``<real name> + <matn/bio
tail>`` span whose tail carries no :data:`_ISNAD_BOUNDARY` verb (so class 5's
truncation never fires): it either DROPS the whole span — deleting the real leading
narrator — or KEEPS it whole, matn and all. #423's provenance rule was deliberately
scoped all-residue precisely so it could never do the former to a rijal biography,
which left the residual for here.

14. **Leading-name recovery** — when a span leads with a real nasab/kunya name and
    the remainder is matn or provenance, :func:`_recover_leading_name` cuts at the
    name/matn boundary (:func:`_leading_name_before_matn`) and re-gates the head
    through :func:`clean_narrator_name`, recovering it: ``"ابو هريره فذهب رسول الله
    وانتم تنتثلونها"`` → ``"ابو هريره"``; ``"ابن عباس …"`` / ``"ابو القاسم …"`` leading
    a matn quote are recovered rather than dropped whole. It is a pure RECOVERY —
    it only ever replaces a *drop* or a *matn-polluted whole span* with a cleaner
    leading name, and NEVER turns a surviving name into a drop (no-worse-than-main).
    The recovered head MUST carry a :data:`_NASAB_CONNECTORS` token, which is what
    keeps a bare matn head (``"دعوني"``, ``"سريه تغزو …"`` — no nasab) from being
    resurrected as a narrator; and the ``ف``-narrative boundary (``فذهب`` "so he
    went") fires only when a matn signal follows it, so a real ``ف``-initial ism
    (``فراس``) with no matn after it is never cut. It does NOT loosen #423's
    all-residue provenance scoping — a substantive-but-matn span like ``"كمن زار
    رسول الله"`` (verb ``زار``) has no nasab lead, so it is left KEPT, exactly as
    #423 requires.

The phrase constants are in **normalized-Arabic** form (post
``normalize_arabic``) because this runs on ``name_normalized``.
"""

from __future__ import annotations

import re

from src.utils.arabic import normalize_arabic, strip_format_marks

__all__ = [
    "clean_narrator_name",
    "clean_narrator_name_display",
    "asserted_name_tokens",
    "cleaner_removed_content",
    "is_recoverable_gloss_tail",
    "is_mubham_relational",
    "split_compound_narrators",
    "strip_markup",
]

# Angle-bracket markup that leaks from source isnad fields (sanadset <NAR>/<IDF>
# tags, frequently unclosed or nested). Stripped, not rejected — the tag usually
# wraps a real name. The trailing ``.replace`` mops up any stray lone bracket so
# no markup character can survive into a canonical name (da#247 acceptance).
_MARKUP_RE = re.compile(r"<[^>]*>")

# Latin / English benediction abbreviations (da#299). English-gloss sources store
# a romanized name carrying a parenthesized benediction — "Prophet Muhammad(saw)",
# "Abu Hurayra (r.a.)", "Ali (as)". normalize_arabic leaves Latin untouched, so
# the Arabic honorific phrases never catch these. Matched ONLY when parenthesized
# or bracketed, so a bare "as"/"ra"/"sa" substring INSIDE a real romanized name is
# never touched (precision-first). Covers the dotted forms ("s.a.w.", "r.a.").
_LATIN_BENEDICTION_RE = re.compile(
    r"[(\[]\s*(?:s\.?a\.?w\.?w?|p\.?b\.?u\.?h|r\.?a\.?a?|r\.?a\.?h|a\.?s|s\.?w\.?t|r\.?t\.?a)\s*\.?\s*[)\]]",
    re.IGNORECASE,
)

# Leading Latin honorific TITLE words (da#299) prefixed to a romanized name in the
# same English-gloss sources — "Prophet Muhammad", "Hazrat Ali", "Sayyidina …".
# Stripped only when they LEAD the span (a title, not a name component); a name
# that merely CONTAINS one of these tokens mid-string is untouched. Closed set of
# unambiguous honorific prefixes — precision-first.
_LATIN_TITLE_RE = re.compile(
    r"^\s*(?:prophet|hazrat|hadhrat|sayyidina|sayyiduna|sayyidna|imam)\s+",
    re.IGNORECASE,
)

# Editorial connective ("that is / i.e.") — never part of a name.
_CONNECTIVE_TOKENS = frozenset({"يعني"})

# Bare waw conjunction ("و" = "and"). MEDIAL it joins compound co-narrators
# ("X و Y"); LEADING (only reachable after a numeric mis-parse strip, da#311) it
# is an isnad continuation fragment ("و عنه", "و روى …"), never a name start.
_WAW_CONJUNCTION = normalize_arabic("و")

# Isnad-formula continuation words (normalized) — "by his/the chain", "from him",
# "was narrated" fragments a numeric mis-parse can leave as the WHOLE remainder
# ("3 و باسناده" → "باسناده", "1 روي عن فلان" → "روي", "1 قد روي" → "قد روي").
# Structural isnad/matn vocabulary (same class as _ISNAD_BOUNDARY / _MATN_OPENERS),
# never a narrator name. Only consulted inside the numeric-mis-parse recovery (via
# :func:`_is_isnad_residue_token`) so ordinary spans are untouched (da#311 round-4b
# — Kavitha's review caught attached-waw "وعنه"/"وباسناده", the passive verb "روي",
# the particle "قد", and bare mubham collectives slipping the round-4 digit-strip).
# da#315 adds the PREPOSITION+PRONOUN back-reference forms — the classical isnad
# shorthand a compiler uses to chain a narration onto the preceding one instead of
# repeating it: "وبه" / "به" ("and by it / by it [same chain]"), "نحوه" ("the like of
# it"), "مثله" ("similar to it"), "منه", "فيه". The de-waw fold below already reached
# "وعنه" → "عنه", but "وبه" folds to "به" — which was NOT in this set, so the whole
# class survived the all-residue floor (13 rows / 18 mentions on the run-5 scrubbed
# set). Every one is a preposition bound to a resumptive pronoun; none is, or contains,
# a proper ism, and rule 5b only drops a span when EVERY token is residue, so a real
# multi-token name is untouched.
_ISNAD_FORMULA_FRAGMENTS = frozenset(
    normalize_arabic(w)
    for w in (
        "باسناده",
        "بإسناده",
        "بالاسناد",
        "اسناده",
        "عنه",
        "روي",
        "روى",
        "قد",
        "به",
        "فيه",
        "منه",
        "نحوه",
        "مثله",
    )
)


def _is_isnad_residue_token(token: str) -> bool:
    """True when *token* is a bare isnad/matn function word, not a name component.

    Folds a LEADING attached-waw proclitic ("وعنه" → "عنه", "وباسناده" → "باسناده")
    before matching, so the و-prefixed continuation forms a numeric mis-parse leaves
    behind ("2 وعنه", "1 وباسناده") are recognised — while a real waw-initial name
    ("وهب" → "هب", "وكيع" → "كيع") folds to a non-residue stem and is NOT matched.
    Also treats a bare mubham collective ("جماعه", "قوم" — a group/people, minted
    without the partitive "من" that guard 5 needs) as residue. da#311 round-4b.
    """
    norm = normalize_arabic(token)
    dewawed = norm[1:] if norm.startswith("و") and len(norm) > 1 else norm
    return (
        norm in _ISNAD_FORMULA_FRAGMENTS
        or dewawed in _ISNAD_FORMULA_FRAGMENTS
        or norm in _MUBHAM_LEADERS
    )


# Honorific / eulogy phrases (normalized) that name no narrator. Stripped from
# anywhere in the span (they appear appended to real names too).
_HONORIFIC_PHRASES: tuple[str, ...] = (
    "صلى الله عليه واله وسلم",
    "صلى الله عليه وسلم",
    # Shia taṣliya without the trailing وسلم (da#271): the dominant benediction in
    # thaqalayn/lk — the bare "صلى الله عليه واله" (and its spaced "و اله" variant)
    # was NOT covered by the two forms above, so "رسول الله صلى الله عليه واله" (a
    # mc-365 false narrator) survived. Listed AFTER the longer وسلم-suffixed form so
    # that one is stripped first where present.
    "صلى الله عليه واله",
    "صلى الله عليه و اله",
    "عليه السلام",
    "عليهم السلام",
    "عليها السلام",
    "رضي الله عنه",
    "رضي الله عنها",
    "رضي الله عنهم",
    # DUAL pronoun form (da#471): "رضي الله عنهما" ("… with both of them"), the
    # standard benediction after a PAIR of Companions ("عبد الله بن عمر رضي الله
    # عنهما"). It was absent, so the strip matched only the "رضي الله عنه" PREFIX
    # and left a dangling "ما"/"ا" fragment — harmless while name_ar_normalized was
    # the only cleaner output, but da#301 surfaces the cleaner on the DISPLAY name,
    # where "عبد الله بن عمر ا" would become a Narrator label. Listed here (both ي
    # and ى spellings); the LONGEST-FIRST strip loop removes it before the shorter
    # عنه/عنها/عنهم forms, so the whole dual benediction goes cleanly.
    "رضي الله عنهما",
    # alif-maqsura spelling (da#308): normalize_arabic does NOT fold ى→ي, so the
    # very common "رضى الله عنه" survives normalization distinct from the ي form
    # above and left an un-scrubbed benediction tail (e.g. "ابي هريره رضى الله عنه
    # ان" — mc-1215). Listed so the residue is stripped and the real leading name
    # ("ابي هريره") is recovered rather than kept honorific-polluted or dropped.
    "رضى الله عنه",
    "رضى الله عنها",
    "رضى الله عنهم",
    "رضى الله عنهما",  # da#471 dual form, alif-maqsura spelling (see ي form above)
    # bare / truncated benediction fragments (da#311) — the source often stores a
    # benediction cut off before its pronoun tail ("… رضي الله", "… صلى الله عليه"
    # with no عنه/وسلم). Stripped LONGEST-FIRST (see the strip loop) so the full
    # "رضي الله عنه" / "صلى الله عليه وسلم" forms above are removed first where
    # present; the bare forms then catch the truncated residue. Neither "رضي الله"
    # ("God is pleased") nor "صلى الله عليه" is ever part of a real narrator name.
    "صلى الله عليه",
    "رضي الله",
    "رضى الله",
    "رحمه الله",
    "عز وجل",
    "تبارك وتعالى",
    # Urdu-folded taṣliya (da#299): normalize_arabic now folds the Urdu yeh ی
    # (U+06CC) to the Arabic yeh ي, so a benediction typed in Urdu script —
    # "صلی اللہ علیہ وسلم" — normalizes to "صلي الله عليه وسلم" (regular yeh),
    # NOT the alif-maqsura "صلى …" spelling the Arabic-sourced forms above carry.
    # Same asymmetry the da#308 رضى/رضي pair already handles: list the yeh spelling
    # so an Urdu-script name folds to the SAME benediction-stripped canonical form
    # as its Arabic twin. Longest-first (the strip loop) so the وسلم/واله forms win
    # before the bare "صلي الله عليه".
    "صلي الله عليه واله وسلم",
    "صلي الله عليه وسلم",
    "صلي الله عليه واله",
    "صلي الله عليه و اله",
    "صلي الله عليه",
    # taṣliya LIGATURE (da#311): the single-codepoint ﷺ (U+FDFA, "ARABIC LIGATURE
    # SALLALLAHOU ALAYHE WASALLAM") is the same benediction as the spelled-out
    # "صلى الله عليه وسلم" above but survives normalize_arabic untouched (it is
    # not diacritics/alif/hamza/taa-marbuta, so none of that pipeline's steps
    # touch it). Left unstripped it counts as an extra token and can push a
    # short matn residue ("نهى النبي ﷺ") at-or-above the sentence-gate's minimum
    # token floor by accident, masking the real (shorter) residue underneath.
    # Stripped here exactly like the phrase it is a ligature FOR.
    "ﷺ",
)

# Leading honorific TITLE phrases (distinct from the eulogy/benediction phrases
# above) — a scholarly epithet prefixed to a real name (da#311), e.g. the common
# Imami title for Ibn Babawayh (Shaykh al-Saduq): "الشيخ الجليل ابو جعفر ...".
# Stripped the same way (anywhere-in-span replace) so the real name underneath is
# recovered, not dropped and not left title-polluted.
_HONORIFIC_TITLE_PHRASES: tuple[str, ...] = ("الشيخ الجليل",)

# Leading collective / anonymous (mubham) descriptors. A span that *begins* with
# one of these AND contains the partitive ``من`` ("a man FROM …", "a group OF …")
# is an unnamed-narrator reference, not a person. The ``من`` condition is what
# keeps a real name that merely starts with a common token (e.g.
# ``"عباد بن زياد من ولد المغيره"``) from being dropped.
_MUBHAM_LEADERS = frozenset(
    {
        "رجل",
        "رجلا",
        "رجال",
        "ناس",
        "اناس",
        "اناسا",
        "ناسا",
        "جماعه",
        "جمله",
        "قوم",
        "نفر",
        "عده",
        "بعض",
        "شيخ",
        "غير",
    }
)

# Bare relational-pronoun (mubham) references — "his father", "my father", "his
# grandfather", "his son", … — minted as named narrators. Like the collective
# descriptors above these are unnamed-narrator references (da#247 root cause 3),
# but they carry no partitive ``من`` so the ``_MUBHAM_LEADERS`` guard misses them.
# On the #723 graph ``ابيه`` ("his father") alone was the single MOST-mentioned
# "narrator" (65,755 mentions) — a spurious node into which every chain's elided
# ancestor wrongly collapsed, fabricating cross-chain links. Normalized-Arabic
# forms (post ``normalize_arabic``: hamza/diacritics stripped). A span is dropped
# only when the WHOLE name is exactly ONE of these (see clean step 6), so a real
# multi-token kunya like ``"ابي اسحاق"`` (Abū Isḥāq) is never touched.
_MUBHAM_RELATIONAL = frozenset(
    {
        "ابيه",  # his father        "ابي",   # my father (bare, incomplete kunya)
        "ابي",
        "ابيها",  # her father
        "امه",  # his mother         "امها",  # her mother
        "امها",
        "ابنه",  # his son           "ابنها"
        "ابنها",
        "ابنته",  # his daughter     "ابنتها"
        "ابنتها",
        "جده",  # his grandfather    "جدها"
        "جدها",
        "جدته",  # his grandmother   "جدتها"
        "جدتها",
        "اخيه",  # his brother       "اخيها"
        "اخيها",
        "اخته",  # his sister        "اختها"
        "اختها",
        "عمه",  # his paternal uncle "عمها","عمته"
        "عمها",
        "عمته",
        "خاله",  # his maternal uncle "خالها","خالته"
        "خالها",
        "خالته",
        "زوجته",  # his wife          "زوجها" his/her spouse
        "زوجها",
        "عنه",  # "from him" — transmission particle, never a name
        # da#391: the 1st-person-possessive (``-ي``) and vocative (``-اه``) kinship
        # forms. The set above is exhaustively 3rd-person ("his/her …") plus the bare
        # 1st-masc ``ابي``; a mention/alias phrased in the narrator's OWN voice
        # ("امتاه" = "O mother!", "عمتي" = "my aunt", "جدي" = "my grandfather") is the
        # same unnamed-relation reference and equally never a proper ism. Closed,
        # kinship-only — no name adjudication. Attested as surviving aliases on the
        # curated set (``عمتي``/``امتاه`` on the ʿĀʾisha chimera; ``جدي``×91, ``عمي``×32).
        "امي",  # my mother
        "ابني",  # my son            "ابنتي"/"بنتي" my daughter
        "ابنتي",
        "بنتي",
        "جدي",  # my grandfather     "جدتي" my grandmother
        "جدتي",
        "اخي",  # my brother         "اختي" my sister
        "اختي",
        "عمي",  # my paternal uncle  "عمتي" my paternal aunt
        "عمتي",
        "خالي",  # my maternal uncle "خالتي" my maternal aunt
        "خالتي",
        "امتاه",  # O mother! (vocative) "اماه" variant
        "اماه",
        "ابتاه",  # O father! (vocative)
    }
)

# Prophet-reference titles (da#271). The Prophet is the matn's ultimate source,
# not an isnad narrator, yet honorific-laden references to him get minted as
# narrator nodes ("رسول الله صلى الله عليه واله" — mc 365 on the run-3 stage —
# "النبي …"). After the taṣliya is stripped (honorific phrases above) the residue
# is a bare title; a span that LEADS with one is such a reference and is dropped.
# Keyed on the leader (definite-article "the Prophet/Messenger", or رسول/نبي only
# when immediately followed by الله) so a real name is never touched by mere
# coincidence of its first token. See :func:`_is_prophet_reference`.
_PROPHET_TITLE_SOLE = frozenset({"النبي", "الرسول", "للنبي", "للرسول"})
# رسول/نبي (and the proclitic-lam "لرسول"/"لنبي" = "to the Messenger/Prophet")
# count only when immediately followed by الله. The ل-forms are unambiguous — no
# real name is "لرسول"/"لنبي" — so "لرسول الله" ("to the Messenger of Allah", a
# matn residue) is caught while لبيد / لطيف and other ل-initial names are not.
_PROPHET_TITLE_LEADERS = frozenset({"رسول", "نبي", "لرسول", "لنبي"})

# Prophet-APPOSITION connectors (da#311). A real narrator described BY her
# relation to the Prophet — "<name> زوج النبي" ("… wife of the Prophet"), "<name>
# بنت النبي" ("… daughter of the Prophet") — surfaces in fawaz/thaqalayn as a
# canonical whose stored name is the trimmed apposition (the trailing قالت + the
# benediction were already stripped at NER time, leaving e.g. "عاءشه زوج النبي",
# mc-196). The REAL narrator is the leading name (Aisha), so we truncate at the
# connector and RECOVER it, rather than drop the row. The discriminator is that
# the token AFTER the connector is a Prophet reference (:func:`_prophet_ref_len`):
# "عاءشه بنت سعد" (بنت followed by a real name سعد) is untouched, "عاءشه بنت النبي"
# (بنت followed by النبي) truncates to "عاءشه". None of these is ever removed from
# a real nasab lineage because a lineage's connector is ALWAYS followed by a name,
# never by a bare Prophet title.
_APPOSITION_CONNECTORS = frozenset(
    {
        "زوج",  # husband/wife of
        "زوجه",  # his wife
        "بنت",  # daughter of
        "ابنه",  # his son / daughter (context)
        "ابنته",  # his daughter
        "ام",  # mother of
        "اخت",  # sister of
        "اهل",  # family/people of
        "مولى",  # client/freedman of
        "مولاه",  # his client
        "خادم",  # servant of
        "خادمه",  # her/his servant
        "صاحب",  # companion of
        "صاحبه",  # his companion
        "عم",  # paternal uncle of
        "خال",  # maternal uncle of
        "جد",  # grandfather of
        "عمه",  # paternal aunt of
    }
)

# English non-name *leader* words. Several English-only sources (halimbahae,
# sunnah translations) put translated isnad prose in the name field, so a "name"
# is really a sentence fragment: "It was narrated…", "The Prophet said…", "his
# father", "This hadith has been…". A span whose FIRST token (case-folded) is one
# of these is such a fragment — on the #723 reload "It"-led spans alone were
# 19,087 mentions. Precision: NO Arabic-name romanization starts with these, and
# the genuine name-leaders are excluded — `abu`, `ibn`, `abd`, `umm`, `bint`,
# `banu`, `dhu`, `al`/`an` (article-assimilation forms like "An-Nawawi"), and
# every actual name token (Anas, Malik, Jabir, …). So real romanized narrators
# are never dropped; only function/pronoun/verb-led prose is.
_EN_NONNAME_LEADERS = frozenset(
    {
        "it",
        "its",
        "this",
        "these",
        "those",
        "that",
        "there",
        "the",
        "he",
        "she",
        "they",
        "we",
        "you",
        "him",
        "his",
        "her",
        "my",
        "our",
        "your",
        "their",
        "was",
        "were",
        "is",
        "are",
        "has",
        "have",
        "had",
        "been",
        "be",
        "will",
        "would",
        "did",
        "narrated",
        "said",
        "says",
        "reported",
        "related",
        "told",
        "mentioned",
        "transmitted",
        "then",
        "when",
        "while",
        "from",
        "to",
        "by",
        "of",
        "for",
        "with",
        "and",
        "but",
        "as",
        "at",
        "upon",
        "about",
        "which",
        "who",
        "whom",
        "whose",
        "what",
    }
)

# A narrator name longer than this many whitespace tokens is a phrase / sentence /
# mis-parsed text body (thaqalayn dumps whole hadith bodies of 50–500+ tokens),
# not a name. The cap is set high (30) on purpose: classical full nasab lineages
# ("ابو القاسم عبد الرحمن بن الحسن بن احمد بن محمد بن عبيد بن عبد الملك …") run
# 20–30 tokens and are REAL narrators that must not be dropped — false-dropping a
# real name is worse than letting a rare mid-length junk span through (the proper
# thaqalayn parser fix, not this backstop, is the real remedy for the text dumps).
_MAX_NAME_TOKENS = 30

# An English/romanized span carrying at least this many WHOLE-token non-name leader
# words (articles, pronouns, prepositions, transmission verbs — "of", "the", "and",
# "then", "narrated", …) is translated isnad PROSE, not a name (da#253). A real
# romanized narrator name carries ZERO such tokens — the genuine name-leaders
# (abu/ibn/abd/umm/al/an/…) are deliberately excluded from _EN_NONNAME_LEADERS —
# and an Arabic-script token can never equal an ASCII word, so this guard has no
# false-positive surface on real names while catching embedded matn prose that
# neither leads with a leader (step 7) nor exceeds the token cap (step 4). Set to 2:
# a lone leading leader is already handled by step 7's first-token check.
_MIN_PROSE_LEADER_TOKENS = 2

# --- Arabic isnad/matn residual (da#258) ---------------------------------------
# Whole-token (normalized-Arabic) boundary markers: the first one seen in a span
# ends the *name* and begins isnad/matn text. Truncating at the first boundary and
# keeping the preceding tokens recovers the real leading narrator (class 5, chain
# connectives + editorial glosses), and dropping when the boundary *leads* the span
# handles the matn body whose name was elided (class 4). NONE of these is ever a
# legitimate name component — they are transmission/matn verbs, the isnad
# subordinators عن ("from") / ان ("that"), relative pronouns, temporal particles,
# and the "هو … / قيل …" identification glosses — so a real name (verified: every
# ibn-lineage nasab in the da#247 test set) contains zero of them and is never cut.
# The compound conjunction و and the co-narrator marker جميعا are deliberately
# EXCLUDED: a name-only "X و Y جميعا" join is a class-6 compound to be SPLIT
# (split_compound_narrators), not truncated/dropped — losing co-narrators regresses.
_ISNAD_BOUNDARY = frozenset(
    {
        # transmission / matn verbs
        "كان",
        "كانت",
        "قال",
        "قالت",
        "قالوا",
        "فقال",
        "فقالت",
        "يقول",
        "تقول",
        "قلت",  # da#311: 1st-person "I said" — an opener too, but as a boundary it
        "قلنا",  # recovers a leading narrator mid-span ("حميد قلت لزينب …" → "حميد").
        "حدثنا",
        "حدثني",
        "اخبرنا",
        "اخبرني",
        "انبانا",
        "سمعت",
        "حدثاه",
        "جاء",
        "قام",
        "سال",
        "فساله",
        "سءل",  # da#311: "was asked" — an opener too, but as a boundary it recovers
        "سالت",  # a leading narrator mid-span ("يحيى سءل مالك" → "يحيى").
        "اساله",  # "I ask him" (the كتبت … اساله correspondence-matn tail verb)
        # 3rd-person transmission verb "narrated" (da#311) — the sibling of the
        # 1st-person حدثنا/حدثني already above; not previously present, so a name
        # followed by حدث/حدثت (e.g. "فلان حدث عن فلان") survived untruncated.
        "حدث",
        "حدثت",
        "روت",  # "she narrated" (fem.) — sibling of روى, the truncation-boundary
        # (not leading-opener) form: a real name followed by "روت …" truncates
        # and recovers the leading name, same as قالت/حدثت above.
        # dual "they two said" + و/ف-prefixed said (da#311) — a real-co-narrator
        # compound "X و Y قالا" or a "<name> وقال …" continuation. NONE is a name
        # component; truncating recovers the leading narrator(s) and drops the
        # trailing speech verb, so "عمرو الناقد وزهير بن حرب قالا" → the preserved
        # co-narrator span "عمرو الناقد وزهير بن حرب" (compound kept, verb gone).
        "قالا",
        "قالتا",
        "وقال",
        "وقالت",
        "وقالوا",
        "فقالوا",
        # bare 3rd-person "heard" (da#311) — sibling of سمعت already above; a real
        # name followed by "سمع فلانا" truncates to the leading name.
        "سمع",
        # 1st/3rd-person "wrote (to)" (da#311) — a maktub/correspondence matn
        # opener ("كتبت الى ابي الحسن …"); كتب/كتبت is never a name component.
        "كتب",
        # subordinating "that he/she/they …" (da#311) — the ه/ها/هم-suffixed ان
        # forms the bare "ان" boundary misses ("عاءشه انها …", "ابي جعفر انه سءل").
        "انه",
        "انها",
        "انهم",
        "انهما",
        # isnad subordinators / relative pronouns / clause-opening particles
        "عن",
        "ان",
        "الذي",
        "التي",
        "الذين",
        "اذ",
        "اذا",  # da#311: "when / if" conditional matn opener ("اذا كان الماء …")
        "حين",
        "حينما",
        "لما",
        "فلما",
        "ثم",
        "حتى",
        "والله",
        # editorial identification glosses ("he is …", "it is said …")
        "هو",
        "وهو",
        "قيل",
        "يقال",
    }
)

# Boundary tokens into which leading proclitics may be folded (da#314). The definite
# relative pronouns ONLY: "والذي" ("and he who") opens the oath formula "والذي نفس
# محمد بيده" and is never a name. Kept as a narrow allowlist rather than folding
# across all of _ISNAD_BOUNDARY, because folding into that set's SHORT function words
# (ان/اذا/عن/هو/ثم) collides with real name material — "وان بن سالم" (mc-251, a بن
# lineage) would false-drop via وان → ان. See :func:`_truncate_at_isnad_boundary`.
_PROCLITIC_FOLDABLE_BOUNDARY = frozenset({"الذي", "التي", "الذين"})

# Proclitics stripped (iteratively, longest-chain-first) when folding into the set
# above. The oath formula stacks them — "فوالذي نفس محمد بيده" (ف+و), "فبالذي ارسلك
# الله" (ف+ب) — so a single-proclitic fold leaves the commonest forms behind. Safe
# ONLY because the fold target is the relative-pronoun allowlist: no Arabic ism is
# "بالذي" / "فوالذي", so no real name can be reached by peeling these.
_BOUNDARY_PROCLITICS = ("و", "ف", "ب", "ل", "ك")


def _fold_proclitics(token: str) -> str:
    """Peel leading proclitics while the remainder is a foldable boundary (da#314).

    ``"فوالذي"`` → ``"والذي"`` → ``"الذي"``. Returns *token* unchanged when no chain
    of proclitics reaches :data:`_PROCLITIC_FOLDABLE_BOUNDARY`, so a real name is
    never partially peeled (the peel is only *kept* if it lands on the allowlist).
    """
    seen = token
    for _ in range(len(_BOUNDARY_PROCLITICS)):
        if seen in _PROCLITIC_FOLDABLE_BOUNDARY:
            return seen
        if seen[:1] not in _BOUNDARY_PROCLITICS or len(seen) < 2:
            break
        seen = seen[1:]
    return seen if seen in _PROCLITIC_FOLDABLE_BOUNDARY else token


# Matn-density backstop: a span whose leading tokens form a real name but whose tail
# is matn NOT anchored by an _ISNAD_BOUNDARY verb (e.g. a bare-preposition run) is
# still a sentence. ≥2 whole-token matn/particle words → drop. A real name carries
# zero; the threshold of 2 protects a lone trailing particle and, crucially, a
# class-6 compound that ends in a single قالوا join-marker (density 1 → preserved).
_MATN_DENSITY = frozenset(
    {
        "كان",
        "كانت",
        "قال",
        "قالت",
        "قالوا",
        "فقال",
        "يقول",
        "اذ",
        "حين",
        "لما",
        "فلما",
        "ثم",
        "حتى",
        "والله",
        "هو",
        "وهو",
        "حدث",  # da#311: 3rd-person "narrated", sibling of قال/كان in this backstop
        "حدثت",
        "روت",
        "قالا",  # da#311: dual "they two said"
        "قالتا",
    }
)

# Compound co-narrator join marker: a span ending in one of these ("… together",
# "… they said") is multiple narrators (class 6). Used by split_compound_narrators
# to confirm a compound and to strip the trailing marker before splitting.
_COMPOUND_TRAILING_MARKERS = frozenset({"جميعا", "قالوا"})

# Trailing / edge punctuation seen on extracted spans ("شيخ من اهل المدينه ,").
# ؟ is included (da#308) so a stray trailing question mark ("الاوزاعي الدمشقي؟")
# is stripped off the token like any other edge mark and does not ride into the
# clustering key; matn ؟ mid-span is still detected via the raw kept-text scan.
#
# Bidi / zero-width FORMAT MARKS are included (da#316). The raw source interleaves
# U+200F RIGHT-TO-LEFT MARK *between* trailing punctuation characters — the real
# stored span is ``'قتاده،‏.‏'``, not ``'قتاده،.'``. ``str.strip`` halts at
# the first character outside its set, so a trailing RLM standing between the marks
# ends the peel before the ``.`` and ``،`` are reached and the name keeps its cruft.
# NOTE this only matters on RAW / voweled text: ``normalize_arabic`` already removes
# these marks (:func:`~src.utils.arabic.strip_format_marks`, da#271), so the
# *normalized* path never sees one. It is the DISPLAY path — which strips edge punct
# from the un-normalized voweled token to keep its diacritics — that needs them here.
_FORMAT_MARK_CHARS = "​‌‍‎‏‪‫‬‭‮﻿"
_EDGE_PUNCT = " \t\r\n,،.;؛:؟-_\"'«»()[]" + _FORMAT_MARK_CHARS

# --- Sentence / matn-body drop-gate (da#308) -----------------------------------
# Nasab (lineage) connectors — the PRECISION GUARD. A string dense in these is a
# genuine ``ibn``-lineage / kunya-nasab name and is SPARED from the matn signals
# below regardless of length. Matched diacritics-insensitively (each token is run
# through ``normalize_arabic`` first) so a vocalized ``بْنُ`` / ``أَبُو`` counts.
_NASAB_CONNECTORS = frozenset(
    {
        "بن",  # ibn (medial)
        "ابن",  # ibn (leading / after a stop)
        "بنت",  # bint (daughter of)
        "ابو",  # abu (kunya)
        "ابي",  # abi (kunya, genitive)
        "ابا",  # aba (kunya, accusative)
        "بني",  # banu (tribe, genitive)
        "بنو",  # banu (tribe, nominative)
    }
)

# Matn / discourse verb-openers that :data:`_ISNAD_BOUNDARY` does NOT already cut
# on (قال/سمعت/حدثنا/عن are boundaries; these are the residual openers). A span that
# LEADS with one is a hadith body / answer, never a name — none is a legitimate
# name component (a real kunya opener is ابو/ابي, kept out of this set). Normalized
# form; matched against the diacritics-folded leading token.
_MATN_OPENERS = frozenset(
    {
        "قلت",  # "I said"
        "قلنا",  # "we said"
        "كنا",  # "we were"
        "كنت",  # "I was"
        "نهى",  # "he forbade"
        "امر",  # "he ordered"
        "صلى",  # "he prayed" (verb; the taṣliya benediction is stripped in step 2)
        "صلت",  # "she prayed"
        "رايت",  # "I saw"
        "بينما",  # "while"
        "بينا",  # "while"
        "نعم",  # "yes" (matn answer particle, minted as a lone junk narrator)
        "بلى",  # "yes indeed"
        # Shia dialogue-hadith openers (da#311) — the dominant matn shape in
        # thaqalayn/fawaz ("I asked Abu al-Hasan (as): ...", "Abu Abd Allah (as)
        # was asked: ..."), tuned on Sunni/Tirmidhī patterns only until now.
        # Normalized forms per ``normalize_arabic``'s alif/hamza folding: أ→ا so
        # سألت → سالت; ئ→ء so سُئل → سءل.
        "سالت",  # سألت "I asked" (1st person; bare سال 3rd-person is already an
        # _ISNAD_BOUNDARY token — this is the distinct 1st-person conjugation)
        "سءل",  # سُئل "was asked" (passive)
        "روى",  # "he narrated" (3rd person; a lone leading opener, distinct
        # from the truncation-boundary سمعت/حدثنا/حدثني/اخبرنا/اخبرني/حدث/حدثت
        # forms above, which recover a preceding real name instead)
        "روي",  # "was narrated" (passive, ya-final — normalize_arabic does NOT
        # fold ى→ي so this is a DISTINCT token from روى above; da#311 round-4b, a
        # lone leading روي slips truncation and mints a 541-mention junk narrator)
        "كتبت",  # "I wrote (to)" — maktub/correspondence matn opener ("كتبت الى
        # ابي الحسن اساله …"); the 3rd-person كتب is an _ISNAD_BOUNDARY token.
        "يا",  # vocative "O …" ("يا رسول الله", "يا محمد", "يا معاذ …") — a
        # matn address, never a name; no Arabic name begins with the particle يا.
        "تزوج",  # "he married" ("تزوج النبي ميمونه" — Prophet-marriage matn); never
        # a name (the noun زوج is handled as an apposition connector separately).
        "رءيا",  # "[I] saw" (رأيا; a رايت variant); a matn eyewitness opener.
        # Oath / adjuration verb (da#314) — "انشدك بالله" ("I adjure you BY GOD …"),
        # the matn interrogation formula ("فانشدك بالله هل تعلم" = "I adjure you by
        # God, do you know …"). The ف-prefixed form folds via the step-2 opener check.
        # A verb of address, never a name component.
        "انشدك",
        "نشدتك",
    }
)

# --- Matn function-word / Qur'anic-verse signature (da#317) --------------------
# Standalone matn function words — prepositions, personal pronouns, demonstratives,
# and conditional / negation / conjunction particles — that a real *proper-noun*
# narrator name NEVER carries as a whole token. The da#308 gate caught verb-led and
# grading-formula matn but MISSED (a) conjunction-initial narrative (``و…``/``ف…``),
# (b) divine-name / Qur'anic-verse phraseology, and (c) short function-word-only
# fragments (``وما`` "and not"). On the #723 reload ~26% of the 150,187 canonical
# "narrators" were such matn/verse sentences in the zero-degree tail (row 1 of the
# id-ordered /narrators list is Qur'an 22:32). These particles are the text-only
# signature of that class (no graph / degree input — the gate runs at NER time).
#
# PRECISION: none is ever a component of a real name, and every rule that consults
# this set runs AFTER the nasab-connector precision guard (so any genuine lineage /
# kunya is already spared). Crucially ``على`` (preposition "on", alif-maqsura final)
# is a DISTINCT token from ``علي`` (the name ʿAlī, ya final) because
# ``normalize_arabic`` does NOT fold ى→ي — so no ʿAlī is ever mistaken for the
# preposition. ``من`` is deliberately EXCLUDED (it is the partitive the mubham
# guards 5/5b key on, and folding it in here would be redundant/entangling).
_MATN_PARTICLES = frozenset(
    {
        # prepositions
        "على",  # "on / upon" (alif-maqsura — NOT the name علي)
        "في",  # "in"
        "الى",  # "to / towards"
        "عند",  # "at / with"
        "بعد",  # "after"
        "قبل",  # "before"
        "مع",  # "with"
        "لدى",  # "at / with"
        "بين",  # "between"
        "نحو",  # "towards / about"
        # personal / attached pronouns
        "نحن",  # "we"
        "انا",  # "I"
        "انت",  # "you (m.)"
        "انتم",  # "you (pl.)"
        "هم",  # "they"
        "هما",  # "they two"
        "اليك",  # "to you"
        "اليه",  # "to him"
        "اليها",  # "to her"
        "اليهم",  # "to them"
        "لك",  # "to you"
        "له",  # "to him"
        "لها",  # "to her"
        "لهم",  # "to them"
        "لنا",  # "to us"
        # demonstratives / place adverbs
        "هذا",  # "this (m.)"
        "هذه",  # "this (f.)"
        "هؤلاء",  # "these"
        "ذلك",  # "that (m.)"
        "تلك",  # "that (f.)"
        "هناك",  # "there"
        "هناكا",  # "there" (pausal/poetic form seen in the evidence set)
        "هنا",  # "here"
        "حيث",  # "where"
        # conditional / negation / conjunction / interrogative particles
        "لو",  # "if"
        "لولا",  # "if not"
        "ما",  # "not / what"
        "لا",  # "no / not"
        "لم",  # "did not"
        "لن",  # "will not"
        "او",  # "or"
        "ام",  # "or (interrog.)"
        "الا",  # "except / but" (exceptive particle; never a name component)
        "لكن",  # "but"
        "بل",  # "rather"
        "قد",  # "already / indeed"
        "كل",  # "all / every"
        "بعض",  # "some"
        "اما",  # "as for / either"
        "انما",  # "only / but"
        "كي",  # "so that"
        "كيف",  # "how"
        "متى",  # "when"
        "اين",  # "where"
        "لذلك",  # "therefore"
        "لذا",  # "hence"
        "كذلك",  # "likewise"
        "كذا",  # "such / thus"
    }
)


# Theophoric compound HEADS (da#317 recall fix) — the first element of an
# ``X + الله`` given name (``عبد الله`` "servant of God", ``هبة الله`` "gift of
# God", …). When counting the bare divine name الله for the Qur'anic-verse
# signature, an الله immediately preceded by one of these is the SECOND half of a
# real theophoric name, not verse repetition — so it must NOT count. Without this,
# a genuine double-theophoric narrator (``عبد الله بن عبد الله القرشي الاموي``)
# false-drops once its nisbas dilute the nasab density below the spare threshold.
# عبد/عبيد are ~99% of the class; the rest are the common Islamic ``…الله`` heads.
# Normalized-Arabic forms (هبة→هبه, امة→امه, رحمة→رحمه).
_THEOPHORIC_HEADS = frozenset(
    {
        "عبد",  # servant of
        "عبيد",  # (diminutive) servant of
        "امه",  # handmaid of (امة)
        "هبه",  # gift of (هبة)
        "سعد",  # good-fortune of
        "نصر",  # victory of
        "فضل",  # grace of
        "رحمه",  # mercy of (رحمة)
        "ضياء",  # light/radiance of
        "نور",  # light of
        "سيف",  # sword of
        "عطاء",  # gift of
        "جار",  # neighbour of
        "حبيب",  # beloved of
        "ذبيح",  # sacrifice of
    }
)


# The partitive / ablative ``من`` ("from", "of", "one who"), optionally carrying a
# ك/و/ف/ل/ب proclitic (``كمن`` "like one who", ``ومن``, ``فمن``). It is deliberately
# NOT a member of :data:`_MATN_PARTICLES` — guard 5 keys the mubham collective on the
# bare partitive and folding it in there would entangle the two — so the
# Prophet-provenance rule (da#314) tests for it through this helper instead.
_PARTITIVE = "من"


def _is_partitive(token: str) -> bool:
    """True when *token* is the partitive ``من``, bare or with a single proclitic (da#314)."""
    if token == _PARTITIVE:
        return True
    return token[:1] in ("و", "ف", "ك", "ل", "ب") and token[1:] == _PARTITIVE


# Collective nouns that head an ANONYMOUS group referred to the Prophet — "the
# companions of…", "the wives of…". With a mubham quantifier in front ("بعض اصحاب
# النبي" = "SOME OF the companions of the Prophet") the span names no individual, so
# it is residue for the provenance rule below. Closed set, and never an ism.
_PROPHET_COLLECTIVE_NOUNS = frozenset({"اصحاب", "ازواج", "نساء", "اهل", "ال", "امه"})

# Bare pronouns left behind by a mis-parse ("ه من رسول الله" mc-55, "ها من رسول الله"
# mc-7, "هن من رسول الله" mc-5). A pronoun is never an ism, so these carry no naming
# content. The one-character check below already caught the bare "ه"; this completes the
# closed set rather than leaving the lexicon to depend on token LENGTH, which is
# arbitrary — "ه" and "ها" are the same class of fragment.
_BARE_PRONOUNS = frozenset({"ه", "ها", "هم", "هن", "هما", "هي", "هو", "انا", "انت"})


def _is_prophet_ref_component(token: str) -> bool:
    """True when *token* is part of a Prophet reference itself (``رسول``/``النبي``/``الله``)."""
    return token in _PROPHET_TITLE_SOLE or token in _PROPHET_TITLE_LEADERS or token == "الله"


def _is_provenance_residue(token: str) -> bool:
    """True when *token* carries no naming content — function word or anonymous collective.

    The membership test for the da#314 provenance rule's all-residue scoping. A token
    is residue when it is the partitive ``من`` (with proclitics), a bare matn particle,
    a mubham collective quantifier (``بعض``, ``رجل``), a collective noun heading an
    anonymous group (``اصحاب``, ``ازواج``), or a stray sub-token left by a mis-parse
    (a one-letter fragment such as ``ه``/``ها``, which no Arabic ism can be).

    The apposition connectors are EXCLUDED from the matn-particle arm (Sofia Cardoso):
    ``أم`` (*umm*) normalizes to ``ام``, homographic with the disjunctive particle
    ``أم`` ("or"), and treating it as residue is what deleted Umm Kulthum bint Muhammad.
    Redundant under the all-token scoping — her span carries substantive tokens too —
    but kept as the explicit, verified guard against the homograph.
    """
    if _is_partitive(token) or token in _MUBHAM_LEADERS or token in _PROPHET_COLLECTIVE_NOUNS:
        return True
    if len(token) == 1 or token in _BARE_PRONOUNS:
        return True
    return _is_matn_particle(token) and token not in _APPOSITION_CONNECTORS


def _is_matn_particle(token: str) -> bool:
    """True when *token* (normalized) is a bare matn function word (da#317).

    Matches directly, or after folding a single leading ``و``/``ف`` proclitic
    ("and / so") — so ``وما`` ("and not") and ``فلو`` ("so if") count, while a real
    ``و``/``ف``-initial name (``وهب`` → ``هب``, ``فاطمه`` → ``اطمه``) folds to a
    non-particle stem and is NOT matched. ``في`` ("in") is matched directly (its
    ``ف`` is not a proclitic), so the fold never strips it to a bare ``ي``.
    """
    if token in _MATN_PARTICLES:
        return True
    return token[:1] in ("و", "ف") and len(token) > 1 and token[1:] in _MATN_PARTICLES


# Grading / commentary formulae as TOKEN sequences (da#308). Matched as a
# contiguous token subsequence, not a raw substring, so "حسن صحيح" does not fire
# inside the single token "الحسن" + "صحيح" of a real name (Kavitha #4).
# al-Tirmidhī-style verdicts ("this is a ḥasan-ṣaḥīḥ ḥadith") dumped into the name
# field — never any part of a person's name.
_GRADING_FORMULAE: tuple[tuple[str, ...], ...] = (
    ("هذا", "حديث"),
    ("حديث", "حسن"),
    ("حديث", "صحيح"),
    ("حديث", "غريب"),
    ("حديث", "ضعيف"),
    ("حسن", "صحيح"),
    ("حسن", "غريب"),
    ("صحيح", "غريب"),
    ("متفق", "عليه"),
    # al-Tirmidhī editorial commentary (da#311): "… روى هذا الحديث …" ("… narrated
    # this hadith …"), "روى غير واحد" ("more than one narrated"). The whole span is
    # the compiler's commentary keyed on his own kunya (ابو عيسى / ابو داود), not a
    # narration BY a distinct narrator — the compiler has his own clean canonical.
    # Neither "هذا الحديث" nor "غير واحد" is ever a component of a real name.
    ("هذا", "الحديث"),
    ("غير", "واحد"),
)


def _contains_token_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    """True when *sequence* appears as a contiguous run of whole tokens in *tokens*."""
    span = len(sequence)
    if span == 0 or span > len(tokens):
        return False
    return any(tuple(tokens[i : i + span]) == sequence for i in range(len(tokens) - span + 1))


# Matn punctuation — quoted speech guillemets and the question mark signal a hadith
# body / dialogue, not a name. (« » are stripped from token EDGES by _EDGE_PUNCT,
# so this is checked against the pre-tokenized text; ؟ is not an edge-punct char.)
_MATN_PUNCT: tuple[str, ...] = ("«", "»", "؟")

# A matn span shorter than this many tokens is not dropped on a signal — the
# token count is only ever a *co-factor* with a matn signal (issue #308: "a bare
# high token count is NOT sufficient"), never a drop reason on its own. A lone
# matn opener (n == 1, e.g. "نعم") is handled separately.
_MATN_SENTENCE_MIN_TOKENS = 3

# Nasab density at/above which a span is spared as a genuine lineage even if a weak
# matn signal (punctuation / particle density) is present. One connector in a short
# name (ابو X, X بن Y) already clears this; ``_NASAB_CONNECTORS`` count >= 2 spares
# unconditionally.
_NASAB_SPARE_DENSITY = 0.15

# Arabic-Indic digits (U+0660–U+0669) alongside ASCII 0–9, for the numeric-leading
# mis-parse guard (da#311).
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def _is_number_token(token: str) -> bool:
    """True when *token* is a bare number (ASCII or Arabic-Indic digits) (da#311)."""
    stripped = token.strip(_EDGE_PUNCT)
    if not stripped:
        return False
    return all(c.isdigit() or c in _ARABIC_INDIC_DIGITS for c in stripped)


# "Asked" verb conjugations (da#311) — the classic Shia dialogue-hadith opener
# "سألته / سألتها / وسألته / سُئلت …" ("I asked him / her", "was asked"), by far the
# single highest-mc matn opener (سالته mc-1035). PRECISION: the stem سال collides
# with the extremely common name سالم (Salim, mc-4072) — so we match the stem ONLY
# when the remainder is a pronoun/tense SUFFIX from a closed set (never "م" →
# Salim, "مه", "ار", …). A leading proclitic و ("and he asked") is stripped first.
_ASK_STEMS = ("سال", "سءل")
_ASK_SUFFIXES = frozenset(
    {
        "",  # bare "he asked" (سال / سءل)
        "ت",  # I / she asked
        "ته",
        "تها",
        "تهم",
        "تهما",
        "تك",
        "تكم",
        "تموه",
        "تموها",
        "ه",  # he asked him
        "ها",
        "هم",
        "هما",
        "ني",  # he asked me
        "نا",
        "تني",
        "تنا",
        "وا",  # they asked
        "وه",  # they asked him
        "وها",
        "ونه",
        "وني",
    }
)


def _is_ask_verb(token: str) -> bool:
    """True when *token* is an "asked" verb conjugation (da#311), NOT the name سالم.

    Matched diacritics-insensitively (the token is normalized first). The stem
    ``سال``/``سءل`` counts only when what follows is a closed pronoun/tense suffix,
    so ``سالم`` (Salim, remainder ``م``), ``سالمه``, ``سالار`` are never matched.
    """
    t = normalize_arabic(token).strip(_EDGE_PUNCT)
    for candidate in (t, t[1:] if t.startswith("و") and len(t) > 1 else t):
        for stem in _ASK_STEMS:
            if candidate.startswith(stem) and candidate[len(stem) :] in _ASK_SUFFIXES:
                return True
    return False


def _is_degenerate_name(name: str) -> bool:
    """True when *name* is too short / empty / punctuation / latin-noise (da#311).

    A recovered value with fewer than 2 significant Arabic letters AND fewer than 2
    Latin letters is garbage (``"ه"``, ``"ف"``, a stray ``"p"``, a zero-width mark,
    pure punctuation) — drop the row rather than write it. Real romanized names
    (``"Malik"``, ``"Abu"``) clear the Latin-letter floor and are kept; a genuine
    2+-letter Arabic name (``"من"`` notwithstanding — 2 letters) clears the Arabic
    floor.
    """
    stripped = name.strip().strip(_EDGE_PUNCT).strip()
    if not stripped:
        return True
    arabic = sum(1 for c in stripped if "ء" <= c <= "ي")
    latin = sum(1 for c in stripped if c.isascii() and c.isalpha())
    return arabic < 2 and latin < 2


def strip_markup(name: str | None) -> str:
    """Clean the voweled DISPLAY name: markup + colon-joined prose + edge punct.

    Display-safe cleaner for the voweled ``name_raw`` (keeps diacritics). It strips
    angle-bracket markup tags + stray brackets, then truncates a
    ``<name>:<English-prose>`` colon-join at the prose boundary (da#253) via
    :func:`_truncate_colon_prose` — so the DISPLAY field never carries a hadith matn
    (the reported prod node ``nar:00063b2c…`` was an English fallback narrator whose
    ``name_ar`` was the matn). The parallel normalized clustering key goes through
    :func:`clean_narrator_name`; this keeps the two in lock-step. Only colon spans
    whose tail begins with an English leader are cut — Arabic voweled names and
    ordinary names (no such colon) are returned unchanged.

    Bidi / zero-width format marks are stripped FIRST (da#316). This is the raw,
    un-normalized entry point — the only path that never runs through
    ``normalize_arabic`` (which would remove them) because it must preserve the
    diacritics. The marks appear *interleaved* with punctuation (``'قتاده،‏.‏'``)
    and *interior* to the span (``'…تعالى‏:‏ ‏(‏لقد'``), so an edge-strip alone
    cannot clear them; removing them here makes the subsequent edge-punct strip —
    and the token alignment in :func:`clean_narrator_name_display` — behave exactly
    as it does on normalized text. Diacritics are untouched
    (:func:`~src.utils.arabic.strip_format_marks` removes only bidi/zero-width
    controls), so the display name keeps its vowels.
    """
    if not name:
        return ""
    cleaned = strip_format_marks(name)
    cleaned = _MARKUP_RE.sub(" ", cleaned).replace("<", " ").replace(">", " ")
    cleaned = _truncate_colon_prose(" ".join(cleaned.split()))
    return cleaned.strip(_EDGE_PUNCT).strip()


def _truncate_colon_prose(text: str) -> str:
    """Cut a ``<name>:<English-prose>`` colon-join at the prose boundary (da#253).

    Returns ``text`` up to (excluding) the first ``:`` whose following segment
    begins with an English non-name leader word; if no such colon exists the text
    is returned unchanged. This recovers the pre-colon name (``"Thawban"``) from a
    ``"Thawban:The Messenger …"`` matn-join, while leaving colon-free names and
    ``"name:name"`` joins (whose tail is a real name, not a leader) untouched.
    """
    if ":" not in text:
        return text
    segments = text.split(":")
    kept = [segments[0]]
    for seg in segments[1:]:
        head = seg.strip(_EDGE_PUNCT).split()
        if head and head[0].lower() in _EN_NONNAME_LEADERS:
            break
        kept.append(seg)
    return ":".join(kept)


def _truncate_at_isnad_boundary(tokens: list[str]) -> list[str] | None:
    """Cut a span at the first isnad/matn boundary token (da#258 classes 4+5).

    Returns the leading tokens up to (excluding) the first token in
    :data:`_ISNAD_BOUNDARY` — recovering the real narrator that precedes an isnad
    connective / matn verb / editorial gloss (``"ابو محمد بصري عن محمد بن علي"`` →
    ``["ابو","محمد","بصري"]``; ``"عاءشه قالت كان"`` → ``["عاءشه"]``). Returns
    ``None`` when the boundary *leads* the span (``"كان علي …"``, ``"قالوا"``) —
    the real narrator sits after an elided-grammar verb and cannot be recovered
    here. When no boundary token is present the tokens are returned unchanged.

    Leading proclitics are folded before matching (:func:`_fold_proclitics`), but ONLY
    into the definite RELATIVE PRONOUNS (:data:`_PROCLITIC_FOLDABLE_BOUNDARY`) — so the
    oath formula ``"والذي نفس محمد بيده"`` ("by Him in whose hand is Muhammad's soul",
    da#314) leads with ``والذي`` → folds to the boundary ``الذي`` → drops. The formula
    stacks proclitics in the corpus (``فوالذي``, ``فبالذي``), so the fold is iterative.

    The fold is deliberately NOT applied to the whole boundary set. Folding it into
    the SHORT function words there (``ان``, ``اذا``, ``عن``, ``هو``, ``ثم``) collides
    with real name material: it false-drops ``"وان بن سالم"`` (mc-251 — a ``بن``
    lineage) via ``وان`` → ``ان``, and ``"فان"``/``"فاذا"`` likewise. The definite
    relative pronouns have no such collision — no Arabic ism is ``والذي``/``فالتي`` —
    which is what makes them, and only them, safe to fold.

    The fold is applied at index 0 ONLY. This function has *dual* semantics — a
    boundary LEADING the span drops it, a boundary at index > 0 *truncates* and keeps
    the head — and the oath is a matn SIGNAL, not a name TERMINATOR. In the corpus the
    oath is overwhelmingly mid-matn ("تعاهدوا هذا القران فوالذي نفس محمد بيده…");
    folding it at i > 0 would make it a truncation point and hand the matn head back
    AS A NARRATOR NAME, minting 52 new zero-degree matn nodes ("دعوني" = "leave me",
    "كلا" = "nay", "قول الله") — exactly the class da#317 exists to remove. Restricted
    to the leading position, a mid-matn oath instead leaves the span whole and the
    matn-sentence gate (step 10) drops it as the matn body it is.
    """
    for i, raw_tok in enumerate(tokens):
        tok = raw_tok if (raw_tok in _ISNAD_BOUNDARY or i > 0) else _fold_proclitics(raw_tok)
        if tok in _ISNAD_BOUNDARY or _is_ask_verb(tok):
            # Nisba-transition ثم (da#308): "…الليثي ثم الجندعي" ("al-Laythī then
            # al-Jundaʿī") is a real tribal-affiliation nisba tail, not a temporal
            # matn connective. Recognise it by ثم flanked on BOTH sides by an
            # ال-prefixed nisba and do NOT cut there — a bare / verb-flanked ثم
            # ("ثم", "ثم قال …") still truncates as before.
            if (
                tok == "ثم"
                and 0 < i < len(tokens) - 1
                and tokens[i - 1].startswith("ال")
                and tokens[i + 1].startswith("ال")
            ):
                continue
            if i == 0:
                return None
            return tokens[:i]
    return tokens


def _is_prophet_reference(tokens: list[str]) -> bool:
    """True when the span is a reference to the Prophet by title, not a narrator (da#271).

    A span LEADING with a sole Prophet title (``النبي`` / ``الرسول``) or with
    ``رسول``/``نبي`` immediately followed by ``الله`` ("Messenger/Prophet of Allah")
    is a reference to the Prophet — the matn source, never an isnad narrator. The
    ``+ الله`` requirement means a real name whose first token merely equals
    ``رسول``/``نبي`` (without ``الله`` after) is never dropped.
    """
    if not tokens:
        return False
    if tokens[0] in _PROPHET_TITLE_SOLE:
        return True
    return tokens[0] in _PROPHET_TITLE_LEADERS and len(tokens) >= 2 and tokens[1] == "الله"


def _prophet_ref_len(tokens: list[str], i: int) -> int:
    """Length of a Prophet reference STARTING at index *i*, else 0 (da#311).

    Returns 1 for a sole title (``النبي`` / ``الرسول``) and 2 for ``رسول``/``نبي``
    immediately followed by ``الله`` (``رسول الله`` "Messenger of Allah"). The
    positional sibling of :func:`_is_prophet_reference`, which only inspects the
    leader; this locates a reference anywhere in the span.
    """
    if i >= len(tokens):
        return 0
    if tokens[i] in _PROPHET_TITLE_SOLE:
        return 1
    if tokens[i] in _PROPHET_TITLE_LEADERS and i + 1 < len(tokens) and tokens[i + 1] == "الله":
        return 2
    return 0


def _truncate_at_prophet_apposition(tokens: list[str]) -> list[str] | None:
    """Cut a ``<name> <connector> <Prophet-ref>`` apposition, keeping the name (da#311).

    A real narrator described BY her relation to the Prophet — ``"عاءشه زوج النبي"``
    ("Aisha, wife of the Prophet", mc-196 fawaz), ``"X بنت النبي"`` — is stored as a
    canonical whose name is the trimmed apposition (the trailing قالت + benediction
    were already stripped at NER time). The real narrator is the LEADING name, so we
    truncate at the :data:`_APPOSITION_CONNECTORS` token whose next token opens a
    Prophet reference (:func:`_prophet_ref_len`) and recover it.

    * ``["عاءشه","زوج","النبي"]`` → ``["عاءشه"]`` (recover Aisha)
    * ``["عاءشه","بنت","سعد"]`` → unchanged (بنت followed by a real name, not a
      Prophet ref — a genuine nasab, never cut)
    * ``["زوج","النبي"]`` (connector LEADS, no name before it) → ``None`` (drop —
      a bare "wife of the Prophet" names no recoverable narrator)

    Returns the truncated leading tokens, ``None`` to drop, or the tokens unchanged
    when no Prophet-apposition is present.
    """
    for i, tok in enumerate(tokens):
        if tok in _APPOSITION_CONNECTORS and _prophet_ref_len(tokens, i + 1) > 0:
            if i == 0:
                return None
            return tokens[:i]
    return tokens


def _is_matn_sentence(tokens: list[str], kept_text: str, was_truncated: bool) -> bool:
    """True when the span is a hadith-matn body / grading commentary, not a name (da#308).

    Precision-first drop-gate for the residual matn that survives the da#258
    boundary truncation (its opener is not an :data:`_ISNAD_BOUNDARY` token) — a
    whole hadith body (``"قلت لابي عبد الله ذهبت …"``), an answer particle
    (``"نعم"``), or a grading verdict (``"ابو عيسى هذا حديث حسن صحيح"``).

    Signals are combined precision-first; a bare token count is never a signal on
    its own (issue #308). *tokens* are the RETAINED (post-truncation) name tokens;
    *kept_text* is the raw text of that retained span ONLY (the truncated-off isnad
    / matn tail is excluded) so ``«…»`` / ``؟`` in a tail the da#258 truncation
    already removed do NOT count — a real name recovered from ``<name> قال …؟`` is
    not dropped for punctuation that was never part of the name (Nikolaos, #310).
    *was_truncated* is True when :func:`_truncate_at_isnad_boundary` actually cut
    something off (i.e. *tokens* is a RECOVERED leading fragment, not the whole
    original span) — see the da#311 subject-led signal below for why this matters.

    Matn punctuation is a **co-factor**, not a standalone drop reason: a lone ``؟``
    on an otherwise clean name never drops it; it drops only alongside a matn /
    particle word. The **precision guard** measures nasab-connector density
    diacritics-insensitively and spares a genuine lineage / kunya-nasab regardless
    of length, so real nasab chains, nisba tails, and kunya compounds pass through.

    da#311 adds two more signals ahead of the precision guard: a 2-token
    verb-opener residue (``"نهى النبي"``) and — ONLY when *was_truncated* — a bare
    Prophet/Messenger reference surviving anywhere in the recovered span
    (subject-led matn like ``"عاءشه زوج النبي قالت"``). The ``was_truncated`` gate
    is a precision guard in its own right: ``"محمد رسول الله"`` (Muhammad, the
    Messenger of Allah — a real narrator's own name+title, muhaddithat fixture)
    never triggers a boundary truncation at all (no isnad/matn verb present), so
    it is never re-scrutinized by this signal and survives untouched; only a span
    where truncation *actually recovered a fragment* — meaning something after it
    was cut as matn — gets the extra scrutiny. See the module docstring's da#311
    section for the full rationale.
    """
    n = len(tokens)
    if n == 0:
        return False

    # Diacritics-insensitive, hamza/alif-normalized token forms (so a vocalized
    # ``بْنُ`` counts toward nasab density, and the matn sets match reliably).
    bare = [normalize_arabic(t) for t in tokens]

    # (0) Vocative particle anywhere (da#311): the standalone token يا ("O …")
    #     opens a matn address ("… قلت يا رسول الله", "… صدقت يا محمد") that the
    #     leading-only opener check misses when a matn word precedes it. No Arabic
    #     name carries the vocative particle يا as a whole token at any position.
    if "يا" in bare:
        return True

    # (1) Grading / commentary formula (token-anchored) — never a name component.
    if any(_contains_token_sequence(bare, formula) for formula in _GRADING_FORMULAE):
        return True

    # (2) Matn / discourse verb leading the span: a bare opener ("نعم") or a
    #     verb-led sentence ("قلت … / صلى النبي …"). da#317 folds a single leading
    #     و/ف proclitic before the opener check so a conjunction-initial narrative
    #     opener ("فقلت" = "so I said", "وقلت", "فنهى") is caught — none of the
    #     openers is a real name, and no و/ف-initial real name folds to an opener
    #     stem (وهب→هب, فاطمه→اطمه), so this has no false-positive surface.
    lead = bare[0]
    lead_folded = lead[1:] if lead[:1] in ("و", "ف") and len(lead) > 1 else lead
    if lead in _MATN_OPENERS or lead_folded in _MATN_OPENERS:
        if n == 1 or n >= _MATN_SENTENCE_MIN_TOKENS:
            return True
        # (2s) Short residue (da#311): n == 2 falls under the general min-token
        #      floor above, but a verb-opener + single trailing noun ("نهى
        #      النبي" — the taṣliya-ligature residue left after step 2 strips
        #      the honorific tail) is still matn, not a name. _MATN_OPENERS
        #      contains no nasab/kunya opener by construction (the real kunya
        #      openers ابو/ابي/ابن are deliberately excluded — see the set's
        #      docstring), so this can never fire on a genuine 2-token kunya
        #      ("ابو هريره"); the nasab-density guard is reused anyway as an
        #      explicit belt-and-suspenders check against a future opener
        #      addition that might coincide with a nasab connector.
        if not any(t in _NASAB_CONNECTORS for t in bare):
            return True

    # (2p) Subject-led matn (da#311): a bare reference to the Prophet/Messenger
    #      surviving INTO a RECOVERED (post-truncation) span — anywhere, not
    #      only as the leader — signals the span is matn / an appositive
    #      description of a person's relation to the Prophet ("عاءشه زوج النبي
    #      قالت" = "Aisha, wife of the Prophet, said" — mc-196, fawaz), not a
    #      standalone recoverable name. This is the sibling of da#271's
    #      leader-only :func:`_is_prophet_reference` (step 6b, which runs
    #      before this and only catches a LEADING reference). Gated on
    #      ``was_truncated``: a person's OWN name+title ("محمد رسول الله" —
    #      Muhammad, the Messenger of Allah, a real narrator entry in its own
    #      right) never trips an isnad/matn boundary truncation at all, so it
    #      is never re-scrutinized here; only a span where something WAS
    #      actually cut off as matn (a trailing speech verb, in these cases)
    #      gets the extra check. Checked BEFORE the nasab guard: a
    #      relational-idafa appositive ("زوج النبي", "بنت النبي") often carries
    #      a nasab connector (``زوج``/``بنت``) that would otherwise spare it.
    if was_truncated and ("النبي" in bare or "الرسول" in bare):
        return True
    if was_truncated:
        for i in range(len(bare) - 1):
            if bare[i] in _PROPHET_TITLE_LEADERS and bare[i + 1] == "الله":
                return True

    # PRECISION GUARD — spare a genuine lineage from the remaining (weaker)
    # punctuation / density signals, regardless of how long it is.
    nasab = sum(1 for t in bare if t in _NASAB_CONNECTORS)
    if nasab >= 2 or nasab / n >= _NASAB_SPARE_DENSITY:
        return False

    # (2d) Prophet-reference PROVENANCE (da#314): a narration-provenance tail — "…
    #      من رسول الله" ("… FROM the Messenger of Allah"), "هذا من رسول الله", "كمن
    #      زار رسول الله" — kept as a canonical "narrator" (mc-55 "ه من رسول الله",
    #      mc-38 bare "من رسول الله"). The existing Prophet detectors both miss it:
    #      step 6b's :func:`_is_prophet_reference` is LEADER-anchored (here the title
    #      sits in the TAIL, behind a preposition), and signal (2p) above is gated on
    #      ``was_truncated`` (these spans carry no isnad/matn verb, so nothing ever
    #      truncated and they are never re-scrutinized).
    #
    #      DISCRIMINATOR — what may legitimately precede "رسول الله" in a real name is
    #      an ism, and nothing else: "محمد رسول الله" (Muhammad, the Messenger of Allah
    #      — a REAL narrator, the muhaddithat fixture) carries ZERO matn particles and
    #      no partitive, so it is kept. A provenance fragment, by contrast, always
    #      carries the very preposition/particle that makes it provenance ("من" — the
    #      ablative — or a demonstrative "هذا"/"ذلك"). So: a Prophet reference ANYWHERE
    #      in the span, co-occurring with at least one partitive or matn particle, is a
    #      provenance/matn fragment, never a name. Runs AFTER the nasab guard, so a
    #      genuine lineage is already spared before this can see it.
    #      The particle disjunct EXCLUDES the apposition connectors (Sofia Cardoso, PR
    #      review). "أم" (umm — the commonest female kunya connector) normalizes to
    #      "ام", which is HOMOGRAPHIC with the disjunctive particle "أم" ("or") and is
    #      therefore in _MATN_PARTICLES; it is the only token in this module that is
    #      both. Without the exclusion, any real narrator shaped "أم X <role> رسول الله"
    #      — where the role noun is not already an apposition connector — reads as a
    #      Prophet reference PLUS a particle and is deleted. That is not hypothetical:
    #      it dropped "أم كلثوم بنت سيد البشر رسول الله" (Umm Kulthum bint Muhammad, the
    #      Prophet's DAUGHTER) and "أم أيمن حاضنة رسول الله" (Umm Ayman, his nurse and a
    #      transmitter) — both bio-promoted rows at mention_count 0, which is precisely
    #      why a mention-weighted sweep could not see them. The mubham collectives this
    #      rule targets are untouched: "بعض" is not an apposition connector, so
    #      "بعض اصحاب النبي" / "بعض ازواج النبي" still drop.
    #      SCOPING (second review round): the rule fires only when EVERY token is
    #      function-word residue or part of the Prophet reference itself — the same
    #      all-residue shape as rule 5b, which is the only shape proven safe here. A
    #      span carrying ANY substantive token (an ism, a nisba, a common noun) is
    #      structurally out of reach, which is what a "does it contain a real name?"
    #      test needs to be, absent a name lexicon.
    #
    #      This is not belt-and-braces; it is load-bearing. A rijal BIOGRAPHY entry
    #      names a real narrator and then describes him — and a companion's biography
    #      mentions the Prophet *by nature*: "شهد النبي في حجة الوداع" ("he witnessed
    #      the Prophet at the Farewell Pilgrimage"), "وفد على رسول الله" ("he came as a
    #      delegate to the Messenger of Allah"). Those carry a Prophet reference AND a
    #      preposition, so the earlier any()-form of this rule deleted 44 real
    #      bio-promoted narrators — among them ابو بكر الصديق … خليفه رسول الله (Abu
    #      Bakr al-Siddiq, the first Caliph), الطيب ولد رسول الله (the Prophet's son)
    #      and ذواليدين (Dhu al-Yadayn). All are mention_count 0, so — exactly like Umm
    #      Kulthum — a mention-weighted sweep could not see them.
    if any(_prophet_ref_len(bare, i) > 0 for i in range(n)) and all(
        _is_prophet_ref_component(t) or _is_provenance_residue(t) for t in bare
    ):
        return True

    # (2a) Divine-name / Qur'anic-verse signature (da#317): the bare divine name
    #      الله occurring TWICE OR MORE is a verse / oath / matn signature, never a
    #      real name. An الله that is the SECOND half of a theophoric compound name
    #      (``عبد الله``, ``هبه الله`` — preceded by a _THEOPHORIC_HEADS token) does
    #      NOT count: a double-theophoric narrator (``عبد الله بن عبد الله القرشي
    #      الاموي``) is a real name whose nasab density can dip below the spare
    #      threshold once nisbas are added, so counting its theophoric الله would
    #      false-drop it. The verse signal still fires on Qur'an 22:32 ("الله ومن …
    #      شعاءر الله …"): its first الله is sentence-initial (no head) and the
    #      second follows شعاءر (not a theophoric head) → still counts 2. رسول/نبي +
    #      الله is a Prophet reference already dropped by step 6b upstream.
    allah_count = sum(
        1
        for i, t in enumerate(bare)
        if t == "الله" and not (i > 0 and bare[i - 1] in _THEOPHORIC_HEADS)
    )
    if allah_count >= 2:
        return True

    # (2b) Matn function-word density (da#317): a span (past the nasab guard) with
    #      TWO OR MORE standalone matn particles — a preposition (على/في/الى),
    #      pronoun (نحن/اليك/له), conditional/negation (لو/ما/لا), or demonstrative
    #      (هذا/ذلك/هناك), each optionally carrying a leading و/ف proclitic — is
    #      narrative matn, not a name ("وتهيات للحج … على الخروج … نحن لذلك … اليك";
    #      "فقلت لو كنت هناك ما قتلتهم"). A real proper-noun name carries ZERO such
    #      tokens, so the ≥2 threshold has no false-positive surface on real names.
    if sum(1 for t in bare if _is_matn_particle(t)) >= 2:
        return True

    # (2c) All-function-word fragment (da#317): a span whose EVERY token is a matn
    #      particle (leading و/ف proclitic folded) is a bare narrative fragment, not
    #      a name — "وما" ("and not"), a lone "على", "ثم" already handled elsewhere.
    #      Distinct from clean_narrator_name's all-residue guard (isnad-formula
    #      words); this covers the pure prepositional/pronominal fragments.
    if all(_is_matn_particle(t) for t in bare):
        return True

    if n < _MATN_SENTENCE_MIN_TOKENS:
        return False

    # (3) Verb / particle density in the RETAINED span (comma-run sentence whose
    #     opener is a real-looking token but whose body carries matn/particle words
    #     — _MATN_DENSITY: كان/قال/اذ/لما/ثم/هو/…). A real name carries zero. A
    #     matn ``«…»`` / ``؟`` within the kept span (NOT a truncated-off tail) is a
    #     co-factor: it lowers the density needed from 2 to 1, but never drops on
    #     its own.
    density = sum(1 for t in bare if t in _MATN_DENSITY)
    has_matn_punct = any(p in kept_text for p in _MATN_PUNCT)
    return density >= 2 or (has_matn_punct and density >= 1)


def _is_name_matn_signal(token: str) -> bool:
    """True when *token* ends the name and begins matn/provenance (da#424).

    The non-positional half of the name/matn boundary test: the partitive ``من``,
    an "asked" verb, a matn particle (``على``/``في``/``هذا``/``انتم`` …), a
    matn-density verb (``كان``/``قال`` …), a matn opener (``قلت``/``نهى`` …, a
    single ``و``/``ف`` proclitic folded), or a mubham collective (``رجل``/``بعض`` —
    "a man / some of …", the head of a bio descriptor). Prophet references
    (``رسول الله``/``النبي``) are handled POSITIONALLY by the caller via
    :func:`_prophet_ref_len`, because ``رسول``/``نبي`` are only a reference when
    ``الله`` follows. The bare divine name ``الله`` is deliberately NOT a signal
    here — it is the second half of a theophoric ism (``عبد الله``) far more often
    than a matn token, and the provenance forms that carry it (``من رسول الله``)
    are already cut at ``من`` / ``رسول``.
    """
    if _is_partitive(token) or _is_ask_verb(token) or _is_matn_particle(token):
        return True
    if token in _MATN_DENSITY or token in _MUBHAM_LEADERS:
        return True
    folded = token[1:] if token[:1] in ("و", "ف") and len(token) > 1 else token
    return token in _MATN_OPENERS or folded in _MATN_OPENERS


def _is_fa_narrative(token: str) -> bool:
    """True when *token* is a ``ف``-proclitic narrative-continuation verb (da#424).

    The ``ف`` ("so / then") narrative proclitic on a ≥3-letter verb stem opens a
    matn clause (``فذهب`` "so he went", ``فقام`` "so he stood"). Shape-only and
    deliberately NOT self-sufficient: a ``ف``-initial ism (``فراس``, ``فيروز``,
    ``فديك``) has the same shape, so :func:`_is_fa_narrative_boundary` adds the two
    locality guards that tell a verb from an ism (preceding kunya / nasab connector,
    and LOCAL matn corroboration). A ``ف``-proclitic on a *particle* (``فلو``,
    ``فما``) is excluded here — it is already a :func:`_is_matn_particle` signal —
    and a ``ف``-stem that is itself a nasab connector is excluded so nothing name-
    bearing is mistaken for a verb.
    """
    if len(token) < 4 or token[0] != "ف":
        return False
    stem = token[1:]
    if _is_matn_particle(token) or stem in _NASAB_CONNECTORS:
        return False
    return True


def _is_matn_verb(token: str) -> bool:
    """True when *token* is a matn VERB signal surviving to step 10 (da#424).

    A matn opener (``قلت``/``نهى``/``صلى``/``روى`` …, a single ``و``/``ف`` proclitic
    folded) or an "asked" verb. The :data:`_ISNAD_BOUNDARY` verbs (``قال``/``كان``/
    ``حدثنا`` …) are already cut at step 3c, so the residual verb signals at this
    point are the openers + ask-verbs. This is the ONLY boundary class the
    KEPT-WHOLE recovery path truncates on — never a bare Prophet reference, particle
    or mubham noun — so a rijal biography that mentions the Prophet nominally
    (``خليفه رسول الله`` "Successor of the Messenger of Allah", ``شهد النبي في حجة
    الوداع`` "witnessed the Prophet at the Farewell Pilgrimage") is left whole, not
    lopped. A genuine matn NARRATIVE (``فذهب رسول الله …`` "so he went …") is caught
    via the ``ف``-narrative rule that the caller ORs in.
    """
    if _is_ask_verb(token):
        return True
    folded = token[1:] if token[:1] in ("و", "ف") and len(token) > 1 else token
    return token in _MATN_OPENERS or folded in _MATN_OPENERS


def _fa_narrative_corroborated(rest: list[str]) -> bool:
    """True when a NARRATION signal LOCALLY corroborates a ``ف``-narrative verb (da#424).

    *rest* is the token slice immediately AFTER a candidate ``ف``-token. Two guards:

    * **Locality** — the corroborating signal must appear BEFORE any intervening
      nasab connector. A ``ف``-initial ism inside a lineage (``… بن فراس بن مالك …``)
      is followed by ``بن``, which stops the scan and refuses corroboration, so the
      ism is not mistaken for a verb. (Oyunbileg, PR #474 review — replaced an
      ``any()`` over the WHOLE remainder that severed such names into ``X بن``.)
    * **Narration-only evidence** — the signal must be a genuine narration marker: a
      Prophet reference, an "asked" verb, a matn opener, or a matn-density discourse
      word (:func:`_is_matn_verb` + :data:`_MATN_DENSITY`). A bare PREPOSITION is
      NOT accepted (Nikolaos, PR #474 review): ``في``/``على``/``الى`` attach to a
      noun as readily as a verb, so a plain ``ف``-initial NISBA / ethnonym
      (``فارسي``, ``فلسطيني``, ``فرغاني``) followed by ``… في المدينه`` was read as a
      narrative verb and clipped — dropping the very nisba that disambiguates two
      otherwise-identical nasab leads (``عبد الله بن عمر الفارسي`` vs ``… الفلسطيني``),
      a wrong-cross-narrator-MERGE hazard. Requiring a narration marker still
      corroborates the genuine ``فذهب رسول الله`` (via the Prophet reference).
    """
    for j, tok in enumerate(rest):
        if tok in _NASAB_CONNECTORS:
            return False
        if _prophet_ref_len(rest, j) > 0 or _is_matn_verb(tok) or tok in _MATN_DENSITY:
            return True
    return False


def _is_fa_narrative_boundary(tokens: list[str], i: int) -> bool:
    """True when ``tokens[i]`` is a ``ف``-narrative verb that ENDS the name (da#424).

    Three conditions, all required — the ``ف``-narrative rule's precision guards:

    1. ``tokens[i]`` looks like a ``ف``-proclitic verb (:func:`_is_fa_narrative`).
    2. It is NOT the ism of an immediately-preceding kunya / nasab connector: a
       token right after ``ابو``/``ابي``/``ابا``/``بن``/``ابن`` is that name's ism,
       never a verb (``ابو فراس`` = "Abū Firās", ``بن فراس`` = "ibn Firās"), so a
       ``ف``-initial ism there must never truncate. (Oyunbileg, PR #474.)
    3. A matn signal corroborates it LOCALLY (:func:`_fa_narrative_corroborated`) —
       before any intervening nasab connector.
    """
    if not _is_fa_narrative(tokens[i]):
        return False
    if i > 0 and tokens[i - 1] in _NASAB_CONNECTORS:
        return False
    return _fa_narrative_corroborated(tokens[i + 1 :])


def _leading_name_before_matn(tokens: list[str], verb_only: bool) -> list[str]:
    """The leading run of name tokens, cut at the first matn boundary (da#424).

    Scans left to right and stops at the first matn boundary. A ``ف``-narrative verb
    (``فذهب`` "so he went") that clears the :func:`_is_fa_narrative_boundary` locality
    guards always ends the name. The rest of the boundary set depends on *verb_only*:

    * ``verb_only=True`` (the KEPT-WHOLE path — the span would otherwise SURVIVE):
      stop ONLY at a matn VERB (:func:`_is_matn_verb`). A bare Prophet reference,
      particle or mubham noun is NOT a boundary here, so a bio apposition
      (``خليفه رسول الله``, ``شهد النبي …``, ``وفد الى رسول الله``) is left whole.
      This confines kept-span truncation to genuine narrative matn.
    * ``verb_only=False`` (the DROP path — the span would otherwise be DELETED):
      also stop at a Prophet reference (:func:`_prophet_ref_len`) or any
      :func:`_is_name_matn_signal` token. Recovering here is purely additive — the
      span was going to be dropped — so it may truncate aggressively.

    Returns the tokens unchanged when no boundary is present. A theophoric ``الله``
    (``عبد الله``) is never a boundary, so an idafa ism is kept whole; the provenance
    ``من رسول الله`` is cut at ``من`` (only on the drop path).
    """
    for i, tok in enumerate(tokens):
        if _is_fa_narrative_boundary(tokens, i):
            return tokens[:i]
        if verb_only:
            if _is_matn_verb(tok):
                return tokens[:i]
        elif _prophet_ref_len(tokens, i) > 0 or _is_name_matn_signal(tok):
            return tokens[:i]
    return tokens


def _recover_leading_name(tokens: list[str], verb_only: bool) -> str | None:
    """Truncate-and-re-gate: recover a real leading name from a matn tail (da#424).

    Cuts *tokens* at the first matn boundary (:func:`_leading_name_before_matn`,
    passing *verb_only*) and re-gates the head through :func:`clean_narrator_name`.
    Returns the recovered clean name, or ``None`` when nothing is recoverable.

    Two guards make this safe and additive:

    * The head must carry a :data:`_NASAB_CONNECTORS` token (a kunya/ibn/bint/bn).
      This is what distinguishes a real leading narrator (``ابو هريره``, ``ابن عباس``)
      from a bare matn head that merely precedes the first matn word (``دعوني``,
      ``سريه تغزو``) — the latter has no nasab connector and is NOT resurrected.
    * A boundary must actually exist (``0 < len(head) < len(tokens)``) — a clean
      name with no matn tail returns ``None`` and is left untouched by the caller.

    The final :func:`clean_narrator_name` re-gate applies every downstream guard to
    the recovered head, so a head that is itself mubham / degenerate still drops.

    A grading-COMMENTARY span (``"ابو عيسى هذا حديث حسن صحيح"`` — al-Tirmidhī's own
    ḥadith verdict, keyed on his kunya) is deliberately NOT recovered: it is the
    compiler's editorial commentary, not a narration by a distinct narrator, and the
    compiler has a clean canonical of his own (class 9). Recovering the leading kunya
    would re-mint a redundant commentary mention, so these are left to drop.
    """
    if any(_contains_token_sequence(tokens, formula) for formula in _GRADING_FORMULAE):
        return None
    head = _leading_name_before_matn(tokens, verb_only)
    if not (0 < len(head) < len(tokens)):
        return None
    if not any(t in _NASAB_CONNECTORS for t in head):
        return None
    return clean_narrator_name(" ".join(head))


def split_compound_narrators(name_normalized: str | None) -> list[str]:
    """Split a compound co-narrator join into its member names (da#258 class 6).

    Detection + split primitive for the ``X و Y … جميعا`` / ``X و Y و Z قالوا``
    form — multiple REAL narrators joined by the conjunction ``و`` ("and") and a
    trailing co-narrator marker (``جميعا`` "together" / ``قالوا`` "they said").
    Returns the list of member name strings (each still to be run through
    :func:`clean_narrator_name` by the caller); returns a single-element list
    ``[name]`` when the span is not a compound, and ``[]`` for empty input.

    The trailing marker is stripped first, then the span is split on the
    conjunction in both its shapes: a standalone ``و`` token (``"سعد … و عبد الله
    …"``) and a proclitic ``و`` prefix on a member (``"… وقتيبه وابن حجر"``). The
    proclitic split fires **only inside a confirmed compound** (a standalone ``و``
    or a trailing marker is present) so it never mutates a lone ``و``-initial real
    name (``وكيع`` "Waki'", ``وهب`` "Wahb") outside that context.

    NOTE (scope): this is the detection/split *primitive*. Emitting one narrator
    mention row per member is a resolve-stage change to ``src/resolve/ner.py``'s two
    call sites (they currently map one span → one row) and is tracked as a scoped
    follow-up; until it is wired in, an un-split compound survives unchanged (it is
    never dropped — see :func:`clean_narrator_name`), so no co-narrator is lost.
    """
    if not name_normalized:
        return []
    tokens = [s for t in name_normalized.split() if (s := t.strip(_EDGE_PUNCT))]
    if not tokens:
        return []

    has_standalone_waw = "و" in tokens
    has_trailing_marker = tokens[-1] in _COMPOUND_TRAILING_MARKERS
    # A single proclitic-و member alone (e.g. "وقتيبه") is NOT a compound; require a
    # standalone-و OR a trailing جميعا/قالوا marker to confirm a multi-narrator join.
    if not (has_standalone_waw or has_trailing_marker):
        return [name_normalized.strip()]

    # Strip the trailing co-narrator marker(s) before splitting.
    while tokens and tokens[-1] in _COMPOUND_TRAILING_MARKERS:
        tokens.pop()

    members: list[list[str]] = [[]]
    for tok in tokens:
        if tok == "و":  # standalone conjunction → member boundary
            members.append([])
        elif tok.startswith("و") and len(tok) > 1 and members[-1]:
            # proclitic-و on a non-first member ("وقتيبه") → boundary + keep tail
            members.append([tok[1:]])
        else:
            members[-1].append(tok)

    parts = [" ".join(m) for m in members if m]
    return parts or [name_normalized.strip()]


def _strip_markup_and_honorifics(name_normalized: str) -> str:
    """Steps 1–2a of the cleaner: remove markup and honorific/title phrases.

    NORMALIZATION, not truncation — every one of these removals is an affix the
    source wrapped around a name it still asserts in full.
    """
    # 1. Strip markup tags + any stray angle brackets (no markup char survives).
    text = _MARKUP_RE.sub(" ", name_normalized).replace("<", " ").replace(">", " ")

    # 2. Strip honorific / eulogy phrases anywhere in the span. LONGEST-FIRST
    #    (da#311): a shorter phrase can be a PREFIX of a longer one — "رضى الله عنه"
    #    (masc.) is a prefix of "رضى الله عنها" (fem.); stripping the short form
    #    first left the fem. suffix "ا" as a stray one-letter token ("عاءشه، رضى
    #    الله عنها تقول" → "عاءشه ا"). Sorting by length descending strips the
    #    longest matching phrase first, so the whole benediction is removed cleanly.
    for phrase in sorted(_HONORIFIC_PHRASES, key=len, reverse=True):
        if phrase in text:
            text = text.replace(phrase, " ")

    # 2a. Strip leading honorific TITLE phrases (da#311) — a scholarly epithet
    #     prefixed to a real name ("الشيخ الجليل ابو جعفر …" = Ibn Babawayh /
    #     Shaykh al-Saduq). Distinct class from the eulogy/benediction phrases
    #     above (a title, not a prayer-formula), but stripped the same way so
    #     the real name underneath is recovered rather than kept title-polluted.
    for phrase in _HONORIFIC_TITLE_PHRASES:
        if phrase in text:
            text = text.replace(phrase, " ")

    # 2c. Latin / English benediction + leading-title residue (da#299). A romanized
    #     name from an English-gloss source carries a parenthesized benediction
    #     ("Muhammad(saw)") and/or a leading honorific title ("Prophet Muhammad");
    #     normalize_arabic leaves Latin untouched, so neither is caught by the Arabic
    #     honorific phrases above. Strip the parenthesized benediction anywhere and
    #     the honorific title only where it LEADS. (For the kaggle bio path the
    #     Arabic-span extraction has already dropped all Latin, so this is a no-op
    #     there; it recovers the name for the other English-gloss sources.)
    text = _LATIN_BENEDICTION_RE.sub(" ", text)
    text = _LATIN_TITLE_RE.sub("", text)
    return text


def _tokenize_and_strip_affixes(text: str) -> tuple[list[str], list[str], bool]:
    """Steps 3–3b of the cleaner: tokenize, drop connectives, pop leading ordinals.

    Returns ``(raw_tokens, tokens, stripped_number)``. Like
    :func:`_strip_markup_and_honorifics` these are all NORMALIZATIONS: the token
    sequence that survives is still the name the source asserted, merely unwrapped.
    """
    # 3. Tokenize: strip edge punctuation from each token, then drop the editorial
    #    connective and any pure-punctuation tokens. Without the per-token strip a
    #    stray trailing mark ("ابيه،", "ابي ،", "ابيه :") shields a mubham token
    #    from the relational / collective guards below (da#247 residual: 7 such
    #    refs survived the first scrub purely on a trailing Arabic comma).
    #    `raw_tokens` keeps each surviving token in its ORIGINAL form (with any
    #    «…» / ؟ that _EDGE_PUNCT strips off `tokens`), index-aligned to `tokens`,
    #    so the da#308 gate can scan matn punctuation over the RETAINED span only.
    raw_tokens: list[str] = []
    tokens: list[str] = []
    for t in text.split():
        stripped = t.strip(_EDGE_PUNCT)
        if not stripped or stripped in _CONNECTIVE_TOKENS:
            continue
        raw_tokens.append(t)
        tokens.append(stripped)
    if not tokens:
        return raw_tokens, tokens, False

    # 3b. Numeric-leading mis-parse (da#311). A span whose FIRST token is a bare
    #     number (ASCII or Arabic-Indic digits) is a thaqalayn parser artifact — a
    #     hadith/enumeration ordinal leaked in front of the real name or an isnad
    #     fragment ("1 علي بن ابراهيم", "5659 و روى محمد بن ابي عمير"). No real
    #     narrator name begins with a digit. An earlier pass DROPPED such rows on
    #     the assumption the trailing narrator had a clean canonical elsewhere — but
    #     that vanished real transmitters whose ONLY stored forms are numbered:
    #     Ali ibn Ibrahim al-Qummi (al-Kafi's most prolific isnad head) exists SOLELY
    #     as "1 علي بن ابراهيم" / "10 علي بن ابراهيم" / … so the drop erased him from
    #     the graph entirely (da#311 round-4). Instead STRIP the leading number
    #     token(s) and re-gate the remainder through every guard below: a real name
    #     ("علي بن ابراهيم") is recovered, while a mubham remainder ("عده من اصحابنا")
    #     still drops at the collective guard (5) and an isnad fragment ("و روى …")
    #     is still truncated/dropped by 3c. A span that is ONLY digits drops here.
    stripped_number = False
    while tokens and _is_number_token(tokens[0]):
        tokens.pop(0)
        raw_tokens.pop(0)
        stripped_number = True
    #     A number-mis-parse whose remainder now LEADS with a bare waw conjunction
    #     ("2 و عنه", "3 و باسناده", "5659 و روى …") is an isnad connective fragment,
    #     never a name — the leaked ordinal sat in front of a *continuation* clause,
    #     not a narrator. Strip the leading lone waw so the remainder is re-gated on
    #     its merits (dropping "و عنه"→"عنه", "و روى …"→boundary-drop) instead of the
    #     junk surfacing. Scoped to the mis-parse case (stripped_number) so ordinary
    #     compound "X و Y" spans — where the waw is MEDIAL — are never touched.
    if stripped_number:
        while tokens and normalize_arabic(tokens[0]) == _WAW_CONJUNCTION:
            tokens.pop(0)
            raw_tokens.pop(0)
    return raw_tokens, tokens, stripped_number


def is_mubham_relational(name_normalized: str | None) -> bool:
    """True when the WHOLE name is a single bare relational-pronoun (mubham) token.

    "his father", "my aunt", "O mother!" — an unnamed-relation reference, never a
    person. This is exactly the drop :func:`clean_narrator_name` step 6 already applies
    at NER/bio time; it is exposed so the resolve **alias-union** sites can apply the
    SAME drop to the ``aliases`` ``list<string>`` — the field ``clean_narrator_name``
    never reaches, and the da#391 blind spot (a relational deixis surviving as an alias
    of a real narrator, so any alias lookup resolves the wrong person).

    Normalizes internally (``normalize_arabic`` is idempotent) so a raw or already-
    normalized alias both resolve. The single-token guard is the precision boundary:
    a real multi-token kunya (``"ابي اسحاق"`` = Abū Isḥāq) returns ``False``, and the
    lexicon is kinship-only, so no real single-token ism is ever matched.
    """
    if not name_normalized:
        return False
    tokens = normalize_arabic(name_normalized).split()
    return len(tokens) == 1 and tokens[0] in _MUBHAM_RELATIONAL


def clean_narrator_name(name_normalized: str | None) -> str | None:
    """Clean a normalized narrator name; return the cleaned name or ``None`` to drop.

    Strips markup tags, the editorial ``يعني`` connective and honorific phrases,
    truncates a colon-joined English matn (da#253) or an Arabic isnad/matn tail
    (da#258 classes 4+5, recovering the leading narrator), then drops the span when
    it is empty, over-long (phrase / text body), a mubham collective descriptor, an
    English prose fragment, or a matn body. Operates in normalized-Arabic space.
    An un-split class-6 compound (``X و Y جميعا``) is preserved, not dropped — see
    :func:`split_compound_narrators`.
    """
    if not name_normalized:
        return None

    text = _strip_markup_and_honorifics(name_normalized)

    # 2b. Colon-joined English prose (da#253). Some sources emit "<name>:<matn>" —
    #     a companion name colon-joined to a hadith body ("Thawban:The Messenger of
    #     Allah … sacrificed …"). The colon is INTERNAL to the whitespace token
    #     ("Thawban:The"), so it hides the prose leader "The" from step 7's tokens[0]
    #     check, and an ~11-token body stays under the step-4 cap. Truncate at the
    #     first colon whose tail begins with an English leader, keeping the pre-colon
    #     name to be re-validated by every guard below (recovers "Thawban").
    #     This is a TRUNCATION, not a normalization — it is deliberately applied here
    #     and NOT in :func:`asserted_name_tokens` (da#379).
    text = _truncate_colon_prose(text)

    raw_tokens, tokens, stripped_number = _tokenize_and_strip_affixes(text)
    if not tokens:
        return None

    # 3c. Arabic isnad/matn boundary truncation (da#258 classes 4+5). Cut the span
    #     at the first matn-verb / isnad-connective / relative-pronoun / gloss token
    #     ("عن", "ان", "الذي", "كان", "قالت", "هو", …), keeping the real leading
    #     narrator that precedes it ("ابو محمد بصري عن محمد بن علي" → "ابو محمد
    #     بصري"; "عاءشه قالت كان" → "عاءشه"; "بندار هو محمد بن بشار" → "بندار"). A
    #     boundary that LEADS the span ("كان علي …", bare "قالوا"/"ثم") leaves no
    #     name → drop. None of these tokens is ever a name component, so real nasab
    #     lineages are never cut. Runs before the token cap so a truncated lead is
    #     re-measured, not the raw sentence.
    truncated = _truncate_at_isnad_boundary(tokens)
    if truncated is None:
        return None
    # da#311: whether truncation actually cut something off. A recovered
    # fragment (truncation DID fire) is a leading-name RECOVERY that must be
    # re-validated more strictly than a span that was never touched by the
    # boundary rule at all — see the subject-led signal in _is_matn_sentence.
    was_truncated = stripped_number or len(truncated) < len(tokens)
    tokens = truncated

    # 3d. Prophet-apposition truncation (da#311). A real narrator stored as the
    #     trimmed apposition "<name> زوج/بنت النبي" ("Aisha, wife of the Prophet"
    #     — mc-196 fawaz, the trailing قالت + benediction already stripped at NER
    #     time) has no isnad-boundary verb, so 3c leaves it whole. Cut at the
    #     relational connector whose next token is a Prophet reference, recovering
    #     the leading name ("عاءشه زوج النبي" → "عاءشه"); a bare "زوج النبي" (no
    #     name before the connector) drops. "عاءشه بنت سعد" (بنت + a real name, not
    #     a Prophet ref) is never touched.
    appos = _truncate_at_prophet_apposition(tokens)
    if appos is None:
        return None
    if len(appos) < len(tokens):
        was_truncated = True
        tokens = appos

    # 3e. Strip any TRAILING isnad-boundary connective the truncation left dangling
    #     (da#311, idempotency). The nisba-transition ثم skip (see
    #     _truncate_at_isnad_boundary) can leave a ثم at the end when the following
    #     ال-token was itself truncated ("اتموا الصف الاول ثم") — a lone trailing
    #     ثم/عن/ان/… is never a name component, so pop it. This makes the recovery a
    #     fixed point (clean(clean(x)) == clean(x)) in a single pass.
    #
    #     A trailing bare waw CONJUNCTION is popped the same way (da#316): the source
    #     stores a dangling co-narrator join whose second member was never captured
    #     ("مالك،‏.‏ و" — Malik, followed by an "and" that leads nowhere). A waw with
    #     no following member conjoins nothing and is never a name component. Only the
    #     TRAILING position is touched, so a MEDIAL waw — the class-6 compound join
    #     "X و Y" that :func:`split_compound_narrators` must still see — is untouched.
    while tokens and tokens[-1] in _ISNAD_BOUNDARY:
        tokens.pop()
        was_truncated = True
    #     The dangling waw is popped WITHOUT setting was_truncated. That flag means "a
    #     tail was cut off, so this is a RECOVERED fragment and needs stricter scrutiny"
    #     — and it gates signal (2p), which drops any span carrying a bare Prophet
    #     reference. A trailing conjunction is an affix, not name material: popping it
    #     removes nothing the source asserted, so it must not escalate scrutiny. Setting
    #     it here deleted "ابو بكر الصديق … خليفه رسول الله و" — ABU BAKR AL-SIDDIQ, whose
    #     bio ends in a dangling waw — because the pop flipped was_truncated and (2p)
    #     then fired on the "رسول الله" in "خليفه رسول الله" ("Successor of the Messenger
    #     of Allah"). Without the flag he is spared by the nasab guard (ابو/بن/ابي), as he
    #     should be. Same reasoning as :func:`cleaner_removed_content`: an affix is not a
    #     tail.
    while tokens and tokens[-1] == _WAW_CONJUNCTION:
        tokens.pop()
    if not tokens:
        return None

    # 4. Over-long span → phrase / sentence / mis-parsed hadith body.
    if len(tokens) > _MAX_NAME_TOKENS:
        return None

    # 5. Mubham (anonymous collective) descriptor → not a named narrator.
    if tokens[0] in _MUBHAM_LEADERS and "من" in tokens:
        return None

    # 5b. All-residue span (da#311 round-4b — Kavitha's review). A span whose EVERY
    #     token is bare isnad/matn residue — a chain-formula word ("باسناده"), its
    #     attached-waw form ("وعنه"/"وباسناده"), the passive narrate verb ("روي"),
    #     the particle ("قد"/"قد روي"), or a bare mubham collective *without* the
    #     partitive "من" that guard 5 requires ("رجل" = "a man" mc-4920, "بعض",
    #     "شيخ", "جماعه", "ناس") — is an isnad continuation / anonymous descriptor,
    #     never a named narrator. This closes the bare-collective gap guard 5 left
    #     open (it minted high-mention pollution nodes — the same أبيه class this
    #     series exists to eliminate) AND drops the residue a numeric mis-parse
    #     leaves behind. _is_isnad_residue_token folds a leading attached-waw so a
    #     real waw-initial name ("وهب", "وكيع") is NOT matched; a mixed span with any
    #     real name token ("رجل من الانصار" already caught by 5; "شيخ الطاءفه")
    #     survives here because not ALL its tokens are residue.
    if all(_is_isnad_residue_token(t) for t in tokens):
        return None

    # 6. Bare relational-pronoun reference ("his father", "his grandfather", "my
    #    father") → an unnamed (mubham) narrator, not a person. Drop ONLY when the
    #    WHOLE name is exactly one such token, so multi-token kunya names
    #    ("ابي اسحاق" = Abū Isḥāq) survive untouched (precision guard).
    if len(tokens) == 1 and tokens[0] in _MUBHAM_RELATIONAL:
        return None

    # 6b. Prophet reference (da#271): after the taṣliya was stripped (step 2) the
    #     residue is a bare title — "رسول الله", "النبي", "نبي الله". The Prophet is
    #     the matn's source, not an isnad narrator, so a span leading with one of
    #     these is dropped. Leader-anchored + the "الله" requirement keep it from
    #     touching a real name.
    if _is_prophet_reference(tokens):
        return None

    # 7. English non-name fragment (translated isnad prose in the name field):
    #    first token is a function/pronoun/verb leader ("It was…", "The Prophet…",
    #    "his father"). Name-leaders (abu/ibn/abd/al/…) are excluded from the set,
    #    so romanized narrators are never touched. Case-folded; a no-op on Arabic.
    if tokens[0].lower() in _EN_NONNAME_LEADERS:
        return None

    # 8. Embedded English prose without a leading leader (da#253). A "<name>:<matn>"
    #    join whose tail step 2b could not fully strip, or a name+prose run, still
    #    reads as a sentence: it carries multiple English function/stop words ("of",
    #    "and", "then", "narrated"). A real romanized name carries ZERO leader tokens
    #    and an Arabic name can never match an ASCII word — so >= 2 signals prose.
    leader_tokens = sum(1 for t in tokens if t.lower() in _EN_NONNAME_LEADERS)
    if leader_tokens >= _MIN_PROSE_LEADER_TOKENS:
        return None

    # 9. Arabic matn-density backstop (da#258 class 4). A span whose leading tokens
    #    read like a name but whose tail is matn NOT anchored by an _ISNAD_BOUNDARY
    #    verb (a bare-particle run the step-3c truncation could not cut on) still
    #    carries >= 2 whole-token matn/particle words. A real name carries zero; a
    #    class-6 compound ending in a single قالوا marker carries one (density 1) and
    #    is preserved for splitting.
    if sum(1 for t in tokens if t in _MATN_DENSITY) >= _MIN_PROSE_LEADER_TOKENS:
        return None

    # 10. Sentence / matn-body drop-gate (da#308). The residual matn whose opener is
    #     not an _ISNAD_BOUNDARY token (so step 3c did not truncate it): a whole
    #     hadith body ("قلت …"/"نهى رسول الله …"), an answer particle ("نعم"), a
    #     grading verdict ("… هذا حديث حسن صحيح"), or a quoted-speech / question span.
    #     Precision-guarded by nasab-connector density so genuine lineages, nisba
    #     tails, and kunya compounds are never dropped. `kept_text` is the raw text
    #     of the RETAINED tokens only, so «…» / ؟ in an isnad/matn tail that step 3c
    #     already truncated off does NOT drop the recovered leading name (#310).
    kept_text = " ".join(raw_tokens[: len(tokens)])
    matn = _is_matn_sentence(tokens, kept_text, was_truncated)

    # 10b. Truncate-and-re-gate leading-name recovery (da#424). A span leading with
    #     a real nasab/kunya name whose tail is matn or provenance — carrying no
    #     _ISNAD_BOUNDARY verb, so step 3c never truncated it — is recovered to the
    #     leading name rather than dropped whole or kept matn-polluted:
    #     "ابو هريره فذهب رسول الله وانتم تنتثلونها" → "ابو هريره" (kept-whole →
    #     cleaned); "ابن عباس …" / "ابو القاسم …" leading a matn quote (dropped by
    #     the gate above → recovered). Purely a RECOVERY: it only replaces a drop or
    #     a matn-polluted whole span with a cleaner name, never turns a survivor into
    #     a drop. The recovered head must carry a nasab connector (so a bare matn
    #     head — "دعوني", "سريه تغزو" — is not resurrected), and it re-gates through
    #     the full cleaner. It does NOT loosen #423's all-residue provenance scoping:
    #     a substantive-but-matn residual with no nasab lead ("كمن زار رسول الله") is
    #     left KEPT, exactly as #423 requires.
    #     The recovery is aggressive on the DROP path (matn — the span would be
    #     deleted anyway, so any recovered name is a strict gain) and conservative on
    #     the KEPT-WHOLE path (verb_only — truncate only at a genuine narrative verb,
    #     so a bio apposition mentioning the Prophet nominally is left whole).
    recovered = _recover_leading_name(tokens, verb_only=not matn)
    if recovered is not None:
        return recovered
    if matn:
        return None

    # 11. Degenerate-result floor (da#311). A recovery that boils down to a single
    #     stray letter / zero-width mark / punctuation / lone latin char ("ه", "و",
    #     "p") is garbage, not a name — drop rather than emit it. Real 2+-letter
    #     Arabic names and romanized names (Malik) clear the floor.
    result = " ".join(tokens)
    if _is_degenerate_name(result):
        return None

    return result


def asserted_name_tokens(name_normalized: str | None) -> tuple[str, ...]:
    """The tokens the SOURCE asserted as the name — the cleaner's normalizations, no truncation.

    Runs exactly the prefix of :func:`clean_narrator_name` that *unwraps* a name
    without cutting it: markup strip, honorific/title strip, edge-punctuation strip,
    connective drop, and the leading-ordinal pop. Everything the cleaner does after
    that point — colon-prose truncation, the isnad/matn boundary cut, the Prophet
    apposition cut, the trailing-connective pop — removes name material and is
    deliberately excluded.
    """
    if not name_normalized:
        return ()
    _raw, tokens, _stripped_number = _tokenize_and_strip_affixes(
        _strip_markup_and_honorifics(name_normalized)
    )
    return tuple(tokens)


def cleaner_removed_content(name_normalized: str | None, cleaned: str | None) -> bool:
    """True when :func:`clean_narrator_name` removed name-span material, not just an affix (da#379).

    ``clean_narrator_name`` does two categorically different things that its return
    value alone does not distinguish:

    * **Normalization** — markup, honorifics, editorial connectives and a bracketed
      catalog entry number are affixes *around* a name. Stripping them leaves the
      name the source asserted, whole: ``"( 4875 ) عبد الله بن موسى"`` →
      ``"عبد الله بن موسى"``, still that narrator's full name. This returns ``False``.
    * **Truncation** — an isnad/matn tail or a colon-joined body is cut off, keeping
      only a head. This returns ``True``. Note it does NOT imply the cleaner "cut into
      the name" (da#398): truncation splits into two sub-cases the return value again
      does not separate —

        - *cut into the name*: the residue is a fragment, frequently a bare ism —
          ``"عبيدة مولى رسول الله ذكره بن شاهين …"`` (an obscure client of the Prophet)
          reduces to ``"عبيده"``, the name of a different, heavily-attested narrator.
        - *tail after a complete name*: the asserted name is INTACT and only a chain
          continuation after it was removed — ``"اسحاق بن مره عن انس"`` → ``"اسحاق بن
          مره"`` (the full nasab; only the teacher-key ``عن انس`` cut). The name was
          not cut *into*; a tail *after* it was removed. Roughly 40% of the corpus's
          truncations are this class (da#397/da#398).

    So a ``True`` means "removed more than an affix", NOT "the residue is not a name".
    A caller keying identity on the cleaner's output must not over-read it (da#398):
    it separates affix-normalization from name-material removal, and no finer. Deciding
    whether a tail-after-complete-name residue is the SAME person as an existing record
    is a homonymy question this function cannot answer — it needs biographical
    disambiguation, not a token diff.

    The finer *gloss-tail vs name-cut* split this boolean deliberately withholds is
    carried by :func:`is_recoverable_gloss_tail` (da#397): it recognises the
    tail-after-a-complete-name class (a complete nasab that lost only a teacher-key
    isnad tail) as the recoverable-pending-disambiguation subset. This predicate stays
    the coarse affix/removal line; that one adds the sub-classification — and neither
    decides identity, which still needs biographical disambiguation.

    Comparing against :func:`asserted_name_tokens` — rather than a re-tokenization of
    the raw input — is what draws the affix/removal line; a re-tokenization sees the
    entry number, the connective and the honorific as "content removed" and refuses a
    merge the source fully supports.
    """
    if not name_normalized or not cleaned:
        return False
    return tuple(cleaned.split()) != asserted_name_tokens(name_normalized)


# --- Gloss-tail vs name-cut discrimination (da#397) -----------------------------
# Isnad TRANSMISSION connectors that open a "teacher key" — a rijāl compiler's
# "<narrator> عن <teacher>" / "<narrator> حدثنا <student>" chain fragment appended
# AFTER a complete name to disambiguate a narrator known through one chain. A removed
# tail opening on one of these is a GLOSS about how the narrator transmitted, not
# alternative name material. Deliberately a SUBSET of _ISNAD_BOUNDARY: it holds the
# transmission verbs + عن and EXCLUDES the dispute/speech openers (قيل / قال / يقال /
# وقيل — "it is said" / "he said"), which introduce a DISPUTED or alternative identity
# and must not read as a mere gloss, and the relative-pronoun / temporal boundaries.
_TEACHER_KEY_OPENERS = frozenset(
    {
        "عن",  # "from" — the canonical teacher key
        "عنه",  # "from him"
        "حدثنا",
        "حدثني",
        "حدث",
        "حدثت",
        "حدثاه",
        "اخبرنا",
        "اخبرني",
        "انبانا",
        "سمعت",
        "سمع",
        "روت",  # "she narrated" (روى sibling)
    }
)

# Explicit disputed-identity markers (da#397). Even when a removed tail opens on a
# teacher key AND the residue is a complete nasab, a tail carrying one of these grades
# the SUBJECT unidentified/disputed — the ``محمد بن شرحبيل … وقيل اسمه عمرو مجهول`` case
# — so the residue is NOT a safely-discriminating name. Checked against the removed
# tail only, keeping the graded-مجهول / ``وقيل اسمه`` case conservative (class B /
# refused). ``يعرف`` catches the ``لا يعرف`` ("not known") tail; the residue itself is
# never inspected for these (a real name never carries them).
_DISPUTED_IDENTITY_MARKERS = frozenset({"مجهول", "يعرف", "وقيل", "قيل", "يقال"})

# Patronymic nasab connectors (da#397). A "complete nasab" residue must carry an
# ``ibn`` link — narrower than :data:`_NASAB_CONNECTORS` (which also spares kunya
# ``ابو``/``ابي`` and tribal ``بنو``): da#397's discriminating-name test is a
# lineage ``<ism> بن <father> …``, not a bare kunya.
_PATRONYMIC_CONNECTORS = frozenset({"بن", "ابن"})


def is_recoverable_gloss_tail(name_normalized: str | None, cleaned: str | None) -> bool:
    """True when a truncation is a class-A *gloss-tail* cut (da#397).

    Discriminates da#397's two categorically different truncations, which
    :func:`cleaner_removed_content` alone conflates:

    * **Class A — gloss-tail (this returns True):** the removed tail opens on an isnad
      TRANSMISSION connector (``عن`` / ``حدثنا`` / … — a teacher key, not name
      material) AND the residue is a *complete nasab* (``>= 3`` tokens carrying a
      patronymic ``بن``/``ابن``) AND the removed tail carries no disputed-identity
      marker. The residue IS the discriminating name the source asserted
      (``اسحاق بن مره عن انس`` → ``اسحاق بن مره``), so it is the
      RECOVERABLE-pending-disambiguation subset.
    * **Class B — name-cut (this returns False):** a bare ism/kunya residue
      (``عبيده``, ``ابو نهشل``), a non-teacher-key cut (``… يقال`` / ``… الذي``), or a
      tail/span carrying a disputed-identity marker (``مجهول`` / ``وقيل`` — the
      ``محمد بن شرحبيل … مجهول`` homonym). Refusing these stays correct.

    IMPORTANT — this does NOT decide the merge. Recovering class A onto an attested
    narrator is NOT provably safe: :func:`~src.parse.identity.make_canonical_id` folds
    Arabic inflection but NOT homonymy, and a complete nasab can still collide with a
    homonym — measured, 45 of 240 class-A accept-target name-keys carry conflicting
    jarḥ grades (one key graded both *thiqa* and *kadhdhab* = provably >1 person; da#397
    PR#441 / da#423). So :func:`~src.resolve.bio_promote.promote_bios_to_canonical`
    still REFUSES class A; this predicate only *quantifies* the recoverable subset
    (splitting the refusal counter, da#398 scope-1) and is the precondition for a
    future disambiguation-gated recovery. Same coarseness caveat as
    :func:`cleaner_removed_content`: a token diff cannot answer the homonymy question.

    Returns ``False`` for a non-truncation (nothing was cut) and for class B.
    """
    if not cleaned or not cleaner_removed_content(name_normalized, cleaned):
        return False
    asserted = asserted_name_tokens(name_normalized)
    residue = tuple(cleaned.split())
    # The removed material must be a TRAILING cut — residue is a strict prefix of the
    # asserted tokens. When it is not (a colon-prose cut, or any unexpected shape) the
    # removed-tail opener cannot be identified, so classify conservatively as name-cut.
    if len(residue) >= len(asserted) or asserted[: len(residue)] != residue:
        return False
    removed_tail = asserted[len(residue) :]
    # (a) tail opens on a teacher-key / transmission connector (not a dispute/speech
    #     opener, not a relative pronoun / temporal boundary).
    if removed_tail[0] not in _TEACHER_KEY_OPENERS:
        return False
    # (b) no disputed-identity marker anywhere in the removed tail — the graded-مجهول /
    #     ``وقيل اسمه`` case stays conservative (class B).
    if any(t in _DISPUTED_IDENTITY_MARKERS for t in removed_tail):
        return False
    # (c) residue is a COMPLETE nasab: >= 3 tokens carrying a patronymic connector. A
    #     bare ism/kunya residue may not merge regardless of tail (da#397).
    if len(residue) < 3:
        return False
    return any(t in _PATRONYMIC_CONNECTORS for t in residue)


def clean_narrator_name_display(name_ar: str | None) -> str | None:
    """Clean the VOWELED, load-bearing ``name_ar`` display field; drop or recover (da#311).

    The graph loader keys a Narrator node's ``name`` on ``name_ar`` first, only
    falling back to ``name_ar_normalized`` when ``name_ar`` is empty
    (``src/graph/load_nodes.py``: ``name_ar = _val(row,"name_ar","") or
    _val(row,"name_ar_normalized","")``; ``SET n.name_ar = row.name_ar``). So the
    matn scrub MUST clean ``name_ar`` — cleaning only the normalized field leaves
    the load-bearing matn in place. ``name_ar`` also frequently carries the FULL
    matn body while ``name_ar_normalized`` was already truncated at NER time, so
    this cleans ``name_ar`` on its own terms.

    Reuses the full :func:`clean_narrator_name` gate (normalizing internally for
    matching) and maps its recovered value BACK onto the voweled tokens so the
    display keeps its diacritics where possible:

    * ``clean_narrator_name`` returns ``None`` → return ``None`` (drop the row).
    * otherwise align the cleaned normalized tokens to a contiguous run of the
      voweled tokens and return that run (``"أبو عائشة"`` unchanged; the mc-196
      ``"عاءشه، زوج النبي صلى الله عليه وسلم قالت"`` → ``"عاءشه"``). When alignment
      fails (a mid-name honorific broke contiguity) fall back to the cleaned
      normalized value — denatured, but clean (never matn).
    """
    if not name_ar or not name_ar.strip():
        return None
    display = strip_markup(name_ar)
    if not display:
        return None

    cleaned = clean_narrator_name(normalize_arabic(display))
    if cleaned is None:
        return None

    # Build voweled display tokens + their normalized forms, dropping the same
    # empty-after-edge-strip / editorial-connective tokens ``clean_narrator_name``
    # drops during tokenization — so a standalone internal ``:`` / ``،`` / ``يعني``
    # (which becomes an empty normalized token) does NOT break the contiguous-run
    # alignment and force a denaturing fallback (``"سالم بن أبي الجعد : رافع"`` must
    # keep its diacritics, not collapse to the normalized form).
    raw_tokens: list[str] = []
    norm_tokens: list[str] = []
    for t in display.split():
        norm = normalize_arabic(t).strip(_EDGE_PUNCT)
        if not norm or norm in _CONNECTIVE_TOKENS:
            continue
        raw_tokens.append(t.strip(_EDGE_PUNCT))
        norm_tokens.append(norm)

    target = [t.strip(_EDGE_PUNCT) for t in cleaned.split()]
    span = len(target)
    if span == 0:
        return None

    for start in range(len(norm_tokens) - span + 1):
        if norm_tokens[start : start + span] == target:
            recovered = " ".join(raw_tokens[start : start + span]).strip()
            return recovered or cleaned
    # No contiguous voweled run (e.g. a mid-name honorific was removed): the cleaned
    # normalized value is still clean (never matn), so use it rather than risk
    # re-emitting the voweled matn.
    return cleaned
