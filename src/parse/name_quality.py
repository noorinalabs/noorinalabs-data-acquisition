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

The phrase constants are in **normalized-Arabic** form (post
``normalize_arabic``) because this runs on ``name_normalized``.
"""

from __future__ import annotations

import re

__all__ = ["clean_narrator_name", "split_compound_narrators", "strip_markup"]

# Angle-bracket markup that leaks from source isnad fields (sanadset <NAR>/<IDF>
# tags, frequently unclosed or nested). Stripped, not rejected — the tag usually
# wraps a real name. The trailing ``.replace`` mops up any stray lone bracket so
# no markup character can survive into a canonical name (da#247 acceptance).
_MARKUP_RE = re.compile(r"<[^>]*>")

# Editorial connective ("that is / i.e.") — never part of a name.
_CONNECTIVE_TOKENS = frozenset({"يعني"})

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
    "رحمه الله",
    "عز وجل",
    "تبارك وتعالى",
)

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
_PROPHET_TITLE_SOLE = frozenset({"النبي", "الرسول"})
_PROPHET_TITLE_LEADERS = frozenset({"رسول", "نبي"})

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
        # isnad subordinators / relative pronouns / clause-opening particles
        "عن",
        "ان",
        "الذي",
        "التي",
        "الذين",
        "اذ",
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
    }
)

# Compound co-narrator join marker: a span ending in one of these ("… together",
# "… they said") is multiple narrators (class 6). Used by split_compound_narrators
# to confirm a compound and to strip the trailing marker before splitting.
_COMPOUND_TRAILING_MARKERS = frozenset({"جميعا", "قالوا"})

# Trailing / edge punctuation seen on extracted spans ("شيخ من اهل المدينه ,").
_EDGE_PUNCT = " \t\r\n,،.;؛:-_\"'«»()[]"


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
    """
    if not name:
        return ""
    cleaned = _MARKUP_RE.sub(" ", name).replace("<", " ").replace(">", " ")
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
    """
    for i, tok in enumerate(tokens):
        if tok in _ISNAD_BOUNDARY:
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

    # 1. Strip markup tags + any stray angle brackets (no markup char survives).
    text = _MARKUP_RE.sub(" ", name_normalized).replace("<", " ").replace(">", " ")

    # 2. Strip honorific / eulogy phrases anywhere in the span.
    for phrase in _HONORIFIC_PHRASES:
        if phrase in text:
            text = text.replace(phrase, " ")

    # 2b. Colon-joined English prose (da#253). Some sources emit "<name>:<matn>" —
    #     a companion name colon-joined to a hadith body ("Thawban:The Messenger of
    #     Allah … sacrificed …"). The colon is INTERNAL to the whitespace token
    #     ("Thawban:The"), so it hides the prose leader "The" from step 7's tokens[0]
    #     check, and an ~11-token body stays under the step-4 cap. Truncate at the
    #     first colon whose tail begins with an English leader, keeping the pre-colon
    #     name to be re-validated by every guard below (recovers "Thawban").
    text = _truncate_colon_prose(text)

    # 3. Tokenize: strip edge punctuation from each token, then drop the editorial
    #    connective and any pure-punctuation tokens. Without the per-token strip a
    #    stray trailing mark ("ابيه،", "ابي ،", "ابيه :") shields a mubham token
    #    from the relational / collective guards below (da#247 residual: 7 such
    #    refs survived the first scrub purely on a trailing Arabic comma).
    tokens = [
        stripped
        for t in text.split()
        if (stripped := t.strip(_EDGE_PUNCT)) and stripped not in _CONNECTIVE_TOKENS
    ]
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
    tokens = truncated

    # 4. Over-long span → phrase / sentence / mis-parsed hadith body.
    if len(tokens) > _MAX_NAME_TOKENS:
        return None

    # 5. Mubham (anonymous collective) descriptor → not a named narrator.
    if tokens[0] in _MUBHAM_LEADERS and "من" in tokens:
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

    return " ".join(tokens)
