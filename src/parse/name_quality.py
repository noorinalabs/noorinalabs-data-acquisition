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
3. mubham (unnamed) descriptors minted as named narrators — both *collective*
   phrases (``"رجل من اصحاب النبي"``, ``"جماعه من اصحاب"``) and bare *relational
   pronouns* (``"ابيه"`` "his father", ``"جده"`` "his grandfather"), plus bare
   honorific phrases. The relational-pronoun sub-class was the residual after the
   first da#247 pass: ``ابيه`` alone remained the single most-mentioned "narrator"
   (65,755) on the #723 stage reload because it has no partitive ``من`` for the
   collective guard to catch.

The phrase constants are in **normalized-Arabic** form (post
``normalize_arabic``) because this runs on ``name_normalized``.
"""

from __future__ import annotations

import re

__all__ = ["clean_narrator_name", "strip_markup"]

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

# A narrator name longer than this many whitespace tokens is a phrase / sentence /
# mis-parsed text body (thaqalayn dumps whole hadith bodies of 50–500+ tokens),
# not a name. The cap is set high (30) on purpose: classical full nasab lineages
# ("ابو القاسم عبد الرحمن بن الحسن بن احمد بن محمد بن عبيد بن عبد الملك …") run
# 20–30 tokens and are REAL narrators that must not be dropped — false-dropping a
# real name is worse than letting a rare mid-length junk span through (the proper
# thaqalayn parser fix, not this backstop, is the real remedy for the text dumps).
_MAX_NAME_TOKENS = 30

# Trailing / edge punctuation seen on extracted spans ("شيخ من اهل المدينه ,").
_EDGE_PUNCT = " \t\r\n,،.;؛:-_\"'«»()[]"


def strip_markup(name: str | None) -> str:
    """Remove angle-bracket markup tags + stray brackets and trim edge punctuation.

    Display-safe cleaner for the voweled ``name_raw`` (keeps diacritics); the
    normalized name goes through :func:`clean_narrator_name` instead.
    """
    if not name:
        return ""
    cleaned = _MARKUP_RE.sub(" ", name).replace("<", " ").replace(">", " ")
    return " ".join(cleaned.split()).strip(_EDGE_PUNCT).strip()


def clean_narrator_name(name_normalized: str | None) -> str | None:
    """Clean a normalized narrator name; return the cleaned name or ``None`` to drop.

    Strips markup tags, the editorial ``يعني`` connective and honorific phrases,
    then drops the span when it is empty, over-long (phrase / text body), or a
    mubham collective descriptor. Operates in normalized-Arabic space.
    """
    if not name_normalized:
        return None

    # 1. Strip markup tags + any stray angle brackets (no markup char survives).
    text = _MARKUP_RE.sub(" ", name_normalized).replace("<", " ").replace(">", " ")

    # 2. Strip honorific / eulogy phrases anywhere in the span.
    for phrase in _HONORIFIC_PHRASES:
        if phrase in text:
            text = text.replace(phrase, " ")

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

    return " ".join(tokens)
