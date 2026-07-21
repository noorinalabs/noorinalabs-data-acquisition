"""Pure-Python Arabic text processing utilities for hadith isnad analysis.

All functions use only ``re`` and ``unicodedata`` from the standard library.
Compiled regex patterns are defined at module level for performance.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_diacritics",
    "strip_format_marks",
    "normalize_alif",
    "normalize_hamza",
    "normalize_taa_marbuta",
    "normalize_urdu",
    "normalize_arabic",
    "clean_whitespace",
    "strip_to_letters",
    "is_arabic",
    "extract_arabic_span",
    "extract_transmission_phrases",
    "contains_transmission_marker",
    "canonical_surface",
    "transliterate",
]

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Arabic tashkeel (diacritics): U+064B–U+065F and U+0670 (superscript alef)
_DIACRITICS_RE: re.Pattern[str] = re.compile(r"[\u064B-\u065F\u0670]")

# Alif variants: أ (U+0623), إ (U+0625), آ (U+0622), ٱ (U+0671)
_ALIF_VARIANTS_RE: re.Pattern[str] = re.compile(r"[\u0623\u0625\u0622\u0671]")

# Hamza-on-carrier variants: ؤ (U+0624), ئ (U+0626)
_HAMZA_VARIANTS_RE: re.Pattern[str] = re.compile(r"[\u0624\u0626]")

# Taa marbuta: ة (U+0629)
_TAA_MARBUTA_RE: re.Pattern[str] = re.compile(r"\u0629")

# Alif maqsura: U+0649 (the dotless final yaa), folded to yaa U+064A (da#427)
_ALIF_MAQSURA_RE: re.Pattern[str] = re.compile(r"\u0649")

# Tatweel / kashida: ـ (U+0640)
_TATWEEL_RE: re.Pattern[str] = re.compile(r"\u0640")

# Bidi / zero-width / format control marks that leak from source text: zero-width
# space + ZWNJ/ZWJ (U+200B-U+200D), LRM/RLM (U+200E/F), the bidi embedding /
# override / isolate controls (U+202A-U+202E, U+2066-U+2069), BOM (U+FEFF), and
# the Arabic end-of-ayah mark (U+06DD). None carry name meaning; a stray one
# otherwise splits an identical name into a distinct canonical node (da#271: 3,114
# canonical names carried one, e.g. an RLM embedded mid-name).
_FORMAT_MARKS_RE: re.Pattern[str] = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u06dd]"
)

# Multiple whitespace
_MULTI_WS_RE: re.Pattern[str] = re.compile(r"\s+")

# Urdu / Persian letter variants -> their Arabic-script equivalents (da#299). The
# kaggle rijal bios spell names (and their benediction tails, "salla llahu alayhi
# wa-sallam") with Urdu keyboard letterforms -- the Urdu yeh U+06CC, keheh
# U+06A9, heh-goal U+06C1, ... -- which are DISTINCT codepoints from the Arabic
# letters they render (yeh U+064A, kaf U+0643, heh U+0647). Left un-folded, an
# Urdu-script name never collapses onto the canonical form of its Arabic twin, so
# the honorific-strip and canonical-id keying both miss it. This fold is
# precision-safe by construction: every key is a Urdu/Persian-specific codepoint
# that does not occur in real Arabic name text, so folding it can only touch
# Urdu-script input -- an all-Arabic name passes through untouched. Only variants
# with an unambiguous Arabic target are folded; Urdu/Persian consonants that have
# NO Arabic counterpart (peh, cheh, jeh, gaf, retroflexes) are deliberately left
# alone rather than invented a fold for (precision-first).
_URDU_FOLD_MAP: dict[str, str] = {
    "ک": "ك",  # KEHEH                 -> KAF
    "ڪ": "ك",  # SWASH KAF             -> KAF
    "ی": "ي",  # FARSI YEH             -> YEH
    "ے": "ي",  # YEH BARREE            -> YEH
    "ۓ": "ي",  # YEH BARREE WITH HAMZA -> YEH
    "ہ": "ه",  # HEH GOAL              -> HEH
    "ۂ": "ه",  # HEH GOAL WITH HAMZA   -> HEH
    "ۀ": "ه",  # HEH WITH YEH ABOVE    -> HEH
    "ھ": "ه",  # HEH DOACHASHMEE       -> HEH
    "ں": "ن",  # NOON GHUNNA           -> NOON
}
_URDU_TRANSLATION: dict[int, str] = str.maketrans(_URDU_FOLD_MAP)

# Arabic-script characters (base block U+0600-06FF, supplement U+0750-077F,
# extended-A U+08A0-08FF, plus the presentation-forms ranges U+FB50-FDFF /
# U+FE70-FEFF). Used to pull the Arabic name out of a mixed Latin+Arabic bio blob
# (da#299). Including the presentation-forms ranges keeps a stray ligature (e.g.
# the tasliya U+FDFA) riding with the span rather than torn out mid-name.
_ARABIC_SPAN_CLASS: str = r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
# A maximal run of characters that are NEITHER Arabic-script NOR whitespace -- a
# Latin word, an ASCII digit run, a parenthesis, punctuation. Replaced by a space
# so the Arabic runs on either side of a dropped Latin chunk join into one span
# rather than fusing across it.
_NON_ARABIC_RUN_RE: re.Pattern[str] = re.compile(rf"[^{_ARABIC_SPAN_CLASS}\s]+")

# Arabic script block: U+0600–U+06FF
_ARABIC_CHAR_RE: re.Pattern[str] = re.compile(r"[\u0600-\u06FF]")

# ---------------------------------------------------------------------------
# Transmission phrase patterns
# ---------------------------------------------------------------------------

# Optional run of Arabic diacritics (tashkeel U+064B–U+065F, superscript alef
# U+0670) and tatweel/kashida (U+0640). Real isnad text is fully voweled
# (e.g. ``حَدَّثَنَا``), so a bare keyword like ``حدثنا`` will never match unless
# we tolerate diacritics interleaved *between* the base letters. Without this
# the extractor found zero transmission phrases on every voweled chain and fell
# back to emitting the whole isnad as a single un-segmented "narrator" (da#146).
_OPT_DIAC: str = r"[\u064b-\u065f\u0670\u0640]*"

# Base transmission terms keyed to their canonical label. Written with the
# classical hamza-bearing forms (أ/إ) so they match raw, un-normalized isnad
# text; positions returned therefore index into the caller's original string.
# Singular (-ني) and 3rd-person (سمع) variants are included alongside the plural
# (-نا) forms because real chains alternate between them ("حدثني" / "أخبرني" /
# "سمع فلانٌ") — omitting them under-segments the chain (da#146).
_TRANSMISSION_TERMS: dict[str, str] = {
    "حدثنا": "haddathana",
    "حدثني": "haddathani",
    "أخبرنا": "akhbarana",
    "أخبرني": "akhbarani",
    "أنبأنا": "anba_ana",
    "أنبأني": "anba_ani",
    "سمعت": "samitu",
    "سمع": "samia",
    "عن": "an",
    "قال": "qala",
    "ناولني": "nawalani",
    "كتب إلي": "kataba_ilayya",
}

# Short transmission particles that must be anchored on a word boundary. Unlike
# the long forms (حدثنا/أخبرنا/…), these one-to-three-letter particles occur as
# substrings *inside* real narrator names — عن in عَنْبَسَة / مَعْن / يَعْنِي,
# قال in مَقَالَة — so an un-anchored match over-segments a genuine name into a
# spurious mention (da#155 / da#154). The long forms are distinctive enough that
# substring collisions don't occur in practice, so they stay un-anchored.
_SHORT_PARTICLES: frozenset[str] = frozenset({"عن", "قال", "سمع"})

# Arabic letters + combining marks + tatweel, as a regex class body (no
# brackets). Used to build the word-boundary lookarounds below: a particle is a
# standalone transmission term only when it is NOT flanked by another Arabic
# letter or diacritic. Punctuation (، ؛ …) and whitespace fall outside this
# class, so they correctly count as boundaries. Excludes Arabic-Indic digits.
_AR_LETTER_OR_MARK: str = "\u0621-\u063a\u0640-\u065f\u0670-\u06d3"

# Fixed-width (single-char) lookbehind/lookahead asserting a non-letter boundary.
_LB: str = rf"(?<![{_AR_LETTER_OR_MARK}])"
_LA: str = rf"(?![{_AR_LETTER_OR_MARK}])"

# Normalization equivalence classes mirroring :func:`normalize_arabic`. A term
# written with one orthographic variant (e.g. أخبرنا with hamza-alif U+0623)
# must also match the bare-alif form found throughout the real corpus (اخبرنا
# U+0627) — the production isnads mix both, and matching only the hamza form
# left whole sub-chains un-split into blobs (da#158).
_ALIF_CLASS: str = "اأإآٱ"  # ا أ إ آ ٱ
_HAMZA_CLASS: str = "ءؤئ"  # ء ؤ ئ
_TAA_CLASS: str = "ةه"  # ة ه
_NORM_EQUIV: dict[str, str] = {}
for _cls in (_ALIF_CLASS, _HAMZA_CLASS, _TAA_CLASS):
    for _ch in _cls:
        _NORM_EQUIV[_ch] = _cls


def _variant_class(ch: str) -> str:
    """Return a regex fragment matching *ch* and its normalization-equivalents."""
    equiv = _NORM_EQUIV.get(ch)
    if equiv is not None:
        return "[" + equiv + "]"
    return re.escape(ch)


def _compile_diacritic_tolerant(term: str, *, anchor: bool) -> re.Pattern[str]:
    """Compile *term* into a regex that tolerates interleaved diacritics.

    Each base letter is expanded to a normalization-equivalence class (so alif
    and hamza/taa variants all match) followed by an optional diacritics/tatweel
    run, and inter-word whitespace is matched flexibly. The compiled pattern
    matches the fully-voweled, orthographically-variant forms found in real
    isnads while preserving match positions in the original (un-normalized) text.

    When *anchor* is true the pattern is wrapped in non-letter-boundary
    lookarounds so a short particle (عن/قال/سمع) only matches as a standalone
    transmission term, never as a substring of a longer name (da#155).
    """
    parts: list[str] = []
    for ch in term:
        if ch.isspace():
            parts.append(r"\s+")
        else:
            parts.append(_variant_class(ch) + _OPT_DIAC)
    body = "".join(parts)
    if anchor:
        body = _LB + body + _LA
    return re.compile(body)


TRANSMISSION_PATTERNS: dict[re.Pattern[str], str] = {
    _compile_diacritic_tolerant(term, anchor=term in _SHORT_PARTICLES): label
    for term, label in _TRANSMISSION_TERMS.items()
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel marks (U+064B–U+065F, U+0670)."""
    return _DIACRITICS_RE.sub("", text)


def strip_format_marks(text: str) -> str:
    """Remove bidi / zero-width / format control marks (da#271).

    Strips LRM/RLM, ZWNJ/ZWJ, zero-width space, the bidi embedding/isolate
    controls, BOM, and the Arabic end-of-ayah mark — none carry name meaning, and
    a stray one otherwise splits an identical name into a distinct canonical node.
    """
    return _FORMAT_MARKS_RE.sub("", text)


def normalize_alif(text: str) -> str:
    """Normalize أ إ آ ٱ to bare alif ا."""
    return _ALIF_VARIANTS_RE.sub("\u0627", text)


def normalize_hamza(text: str) -> str:
    """Normalize ؤ ئ to standalone hamza ء."""
    return _HAMZA_VARIANTS_RE.sub("\u0621", text)


def normalize_taa_marbuta(text: str) -> str:
    """Normalize ة to ه."""
    return _TAA_MARBUTA_RE.sub("\u0647", text)


def normalize_alif_maqsura(text: str) -> str:
    """Normalize alif maqsura U+0649 to yaa U+064A (da#427).

    Alif maqsura is the dotless final yaa -- orthographically the SAME letter as
    yaa, written U+0649 only in word-final position (yahya, musa, al-ladhi, ala).
    Corpora spell it inconsistently, so a name or particle ending in it never
    collapses onto its yaa-spelled twin without this fold -- the ~13 al-ladhi /
    al-lati / wa-l-ladhi matn pronouns never reach the al-ladhi / al-lati boundary
    set, and yahya (maqsura) / yahyi (yaa) mint two canonical ids for one narrator.

    The fold is a single codepoint substitution U+0649 -> U+064A. It is
    homographic-safe on names -- no real ism / nisba folds onto a particle or
    boundary token (asserted set-level in tests/test_parse/test_name_quality.py) --
    with ONE documented collision it does NOT resolve in this layer: the preposition
    ``\u0639\u0644\u0649`` ("on / upon") folds to ``\u0639\u0644\u064a``, the name Ali. That
    collision is handled downstream (``\u0639\u0644\u0649`` is removed from
    ``name_quality._MATN_PARTICLES`` so a bare ``\u0639\u0644\u064a`` narrator is never dropped as a
    matn particle); it is called out here because a caller keying identity on the
    surface must know the two are no longer distinguishable after normalization.

    Composes with the da#299 Urdu-yeh fold (``normalize_urdu``: U+06CC / U+06D2 /
    U+06D3 -> U+064A): distinct SOURCE codepoints, same yaa target, and neither
    produces the other's input, so the two are order-independent -- an Urdu-script
    ``\u0635\u0644\u06cc`` and an Arabic ``\u0635\u0644\u0649`` both converge on the same
    ``\u0635\u0644\u064a``. Idempotent (U+0649 is fully replaced, and U+064A is a fixed point).
    """
    return _ALIF_MAQSURA_RE.sub("\u064a", text)


def normalize_urdu(text: str) -> str:
    """Fold Urdu/Persian letter variants to their Arabic equivalents (da#299).

    Maps the Urdu-specific letterforms that render the SAME letter as an Arabic
    codepoint (keheh->kaf, Farsi/barree yeh->yeh, heh-goal/doachashmee->heh,
    noon-ghunna->noon) onto that Arabic letter (see :data:`_URDU_FOLD_MAP`). Every
    mapped codepoint is Urdu/Persian-specific and never occurs in real Arabic name
    text, so this is a no-op on all-Arabic input and only ever touches Urdu-script
    names -- the property that makes it safe to run inside :func:`normalize_arabic`.
    """
    return text.translate(_URDU_TRANSLATION)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to a single space and strip edges."""
    return _MULTI_WS_RE.sub(" ", text).strip()


def strip_to_letters(text: str) -> str:
    """Reduce *text* to Unicode letters and single spaces (da#373).

    Every character that is neither a letter nor whitespace — punctuation,
    digits, editorial dashes, quotation marks — is replaced by a space, and
    runs of whitespace are then collapsed. Replacing (not deleting) means a
    mark sitting *between* two words never fuses them, with or without an
    adjacent space (``"متن،متن"`` and ``"متن، متن"`` both yield ``"متن متن"``).

    This is the punctuation invariance :func:`normalize_arabic` deliberately
    omits (it folds orthography but keeps punctuation): two editions of one
    tradition differ by commas / dashes / quotes, so an *exact* matn hash must
    strip them or it never collides. Unicode-letter aware, so it also serves
    the English matn path. Idempotent — its output is only letters and single
    spaces, both fixed points.
    """
    stripped = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in text)
    return _MULTI_WS_RE.sub(" ", stripped).strip()


def normalize_arabic(text: str) -> str:
    """Full Arabic normalization pipeline.

    Steps:
    1. Strip bidi / zero-width / format control marks
    2. Strip diacritics
    3. Fold Urdu/Persian letter variants to Arabic (da#299)
    4. Normalize alif variants
    5. Normalize hamza variants
    6. Normalize taa marbuta
    6b. Normalize alif maqṣūra ى -> yāʾ ي (da#427)
    7. Strip tatweel (kashida)
    8. Collapse whitespace

    The Urdu fold (step 3) runs BEFORE the alif/hamza/taa steps so a folded
    letter is then carried through the rest of the pipeline identically to its
    Arabic twin — e.g. Urdu ``ہ`` -> Arabic ``ه`` is indistinguishable from a
    sourced ``ه`` downstream. It is a no-op on all-Arabic input (see
    :func:`normalize_urdu`), so no existing normalization result changes.
    """
    text = strip_format_marks(text)
    text = strip_diacritics(text)
    text = normalize_urdu(text)
    text = normalize_alif(text)
    text = normalize_hamza(text)
    text = normalize_taa_marbuta(text)
    text = normalize_alif_maqsura(text)
    text = _TATWEEL_RE.sub("", text)
    text = clean_whitespace(text)
    return text


def is_arabic(text: str) -> bool:
    """Return True if *text* contains at least one Arabic script character (U+0600–U+06FF)."""
    return bool(_ARABIC_CHAR_RE.search(text))


def extract_arabic_span(text: str) -> str:
    """Pull the Arabic-script name out of a mixed Latin+Arabic blob (da#299).

    kaggle rijal bios store a name as a mixed blob — a Latin gloss, its
    benediction abbreviation, then the Arabic name and an Arabic benediction tail
    (``"Prophet Muhammad(saw) ( <arabic name> ( <arabic benediction>"``). Every
    run of non-Arabic, non-space characters (Latin words, ASCII digits,
    parentheses, punctuation) is replaced by a space and the whitespace collapsed,
    so the surviving Arabic runs join into one span. The raw (voweled, still
    Urdu-script) form is preserved — only :func:`normalize_arabic` folds it — so a
    caller keeps the source's diacritics for display.

    Returns the input (whitespace-collapsed) unchanged when it carries no Arabic
    at all, so a pure-Latin name is never blanked. The benediction tail is NOT
    stripped here (that is the name-quality cleaner's job); this only discards the
    non-Arabic noise around the Arabic span.
    """
    if not text:
        return text
    span = clean_whitespace(_NON_ARABIC_RUN_RE.sub(" ", text))
    return span if span else text.strip()


def extract_transmission_phrases(text: str) -> list[tuple[int, int, str]]:
    """Find transmission formula positions in *text*.

    Returns a list of ``(start, end, label)`` tuples for each non-overlapping
    match found via :data:`TRANSMISSION_PATTERNS`, in ascending start order.

    Overlapping matches are resolved by preferring the longer (more specific)
    term: e.g. where both ``سمعت`` (samitu) and the bare ``سمع`` (samia) match at
    the same position, only ``سمعت`` is kept. This keeps segment boundaries in
    :func:`src.parse.narrator_extraction._extract_arabic` clean when overlapping
    variants are present in the pattern set.
    """
    matches: list[tuple[int, int, str]] = []
    for pattern, label in TRANSMISSION_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), label))

    # Sort by start asc, then by span length desc so that at any shared start the
    # longer match is considered first and shorter overlapping ones are dropped.
    matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))

    results: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, label in matches:
        if start >= last_end:
            results.append((start, end, label))
            last_end = end
    return results


# ---------------------------------------------------------------------------
# Transliteration (Arabic name -> Latin display form)
# ---------------------------------------------------------------------------
#
# Narrator records carry a required ``name_ar`` but almost never a sourced
# ``name_en`` (113 / 47,199 at P5W3 — da#159), so English search and display are
# hollow. This is a deterministic, dependency-free Arabic->Latin transliterator
# used to synthesize a readable display name whenever a sourced English name is
# absent.
#
# Two layers, in priority order:
#   1. A token lexicon of the formulaic building blocks of classical narrator
#      names — kunya/nasab particles (Abu, ibn, bint, Umm), theophorics
#      (Abd Allah, al-Rahman), the most common given names (Muhammad, Ali,
#      Husayn, ...) and high-frequency nisbas (al-Bukhari, al-Kufi, ...).
#      Arabic names are highly repetitive, so a modest lexicon renders the
#      overwhelming majority of *tokens* correctly.
#   2. A consonant-skeleton letter map for any token outside the lexicon. The
#      result is imperfect for rare, un-voweled tokens (Arabic script omits short
#      vowels) but is always non-empty, ASCII, and deterministic — turning 0%
#      coverage into ~100% with sane, stable output.
#
# Output is intentionally plain ASCII (no ʿayn/macrons) so it is friendly to
# fulltext search and case-insensitive matching.

# Per-letter consonant/long-vowel map. Carriers that contribute no Latin glyph
# in plain ASCII (hamza, ayn) map to the empty string. The input is expected to
# be normalized (diacritics already stripped by :func:`normalize_arabic`).
_TRANSLIT_LETTERS: dict[str, str] = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "a",
    "و": "w",
    "ي": "y",
    "ى": "a",
    "ء": "",
}

# Token lexicon, written with natural spellings; keys are normalized at import so
# a normalized lookup token matches regardless of diacritics / alif-hamza form.
# Values are pre-cased for their typical role (particles ``ibn``/``bint`` stay
# lowercase for mid-name use; the article prefix stays ``al-``).
_RAW_NAME_LEXICON: dict[str, str] = {
    # Kunya / nasab particles
    "أبو": "Abu",
    "أبا": "Aba",
    "أبي": "Abi",
    "ابن": "ibn",
    "بن": "ibn",
    "بنت": "bint",
    "أم": "Umm",
    # Theophorics & honorifics
    "عبد": "Abd",
    "الله": "Allah",
    "عبدالله": "Abd Allah",
    "عبدالرحمن": "Abd al-Rahman",
    "الرحمن": "al-Rahman",
    "الرحيم": "al-Rahim",
    "الصديق": "al-Siddiq",
    "الفاروق": "al-Faruq",
    # Common given names
    "محمد": "Muhammad",
    "أحمد": "Ahmad",
    "علي": "Ali",
    "حسن": "Hasan",
    "الحسن": "al-Hasan",
    "حسين": "Husayn",
    "الحسين": "al-Husayn",
    "عمر": "Umar",
    "عثمان": "Uthman",
    "إبراهيم": "Ibrahim",
    "إسماعيل": "Ismail",
    "إسحاق": "Ishaq",
    "يعقوب": "Yaqub",
    "يوسف": "Yusuf",
    "موسى": "Musa",
    "عيسى": "Isa",
    "داود": "Dawud",
    "سليمان": "Sulayman",
    "يحيى": "Yahya",
    "خالد": "Khalid",
    "جعفر": "Jafar",
    "صالح": "Salih",
    "حماد": "Hammad",
    "حمزة": "Hamza",
    "زيد": "Zayd",
    "سعيد": "Said",
    "سعد": "Sad",
    "سفيان": "Sufyan",
    "سلمان": "Salman",
    "طلحة": "Talha",
    "عمرو": "Amr",
    "عباس": "Abbas",
    "العباس": "al-Abbas",
    "معاوية": "Muawiya",
    "مالك": "Malik",
    "أنس": "Anas",
    "بكر": "Bakr",
    "هريرة": "Hurayra",
    "الزبير": "al-Zubayr",
    "عوف": "Awf",
    "الخطاب": "al-Khattab",
    "شعبة": "Shuba",
    "قتادة": "Qatada",
    "نافع": "Nafi",
    "مجاهد": "Mujahid",
    "عطاء": "Ata",
    "هشام": "Hisham",
    "عروة": "Urwa",
    "معمر": "Mamar",
    "وكيع": "Waki",
    "الوليد": "al-Walid",
    "يزيد": "Yazid",
    "جابر": "Jabir",
    "ثابت": "Thabit",
    "حذيفة": "Hudhayfa",
    "عمار": "Ammar",
    "بلال": "Bilal",
    # Women narrators
    "عائشة": "Aisha",
    "فاطمة": "Fatima",
    "خديجة": "Khadija",
    "سلمى": "Salma",
    # High-frequency nisbas
    "البخاري": "al-Bukhari",
    "المدني": "al-Madani",
    "الكوفي": "al-Kufi",
    "البصري": "al-Basri",
    "الدمشقي": "al-Dimashqi",
    "الأنصاري": "al-Ansari",
    "القرشي": "al-Qurashi",
    "التميمي": "al-Tamimi",
    "الثقفي": "al-Thaqafi",
    "الهمداني": "al-Hamdani",
    "الزهري": "al-Zuhri",
    "الثوري": "al-Thawri",
    "الأعمش": "al-Amash",
    "الشعبي": "al-Shabi",
    "الأوزاعي": "al-Awzai",
    "الليث": "al-Layth",
    "الأشعري": "al-Ashari",
    "السلمي": "al-Sulami",
}

_NAME_LEXICON: dict[str, str] = {normalize_arabic(k): v for k, v in _RAW_NAME_LEXICON.items()}

# Particles that read lowercase in the middle of a name but are capitalized when
# they lead it (e.g. "Muhammad ibn Ali" vs "Ibn Sirin").
_LOWERCASE_PARTICLES: frozenset[str] = frozenset({"ibn", "bint"})


def _capitalize(token: str) -> str:
    """Uppercase the first alphabetic character of *token*, leaving the rest."""
    for i, ch in enumerate(token):
        if ch.isalpha():
            return token[:i] + ch.upper() + token[i + 1 :]
    return token


def _map_letters(token_norm: str) -> str:
    """Map a normalized Arabic token to its consonant/vowel skeleton."""
    return "".join(_TRANSLIT_LETTERS.get(ch, "") for ch in token_norm)


def _transliterate_token(token: str) -> str:
    """Transliterate a single whitespace-delimited token to a Latin form.

    Lexicon hit wins; otherwise the definite article ``al-`` is split off and the
    remainder is rendered with the consonant-skeleton map. Tokens with no Arabic
    content are returned trimmed (already-Latin input passes through).
    """
    norm = normalize_arabic(token)
    if not norm:
        return ""
    if norm in _NAME_LEXICON:
        return _NAME_LEXICON[norm]
    if not is_arabic(norm):
        return token.strip()
    # Definite article prefix (alif-lam) -> "al-" + capitalized remainder.
    if norm.startswith("ال") and len(norm) > 2:
        body = _map_letters(norm[2:])
        return "al-" + _capitalize(body) if body else "al-"
    return _capitalize(_map_letters(norm))


def transliterate(text: str) -> str:
    """Transliterate an Arabic name to a readable, ASCII Latin display form.

    Deterministic and dependency-free. Returns ``""`` for empty/blank input.
    Intended as a *fallback* display name when a record has no sourced
    ``name_en`` — see :mod:`src.graph.load_nodes` (da#159). The output is not a
    scholarly romanization; it favours readable, stable, search-friendly forms.
    """
    if not text or not text.strip():
        return ""
    text = _TATWEEL_RE.sub("", text)
    out: list[str] = []
    for token in _MULTI_WS_RE.sub(" ", text).strip().split(" "):
        stripped = token.strip("،.,;:()[]{}\"'`-")
        if not stripped:
            continue
        latin = _transliterate_token(stripped)
        if not latin:
            continue
        if latin in _LOWERCASE_PARTICLES and out:
            out.append(latin)
        elif latin in _LOWERCASE_PARTICLES:
            out.append(_capitalize(latin))
        else:
            out.append(latin)
    return " ".join(out).strip()


# Independent, normalization-anchored detector for whether a string carries a
# transmission marker at all. It runs on :func:`normalize_arabic` output and
# anchors *every* term on a non-letter boundary, so it neither misses an
# orthographic variant nor false-matches a marker buried inside a longer name.
# Built off the canonical normalizer (not the hand-built char-classes used by
# TRANSMISSION_PATTERNS) so it stays a valid fail-loud signal even if those
# pattern classes ever drift — see src.parse.narrator_extraction._extract_arabic.
_NORM_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(_LB + re.escape(normalize_arabic(term)) + _LA) for term in _TRANSMISSION_TERMS
)


def contains_transmission_marker(text: str) -> bool:
    """Return True if *text* contains a transmission term as a bounded word.

    Used by the Arabic narrator extractor to *fail loud* — a chain that carries a
    transmission marker but yields no segmentable phrases means the segmenter
    failed, and the whole chain must never be minted as a single "narrator" blob
    (da#158). Matching is diacritic/variant-insensitive (via normalization) and
    word-boundary anchored (so ``عن`` in ``عنبسة`` does not count).
    """
    normalized = normalize_arabic(text)
    return any(pattern.search(normalized) for pattern in _NORM_MARKER_PATTERNS)


# ---------------------------------------------------------------------------
# Identity surface — inflection folding for canonical-id minting (da#376)
# ---------------------------------------------------------------------------
# ``normalize_arabic`` is orthographic: it folds hamza/alif/taa-marbuta and strips
# diacritics, but it is neither case- nor spacing-invariant. That was harmless while
# ``disambiguate`` keyed canonical identity on the matched *bio's* spelling, because
# the bio key incidentally absorbed inflection. Once identity became a pure function
# of the mention (da#356), Arabic case endings started minting one node per case:
#
#     ابي هريره  46,563   ابا هريره  5,691   ابو هريره  2,179    -> three Abu Hurayras
#
# and ``fuzzy_cluster`` cannot repair it: a name with fewer than two *significant*
# tokens produces no ``combinations(tokens, 2)`` blocking key, joins no block, and is
# never scored at any threshold. Structural, not statistical. The fold below is
# therefore the only place these can converge, and it must run *before* the id is
# minted — see :func:`src.parse.identity.make_canonical_id`, which applies it.
#
# PRECISION-FIRST, like the da#248 mononym registry: every rule is a closed set or a
# lexicon, never a productive orthographic pattern. Measured on the production corpus,
# a naive "strip a trailing alif from any token of length >= 4" rule would fold
# ``زكريا`` (Zakariyya, 9,207 mentions) to ``زكري``, and ``عمرا`` (accusative of عمرو,
# 'Amr) to ``عمر`` ('Umar) — two different men. No orthographic rule separates
# ``عليا``/``علي`` from ``زكريا``/``زكري``; only a lexicon does. So we abstain by default.

# Accusative / genitive of the kunya ``ابو`` ("father of"). Folding these is
# unconditionally safe: no two narrators are distinguished by the case of their kunya.
_KUNYA_VARIANTS: frozenset[str] = frozenset({"ابا", "ابي", "ابى"})
_KUNYA_CANONICAL = "ابو"

# ``عبدالله`` -> ``عبد الله``. The trigger is the *definite article*, not a bare
# ``عبد`` prefix: ``عبدان`` (3,974 mentions), ``عبدوس`` and ``عبدويه`` are names in
# their own right and must not be split.
_ABD_GLUED_PREFIX = "عبدال"

# Given names whose accusative is stem + alif. Hand-curated from the production
# corpus, then FROZEN — a corpus-derived list would not be a pure function and would
# drift between runs. Deliberately EXCLUDED, with reasons:
#   عمر    - ``عمرا`` is the accusative of عمرو ('Amr), a different man from عمر ('Umar)
#   زكري   - ``زكريا`` is Zakariyya's nominative; its "stem" is not a name
#   جميع   - ``جميعا`` is the adverb "altogether"
#   ابن    - ``ابنا`` is a genealogical connector, not a name
#   البن, عليهم - artefacts / pronouns
_ACCUSATIVE_STEMS: frozenset[str] = frozenset(
    {
        "علي",
        "انس",
        "جابر",
        "مالك",
        "نافع",
        "سعد",
        "بلال",
        "سالم",
        "مكحول",
        "ثابت",
        "معاذ",
        "جندب",
        "مطرف",
        "خباب",
        "زيد",
        "محمد",
        "حميد",
        "سعيد",
        "عمار",
        "صهيب",
        "معمر",
        "عامر",
        "سهل",
        "كعب",
        "تميم",
    }
)


def _fold_token(token: str) -> list[str]:
    """Fold one token into its identity surface. May emit two tokens (``عبدال`` split)."""
    if token in _KUNYA_VARIANTS:
        return [_KUNYA_CANONICAL]
    if token.startswith(_ABD_GLUED_PREFIX) and len(token) > len(_ABD_GLUED_PREFIX):
        return ["عبد", token[3:]]
    if token.endswith("ا") and token[:-1] in _ACCUSATIVE_STEMS:
        return [token[:-1]]
    return [token]


# Two residual letter-level noise classes that :func:`normalize_arabic` leaves
# behind and that fragment one name across canonical ids (da#434). Pure
# normalization, never an identity merge of distinct people:
#   ى (U+0649 alif-maqsura) → ي (U+064A ya) — the same name mints two ids when a
#      final ي is spelled dotless.
#   ء (U+0621 standalone hamza) dropped — after normalize_arabic has folded
#      ؤ ئ → ء, the bare hamza that remains is orthographic noise, so e.g.
#      ``عاءشه`` and ``عائشة`` collapse to the same key instead of two.
# Applied at the mint surface only, NOT in normalize_arabic (which also feeds
# display-name normalization). Idempotent: ي is not itself a fold trigger and no
# ء survives, so both are fixed points on their own output.
_RESIDUAL_NOISE_MAP = str.maketrans({"ى": "ي", "ء": ""})


def _fold_residual_noise(text: str) -> str:
    """Fold alif-maqsura ى→ي and drop the standalone hamza ء (da#434)."""
    return text.translate(_RESIDUAL_NOISE_MAP)


def canonical_surface(name: str) -> str:
    """The string a canonical narrator id is minted from.

    ``normalize_arabic`` plus a precision-first fold of the inflections that would
    otherwise mint one node per Arabic case ending (da#376). Idempotent:
    ``canonical_surface(canonical_surface(x)) == canonical_surface(x)`` — the fold
    targets are all closed sets whose outputs are not themselves fold triggers.

    This is the *identity* surface, not a display name. Callers keep the mention's
    own raw spelling for ``name_ar``; only the key and ``name_ar_normalized`` are
    folded.
    """
    normalized = normalize_arabic(name)
    if not normalized:
        return ""
    normalized = _fold_residual_noise(normalized)
    folded: list[str] = []
    for token in normalized.split():
        folded.extend(_fold_token(token))
    return " ".join(folded)
