"""Pure-Python Arabic text processing utilities for hadith isnad analysis.

All functions use only ``re`` and ``unicodedata`` from the standard library.
Compiled regex patterns are defined at module level for performance.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_diacritics",
    "normalize_alif",
    "normalize_hamza",
    "normalize_taa_marbuta",
    "normalize_arabic",
    "clean_whitespace",
    "is_arabic",
    "extract_transmission_phrases",
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

# Tatweel / kashida: ـ (U+0640)
_TATWEEL_RE: re.Pattern[str] = re.compile(r"\u0640")

# Multiple whitespace
_MULTI_WS_RE: re.Pattern[str] = re.compile(r"\s+")

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


def _compile_diacritic_tolerant(term: str) -> re.Pattern[str]:
    """Compile *term* into a regex that tolerates interleaved diacritics.

    Each base letter is followed by an optional diacritics/tatweel run, and
    inter-word whitespace is matched flexibly. The compiled pattern matches the
    fully-voweled form found in real isnads while preserving match positions in
    the original (un-normalized) text.
    """
    parts: list[str] = []
    for ch in term:
        if ch.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(ch) + _OPT_DIAC)
    return re.compile("".join(parts))


TRANSMISSION_PATTERNS: dict[re.Pattern[str], str] = {
    _compile_diacritic_tolerant(term): label for term, label in _TRANSMISSION_TERMS.items()
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel marks (U+064B–U+065F, U+0670)."""
    return _DIACRITICS_RE.sub("", text)


def normalize_alif(text: str) -> str:
    """Normalize أ إ آ ٱ to bare alif ا."""
    return _ALIF_VARIANTS_RE.sub("\u0627", text)


def normalize_hamza(text: str) -> str:
    """Normalize ؤ ئ to standalone hamza ء."""
    return _HAMZA_VARIANTS_RE.sub("\u0621", text)


def normalize_taa_marbuta(text: str) -> str:
    """Normalize ة to ه."""
    return _TAA_MARBUTA_RE.sub("\u0647", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to a single space and strip edges."""
    return _MULTI_WS_RE.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    """Full Arabic normalization pipeline.

    Steps:
    1. Strip diacritics
    2. Normalize alif variants
    3. Normalize hamza variants
    4. Normalize taa marbuta
    5. Strip tatweel (kashida)
    6. Collapse whitespace
    """
    text = strip_diacritics(text)
    text = normalize_alif(text)
    text = normalize_hamza(text)
    text = normalize_taa_marbuta(text)
    text = _TATWEEL_RE.sub("", text)
    text = clean_whitespace(text)
    return text


def is_arabic(text: str) -> bool:
    """Return True if *text* contains at least one Arabic script character (U+0600–U+06FF)."""
    return bool(_ARABIC_CHAR_RE.search(text))


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
