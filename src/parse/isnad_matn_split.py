"""Recover a matn-embedded isnad from sanadset's "No SANAD" rows (da#366).

Sanadset's 159,558 rows whose ``Sanad`` column reads literally ``No SANAD`` are
NOT chainless: the dataset author simply failed to segment them, and ~122,000
carry an isnad embedded in the running matn text. This module recovers that
isnad **without** the NER ``full_text_ar`` fallback (``src/resolve/ner.py``,
da#369) — that fallback mines the whole body and mints matn sentences as
narrators (measured on ``lk``: mentions/hadith 5.49 → 7.91, 1.44×; 53.7% of the
extra names are substrings of the *matn*, 0.1% of the isnad).

Two isnad conventions occur and no single probe sees both (da#366):

  1. RECEIPT-VERB opener   ``حدثنا فلان، حدثنا فلان، عن فلان``   (``lk``-shaped)
  2. BARE-NAME + ``عن``     ``علي بن إبراهيم عن أبيه عن ابن أبي عمير``  (``tusi``/sanadset-shaped)

An opener-only probe is blind to (2); a ``عن``-only probe is noisy. The
calibrated detector (da#366) takes either:

    D(text) = head_opener(text) OR (>= 2 standalone عن within the first 25 tokens)

measured at 66.5% / 77.2% on the ``No SANAD`` rows against 0.1% / 3.1% on known
matn.

**Precision is paramount.** A splitter that pulls matn tokens into the isnad
fabricates narrators — the exact #317/#369 pollution this recovery exists to
avoid. So the isnad→matn boundary is found conservatively and the module
**fails closed** (recovers nothing) whenever the boundary is not locatable or
the isolated head is not cleanly segmentable by the existing fail-loud
segmenter. Recovering fewer chains is the correct failure mode; recovering a
matn sentence as a chain is not. Recovery is therefore an accepted lower bound.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.parse.narrator_extraction import (
    IsnadSegmentationError,
    NarratorSpan,
    extract_narrator_mentions,
)
from src.utils.arabic import extract_transmission_phrases, normalize_arabic

__all__ = [
    "DETECTOR_TOKEN_WINDOW",
    "SplitResult",
    "detect_isnad",
    "split_isnad_matn",
]

# --- detector -------------------------------------------------------------

# Head openers (receipt verbs) that mark convention (1). Includes the abbreviated
# forms ``ثنا``/``ثني`` (clipped ``حدثنا``/``حدثني``) and ``أبنا`` (clipped
# ``أنبأنا``) that the shared ``_TRANSMISSION_TERMS`` set omits but the corpus
# uses (da#366). Compared against the *normalized* head token, so orthographic
# and diacritic variants collapse (``أخبرنا`` → ``اخبرنا``); the comparison is
# therefore exact-token, never a substring match.
_RAW_HEAD_OPENERS: tuple[str, ...] = (
    "حدثنا",
    "حدثني",
    "ثنا",
    "ثني",
    "أخبرنا",
    "أخبرني",
    "أنبأنا",
    "أبنا",
    "سمعت",
)
_HEAD_OPENERS: frozenset[str] = frozenset(normalize_arabic(t) for t in _RAW_HEAD_OPENERS)

# Detector window and the ``عن`` count that marks convention (2).
DETECTOR_TOKEN_WINDOW = 25
_MIN_AN_FOR_CHAIN = 2

# --- boundary -------------------------------------------------------------

# ``extract_transmission_phrases`` labels (from ``src.utils.arabic``) that CHAIN
# one narrator to the next — an opener or ``عن``/``سمع`` links the following name
# into the isnad. ``قال`` ("qala") is deliberately absent: it is ambiguous and
# handled separately.
_LINK_LABELS: frozenset[str] = frozenset(
    {
        "haddathana",
        "haddathani",
        "akhbarana",
        "akhbarani",
        "anba_ana",
        "anba_ani",
        "samitu",
        "samia",
        "an",
        "nawalani",
        "kataba_ilayya",
    }
)

# A ``قال`` is a chain LINK (``حدثنا فلان قال حدثنا فلان``) only when a linking
# phrase follows it closely; otherwise it introduces the matn
# (``عن أبي هريرة قال: …``) and marks the boundary. This is the gap, in
# characters, from the ``قال`` match end to the next phrase's start within which
# the next phrase is treated as a continuation. Kept small: matn openers are
# rare (0.1% per calibration), so a genuine boundary ``قال`` is very unlikely to
# have a spurious opener within a few characters.
_QALA_LINK_LOOKAHEAD_CHARS = 12

# Matn-introducing punctuation: a colon or an opening/closing quotation mark
# begins direct speech, so it terminates the isnad.
_MATN_PUNCT: tuple[str, ...] = (":", '"', "«", "»", "“", "”", "‹", "›", "﴿")

# Standalone ``أن`` / ``أنّ`` / ``أنه`` / ``أنها`` ("that …") introduces the
# reported content — the classic bare-name isnad→matn hinge
# (``… عن ابن عمر أن رسول الله``). Diacritic-tolerant and boundary-anchored so it
# never fires inside a name (``أنس`` → followed by a letter, excluded).
_DIAC = r"[ً-ٰٟ]*"
_AR_LETTER = r"ء-ي"
_AN_BOUNDARY_RE: re.Pattern[str] = re.compile(
    rf"(?<![{_AR_LETTER}])"
    rf"[اأإآٱ]{_DIAC}ن{_DIAC}"  # أن / ان
    rf"(?:[ها]{_DIAC})*"  # optional أنه / أنها / أنا tail
    rf"(?![{_AR_LETTER}])"
)

# First Arabic-letter word of a text, skipping any leading punctuation/whitespace.
_FIRST_AR_WORD_RE: re.Pattern[str] = re.compile(rf"[^{_AR_LETTER}]*([{_AR_LETTER}]+)")


@dataclass(frozen=True)
class SplitResult:
    """A recovered isnad/matn split.

    ``spans`` are the narrator mentions segmented from ``isnad_ar`` by the shared
    fail-loud :func:`extract_narrator_mentions`; the caller maps them to mention
    rows (and applies its own narrator-likeness filter).
    """

    isnad_ar: str
    matn_ar: str
    convention: str  # "opener" | "an_chain" | "both"
    spans: tuple[NarratorSpan, ...]


def _head_opener(text: str) -> bool:
    """True if the first Arabic word is a transmission opener.

    Leading punctuation/whitespace is skipped so an opener that follows an
    editorial mark still counts; the comparison is on the *normalized* first
    Arabic word, so it is exact-token, never a substring of a longer word.
    """
    match = _FIRST_AR_WORD_RE.match(normalize_arabic(text))
    if match is None:
        return False
    return match.group(1) in _HEAD_OPENERS


def _window_end(text: str) -> int:
    """Char offset just past the :data:`DETECTOR_TOKEN_WINDOW`-th whitespace token."""
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) <= DETECTOR_TOKEN_WINDOW:
        return len(text)
    return tokens[DETECTOR_TOKEN_WINDOW - 1].end()


def _an_count_in_window(text: str, phrases: list[tuple[int, int, str]]) -> int:
    """Count standalone ``عن`` transmission phrases within the detector window."""
    limit = _window_end(text)
    return sum(1 for start, _end, label in phrases if label == "an" and start < limit)


def detect_isnad(text: str, phrases: list[tuple[int, int, str]]) -> str | None:
    """Return the matched convention, or ``None`` if the detector does not fire.

    ``"both"`` when a head opener AND >= 2 windowed ``عن`` are present,
    ``"opener"`` / ``"an_chain"`` when only one signal fires.
    """
    opener = _head_opener(text)
    an_chain = _an_count_in_window(text, phrases) >= _MIN_AN_FOR_CHAIN
    if opener and an_chain:
        return "both"
    if opener:
        return "opener"
    if an_chain:
        return "an_chain"
    return None


def _matn_boundary(text: str, phrases: list[tuple[int, int, str]]) -> int | None:
    """Char offset where the matn begins, or ``None`` if not locatable.

    The isnad must keep at least the first transmission phrase, so a boundary is
    only valid at or after that phrase's end. The earliest reliable matn
    introducer (a non-linking ``قال``, a standalone ``أن``, or matn punctuation)
    wins. ``None`` (fail closed) when no reliable introducer follows the chain —
    guessing a boundary would risk absorbing matn into the final narrator.
    """
    if not phrases:
        return None
    min_boundary = phrases[0][1]

    candidates: list[int] = []

    # Non-linking ``قال`` → boundary; linking ``قال`` (opener/عن follows closely) → skip.
    for idx, (_start, end, label) in enumerate(phrases):
        if label != "qala":
            continue
        nxt = phrases[idx + 1] if idx + 1 < len(phrases) else None
        if (
            nxt is not None
            and nxt[2] in _LINK_LABELS
            and (nxt[0] - end) <= _QALA_LINK_LOOKAHEAD_CHARS
        ):
            continue
        candidates.append(_start)

    # Standalone ``أن`` / ``أنه`` introducing the reported content.
    candidates.extend(m.start() for m in _AN_BOUNDARY_RE.finditer(text))

    # Matn punctuation.
    for punct in _MATN_PUNCT:
        pos = text.find(punct)
        if pos != -1:
            candidates.append(pos)

    valid = [c for c in candidates if c >= min_boundary]
    return min(valid) if valid else None


def split_isnad_matn(text: str | None) -> SplitResult | None:
    """Recover a matn-embedded isnad from *text*, or ``None`` (fail closed).

    Steps: detect the convention (D), find a conservative isnad→matn boundary,
    isolate the head, and segment ONLY that head with the shared fail-loud
    :func:`extract_narrator_mentions`. Returns ``None`` — recovering nothing —
    at any ambiguity: detector miss, unlocatable boundary, empty head,
    unsegmentable head (a chain blob, da#158), or a head yielding no spans.
    """
    if not text or not text.strip():
        return None

    phrases = extract_transmission_phrases(text)
    convention = detect_isnad(text, phrases)
    if convention is None:
        return None

    boundary = _matn_boundary(text, phrases)
    if boundary is None:
        return None

    isnad_ar = text[:boundary].strip()
    matn_ar = text[boundary:].strip()
    if not isnad_ar:
        return None

    try:
        spans = extract_narrator_mentions(isnad_ar, "ar")
    except IsnadSegmentationError:
        # The isolated head carries a transmission marker but would not segment —
        # a partial chain blob. Fail closed rather than mint a blob (da#158).
        return None
    if not spans:
        return None

    return SplitResult(
        isnad_ar=isnad_ar,
        matn_ar=matn_ar,
        convention=convention,
        spans=tuple(spans),
    )
