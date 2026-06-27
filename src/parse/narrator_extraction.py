"""Shared narrator mention extraction from isnad text.

Supports both English and Arabic isnad strings. English extraction uses
keyword-based splitting; Arabic extraction uses transmission phrase detection
from ``src.utils.arabic``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.arabic import (
    contains_transmission_marker,
    extract_transmission_phrases,
    normalize_arabic,
)

__all__ = ["IsnadSegmentationError", "NarratorSpan", "extract_narrator_mentions"]

# English transmission keywords used as chain delimiters.
_EN_DELIMITERS: re.Pattern[str] = re.compile(
    r"\b(?:Narrated|reported\s+by|on\s+the\s+authority\s+of|from|who\s+heard\s+from)\b",
    re.IGNORECASE,
)

# Trailing matn clause to drop from an English name span (da#154 over-merge). A
# narrator name is followed by a reporting verb / complementizer that introduces
# the matn (``Anas b. Malik reported …``, ``Aishah that the Prophet said``); the
# bare ``reported``/``said``/``that`` is NOT one of the ``_EN_DELIMITERS`` split
# tokens, so without this trim the matn rides along on the name. None of these
# words occurs inside a real narrator name, so the cut is safe.
_EN_MATN_TRIM_RE: re.Pattern[str] = re.compile(
    r"\s+(?:reported|narrat(?:ed|ing)|said|says?|mentioned|relate[ds]?|that)\b.*$",
    re.IGNORECASE,
)

# Trailing punctuation to strip from extracted names. Includes the Arabic comma
# (، U+060C) and semicolon (؛ U+061B), which routinely terminate a narrator span
# in voweled isnads (da#146) and would otherwise pollute the exact-match key.
_TRAILING_PUNCT_RE: re.Pattern[str] = re.compile(r"[.,;:!?()،؛\s]+$")

# Connective particles that trail a narrator name into the next clause and should
# not be part of the name span (da#154 name-boundary precision). Compared against
# the *normalized* trailing token, so they cover diacritic/orthographic variants
# (أَنَّهُ → انه). A trailing transmission *marker* (قال/عن/حدثنا/…) is handled
# separately in ``_clean_ar_name`` via ``contains_transmission_marker`` — it is
# NOT always consumed as a phrase upstream (da#244: a trailing ``قال`` the phrase
# splitter missed survived into a tail span and tripped the fail-loud guard).
_AR_TRAILING_CONNECTIVES: frozenset[str] = frozenset({"انه", "انها", "انهم", "انك", "اني", "يقول"})


class IsnadSegmentationError(ValueError):
    """Raised when an Arabic isnad chain cannot be segmented into narrators.

    Failing loud is deliberate: a chain that carries a transmission marker but
    cannot be split must NEVER be minted as a single whole-chain "narrator" blob
    (da#158 — ~80% of loaded Narrator nodes were exactly such blobs). The error
    surfaces the segmentation failure to the pipeline instead of silently
    polluting the graph.
    """


@dataclass(frozen=True)
class NarratorSpan:
    """A narrator mention extracted from isnad text."""

    name: str
    position: int
    transmission_method: str | None = None


def _clean_name(raw: str) -> str | None:
    """Strip whitespace and trailing punctuation. Return None if empty."""
    cleaned = _TRAILING_PUNCT_RE.sub("", raw.strip())
    return cleaned if cleaned else None


def _clean_ar_name(normalized: str) -> str | None:
    """Clean a normalized Arabic name span: punctuation + trailing connectives/markers.

    Strips trailing punctuation, then drops trailing tokens that bleed the name
    into the following clause:

    - connective particles (انه/يقول/…) that ride the name into the next clause
      (da#154), and
    - a residual standalone transmission marker (قال/عن/حدثنا/…) that the phrase
      splitter failed to consume upstream (da#244).

    Stripping is **trailing-only**. A marker in the *middle* of a span signals a
    genuinely-unsegmented multi-narrator blob and must still trip the fail-loud
    guard in :func:`_extract_arabic`, so it is deliberately left in place here.
    Markers are matched via :func:`contains_transmission_marker` on the whole
    trailing token, which is word-boundary anchored — so a ``قال`` buried inside a
    real name (``مقالة``, ``معن``) is never stripped.
    """
    cleaned = _clean_name(normalized)
    if cleaned is None:
        return None
    tokens = cleaned.split()
    while tokens and (
        tokens[-1] in _AR_TRAILING_CONNECTIVES or contains_transmission_marker(tokens[-1])
    ):
        tokens.pop()
    if not tokens:
        return None
    return _clean_name(" ".join(tokens))


def _extract_english(text: str) -> list[NarratorSpan]:
    """Extract narrator mentions from English isnad text."""
    parts = _EN_DELIMITERS.split(text)
    spans: list[NarratorSpan] = []
    position = 0
    for part in parts:
        # Drop any trailing matn clause before cleaning (da#154 over-merge).
        name = _clean_name(_EN_MATN_TRIM_RE.sub("", part))
        if name:
            spans.append(NarratorSpan(name=name, position=position))
            position += 1
    return spans


def _extract_arabic(text: str) -> list[NarratorSpan]:
    """Extract narrator mentions from Arabic isnad text.

    Raises:
        IsnadSegmentationError: when the text carries a transmission marker but
            cannot be split into per-narrator spans (a whole- or partial-chain
            blob would otherwise be minted — da#158).
    """
    stripped = text.strip()
    phrases = extract_transmission_phrases(text)
    if not phrases:
        # No transmission phrase: either a single bare narrator name (legitimate)
        # or a chain the segmenter failed on. Distinguish by an independent,
        # normalization-anchored marker check and fail loud on the latter rather
        # than mint the whole chain as one "narrator" (da#158).
        if contains_transmission_marker(stripped):
            raise IsnadSegmentationError(
                "isnad carries a transmission marker but no segmentable phrase "
                f"was found (would mint a chain blob): {stripped[:120]!r}"
            )
        name = normalize_arabic(stripped)
        return [NarratorSpan(name=name, position=0)] if name else []

    spans: list[NarratorSpan] = []
    position = 0

    # Text before the first transmission phrase may contain a name.
    if phrases[0][0] > 0:
        prefix = text[: phrases[0][0]]
        prefix_name = _clean_ar_name(normalize_arabic(prefix))
        if prefix_name:
            spans.append(NarratorSpan(name=prefix_name, position=position))
            position += 1

    for idx, (start, end, label) in enumerate(phrases):
        # The name follows the transmission phrase, up to the next phrase or end of text.
        next_start = phrases[idx + 1][0] if idx + 1 < len(phrases) else len(text)
        segment = text[end:next_start]
        seg_name = _clean_ar_name(normalize_arabic(segment))
        if seg_name:
            spans.append(NarratorSpan(name=seg_name, position=position, transmission_method=label))
            position += 1

    # Post-split guard: no emitted span may still carry a transmission marker.
    # If one does, the chain was only partially segmented (a residual blob) —
    # fail loud rather than store it (da#158).
    for span in spans:
        if contains_transmission_marker(span.name):
            raise IsnadSegmentationError(
                "segmented narrator span still contains a transmission marker "
                f"(partial chain blob): {span.name[:120]!r}"
            )

    return spans


def extract_narrator_mentions(isnad_text: str, language: str) -> list[NarratorSpan]:
    """Extract narrator mentions from isnad text.

    Args:
        isnad_text: Raw isnad string.
        language: ``"en"`` for English, ``"ar"`` for Arabic.

    Returns:
        List of :class:`NarratorSpan` in chain order.
    """
    if not isnad_text or not isnad_text.strip():
        return []
    if language == "ar":
        return _extract_arabic(isnad_text)
    return _extract_english(isnad_text)
