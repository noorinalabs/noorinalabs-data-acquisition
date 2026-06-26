"""Extract narrator life-dates from rijāl source notation (da#164).

The rijāl / ṭabaqāt genre (Itqān, Taqrīb al-Tahdhīb, …) records narrator
life-dates as free text — a bare Hijri year (``"168 هـ"``), an attested span
(``"بين 161 هـ إلى 170 هـ"`` — "between 161 and 170 AH"), an open bound
(``"بعد 130 هـ"`` — "after 130 AH"), an approximation (``"180 هـ تقريبا"``), or
a list of competing attestations (``"230 هـ ، أو 232 هـ"`` — "230, or 232"). This
module turns those notations into the structured ``earliest/latest`` + precision
bounds that da#161 added to :class:`~src.models.narrator.Narrator`.

Two design rules govern the extraction:

* **Death-anchored.** ``wafāt`` is the well-attested spine of the genre; birth is
  the rare exception. :func:`extract_narrator_dates` parses death as the primary
  signal and parses birth only when the source states it explicitly. The
  ṭabaqa-derived *fallback* window for a narrator the source leaves undated is
  **not** this module's job — that is da#166. We emit only what the text states
  and leave the rest ``None`` (no fabrication), so da#165 (reconcile) and da#166
  (fallback) fill the gaps cleanly.

* **Always a concrete precision.** Every emitted date carries a concrete
  :class:`~src.models.enums.DatePrecision`; a silent source maps to
  ``UNKNOWN`` (the string ``"unknown"``), never a null precision. This honours
  the foundation-review carry-forward that a null precision must read as
  ``UNKNOWN`` and never let a downstream ``DatePrecision(None)`` occur (the
  producer-side null-vs-unknown disagreement is tracked by da#239; this module's
  output never participates in it).

CE conversion (:func:`death_ce_bounds`) uses the **inclusive-window** converter
:func:`~src.utils.hijri.ah_year_to_ce_range`, not the point
:func:`~src.utils.hijri.ah_to_ce`: a death late in an AH year would convert ~1
CE year low through the point function (which returns the CE year of 1
Muharram). This is the second foundation-review carry-forward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.enums import DatePrecision
from src.utils.hijri import ah_year_to_ce_range

__all__ = [
    "MAX_PLAUSIBLE_AH",
    "MIN_PLAUSIBLE_AH",
    "NarratorDates",
    "ParsedDate",
    "death_ce_bounds",
    "extract_narrator_dates",
    "parse_year_notation",
    "to_field_dict",
]

# A plausible Hijri *year* for a hadith narrator. The Hijri era starts at 1; the
# rijāl corpus tops out well under 1500 AH. Numbers outside this band in a date
# field are noise (page refs, hadith numbers leaking into prose) and are dropped.
MIN_PLAUSIBLE_AH = 1
MAX_PLAUSIBLE_AH = 1500

# Itqān (and kin) use ``"-"`` as a "no value" placeholder.
_PLACEHOLDERS = frozenset({"", "-", "—", "–", "?", "غير معروف", "مجهول"})

# Map Arabic-Indic (U+0660..) and Extended/Persian (U+06F0..) digits to ASCII so a
# single ASCII ``\d`` / ``int()`` path handles every numeral form ("توفي سنة ١٥٠").
_DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_YEAR_RE = re.compile(r"\d+")

# "between A … (and|to|–) B" — Arabic ``بين A … (إلى|الى|و|-) B`` and ASCII
# ``between A and B``. ``\D*?`` (lazy non-digits) skips an interposed "هـ"/comma
# without crossing into the next number, so ``و200`` (waw glued to the digit)
# parses as well as ``إلى 170``.
_RANGE_RE = re.compile(
    r"^\s*(?:بين|between)\s*(\d+)\D*?(?:إلى|الى|الي|و|to|-|–|—)\s*(\d+)",
)
# A bare ASCII "A - B" span (e.g. "130-135"), anchored so it cannot swallow a
# comma-separated alternatives list.
_ASCII_SPAN_RE = re.compile(r"^\s*(\d+)\s*[-–—]\s*(\d+)\s*$")
# "after A" — Arabic ``بعد A`` / ASCII ``after A``. Anchored to the start: a
# trailing ``بعد`` buried inside a competing-attestations list does not hijack the
# primary (leading) clause. The digit must follow the marker, so ``بعدها`` ("after
# it", no year) never matches.
_AFTER_RE = re.compile(r"^\s*(?:بعد|after)\s*(\d+)")
# "before A" — Arabic ``قبل A`` / ASCII ``before A``. Same anchoring; ``قبلها``
# ("before it") has no digit and is correctly skipped.
_BEFORE_RE = re.compile(r"^\s*(?:قبل|before)\s*(\d+)")
# Approximation markers — Arabic ``تقريبا/نحو/حوالي/قرابة/زهاء`` and ASCII
# ``circa/c./ca./~/approx``. Position-independent: "180 هـ تقريبا" trails the year.
_CIRCA_RE = re.compile(r"تقريبا|تقريباً|نحو|حوالي|قرابة|زهاء|circa|\bca?\.|~|approx")


@dataclass(frozen=True)
class ParsedDate:
    """A single life-event (birth or death) extracted from source notation.

    ``earliest``/``latest`` are the inclusive AH bounds that map 1:1 onto the
    ``*_year_ah_earliest`` / ``*_year_ah_latest`` model fields; ``point`` is the
    "best single value" that maps onto the existing ``*_year_ah`` point estimate.
    ``precision`` is **always** concrete (never ``None``) — a date this object
    exists for was attested, so its precision is ``EXACT``/``RANGE``/``CIRCA``/
    ``AFTER``/``BEFORE``. The ``UNKNOWN`` precision is reserved for the
    *absence* of a date and is produced by :func:`to_field_dict` from a ``None``
    parse, never by this object.

    Invariant: when both bounds are present, ``earliest <= latest`` (the same
    ordering the Narrator model validator enforces).
    """

    point: int | None
    earliest: int | None
    latest: int | None
    precision: DatePrecision

    def __post_init__(self) -> None:
        if self.earliest is not None and self.latest is not None and self.earliest > self.latest:
            msg = f"ParsedDate earliest ({self.earliest}) must not exceed latest ({self.latest})"
            raise ValueError(msg)


@dataclass(frozen=True)
class NarratorDates:
    """The death (primary) and birth (optional) dates parsed for one narrator."""

    death: ParsedDate | None = None
    birth: ParsedDate | None = None


def _plausible(year: int) -> bool:
    return MIN_PLAUSIBLE_AH <= year <= MAX_PLAUSIBLE_AH


def _years(text: str) -> list[int]:
    """All plausible AH years in ``text``, in order of appearance."""
    return [y for tok in _YEAR_RE.findall(text) if _plausible(y := int(tok))]


def parse_year_notation(value: object) -> ParsedDate | None:
    """Parse a free-text rijāl date field into a :class:`ParsedDate`.

    Returns ``None`` when the field states no usable year — a placeholder
    (``"-"``), empty, or pure prose with no plausible AH numeral
    (``"في خلافة عبد الملك بن مروان"`` — "in the caliphate of ʿAbd al-Malik").
    Recognised notation, in priority order:

    * **range** — ``"بين 161 هـ إلى 170 هـ"`` / ``"between 130 and 135"`` /
      ``"130-135"`` → ``RANGE`` ``[a, b]``.
    * **after** — leading ``"بعد 130 هـ"`` / ``"after 130"`` → ``AFTER``,
      ``earliest = 130``, ``latest = None``.
    * **before** — leading ``"قبل 95 هـ"`` / ``"before 95"`` → ``BEFORE``,
      ``latest = 95``, ``earliest = None``.
    * **circa** — ``"180 هـ تقريبا"`` / ``"circa 150"`` → ``CIRCA`` on the year
      (bounds stay at the point — we do not fabricate a ± window the source did
      not state; the fuzziness is carried by the precision).
    * **exact** — a single attested year ``"168 هـ"`` → ``EXACT`` ``[y, y]``.
    * **alternatives** — several competing years with no governing marker
      (``"230 هـ ، أو 232 هـ"``, ``"73 هـ ، أو 74 هـ ، وقيل 59 هـ"``) → ``RANGE``
      spanning ``[min, max]`` of the attested candidates, with the first-stated
      year as the point. The true date lies within the attested span; this is an
      honest envelope, not a fabricated midpoint.
    """
    if value is None:
        return None
    text = str(value).translate(_DIGIT_TRANS).strip()
    if text in _PLACEHOLDERS:
        return None

    years = _years(text)
    if not years:
        return None  # prose / era reference with no plausible AH year → silent

    # 1) Explicit range (leading ``بين``/``between``, or a bare ASCII "A-B" span).
    rmatch = _RANGE_RE.match(text) or _ASCII_SPAN_RE.match(text)
    if rmatch:
        a, b = int(rmatch.group(1)), int(rmatch.group(2))
        if _plausible(a) and _plausible(b):
            lo, hi = sorted((a, b))
            return ParsedDate(point=a, earliest=lo, latest=hi, precision=DatePrecision.RANGE)

    # 2) Open lower bound: "after A" governing the leading clause.
    amatch = _AFTER_RE.match(text)
    if amatch:
        a = int(amatch.group(1))
        if _plausible(a):
            return ParsedDate(point=a, earliest=a, latest=None, precision=DatePrecision.AFTER)

    # 3) Open upper bound: "before A" governing the leading clause.
    bmatch = _BEFORE_RE.match(text)
    if bmatch:
        b = int(bmatch.group(1))
        if _plausible(b):
            return ParsedDate(point=b, earliest=None, latest=b, precision=DatePrecision.BEFORE)

    # 4) Approximation marker on a single year ("180 هـ تقريبا", "circa 150").
    if _CIRCA_RE.search(text):
        y = years[0]
        return ParsedDate(point=y, earliest=y, latest=y, precision=DatePrecision.CIRCA)

    # 5) A single attested year → exact.
    distinct = sorted(set(years))
    if len(distinct) == 1:
        y = distinct[0]
        return ParsedDate(point=y, earliest=y, latest=y, precision=DatePrecision.EXACT)

    # 6) Several competing attestations with no governing marker → honest span.
    return ParsedDate(
        point=years[0],
        earliest=distinct[0],
        latest=distinct[-1],
        precision=DatePrecision.RANGE,
    )


def extract_narrator_dates(
    *,
    death_text: object = None,
    birth_text: object = None,
) -> NarratorDates:
    """Extract a narrator's death (primary) and birth (optional) dates.

    Death-anchored: ``death_text`` is the well-attested ``wafāt`` field and is the
    primary signal; ``birth_text`` is parsed only when the source states it (birth
    is rare in the genre — the ṭabaqa-derived birth *fallback* is da#166, not
    here). Either side is ``None`` in the result when its source field is silent.
    """
    return NarratorDates(
        death=parse_year_notation(death_text),
        birth=parse_year_notation(birth_text),
    )


def death_ce_bounds(parsed: ParsedDate | None) -> tuple[int | None, int | None]:
    """CE ``(earliest, latest)`` for a death date via the inclusive-window converter.

    Uses :func:`~src.utils.hijri.ah_year_to_ce_range` — **not** the point
    :func:`~src.utils.hijri.ah_to_ce` — so the window spans from the *start* of
    the earliest AH year to the *end* of the latest AH year. The point converter
    returns the CE year of 1 Muharram, which renders a death late in an AH year
    ~1 CE year low; the range converter avoids that systematic bias. An open bound
    (``AFTER`` has no ``latest``; ``BEFORE`` has no ``earliest``) yields ``None``
    on that side.
    """
    if parsed is None:
        return None, None
    ce_earliest = ah_year_to_ce_range(parsed.earliest)[0] if parsed.earliest is not None else None
    ce_latest = ah_year_to_ce_range(parsed.latest)[1] if parsed.latest is not None else None
    return ce_earliest, ce_latest


def to_field_dict(prefix: str, parsed: ParsedDate | None) -> dict[str, str | int | None]:
    """Map a parse result onto the ``Narrator`` / bio-schema date fields.

    ``prefix`` is ``"birth"`` or ``"death"``. Returns the four columns
    (``{prefix}_year_ah``, ``{prefix}_year_ah_earliest``,
    ``{prefix}_year_ah_latest``, ``{prefix}_date_precision``). A ``None`` parse
    (silent source) yields ``None`` bounds and the concrete ``"unknown"``
    precision string — **never** a null precision, so no downstream
    ``DatePrecision(None)`` can arise.
    """
    if parsed is None:
        return {
            f"{prefix}_year_ah": None,
            f"{prefix}_year_ah_earliest": None,
            f"{prefix}_year_ah_latest": None,
            f"{prefix}_date_precision": DatePrecision.UNKNOWN.value,
        }
    return {
        f"{prefix}_year_ah": parsed.point,
        f"{prefix}_year_ah_earliest": parsed.earliest,
        f"{prefix}_year_ah_latest": parsed.latest,
        f"{prefix}_date_precision": parsed.precision.value,
    }
