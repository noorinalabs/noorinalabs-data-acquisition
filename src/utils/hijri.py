"""Hijri (AH) <-> Gregorian (CE) calendar conversion.

The corpus records narrator birth/death and historical-event dates in the
Islamic (Hijri / *Anno Hegirae*) calendar, but the graph, API, and frontend
reason and display in the Gregorian (*Common Era*) calendar. Bare AH integers
are therefore converted to CE **once, at resolve time**, and both values are
stored so downstream readers never convert (this mirrors how
``data/curated/historical_events.yaml`` already carries both AH and CE bounds).

The conversion is backed by :mod:`convertdate.islamic`, which implements the
standard *tabular / arithmetic* astronomical-Hijri algorithm (not observational
moon-sighting). At the year granularity we actually need, observational variance
(+/-1 day on a month boundary) is irrelevant, and the arithmetic algorithm is
deterministic and dependency-light. As a sanity anchor the conversion satisfies
``CE ~= AH + 622 - AH/33`` (the ~3% drift of the shorter lunar year); the unit
tests pin known reference points (1 AH begins 622 CE) rather than the formula.

Helpers are exposed in both directions and at two granularities so the
consumers can pick what they need:

* :func:`ah_to_ce` / :func:`ce_to_ah` -- point *year* conversion (the common
  case: an attested AH death-year -> its CE year).
* :func:`ah_year_to_ce_range` -- the CE *span* of an AH year (a lunar year
  straddles two Gregorian years), for callers that want an inclusive window.
* :func:`ah_to_ce_date` / :func:`ce_to_ah_date` -- full ``(year, month, day)``
  conversion, for the date-parsing step that has a precise date.

Used by da#164 (date parsing) and da#166 (tabaqa -> estimated-window fallback)
to normalize AH dates to CE windows at resolve time.
"""

from __future__ import annotations

import datetime

from convertdate import islamic

__all__ = [
    "ah_to_ce",
    "ah_to_ce_date",
    "ah_year_to_ce_range",
    "ce_to_ah",
    "ce_to_ah_date",
]


def _validate_ah_year(year_ah: int) -> None:
    """Reject non-positive AH years (the Hijri era starts at year 1)."""
    if not isinstance(year_ah, int) or isinstance(year_ah, bool):
        raise TypeError(f"AH year must be an int, got {type(year_ah).__name__}")
    if year_ah < 1:
        raise ValueError(f"AH year must be >= 1, got {year_ah}")


def ah_to_ce(year_ah: int) -> int:
    """Convert an AH year to the CE year in which it *begins* (1 Muharram).

    This is the point-estimate conversion used when storing a CE mirror of an
    attested AH year: ``ah_to_ce(1) == 622``, ``ah_to_ce(256) == 869`` (the AH
    year of al-Bukhari's death). Because a lunar year is ~11 days shorter than a
    solar one, some AH years begin and end in the same CE year and some straddle
    two; this returns the CE year of 1 Muharram. For the full span use
    :func:`ah_year_to_ce_range`.
    """
    _validate_ah_year(year_ah)
    gregorian_year, _, _ = islamic.to_gregorian(year_ah, 1, 1)
    return int(gregorian_year)


def ce_to_ah(year_ce: int) -> int:
    """Convert a CE year to the AH year in effect on 1 January of that year.

    Inverse companion to :func:`ah_to_ce` at year granularity:
    ``ce_to_ah(622) == 1``. Because the calendars drift, this is the AH year
    that contains 1 January ``year_ce``; it is not an exact round-trip inverse
    for every input (year boundaries do not line up), which the round-trip unit
    tests assert tolerantly (+/-1).
    """
    if not isinstance(year_ce, int) or isinstance(year_ce, bool):
        raise TypeError(f"CE year must be an int, got {type(year_ce).__name__}")
    islamic_year, _, _ = islamic.from_gregorian(year_ce, 1, 1)
    return int(islamic_year)


def ah_year_to_ce_range(year_ah: int) -> tuple[int, int]:
    """Return the inclusive ``(ce_start, ce_end)`` CE years an AH year spans.

    A Hijri year runs from 1 Muharram to the last day of Dhu al-Hijja; mapped to
    the Gregorian calendar it covers one or two CE years. ``ah_year_to_ce_range``
    returns the CE year of its first day and the CE year of its last day:
    ``ah_year_to_ce_range(1) == (622, 623)``. Callers that want an inclusive CE
    window for an AH year (e.g. da#166's estimated-window fallback) can use this
    directly; ``ce_start == ce_end`` when the AH year falls inside a single CE
    year.
    """
    _validate_ah_year(year_ah)
    ce_start, _, _ = islamic.to_gregorian(year_ah, 1, 1)
    last_day = islamic.month_length(year_ah, 12)
    ce_end, _, _ = islamic.to_gregorian(year_ah, 12, last_day)
    return int(ce_start), int(ce_end)


def ah_to_ce_date(year_ah: int, month: int, day: int) -> datetime.date:
    """Convert a full AH ``(year, month, day)`` to a Gregorian :class:`date`.

    ``month`` is 1-12 (Muharram..Dhu al-Hijja) and ``day`` is 1-29/30. Raises
    :class:`ValueError` for an out-of-range month or a day that exceeds the
    length of that Hijri month.
    """
    _validate_ah_year(year_ah)
    if not 1 <= month <= 12:
        raise ValueError(f"AH month must be in 1..12, got {month}")
    max_day = islamic.month_length(year_ah, month)
    if not 1 <= day <= max_day:
        raise ValueError(f"AH day must be in 1..{max_day} for {year_ah}-{month}, got {day}")
    gy, gm, gd = islamic.to_gregorian(year_ah, month, day)
    return datetime.date(gy, gm, gd)


def ce_to_ah_date(value: datetime.date) -> tuple[int, int, int]:
    """Convert a Gregorian :class:`date` to an AH ``(year, month, day)`` tuple."""
    if not isinstance(value, datetime.date):
        raise TypeError(f"value must be a datetime.date, got {type(value).__name__}")
    iy, im, idy = islamic.from_gregorian(value.year, value.month, value.day)
    return int(iy), int(im), int(idy)
