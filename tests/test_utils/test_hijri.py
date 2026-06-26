"""Tests for src.utils.hijri — AH<->CE calendar conversion (da#163)."""

from __future__ import annotations

import datetime

import pytest

from src.utils.hijri import (
    ah_to_ce,
    ah_to_ce_date,
    ah_year_to_ce_range,
    ce_to_ah,
    ce_to_ah_date,
)


class TestAhToCe:
    @pytest.mark.parametrize(
        ("year_ah", "expected_ce"),
        [
            (1, 622),  # The Hijra: 1 Muharram AH 1 == 19 July 622 CE.
            (256, 869),  # al-Bukhari (d. 256 AH) — AH year begins in 869 CE.
            (1430, 2008),  # 1 Muharram 1430 == 29 Dec 2008 CE.
        ],
    )
    def test_known_reference_years(self, year_ah: int, expected_ce: int) -> None:
        assert ah_to_ce(year_ah) == expected_ce

    def test_matches_formula_oracle(self) -> None:
        # CE ~= AH + 622 - AH/33 (the ~3% lunar drift). Library is authoritative;
        # the formula is only a sanity oracle, so allow a couple of years' slack.
        for year_ah in (1, 50, 100, 256, 700, 1200, 1445):
            approx = year_ah + 622 - year_ah / 33
            assert abs(ah_to_ce(year_ah) - approx) <= 2

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_year_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError):
            ah_to_ce(bad)

    @pytest.mark.parametrize("bad", [1.5, "1", None, True])
    def test_non_int_year_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError):
            ah_to_ce(bad)  # type: ignore[arg-type]


class TestCeToAh:
    @pytest.mark.parametrize(
        ("year_ce", "expected_ah"),
        [
            (623, 1),  # 1 Jan 623 falls in AH 1 (the Hijra is mid-622).
            (870, 256),  # 1 Jan 870 falls in AH 256 (al-Bukhari's death year).
            (2024, 1445),
        ],
    )
    def test_known_reference_years(self, year_ce: int, expected_ah: int) -> None:
        assert ce_to_ah(year_ce) == expected_ah

    def test_pre_hijra_ce_year_maps_to_year_zero(self) -> None:
        # 1 Jan 622 predates the Hijra (mid-622), so no AH year is yet in effect.
        assert ce_to_ah(622) == 0

    @pytest.mark.parametrize("bad", [1.5, "622", None, True])
    def test_non_int_year_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError):
            ce_to_ah(bad)  # type: ignore[arg-type]


class TestRoundTrip:
    @pytest.mark.parametrize("year_ah", [1, 50, 256, 700, 1200, 1445])
    def test_ah_ce_ah_round_trip_within_one(self, year_ah: int) -> None:
        # Year boundaries do not line up across the calendars, so a year-level
        # round-trip is exact +/-1, never further off.
        assert abs(ce_to_ah(ah_to_ce(year_ah)) - year_ah) <= 1

    @pytest.mark.parametrize(
        "value",
        [
            datetime.date(622, 7, 19),
            datetime.date(870, 9, 1),
            datetime.date(2008, 12, 29),
        ],
    )
    def test_date_round_trip_exact(self, value: datetime.date) -> None:
        # Full-precision dates round-trip exactly.
        assert ah_to_ce_date(*ce_to_ah_date(value)) == value


class TestAhYearToCeRange:
    def test_year_one_spans_two_ce_years(self) -> None:
        assert ah_year_to_ce_range(1) == (622, 623)

    def test_start_le_end_and_brackets_point(self) -> None:
        for year_ah in (1, 256, 700, 1200, 1445):
            start, end = ah_year_to_ce_range(year_ah)
            assert start <= end <= start + 1
            assert start == ah_to_ce(year_ah)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_year_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError):
            ah_year_to_ce_range(bad)


class TestAhToCeDate:
    def test_hijra_day(self) -> None:
        assert ah_to_ce_date(1, 1, 1) == datetime.date(622, 7, 19)

    @pytest.mark.parametrize("month", [0, 13, -1])
    def test_bad_month_rejected(self, month: int) -> None:
        with pytest.raises(ValueError):
            ah_to_ce_date(1, month, 1)

    @pytest.mark.parametrize("day", [0, 31])
    def test_bad_day_rejected(self, day: int) -> None:
        with pytest.raises(ValueError):
            ah_to_ce_date(1, 1, day)


class TestCeToAhDate:
    def test_hijra_day(self) -> None:
        assert ce_to_ah_date(datetime.date(622, 7, 19)) == (1, 1, 1)

    def test_non_date_rejected(self) -> None:
        with pytest.raises(TypeError):
            ce_to_ah_date("2008-12-29")  # type: ignore[arg-type]
