"""Tests for narrator life-date extraction from rijāl notation (da#164).

Covers the real notation forms found in the Itqān ``death`` field — bare year,
``بين … إلى/و …`` ranges, ``بعد``/``قبل`` open bounds, ``تقريبا``/circa, and
multi-alternative ``أو``/``وقيل`` lists — plus the two foundation-review
carry-forwards: the death-year→CE conversion uses the inclusive *range* window
(not the point converter), and every emitted date carries a concrete precision
(never ``DatePrecision(None)``).
"""

from __future__ import annotations

import pytest

from src.models.enums import DatePrecision
from src.models.narrator import Narrator
from src.parse.narrator_dates import (
    NarratorDates,
    ParsedDate,
    death_ce_bounds,
    extract_narrator_dates,
    parse_year_notation,
    to_field_dict,
)
from src.utils.hijri import ah_to_ce, ah_year_to_ce_range


class TestBareYear:
    def test_single_hijri_year_is_exact(self) -> None:
        d = parse_year_notation("168 هـ")
        assert d == ParsedDate(point=168, earliest=168, latest=168, precision=DatePrecision.EXACT)

    def test_year_without_marker(self) -> None:
        d = parse_year_notation("214")
        assert d is not None
        assert d.precision is DatePrecision.EXACT
        assert (d.earliest, d.latest) == (214, 214)

    def test_arabic_indic_digits(self) -> None:
        # "توفي سنة ١٥٠" — died in the year 150, Arabic-Indic numerals.
        d = parse_year_notation("توفي سنة ١٥٠ هـ")
        assert d is not None
        assert d.point == 150
        assert d.precision is DatePrecision.EXACT


class TestRange:
    def test_bayna_ila(self) -> None:
        d = parse_year_notation("بين 161 هـ إلى 170 هـ")
        assert d == ParsedDate(point=161, earliest=161, latest=170, precision=DatePrecision.RANGE)

    def test_bayna_waw_glued_digit(self) -> None:
        # "بين 191 هـ و200 هـ" — the connective waw is glued to the second number.
        d = parse_year_notation("بين 191 هـ و200 هـ")
        assert d is not None
        assert (d.earliest, d.latest) == (191, 200)
        assert d.precision is DatePrecision.RANGE

    def test_ascii_between(self) -> None:
        d = parse_year_notation("between 130 and 135")
        assert (d.earliest, d.latest, d.precision) == (130, 135, DatePrecision.RANGE)  # type: ignore[union-attr]

    def test_ascii_dash_span(self) -> None:
        d = parse_year_notation("130-135")
        assert (d.earliest, d.latest, d.precision) == (130, 135, DatePrecision.RANGE)  # type: ignore[union-attr]

    def test_reversed_bounds_are_sorted(self) -> None:
        d = parse_year_notation("بين 170 هـ إلى 161 هـ")
        assert (d.earliest, d.latest) == (161, 170)  # type: ignore[union-attr]


class TestAfterBefore:
    def test_after_sets_only_earliest(self) -> None:
        d = parse_year_notation("بعد 130 هـ")
        assert d == ParsedDate(point=130, earliest=130, latest=None, precision=DatePrecision.AFTER)

    def test_after_wins_over_trailing_alternative(self) -> None:
        # "بعد 200 هـ ، أو : 212 هـ" — the leading "after 200" governs.
        d = parse_year_notation("بعد 200 هـ ، أو : 212 هـ")
        assert d is not None
        assert d.precision is DatePrecision.AFTER
        assert (d.earliest, d.latest) == (200, None)

    def test_before_sets_only_latest(self) -> None:
        d = parse_year_notation("قبل 95 هـ")
        assert d == ParsedDate(point=95, earliest=None, latest=95, precision=DatePrecision.BEFORE)

    def test_ascii_after_before(self) -> None:
        assert parse_year_notation("after 130").precision is DatePrecision.AFTER  # type: ignore[union-attr]
        assert parse_year_notation("before 95").precision is DatePrecision.BEFORE  # type: ignore[union-attr]

    def test_qablaha_does_not_trigger_before(self) -> None:
        # "34 هـ أو قبلها" — "قبلها" ("before it") has no year after the marker, so
        # it must NOT be read as a BEFORE bound; the digits drive an honest span.
        d = parse_year_notation("34 هـ أو قبلها ، أو 35 هـ أو 36 هـ")
        assert d is not None
        assert d.precision is DatePrecision.RANGE
        assert (d.earliest, d.latest) == (34, 36)


class TestCirca:
    def test_trailing_taqriban(self) -> None:
        d = parse_year_notation("180 هـ تقريبا")
        assert d == ParsedDate(point=180, earliest=180, latest=180, precision=DatePrecision.CIRCA)

    def test_ascii_circa(self) -> None:
        assert parse_year_notation("circa 150").precision is DatePrecision.CIRCA  # type: ignore[union-attr]


class TestAlternatives:
    def test_two_alternatives_span(self) -> None:
        # "230 هـ ، أو 232 هـ" — competing attestations → honest [min,max] span.
        d = parse_year_notation("230 هـ ، أو 232هـ")
        assert d == ParsedDate(point=230, earliest=230, latest=232, precision=DatePrecision.RANGE)

    def test_many_alternatives_envelope_with_first_point(self) -> None:
        d = parse_year_notation("73 هـ ، أو 74 هـ ، وقيل : 59 هـ")
        assert d is not None
        assert d.point == 73  # first stated stays the point estimate
        assert (d.earliest, d.latest) == (59, 74)
        assert d.precision is DatePrecision.RANGE


class TestSilent:
    @pytest.mark.parametrize("value", ["-", "", "   ", "?", None])
    def test_placeholders_are_none(self, value: object) -> None:
        assert parse_year_notation(value) is None

    def test_prose_without_year_is_none(self) -> None:
        # "in the caliphate of ʿAbd al-Malik b. Marwan" — no plausible AH year.
        assert parse_year_notation("في خلافة عبد الملك بن مروان") is None

    def test_implausible_number_dropped(self) -> None:
        assert parse_year_notation("3200") is None
        assert parse_year_notation("0") is None


class TestConcretePrecisionInvariant:
    """Carry-forward #2: never emit a null precision / DatePrecision(None)."""

    def test_silent_source_maps_to_unknown_string(self) -> None:
        fields = to_field_dict("death", None)
        assert fields["death_date_precision"] == DatePrecision.UNKNOWN.value
        assert fields["death_date_precision"] is not None
        # The string round-trips back through the enum without DatePrecision(None).
        assert DatePrecision(fields["death_date_precision"]) is DatePrecision.UNKNOWN

    @pytest.mark.parametrize(
        "text",
        ["168 هـ", "بين 161 هـ إلى 170 هـ", "بعد 130 هـ", "قبل 95 هـ", "180 هـ تقريبا"],
    )
    def test_every_parsed_date_has_concrete_precision(self, text: str) -> None:
        d = parse_year_notation(text)
        assert d is not None
        assert d.precision is not DatePrecision.UNKNOWN
        assert DatePrecision(d.precision.value) is d.precision

    def test_to_field_dict_precision_always_a_valid_enum_value(self) -> None:
        d = parse_year_notation("168 هـ")
        fields = to_field_dict("death", d)
        # No null precision can reach a downstream DatePrecision(None).
        assert DatePrecision(fields["death_date_precision"]) is DatePrecision.EXACT


class TestDeathCeViaRange:
    """Carry-forward #1: convert death AH→CE via the inclusive window, not the point."""

    def test_uses_range_end_not_point_for_latest(self) -> None:
        # AH 1 begins in 622 CE (point) but ends in 623 CE: a death late in the year
        # belongs to 623. The range converter must surface 623 as the CE latest, so
        # it is strictly greater than the point converter here.
        d = parse_year_notation("1")
        ce_earliest, ce_latest = death_ce_bounds(d)
        assert ce_earliest == ah_year_to_ce_range(1)[0]
        assert ce_latest == ah_year_to_ce_range(1)[1]
        assert ce_latest == 623
        assert ce_latest > ah_to_ce(1)  # 623 > 622 — the point converter is low

    def test_range_envelope_spans_start_of_earliest_to_end_of_latest(self) -> None:
        d = parse_year_notation("بين 161 هـ إلى 170 هـ")
        ce_earliest, ce_latest = death_ce_bounds(d)
        assert ce_earliest == ah_year_to_ce_range(161)[0]
        assert ce_latest == ah_year_to_ce_range(170)[1]

    def test_after_has_no_ce_latest(self) -> None:
        ce_earliest, ce_latest = death_ce_bounds(parse_year_notation("بعد 130 هـ"))
        assert ce_earliest == ah_year_to_ce_range(130)[0]
        assert ce_latest is None

    def test_before_has_no_ce_earliest(self) -> None:
        ce_earliest, ce_latest = death_ce_bounds(parse_year_notation("قبل 95 هـ"))
        assert ce_earliest is None
        assert ce_latest == ah_year_to_ce_range(95)[1]

    def test_none_parse_yields_none_bounds(self) -> None:
        assert death_ce_bounds(None) == (None, None)


class TestExtractNarratorDates:
    def test_death_anchored_birth_silent(self) -> None:
        dates = extract_narrator_dates(death_text="168 هـ")
        assert isinstance(dates, NarratorDates)
        assert dates.death is not None and dates.death.point == 168
        assert dates.birth is None  # no birth text → silent (da#166 fills later)

    def test_birth_parsed_when_present(self) -> None:
        dates = extract_narrator_dates(death_text="168 هـ", birth_text="نحو 90 هـ")
        assert dates.birth is not None
        assert dates.birth.precision is DatePrecision.CIRCA
        assert dates.birth.point == 90


class TestModelPopulation:
    """The extracted fields populate the da#161 Narrator bound/precision fields."""

    def test_fields_construct_a_valid_narrator(self) -> None:
        d = parse_year_notation("بين 161 هـ إلى 170 هـ")
        fields = to_field_dict("death", d)
        narrator = Narrator(
            id="nar:test-001",
            name_ar="فلان",
            name_en="Fulan",
            generation="tabii",
            gender="male",
            sect_affiliation="sunni",
            trustworthiness_consensus="thiqa",
            death_year_ah_earliest=fields["death_year_ah_earliest"],  # type: ignore[arg-type]
            death_year_ah_latest=fields["death_year_ah_latest"],  # type: ignore[arg-type]
            death_date_precision=DatePrecision(fields["death_date_precision"]),
        )
        assert narrator.death_year_ah_earliest == 161
        assert narrator.death_year_ah_latest == 170
        assert narrator.death_date_precision is DatePrecision.RANGE

    def test_after_bound_satisfies_model_ordering_validator(self) -> None:
        # AFTER leaves latest None; the model's earliest<=latest validator must pass.
        d = parse_year_notation("بعد 130 هـ")
        fields = to_field_dict("death", d)
        narrator = Narrator(
            id="nar:test-002",
            name_ar="فلان",
            name_en="Fulan",
            generation="tabii",
            gender="male",
            sect_affiliation="sunni",
            trustworthiness_consensus="thiqa",
            death_year_ah_earliest=fields["death_year_ah_earliest"],  # type: ignore[arg-type]
            death_year_ah_latest=fields["death_year_ah_latest"],  # type: ignore[arg-type]
            death_date_precision=DatePrecision(fields["death_date_precision"]),
        )
        assert narrator.death_year_ah_earliest == 130
        assert narrator.death_year_ah_latest is None
        assert narrator.death_date_precision is DatePrecision.AFTER


class TestParsedDateInvariant:
    def test_inverted_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            ParsedDate(point=170, earliest=170, latest=161, precision=DatePrecision.RANGE)
