"""Tests for src.utils.grade — raw grade -> HadithGrade normalization (da#148)."""

from __future__ import annotations

import pytest

from src.models.enums import HadithGrade
from src.utils.grade import normalize_grade


class TestNormalizeGrade:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Arabic forms (with and without diacritics).
            ("صحيح", HadithGrade.SAHIH.value),
            ("صَحِيح", HadithGrade.SAHIH.value),
            ("حسن", HadithGrade.HASAN.value),
            ("ضعيف", HadithGrade.DAIF.value),
            ("موضوع", HadithGrade.MAWDU.value),
            ("صحيح لغيره", HadithGrade.SAHIH_LI_GHAYRIHI.value),
            ("حسن لغيره", HadithGrade.HASAN_LI_GHAYRIHI.value),
            # English / transliterated forms.
            ("Sahih", HadithGrade.SAHIH.value),
            ("HASAN", HadithGrade.HASAN.value),
            ("Da'if", HadithGrade.DAIF.value),
            ("Daif", HadithGrade.DAIF.value),
            ("weak", HadithGrade.DAIF.value),
            ("authentic", HadithGrade.SAHIH.value),
            ("fabricated", HadithGrade.MAWDU.value),
            ("Sahih li ghayrihi", HadithGrade.SAHIH_LI_GHAYRIHI.value),
        ],
    )
    def test_known_grades(self, raw: str, expected: str) -> None:
        assert normalize_grade(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "no-clear-grade", 42])
    def test_unknown_or_empty_maps_to_unknown(self, raw: object) -> None:
        assert normalize_grade(raw) == HadithGrade.UNKNOWN.value  # type: ignore[arg-type]

    def test_li_ghayrihi_wins_over_bare_grade(self) -> None:
        # The qualified form must not be matched as the bare grade first.
        assert normalize_grade("صحيح لغيره") == HadithGrade.SAHIH_LI_GHAYRIHI.value
        assert normalize_grade("حسن لغيره") == HadithGrade.HASAN_LI_GHAYRIHI.value

    def test_return_value_is_always_a_valid_enum_member(self) -> None:
        for raw in ["صحيح", "weak", "garbage", "", None]:
            assert normalize_grade(raw) in {g.value for g in HadithGrade}  # type: ignore[arg-type]
