"""Tests for the da#337 generic-name eligibility screen (``is_generic_name``).

The canonical id is a pure function of the normalized name, so a name too generic
to identify one person (bare kunya, mononym, bare patronymic, short fragment)
over-merges distinct people onto one ``nar:`` node and inflates its betweenness.
These lock the screen that flags such names for the PR-2 split stage:

* a full ``ism + nasab`` is never flagged (thin-name gate),
* each generic shape IS flagged once above the mention floor,
* the same shapes below the floor are not (aggregation gate), and
* empty/blank input returns ``False`` without raising.
"""

from __future__ import annotations

import pytest

from src.resolve.generic_name import GENERIC_MIN_MENTIONS, is_generic_name
from src.utils.arabic import normalize_arabic

# Names already run through the production normalizer, exactly as the caller feeds
# them (``name_ar_normalized``). ``mc`` is a mention count comfortably above the
# floor unless a test overrides it.
_MC = GENERIC_MIN_MENTIONS + 100

_FULL_ISM_NASAB = normalize_arabic("محمد بن اسماعيل البخاري")  # ism + nasab + nisba
_BARE_KUNYA = normalize_arabic("أبو عبد الله")  # "father of ʿAbd Allāh"
_BARE_NASAB = normalize_arabic("ابن عمر")  # "son of ʿUmar", no ism
_MONONYM = normalize_arabic("سفيان")  # single token
_FRAGMENT = normalize_arabic("عبد الله")  # bare two-token compound given name
_KUNYA_WITH_NASAB = normalize_arabic("أبو بكر بن محمد")  # kunya + patronymic → specific


class TestThinNameGate:
    def test_full_ism_nasab_is_not_generic(self) -> None:
        # Three significant tokens (محمد / اسماعيل / البخاري) — specific enough.
        assert is_generic_name(_FULL_ISM_NASAB, _MC) is False

    def test_kunya_with_nasab_is_not_generic(self) -> None:
        # A kunya carrying a بن patronymic is disambiguated, so NOT a bare kunya.
        assert is_generic_name(_KUNYA_WITH_NASAB, _MC) is False


class TestGenericShapesAboveFloor:
    @pytest.mark.parametrize(
        ("name", "label"),
        [
            (_BARE_KUNYA, "bare kunya"),
            (_BARE_NASAB, "bare nasab"),
            (_MONONYM, "single-token mononym"),
            (_FRAGMENT, "two-token fragment"),
        ],
    )
    def test_each_generic_shape_is_flagged(self, name: str, label: str) -> None:
        assert is_generic_name(name, _MC) is True, label


class TestAggregationGate:
    @pytest.mark.parametrize("name", [_BARE_KUNYA, _BARE_NASAB, _MONONYM, _FRAGMENT])
    def test_generic_shape_below_floor_is_not_flagged(self, name: str) -> None:
        assert is_generic_name(name, GENERIC_MIN_MENTIONS - 1) is False

    def test_exactly_at_floor_is_flagged(self) -> None:
        # The gate is inclusive: mention_count == floor qualifies.
        assert is_generic_name(_MONONYM, GENERIC_MIN_MENTIONS) is True


class TestDegenerateInput:
    @pytest.mark.parametrize("name", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_is_false_without_crash(self, name: str) -> None:
        assert is_generic_name(name, _MC) is False
