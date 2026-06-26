"""Tests for src.resolve.geography — location normalization + travel plausibility."""

from __future__ import annotations

import pytest

from src.resolve.geography import (
    is_travel_plausible,
    regions_plausible,
    resolve_region,
)


# ---------------------------------------------------------------------------
# resolve_region
# ---------------------------------------------------------------------------
class TestResolveRegion:
    @pytest.mark.parametrize(
        ("location", "expected"),
        [
            ("Kufa", "iraq"),
            ("al-Kufah", "iraq"),  # leading article stripped
            ("Basra", "iraq"),
            ("Baghdad", "iraq"),
            ("Medina", "hijaz"),
            ("Mecca", "hijaz"),
            ("Damascus", "sham"),
            ("Fustat", "egypt"),
            ("Nishapur", "khurasan"),
            ("Bukhara", "transoxiana"),
            ("Cordoba", "andalus"),
            ("Sanaa", "yemen"),
            ("Rayy", "jibal"),
            ("Shiraz", "fars"),
            ("Qayrawan", "maghrib"),
        ],
    )
    def test_known_transliterations(self, location: str, expected: str) -> None:
        assert resolve_region(location) == expected

    @pytest.mark.parametrize(
        ("location", "expected"),
        [
            ("الكوفة", "iraq"),  # الكوفة
            ("بغداد", "iraq"),  # بغداد
            ("دمشق", "sham"),  # دمشق
            ("مكة", "hijaz"),  # مكة
            ("بخارى", "transoxiana"),  # بخارى
        ],
    )
    def test_known_arabic(self, location: str, expected: str) -> None:
        assert resolve_region(location) == expected

    def test_arabic_multiword_token_fallback(self) -> None:
        # "المدينة المنورة" — only the first token is a known alias.
        assert resolve_region("المدينة المنورة") == "hijaz"

    def test_comma_separated_token_fallback(self) -> None:
        assert resolve_region("Basra, Iraq") == "iraq"

    def test_region_name_directly(self) -> None:
        assert resolve_region("Khurasan") == "khurasan"

    @pytest.mark.parametrize("location", [None, "", "   ", "Atlantis", "Timbuktu"])
    def test_unresolvable_returns_none(self, location: str | None) -> None:
        assert resolve_region(location) is None


# ---------------------------------------------------------------------------
# regions_plausible
# ---------------------------------------------------------------------------
class TestRegionsPlausible:
    def test_same_region(self) -> None:
        assert regions_plausible("iraq", "iraq") is True

    def test_adjacent_regions(self) -> None:
        assert regions_plausible("iraq", "sham") is True

    def test_long_but_attested_corridor(self) -> None:
        # Iraq <-> Transoxiana (al-Bukhari's rihla) — within threshold.
        assert regions_plausible("iraq", "transoxiana") is True

    def test_hub_always_plausible(self) -> None:
        # Hijaz (pilgrimage hub) is plausible with the far east.
        assert regions_plausible("hijaz", "transoxiana") is True
        assert regions_plausible("khurasan", "hijaz") is True

    def test_antipodal_implausible(self) -> None:
        assert regions_plausible("andalus", "khurasan") is False
        assert regions_plausible("andalus", "transoxiana") is False
        assert regions_plausible("egypt", "transoxiana") is False

    def test_unknown_region_is_plausible(self) -> None:
        assert regions_plausible(None, "iraq") is True
        assert regions_plausible("iraq", None) is True
        assert regions_plausible("atlantis", "iraq") is True

    def test_symmetric(self) -> None:
        assert regions_plausible("andalus", "khurasan") == regions_plausible("khurasan", "andalus")


# ---------------------------------------------------------------------------
# is_travel_plausible (free-text wrapper)
# ---------------------------------------------------------------------------
class TestIsTravelPlausible:
    def test_plausible_free_text(self) -> None:
        assert is_travel_plausible("Kufa", "Basra") is True

    def test_implausible_free_text(self) -> None:
        assert is_travel_plausible("Cordoba", "Bukhara") is False

    def test_unresolvable_is_plausible(self) -> None:
        # Unknown location -> no geographic signal -> keep (plausible).
        assert is_travel_plausible("Atlantis", "Bukhara") is True
        assert is_travel_plausible(None, "Bukhara") is True
