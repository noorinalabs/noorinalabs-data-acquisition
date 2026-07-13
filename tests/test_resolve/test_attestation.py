"""Tests for src.resolve.attestation.derive_attestation (da#370)."""

from __future__ import annotations

import pytest

from src.models.enums import Attestation
from src.resolve.attestation import derive_attestation


@pytest.mark.parametrize("mention_count", [0, None])
def test_no_isnad_mention_is_biographical_only(mention_count: int | None) -> None:
    """A record with no isnad mention (bio-promoted, mention_count 0 or null) is bio-only."""
    assert derive_attestation(mention_count) == Attestation.BIOGRAPHICAL_ONLY.value
    assert derive_attestation(mention_count) == "biographical_only"


@pytest.mark.parametrize("mention_count", [1, 2, 1695])
def test_any_isnad_mention_is_attested(mention_count: int) -> None:
    """A single real chain mention is enough to make a narrator isnad-attested."""
    assert derive_attestation(mention_count) == Attestation.ISNAD_ATTESTED.value
    assert derive_attestation(mention_count) == "isnad_attested"


def test_never_derives_unknown() -> None:
    """UNKNOWN is only the loader's legacy default, never a derived value."""
    assert derive_attestation(0) != Attestation.UNKNOWN.value
    assert derive_attestation(5) != Attestation.UNKNOWN.value
