"""Tests for src.resolve.mononym_split — da#248 mononym over-merge splitting.

The load-bearing property under test is **precision**: the split must never
fragment a legitimate single-person node. That is enforced two ways, both
asserted here:

* only registered mononyms are ever eligible (a non-registered name — including
  every single-person mononym — always abstains), and
* within a registered mononym, the split fires only when the chain evidence
  uniquely selects one person; ambiguous or absent evidence abstains.
"""

from __future__ import annotations

from src.parse.identity import make_canonical_id
from src.resolve.mononym_split import (
    MONONYM_REGISTRY,
    is_registered_mononym,
    refine_mononym_name,
)
from src.utils.arabic import normalize_arabic

# Normalized registry keys, built the same way the module and the disambiguator
# build names — via normalize_arabic — so the tests key on what the pipeline sees.
_SUFYAN = normalize_arabic("سفيان")
_YAHYA = normalize_arabic("يحيى")


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------
def test_registry_only_holds_multi_person_mononyms() -> None:
    # Every registered mononym must denote >=2 distinct people — a single-person
    # entry would be a fragmentation hazard.
    assert MONONYM_REGISTRY, "registry must not be empty"
    for name, persons in MONONYM_REGISTRY.items():
        assert len(persons) >= 2, f"{name!r} registered with <2 persons"


def test_dominant_single_referent_mononyms_are_not_registered() -> None:
    # شعبة (Shuʿba ibn al-Ḥajjāj) and مالك (Mālik ibn Anas) each have a dominant
    # single referent in the isnad corpus; registering them would split one
    # person. They must stay out of the registry.
    assert not is_registered_mononym(normalize_arabic("شعبة"))
    assert not is_registered_mononym(normalize_arabic("مالك"))


# ---------------------------------------------------------------------------
# Precision guard #1 — only registered names are eligible
# ---------------------------------------------------------------------------
def test_non_registered_name_never_splits_even_with_strong_evidence() -> None:
    # A full name (already disambiguated) with wildly separated neighbour deaths
    # must still abstain — no non-registered node is ever fragmented.
    assert refine_mononym_name(normalize_arabic("انس بن مالك"), [50, 250]) is None
    # A dominant-single-referent mononym is likewise never split.
    assert refine_mononym_name(normalize_arabic("مالك"), [100, 200]) is None


def test_already_split_name_is_stable_idempotent() -> None:
    # A name that is already disambiguated (a person norm_name, not a bare
    # registry key) abstains, so re-running the resolve stage is a no-op on ids.
    assert refine_mononym_name(normalize_arabic("سفيان الثوري"), [100]) is None


# ---------------------------------------------------------------------------
# Precision guard #2 — abstain unless evidence uniquely selects one person
# ---------------------------------------------------------------------------
def test_registered_name_without_evidence_abstains() -> None:
    assert refine_mononym_name(_SUFYAN, []) is None


def test_ambiguous_evidence_abstains() -> None:
    # A neighbour death ~130 AH is a plausible transmission partner for BOTH
    # Sufyāns (|161-130|=31 and |198-130|=68, both within 15-80), so the split
    # must abstain and leave the shared node intact.
    assert refine_mononym_name(_SUFYAN, [130]) is None


# ---------------------------------------------------------------------------
# Recall — a uniquely-selecting neighbour resolves to the right person
# ---------------------------------------------------------------------------
def test_sufyan_resolves_to_thawri_on_early_neighbour() -> None:
    # Neighbour death ~100 AH: plausible for al-Thawrī (d.161, gap 61) but not for
    # ibn ʿUyayna (d.198, gap 98 > 80) → unique al-Thawrī.
    person = refine_mononym_name(_SUFYAN, [100])
    assert person is not None
    assert person.norm_name == normalize_arabic("سفيان الثوري")


def test_sufyan_resolves_to_ibn_uyayna_on_late_neighbour() -> None:
    # Neighbour death ~170 AH: not plausible for al-Thawrī (gap 9 < 15) but
    # plausible for ibn ʿUyayna (gap 28) → unique ibn ʿUyayna.
    person = refine_mononym_name(_SUFYAN, [170])
    assert person is not None
    assert person.norm_name == normalize_arabic("سفيان بن عيينة")


def test_yahya_resolves_at_the_extremes() -> None:
    early = refine_mononym_name(_YAHYA, [100])
    late = refine_mononym_name(_YAHYA, [300])
    assert early is not None and early.norm_name == normalize_arabic("يحيى بن سعيد الانصاري")
    assert late is not None and late.norm_name == normalize_arabic("يحيى بن معين")


# ---------------------------------------------------------------------------
# The split yields DISTINCT graph nodes (the actual cycle-breaking mechanism)
# ---------------------------------------------------------------------------
def test_split_persons_get_distinct_canonical_ids() -> None:
    thawri = refine_mononym_name(_SUFYAN, [100])
    ibn_uyayna = refine_mononym_name(_SUFYAN, [170])
    assert thawri is not None and ibn_uyayna is not None
    base_id = make_canonical_id(_SUFYAN)
    thawri_id = make_canonical_id(thawri.norm_name)
    ibn_uyayna_id = make_canonical_id(ibn_uyayna.norm_name)
    # Three distinct nodes: the (now-residual) shared mononym and the two people.
    assert len({base_id, thawri_id, ibn_uyayna_id}) == 3
