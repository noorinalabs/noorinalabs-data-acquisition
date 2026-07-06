"""Split over-merged mononym narrator nodes by chain-generation evidence (da#248).

The problem
-----------
The canonical narrator id is a **pure function of the normalized name**
(:func:`src.parse.identity.make_canonical_id`). That is exactly the cross-source
collapse da#99 wants — the *same* person spelled the same way in any number of
sources folds into one node. But it also **over-merges**: a bare single-token
name (a *mononym*) that historically denotes several distinct people collapses
them all into one node. The textbook case is ``سفيان`` "Sufyān" (27,118
mentions), which merges

* **Sufyān al-Thawrī** (d. 161 AH, Kūfa) and
* **Sufyān ibn ʿUyayna** (d. 198 AH, Mecca)

— two men a generation apart. When one node stands in for both, the
``TRANSMITTED_TO`` graph gains back-edges (al-Thawrī's students transmit to
ibn ʿUyayna's teachers *through the shared node*) and forms chronologically
impossible cycles — the artifact ``queries/validation/chain_integrity.cypher``
reports. This is distinct from the *non-name* pollution fixed under da#247/#253:
these are real names, merged too coarsely — a disambiguation-**granularity**
problem.

The fix (precision-first)
-------------------------
:func:`refine_mononym_name` re-resolves a bare registered mononym to a *specific*
person using the death-year evidence of its chain neighbours (teacher at
position-1, student at position+1), so the two Sufyāns land on two distinct
canonical nodes. It returns the disambiguated :class:`MononymPerson` (whose
``norm_name`` feeds the SAME :func:`make_canonical_id` contract — no new id
scheme) or ``None`` to leave the mention on the shared node unchanged.

Two structural precision guards make "no legitimate single-person node is
fragmented" (da#248 acceptance) provable rather than hoped-for:

1. **Only registered names are ever eligible.** :data:`MONONYM_REGISTRY` holds
   *only* mononyms that genuinely denote multiple well-attested people in the
   isnād corpus. A mononym with a dominant single referent — ``شعبة`` (Shuʿba
   ibn al-Ḥajjāj), ``مالك`` (Mālik ibn Anas) — is deliberately **absent**:
   splitting it would fragment one person, the exact regression this guard
   forbids. Every non-registered name (the overwhelming majority, and every
   single-person mononym) returns ``None`` → byte-identical to today.
2. **Unanimity-or-abstain within a registered name.** A registered mononym is
   split *only* when the chain evidence is temporally plausible for **exactly
   one** registered person and implausible for all the others. Absent evidence,
   or evidence consistent with two or more persons, abstains (returns ``None``)
   and leaves the mention merged. The worst case is therefore a *residual* cycle
   (an unsplit ambiguous mention), never a *wrong* split — recall is sacrificed
   for precision by design, per da#248's "do NOT force a low-precision split".

Idempotent: a mention already carrying a disambiguated name (``سفيان الثوري``)
is not a bare registry *key*, so a re-run returns ``None`` and the id is stable.

Scope note (da#248 / da#261 serialization): this module is consumed only by
``src/resolve/disambiguate.py`` at canonical-id assignment. It deliberately does
**not** touch ``src/resolve/ner.py`` (compound-narrator splitting lands there
under da#261) so the two changes do not collide.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.enums import NarratorGeneration
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "MONONYM_REGISTRY",
    "MononymPerson",
    "is_registered_mononym",
    "refine_mononym_name",
]

# Plausible teacher-student death-year gap, mirroring the temporal window used by
# src.resolve.disambiguate (_TEMPORAL_MIN_GAP / _TEMPORAL_MAX_GAP). A neighbour
# whose death year sits 15-80 years from a candidate person's death year is a
# plausible transmission partner for that person.
_MIN_GAP = 15
_MAX_GAP = 80


@dataclass(frozen=True)
class MononymPerson:
    """One historically-distinct person that a bare mononym can denote.

    ``norm_name`` is the *disambiguated* normalized name — it feeds
    :func:`src.parse.identity.make_canonical_id` directly, so the split re-uses the
    one canonical-id contract and simply mints the id of the more-specific name
    (``سفيان الثوري`` rather than the bare ``سفيان``). ``death_year_ah`` and
    ``generation`` are the disambiguating evidence used by the temporal gate; they
    are decision inputs only — this module never writes them onto a record.
    """

    norm_name: str
    name_ar: str
    death_year_ah: int
    generation: NarratorGeneration


def _person(name_ar: str, death_year_ah: int, generation: NarratorGeneration) -> MononymPerson:
    """Build a :class:`MononymPerson`, normalizing ``name_ar`` the way the pipeline
    normalizes every mention/candidate name, so registry ``norm_name`` values and
    pipeline-produced names collapse to the same canonical id."""
    return MononymPerson(
        norm_name=normalize_arabic(name_ar),
        name_ar=name_ar,
        death_year_ah=death_year_ah,
        generation=generation,
    )


def _registry() -> dict[str, tuple[MononymPerson, ...]]:
    """Curated bare-mononym → distinct-persons registry.

    Keys are normalized bare mononyms (via :func:`normalize_arabic`, matching how
    ``disambiguate`` normalizes names). Only names that genuinely denote multiple
    well-attested people with *separable* death dates are included — the temporal
    gate can only discriminate persons whose death years differ by more than the
    plausibility window's slack, so tightly-clustered homonyms (e.g. the three
    Hishāms, d. 146/148/152) are intentionally omitted until a stronger
    (teacher/student-identity) signal is wired, rather than registered where the
    gate would almost always abstain.
    """
    registry: dict[str, tuple[MononymPerson, ...]] = {}

    # سفيان "Sufyān" — the da#248 canonical example. Two men one generation apart;
    # a 37-year death-date separation the temporal gate can resolve in the
    # non-overlapping neighbour-death regions.
    registry[normalize_arabic("سفيان")] = (
        _person("سفيان الثوري", 161, NarratorGeneration.TABA_TABII),
        _person("سفيان بن عيينة", 198, NarratorGeneration.TABA_TABII),
    )

    # يحيى "Yaḥyā" — three well-attested transmitters spanning 143-233 AH, wide
    # enough for the gate to separate in the discriminating regions.
    registry[normalize_arabic("يحيى")] = (
        _person("يحيى بن سعيد الانصاري", 143, NarratorGeneration.TABII),
        _person("يحيى بن سعيد القطان", 198, NarratorGeneration.TABA_TABII),
        _person("يحيى بن معين", 233, NarratorGeneration.ATBA_TABA_TABIIN),
    )

    return registry


# Curated registry, built once at import (see _registry for the inclusion rule).
MONONYM_REGISTRY: dict[str, tuple[MononymPerson, ...]] = _registry()


def is_registered_mononym(norm_name: str) -> bool:
    """``True`` iff ``norm_name`` is a bare mononym eligible for splitting.

    This is precision guard #1 in predicate form: any name not in
    :data:`MONONYM_REGISTRY` — including every single-person mononym — is never a
    split candidate.
    """
    return norm_name in MONONYM_REGISTRY


def _temporally_plausible(person: MononymPerson, adjacent_death_years: list[int]) -> bool:
    """Does any chain neighbour sit a plausible transmission gap from ``person``?"""
    return any(
        _MIN_GAP <= abs(person.death_year_ah - year) <= _MAX_GAP for year in adjacent_death_years
    )


def refine_mononym_name(norm_name: str, adjacent_death_years: list[int]) -> MononymPerson | None:
    """Re-resolve a bare registered mononym to a specific person, or abstain.

    Returns the uniquely-selected :class:`MononymPerson` when ``norm_name`` is a
    registered mononym AND the chain-neighbour death years are temporally
    plausible for **exactly one** registered person; returns ``None`` (leave the
    mention on the shared node) in every other case:

    * ``norm_name`` not registered (precision guard #1 — no single-person node is
      ever eligible), or
    * no neighbour death-year evidence, or
    * the evidence fits zero, or two-or-more, registered persons (precision guard
      #2 — abstain rather than guess).

    ``adjacent_death_years`` are the resolved death years of the mention's
    immediate chain neighbours (positions ±1); the caller collects them from the
    disambiguator's death-year index.
    """
    persons = MONONYM_REGISTRY.get(norm_name)
    if persons is None:
        return None
    if not adjacent_death_years:
        return None

    plausible = [p for p in persons if _temporally_plausible(p, adjacent_death_years)]
    if len(plausible) != 1:
        return None

    selected = plausible[0]
    logger.debug(
        "mononym_split_selected",
        mononym=norm_name,
        resolved_to=selected.norm_name,
        adjacent_death_years=adjacent_death_years,
    )
    return selected
