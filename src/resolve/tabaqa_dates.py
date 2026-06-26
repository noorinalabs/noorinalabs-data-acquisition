"""Estimate a death window from a narrator's ṭabaqa (generation) layer (da#166).

The **last-resort** stage of the narrator-dating pipeline. da#164 parses each
rijāl source's free-text life-dates; da#165 reconciles those per-source datings
into one canonical envelope. After both run, some narrators *still* carry no
death date — the rijāl sources simply never dated them. For those, and **only**
those, this stage derives an *estimated* death-year window from the narrator's
**ṭabaqa class** (the generational layer of the ṭabaqāt genre), tagged with the
:class:`~src.models.enums.DatePrecision.TABAQA_ESTIMATE` precision so a derived
window is never confused with an attested one. (Owner decision #3 on meta
noorinalabs-main#673: these estimated windows are displayed, clearly marked.)

**Which ṭabaqa signal?** The canonical narrator table
(:data:`~src.resolve.schemas.NARRATORS_CANONICAL_SCHEMA`) persists the coarse
``generation`` (:class:`~src.models.enums.NarratorGeneration`) layer, not the
fine-grained Taqrīb ordinal that lives only on the in-memory
:class:`~src.models.narrator.Narrator` model. ``generation`` *is* the ṭabaqa
class at canonical granularity — ``src/parse/itqan.py`` already folds Ibn Ḥajar's
twelve Taqrīb ṭabaqāt into these five buckets — and the design note
(``.claude/research/historical-timeline-dates.md`` §1) names generation/ṭabaqa
layering the "universal fallback that bounds an estimated window." So this stage
keys on the canonical ``generation`` column.

Standing contracts honoured (mirroring da#165):

1. **Fill only genuinely-undated narrators.** A ṭabaqa estimate is a last resort:
   it is written only when the death event is *still* undated after da#165 (point
   + both bounds null **and** precision ``UNKNOWN``). A reconciled/parsed death
   date — including an open ``AFTER``/``BEFORE`` bound — is never overwritten.
2. **No fabrication beyond the ṭabaqa model.** A narrator whose ṭabaqa class is
   null/unknown (``generation`` absent or :attr:`NarratorGeneration.UNKNOWN`)
   stays undated — null bounds + ``UNKNOWN`` precision. Only a *known* layer
   yields an estimate; we never invent a window from nothing.
3. **Death AH→CE via** :func:`~src.utils.hijri.ah_year_to_ce_range` (the
   inclusive-window converter), never the point :func:`~src.utils.hijri.ah_to_ce`;
   the emitted precision is **always** the concrete ``TABAQA_ESTIMATE``, never
   ``DatePrecision(None)``.
4. **Sourced, principled mapping.** A ṭabaqa is a *generation*, so the honest
   representation is a wide multi-decade window per the standard ṭabaqāt
   periodization (see :data:`_GENERATION_DEATH_WINDOW_AH`), not an arbitrary
   point.

Idempotent: a row this stage estimates carries ``TABAQA_ESTIMATE`` precision on
the next run, so it no longer reads as undated and is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision, NarratorGeneration
from src.parse.base import safe_str, write_parquet
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.hijri import ah_year_to_ce_range
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    "TabaqaEstimate",
    "apply_tabaqa_fallback",
    "estimate_death_window",
    "generation_from_value",
]

# Generation (ṭabaqa) → approximate Hijri (AH) death-year window.
#
# A ṭabaqa is a *generation*, so "we only know the layer, not the year" is
# honestly a wide multi-decade death window, not a point. The bounds follow the
# standard ṭabaqāt periodization of the rijāl/ḥadīth sciences — Ibn Ḥajar's
# Taqrīb al-Tahdhīb ordering, whose twelve ṭabaqāt ``src/parse/itqan.py`` already
# folds into these five NarratorGeneration buckets:
#
#   * SAHABI — the Companions died between the Prophet's death (11 AH) and the
#     last surviving Companion, Abū al-Ṭufayl ʿĀmir b. Wāthila (d. ~110 AH).
#   * TABII — the Successors met Companions; the layer dies out across the late
#     1st / 2nd century (~90–180 AH).
#   * TABA_TABII — atbāʿ al-tābiʿīn (Taqrīb ṭab. 5–8), ~150–250 AH; the core
#     isnād-bearing generation of the canonical collections.
#   * ATBA_TABA_TABIIN — the layer after (Taqrīb ṭab. 9–10), ~220–300 AH.
#   * LATER — post-canonical transmitters (Taqrīb ṭab. 11–12), ~280–400 AH.
#
# Windows deliberately OVERLAP at the boundaries: generations are not disjoint in
# real life (a long-lived member of one layer outlives a short-lived member of
# the next), and a too-tight window would assert a precision the ṭabaqa does not
# carry. UNKNOWN is absent on purpose — an unknown ṭabaqa yields no estimate.
_GENERATION_DEATH_WINDOW_AH: dict[NarratorGeneration, tuple[int, int]] = {
    NarratorGeneration.SAHABI: (11, 110),
    NarratorGeneration.TABII: (90, 180),
    NarratorGeneration.TABA_TABII: (150, 250),
    NarratorGeneration.ATBA_TABA_TABIIN: (220, 300),
    NarratorGeneration.LATER: (280, 400),
}


@dataclass(frozen=True)
class TabaqaEstimate:
    """A ṭabaqa-derived death dating for one narrator.

    ``death_year_ah`` is the window midpoint (a "best single value" for the point
    column), with ``earliest``/``latest`` the inclusive AH window and
    ``ce_earliest``/``ce_latest`` its inclusive CE span (via
    :func:`~src.utils.hijri.ah_year_to_ce_range`). ``precision`` is **always**
    :attr:`DatePrecision.TABAQA_ESTIMATE`.
    """

    death_year_ah: int
    death_year_ah_earliest: int
    death_year_ah_latest: int
    death_ce_earliest: int
    death_ce_latest: int
    precision: DatePrecision = DatePrecision.TABAQA_ESTIMATE


def generation_from_value(value: object) -> NarratorGeneration:
    """Normalise a stored ``generation`` value to a concrete NarratorGeneration.

    A missing or out-of-domain value reads as :attr:`NarratorGeneration.UNKNOWN`
    (no known ṭabaqa class) — never ``None`` — so the caller's "known layer only"
    gate is a single membership test.
    """
    text = safe_str(value)
    if text is None:
        return NarratorGeneration.UNKNOWN
    try:
        return NarratorGeneration(text)
    except ValueError:
        return NarratorGeneration.UNKNOWN


def estimate_death_window(generation: NarratorGeneration) -> TabaqaEstimate | None:
    """Estimate a death window from a ṭabaqa (generation) layer, or ``None``.

    Returns ``None`` for an unknown/unmapped layer (e.g.
    :attr:`NarratorGeneration.UNKNOWN`) — the contract that a narrator with no
    known ṭabaqa class gets no fabricated window. Otherwise returns a
    :class:`TabaqaEstimate` whose AH window is the documented generational span
    (:data:`_GENERATION_DEATH_WINDOW_AH`), point = its midpoint, and CE span via
    the inclusive-window :func:`~src.utils.hijri.ah_year_to_ce_range`.
    """
    window = _GENERATION_DEATH_WINDOW_AH.get(generation)
    if window is None:
        return None
    lo, hi = window
    point = (lo + hi) // 2
    ce_earliest = ah_year_to_ce_range(lo)[0]
    ce_latest = ah_year_to_ce_range(hi)[1]
    return TabaqaEstimate(
        death_year_ah=point,
        death_year_ah_earliest=lo,
        death_year_ah_latest=hi,
        death_ce_earliest=ce_earliest,
        death_ce_latest=ce_latest,
    )


def _death_is_undated(rec: dict[str, Any]) -> bool:
    """``True`` when da#164/#165 left this row's *death* event genuinely undated.

    Genuinely undated = no point, no earliest bound, no latest bound, and a
    precision that is null/blank or ``UNKNOWN``. Any reconciled/parsed death
    signal — including a lone ``AFTER`` (earliest set) or ``BEFORE`` (latest set)
    open bound — makes this ``False`` so the fallback never overwrites it.
    """
    if rec.get("death_year_ah") is not None:
        return False
    if rec.get("death_year_ah_earliest") is not None:
        return False
    if rec.get("death_year_ah_latest") is not None:
        return False
    precision = rec.get("death_date_precision")
    return precision in (None, "", DatePrecision.UNKNOWN.value)


def apply_tabaqa_fallback(output_dir: Path) -> Path | None:
    """Fill ṭabaqa-estimated death windows on ``narrators_canonical.parquet`` (da#166).

    Runs AFTER da#165 reconciliation: for every canonical row whose death event is
    still undated (:func:`_death_is_undated`) and that carries a *known* ṭabaqa
    class (``generation``), writes the four death columns
    (``death_year_ah`` + ``_earliest``/``_latest`` + ``death_date_precision`` =
    ``tabaqa_estimate``) from :func:`estimate_death_window`. Birth columns and any
    already-dated death are left untouched.

    Idempotent: an estimated row reads as ``tabaqa_estimate`` (not undated) on a
    re-run, so re-running over the same canonical table is a no-op.

    Returns the canonical path, or ``None`` when it does not yet exist (the stage
    is a no-op until the prior resolve stages have produced the table).
    """
    canonical_path = output_dir / "narrators_canonical.parquet"
    if not canonical_path.exists():
        logger.warning("tabaqa_fallback_no_canonical", path=str(canonical_path))
        return None

    rows = pq.read_table(canonical_path).to_pylist()
    estimated = 0
    skipped_no_tabaqa = 0
    for rec in rows:
        if not _death_is_undated(rec):
            continue
        estimate = estimate_death_window(generation_from_value(rec.get("generation")))
        if estimate is None:
            skipped_no_tabaqa += 1
            continue
        rec["death_year_ah"] = estimate.death_year_ah
        rec["death_year_ah_earliest"] = estimate.death_year_ah_earliest
        rec["death_year_ah_latest"] = estimate.death_year_ah_latest
        rec["death_date_precision"] = estimate.precision.value
        estimated += 1

    arrays = {f.name: [r.get(f.name) for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    out_table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_parquet(out_table, canonical_path, schema=NARRATORS_CANONICAL_SCHEMA)

    logger.info(
        "tabaqa_fallback_complete",
        canonical_total=len(rows),
        estimated=estimated,
        undated_without_tabaqa=skipped_no_tabaqa,
    )
    return canonical_path
