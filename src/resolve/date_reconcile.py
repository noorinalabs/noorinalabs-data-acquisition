"""Reconcile per-source narrator life-dates into one canonical envelope (da#165).

da#164 parses each rijāl source's free-text birth/death notation into structured
``earliest/latest`` AH bounds + a concrete :class:`~src.models.enums.DatePrecision`
and writes them onto that source's ``narrators_bio_*`` row. When several sources
biograph the *same* narrator — they collapse onto one ``nar:`` canonical id via
:func:`~src.parse.identity.make_canonical_id` — those per-source datings must be
folded into a **single** canonical set of birth/death bounds + precision. This is
the reconciliation stage between the per-source parse (da#164, merged) and the
ṭabaqa-derived fallback (da#166, next): we reconcile what the sources *state* and
leave the rest for the fallback to fill.

Reconciliation rules (documented so the behaviour is auditable, not magic):

* **Agreement.** Sources that agree (point estimates within
  :data:`AGREEMENT_TOLERANCE` AH years) collapse to the agreed value: two sources
  both "256 AH" → ``EXACT 256``. Classical *wafāt* attestations routinely differ
  by ±1–2 years; that is agreement, not conflict.

* **Conflict → honest tight range.** When point estimates genuinely disagree
  (e.g. source A "died 256 AH", source B "255 AH"), we do **not** invent a winner
  or a midpoint — we keep the honest envelope ``RANGE [255, 256]`` spanning the
  attested values. The true year lies within; a tight range states exactly what
  is known. (We deliberately avoid an authority-ranking tiebreak: the corpus has
  no reliable per-source authority signal at this stage, and a spurious "winner"
  would be less honest than the envelope. Authority-weighting is left as a future
  refinement — see the da#165 PR TechDebt line.)

* **Noise robustness.** A single source's bound can be widened by a stray
  plausible integer leaking into a free-text date field. Reconciliation anchors
  on the **agreement among point estimates** and widens the envelope only as far
  as a *supported* bound reaches: when a **competing** point cluster exists
  (≥2 distinct in-band points), a per-source bound more than :data:`NOISE_GAP` AH
  years beyond the **fixed** consensus band is treated as noise and is **not**
  allowed to expand the envelope. The gate is measured against the fixed
  consensus band (not the running envelope), so it is independent of the order
  the observations are processed in. A lone source's *own* coherent span (no
  competing cluster) is preserved in full — it is never clipped against its own
  point. With ≥3 sources, a lone point more than :data:`OUTLIER_GAP` years (≈ a
  generation) from the median is down-weighted out of the consensus entirely
  ("prefer agreement, down-weight a lone outlier").

* **circa + multi-alternative.** da#164's parser collapses a ``تقريبا/circa``
  marker co-occurring with a multi-alternative (``أو/وقيل``) field to a single
  ``CIRCA`` point, dropping the 2nd candidate (flagged by Nikolaos). The honest
  behaviour is to **keep the envelope**: :func:`parse_source_notation` re-derives
  such a field as ``CIRCA`` spanning ``[min, max]`` of the candidates, and the
  cross-source reconciler likewise preserves a circa band rather than over-
  tightening to a point.

* **sanadset scalar point-years.** A bare death year that arrives with ``unknown``
  precision and null bounds (a sanadset scalar pending reconcile) carries a known
  value — reconciliation folds it in as a point estimate and **upgrades** the
  precision to a concrete ``EXACT`` (the value *is* known; "unknown" was a
  placeholder for "not yet reconciled", not "no information").

Two standing contracts are honoured throughout:

1. **Death AH→CE** uses the inclusive-window :func:`~src.utils.hijri.ah_year_to_ce_range`
   (see :func:`ce_bounds`), never the point :func:`~src.utils.hijri.ah_to_ce`.
2. **Null precision reads as ``UNKNOWN``**, never ``DatePrecision(None)``: every
   reconciled output carries a concrete :class:`~src.models.enums.DatePrecision`,
   and an absent/garbled source precision string is normalised to ``UNKNOWN`` at
   the input boundary (:func:`observation_from_row`).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision
from src.parse.base import safe_str
from src.parse.identity import make_canonical_id
from src.parse.narrator_dates import (
    MAX_PLAUSIBLE_AH,
    MIN_PLAUSIBLE_AH,
    ParsedDate,
    parse_year_notation,
)
from src.resolve._run_record import write_canonical
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.resolve.tabaqa_dates import generation_from_value, is_plausible_narrator_death_ah
from src.utils.arabic import normalize_arabic
from src.utils.hijri import ah_year_to_ce_range
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    "AGREEMENT_TOLERANCE",
    "NOISE_GAP",
    "OUTLIER_GAP",
    "DateObservation",
    "ReconciledDate",
    "ce_bounds",
    "observation_from_row",
    "parse_source_notation",
    "reconcile_canonical_dates",
    "reconcile_event",
    "reconcile_narrator",
]

# Point estimates within this many AH years are "in agreement" — the routine
# ±1–2 year disagreement between classical wafāt attestations is not a conflict.
AGREEMENT_TOLERANCE = 2
# A per-source bound more than this many AH years beyond the consensus point
# cluster is treated as noise and is NOT allowed to widen the reconciled envelope.
NOISE_GAP = 10
# With >=3 sources, a point more than this many AH years (~ a generation) from the
# median — while a tighter consensus of >=2 exists — is a lone outlier, dropped.
OUTLIER_GAP = 25

# Multi-alternative markers ("or", "and it is said") that, combined with a circa
# marker, indicate the parser collapsed an envelope to a point (Nikolaos's case).
# Deliberately word-form markers only — a bare "/" or Arabic comma is NOT treated
# as an alternative separator, so a circa field with an AH/CE slash
# ("circa 180هـ/795م") does not spuriously widen to a [180, 795] band (Kavitha,
# da#165 review).
_ALT_MARKERS = ("أو", "او", "وقيل", "وقئل", " or ")

# Closed (two-sided / point) precisions: they assert a central estimate. Both
# derived-window kinds (``TABAQA_ESTIMATE`` and ``ISNAD_ESTIMATE``, da#340) belong
# here for the same reason — each carries a concrete point + bounds — so this set
# stays exhaustive over the "asserts a central estimate" precisions.
_CLOSED_PRECISIONS = frozenset(
    {
        DatePrecision.EXACT,
        DatePrecision.RANGE,
        DatePrecision.CIRCA,
        DatePrecision.TABAQA_ESTIMATE,
        DatePrecision.ISNAD_ESTIMATE,
    }
)

# Arabic-Indic / Persian digit → ASCII (mirrors src.parse.narrator_dates so this
# module can re-extract candidate years without importing private internals).
_DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_YEAR_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class DateObservation:
    """One source's parsed dating of a single life-event (birth or death).

    Mirrors the four ``{prefix}_year_ah*`` / ``{prefix}_date_precision`` columns
    da#164 writes onto a ``narrators_bio_*`` row. ``precision`` is the
    :class:`DatePrecision` the source carried; ``UNKNOWN`` denotes a source silent
    on the event (or a sanadset scalar whose precision is pending reconcile).
    """

    point: int | None
    earliest: int | None
    latest: int | None
    precision: DatePrecision


@dataclass(frozen=True)
class ReconciledDate:
    """The single canonical dating produced from a narrator's per-source set.

    ``precision`` is **always** concrete (never ``None``): ``UNKNOWN`` when no
    source stated the event. ``source_count`` is how many usable observations
    contributed; ``conflict`` is ``True`` when the closed point estimates spanned
    more than :data:`AGREEMENT_TOLERANCE` AH years (surfaced for logging/QA).
    """

    point: int | None
    earliest: int | None
    latest: int | None
    precision: DatePrecision
    source_count: int = 0
    conflict: bool = False


@dataclass(frozen=True)
class _Eff:
    """Normalised internal view of one observation for the reconciliation math."""

    kind: str  # "closed" | "after" | "before"
    point: int | None
    lo: int | None
    hi: int | None
    precision: DatePrecision


def _plausible(year: int) -> bool:
    return MIN_PLAUSIBLE_AH <= year <= MAX_PLAUSIBLE_AH


def _plausible_years(text: str) -> list[int]:
    """All plausible AH years in ``text`` in order of appearance (ASCII-folded)."""
    folded = text.translate(_DIGIT_TRANS)
    return [y for tok in _YEAR_RE.findall(folded) if _plausible(y := int(tok))]


def _has_alt_marker(text: str) -> bool:
    return any(marker in text for marker in _ALT_MARKERS)


def parse_source_notation(value: object) -> ParsedDate | None:
    """Parse a source's raw date field, preserving a circa+multi-alternative envelope.

    Delegates to da#164's :func:`~src.parse.narrator_dates.parse_year_notation`,
    then corrects the one collapse case Nikolaos flagged: when an approximation
    marker (``تقريبا/circa``) co-occurs with a multi-alternative field
    (``أو/وقيل``), the base parser returns a single ``CIRCA`` *point*, silently
    dropping the competing candidate. Here we keep the honest envelope —
    ``CIRCA`` spanning ``[min, max]`` of the plausible candidates, first-stated
    year as the point — so reconciliation never loses an attested alternative.

    Returns ``None`` for a field with no usable year, exactly as the base parser.
    """
    parsed = parse_year_notation(value)
    if parsed is None or parsed.precision is not DatePrecision.CIRCA:
        return parsed
    text = str(value)
    years = _plausible_years(text)
    if len(set(years)) >= 2 and _has_alt_marker(text):
        return ParsedDate(
            point=years[0],
            earliest=min(years),
            latest=max(years),
            precision=DatePrecision.CIRCA,
        )
    return parsed


def _precision_from_value(value: object) -> DatePrecision:
    """Normalise a stored precision string to a concrete :class:`DatePrecision`.

    A missing or out-of-domain value reads as ``UNKNOWN`` — never
    ``DatePrecision(None)`` — honouring the null-precision contract at the input
    boundary.
    """
    text = safe_str(value)
    if text is None:
        return DatePrecision.UNKNOWN
    try:
        return DatePrecision(text)
    except ValueError:
        return DatePrecision.UNKNOWN


def observation_from_row(row: dict[str, Any], prefix: str) -> DateObservation:
    """Build a :class:`DateObservation` from a ``narrators_bio_*`` row.

    ``prefix`` is ``"birth"`` or ``"death"``. Reads the four da#164 columns; a
    null/garbled precision string is normalised to ``UNKNOWN``.
    """
    return DateObservation(
        point=row.get(f"{prefix}_year_ah"),
        earliest=row.get(f"{prefix}_year_ah_earliest"),
        latest=row.get(f"{prefix}_year_ah_latest"),
        precision=_precision_from_value(row.get(f"{prefix}_date_precision")),
    )


def _effective(o: DateObservation) -> _Eff | None:
    """Normalise an observation to a closed/after/before band, or ``None`` if empty.

    A sanadset scalar — ``UNKNOWN`` precision but a concrete point — is folded in
    as a closed ``EXACT`` point (the value is known; the precision was a "pending
    reconcile" placeholder).
    """
    if o.precision in _CLOSED_PRECISIONS:
        lo = o.earliest if o.earliest is not None else o.point
        hi = o.latest if o.latest is not None else o.point
        point = o.point if o.point is not None else (lo if lo is not None else hi)
        if point is None and lo is None and hi is None:
            return None
        return _Eff("closed", point, lo, hi, o.precision)
    if o.precision is DatePrecision.AFTER:
        x = o.earliest if o.earliest is not None else o.point
        return None if x is None else _Eff("after", x, x, None, DatePrecision.AFTER)
    if o.precision is DatePrecision.BEFORE:
        x = o.latest if o.latest is not None else o.point
        return None if x is None else _Eff("before", x, None, x, DatePrecision.BEFORE)
    # UNKNOWN (or any other) carrying a usable numeric → known point (sanadset).
    if o.point is not None:
        return _Eff("closed", o.point, o.point, o.point, DatePrecision.EXACT)
    if o.earliest is not None or o.latest is not None:
        lo, hi = o.earliest, o.latest
        point = lo if lo is not None else hi
        return _Eff("closed", point, lo, hi, DatePrecision.RANGE)
    return None


def _consensus_band(points: Sequence[int]) -> tuple[int, int]:
    """The agreed (lo, hi) point band, dropping a lone outlier when >=3 sources."""
    pts = sorted(points)
    if len(pts) <= 2:
        return pts[0], pts[-1]
    median = statistics.median(pts)
    core = [p for p in pts if abs(p - median) <= OUTLIER_GAP]
    if len(core) < 2:
        core = pts
    return min(core), max(core)


def _closed_precision(closed: Iterable[_Eff], lo: int, hi: int) -> DatePrecision:
    """Derive the reconciled precision for a closed estimate."""
    kinds = {e.precision for e in closed}
    if lo == hi:
        # A single agreed value: EXACT, unless every contributing source was circa.
        return DatePrecision.CIRCA if kinds == {DatePrecision.CIRCA} else DatePrecision.EXACT
    # A genuine band: keep CIRCA when every source was approximate, else RANGE.
    return DatePrecision.CIRCA if kinds == {DatePrecision.CIRCA} else DatePrecision.RANGE


def _representative_point(closed: Sequence[_Eff], lo: int, hi: int) -> int:
    """Best single year for the reconciled estimate: the mode within the band."""
    pts = [e.point for e in closed if e.point is not None and lo <= e.point <= hi]
    if not pts:
        return (lo + hi) // 2
    counts = Counter(pts)
    # Most-supported point; tie-break on the smaller (earlier) year.
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _reconcile_open(afters: Sequence[_Eff], befores: Sequence[_Eff], n: int) -> ReconciledDate:
    """Reconcile a narrator dated only by open bounds (``after`` / ``before``)."""
    lo = max((e.lo for e in afters if e.lo is not None), default=None)
    hi = min((e.hi for e in befores if e.hi is not None), default=None)
    if lo is not None and hi is not None:
        if lo <= hi:
            # A bracketed "after X .. before Y" is a legitimate range, not a
            # disagreement — reserve ``conflict`` for the genuinely contradictory
            # ``after > before`` case below (Jean-Claude, da#165 review).
            precision = DatePrecision.EXACT if lo == hi else DatePrecision.RANGE
            return ReconciledDate(lo, lo, hi, precision, n)
        # "after X" later than "before Y" → contradictory; keep the lower bound.
        return ReconciledDate(lo, lo, None, DatePrecision.AFTER, n, conflict=True)
    if lo is not None:
        return ReconciledDate(lo, lo, None, DatePrecision.AFTER, n)
    if hi is not None:
        return ReconciledDate(hi, None, hi, DatePrecision.BEFORE, n)
    return ReconciledDate(None, None, None, DatePrecision.UNKNOWN, n)


def reconcile_event(observations: Iterable[DateObservation]) -> ReconciledDate:
    """Reconcile one life-event's per-source observations into a single dating.

    Returns a :class:`ReconciledDate` with an always-concrete precision
    (``UNKNOWN`` when no source stated the event). See the module docstring for
    the agreement / conflict / noise / circa / sanadset rules applied.
    """
    effs = [e for o in observations if (e := _effective(o)) is not None]
    if not effs:
        return ReconciledDate(None, None, None, DatePrecision.UNKNOWN, 0)

    closed = [e for e in effs if e.kind == "closed"]
    afters = [e for e in effs if e.kind == "after"]
    befores = [e for e in effs if e.kind == "before"]

    if not closed:
        return _reconcile_open(afters, befores, len(effs))

    points = [e.point for e in closed if e.point is not None]
    if points:
        core_lo, core_hi = _consensus_band(points)
    else:  # closed-by-bounds only (no points) — span the available bounds.
        los = [e.lo for e in closed if e.lo is not None]
        his = [e.hi for e in closed if e.hi is not None]
        core_lo = min(los) if los else min(his)
        core_hi = max(his) if his else max(los)

    lo, hi = core_lo, core_hi
    in_band = [
        e
        for e in closed
        if e.point is None
        or core_lo - AGREEMENT_TOLERANCE <= e.point <= core_hi + AGREEMENT_TOLERANCE
    ]
    # Widen the envelope by SUPPORTED bounds. Two rules keep this honest and
    # order-independent (da#165 review):
    #   * The NOISE_GAP gate is anchored on the FIXED consensus band
    #     (``core_lo``/``core_hi``), never the running ``lo``/``hi`` — so a chain
    #     of admitted bounds cannot walk the envelope arbitrarily far past
    #     consensus, and the result does not depend on observation iteration
    #     order (Jean-Claude).
    #   * A bound is clipped only when a COMPETING point cluster contradicts it.
    #     With a single distinct in-band point there is no competition, so a
    #     source's own coherent self-band (e.g. da#164's "between 200 and 230"
    #     RANGE) is preserved in full rather than collapsed against its own point
    #     into a spurious EXACT (Kavitha).
    competing = len({e.point for e in in_band if e.point is not None}) > 1
    for e in in_band:
        if e.lo is not None and e.lo < lo and (not competing or (core_lo - e.lo) <= NOISE_GAP):
            lo = e.lo
        if e.hi is not None and e.hi > hi and (not competing or (e.hi - core_hi) <= NOISE_GAP):
            hi = e.hi
    # Fold consistent open bounds — tighten only, never fabricate beyond the band.
    for e in afters:
        if e.lo is not None and lo <= e.lo <= hi:
            lo = max(lo, e.lo)
    for e in befores:
        if e.hi is not None and lo <= e.hi <= hi:
            hi = min(hi, e.hi)

    conflict = bool(points) and (max(points) - min(points) > AGREEMENT_TOLERANCE)
    precision = _closed_precision(in_band, lo, hi)
    point = _representative_point(in_band, lo, hi)
    return ReconciledDate(point, lo, hi, precision, len(effs), conflict)


def reconcile_narrator(
    birth: Iterable[DateObservation],
    death: Iterable[DateObservation],
) -> dict[str, Any]:
    """Reconcile a narrator's birth + death observation sets into canonical columns.

    Returns the eight ``narrators_canonical`` date columns (point + 3 bounds for
    each event), with concrete precision strings.
    """
    b = reconcile_event(birth)
    d = reconcile_event(death)
    return {
        "birth_year_ah": b.point,
        "birth_year_ah_earliest": b.earliest,
        "birth_year_ah_latest": b.latest,
        "birth_date_precision": b.precision.value,
        "death_year_ah": d.point,
        "death_year_ah_earliest": d.earliest,
        "death_year_ah_latest": d.latest,
        "death_date_precision": d.precision.value,
    }


def ce_bounds(rec: ReconciledDate) -> tuple[int | None, int | None]:
    """Inclusive CE ``(earliest, latest)`` for a reconciled dating.

    Uses :func:`~src.utils.hijri.ah_year_to_ce_range` — the inclusive-window
    converter — for **both** ends, never the point :func:`~src.utils.hijri.ah_to_ce`:
    the window runs from the *start* of the earliest AH year to the *end* of the
    latest. An open bound yields ``None`` on that side.
    """
    ce_earliest = ah_year_to_ce_range(rec.earliest)[0] if rec.earliest is not None else None
    ce_latest = ah_year_to_ce_range(rec.latest)[1] if rec.latest is not None else None
    return ce_earliest, ce_latest


def _bio_canonical_id(row: dict[str, Any]) -> str | None:
    """The ``nar:`` canonical id a bio row keys onto (mirrors bio_promote)."""
    name_ar = safe_str(row.get("name_ar"))
    norm = safe_str(row.get("name_ar_normalized")) or (
        normalize_arabic(name_ar) if name_ar else None
    )
    return make_canonical_id(norm) if norm else None


def reconcile_canonical_dates(
    staging_dir: Path,
    output_dir: Path,
    *,
    sources: set[str] | None = None,
) -> Path | None:
    """Fold per-source bio dates into ``narrators_canonical.parquet`` (da#165 stage).

    Groups every ``narrators_bio_*`` row by the same ``nar:`` canonical id
    bio_promote uses, reconciles each narrator's birth + death observations, and
    writes the eight reconciled date columns onto the matching canonical record.
    Every canonical row — including mention-only narrators with no bio — is left
    with a **concrete** precision (``UNKNOWN`` when nothing was stated), enforcing
    the null-precision contract on the emitted table.

    Idempotent: reconciliation is a pure function of the bio inputs, so re-running
    over the same staging produces the same canonical dates.

    Returns the canonical path, or ``None`` when it does not yet exist (the stage
    is a no-op until disambiguate/bio_promote have produced the table).
    """
    # OPTIONAL input (da#361): unlike the raising sites, ``narrators_canonical.parquet``
    # is a legitimately optional input HERE. This stage is a pure date-enricher: with
    # no canonical table there are simply no narrators to date, so an absent table is
    # an honest no-op, not an upstream defect. Confirmed as intended (da#361 carved
    # date_reconcile + tabaqa_dates out of the fail-loud set): it does NOT call
    # ``require_input`` and returns ``None`` by design. See module docstring.
    canonical_path = output_dir / "narrators_canonical.parquet"
    if not canonical_path.exists():
        logger.warning("reconcile_dates_no_canonical", path=str(canonical_path))
        return None

    # Accumulate per-source observations grouped by canonical id.
    obs_by_cid: dict[str, dict[str, list[DateObservation]]] = {}
    for bf in sorted(staging_dir.glob("narrators_bio_*.parquet")):
        for row in pq.read_table(bf).to_pylist():
            if sources is not None and safe_str(row.get("source")) not in sources:
                continue
            cid = _bio_canonical_id(row)
            if cid is None:
                continue
            slot = obs_by_cid.setdefault(cid, {"birth": [], "death": []})
            slot["birth"].append(observation_from_row(row, "birth"))
            slot["death"].append(observation_from_row(row, "death"))

    rows = pq.read_table(canonical_path).to_pylist()
    reconciled = 0
    for rec in rows:
        cid = rec.get("canonical_id")
        obs_slot = obs_by_cid.get(cid) if cid else None
        if obs_slot is not None:
            merged = reconcile_narrator(obs_slot["birth"], obs_slot["death"])
            # da#454: a reconciled DEATH point outside the isnad-narrator plausibility
            # envelope (the d. ~700-780 AH late-collector/commentator pollution welded
            # onto an early node — see tabaqa_dates.is_plausible_narrator_death_ah) must
            # never be written onto the canonical record. Reconciliation folds per-
            # source OBSERVATIONS (agreement/conflict/noise-gap) but has no concept of
            # "impossible for THIS narrator's generation" — a single uncontested corrupt
            # source sails straight through as a "conflict-free" EXACT point. Scrub the
            # WHOLE death dating (point + both bounds + precision), not just the point:
            # leaving corrupt bounds/EXACT precision behind a null point would read as
            # "dated" to tabaqa_dates' undated-check and skip the ṭabaqa fallback that
            # runs next, stranding the narrator on a half-scrubbed, still-wrong dating.
            death_point = merged.get("death_year_ah")
            # Death columns the da#454 scrub cleared with EXPLICIT intent, so the
            # back-fill-protection loop below honours the clear instead of swallowing
            # it (empty on the normal path — populated only when the scrub fires).
            scrub_cleared: set[str] = set()
            if death_point is not None and not is_plausible_narrator_death_ah(
                death_point, generation_from_value(rec.get("generation"))
            ):
                merged["death_year_ah"] = None
                merged["death_year_ah_earliest"] = None
                merged["death_year_ah_latest"] = None
                merged["death_date_precision"] = DatePrecision.UNKNOWN.value
                scrub_cleared = {
                    "death_year_ah",
                    "death_year_ah_earliest",
                    "death_year_ah_latest",
                }
            for key, value in merged.items():
                # Never null out a point a prior producer back-filled (defensive —
                # a contributing bio would itself have surfaced that point here) —
                # UNLESS the da#454 scrub above cleared it with explicit intent. When
                # the canonical record ALREADY holds a death point (a persisted
                # pre-fix corrupt point on an idempotent `resolve --from-step
                # reconcile` re-run, or a bio_promote-stamped point reconciliation
                # recomputes as implausible from the fuller unscrubbed bio set), the
                # guard would otherwise keep that point while its bounds + precision
                # are nulled — the self-inconsistent half-scrub this scrub exists to
                # prevent, which then reads as "dated" to tabaqa_dates' undated-check
                # and skips the ṭabaqa fallback (da#454).
                if (
                    key.endswith("_year_ah")
                    and value is None
                    and rec.get(key) is not None
                    and key not in scrub_cleared
                ):
                    continue
                rec[key] = value
            reconciled += 1
        # Enforce concrete precision on EVERY row (mention-only rows included).
        for prec_col in ("birth_date_precision", "death_date_precision"):
            if not rec.get(prec_col):
                rec[prec_col] = DatePrecision.UNKNOWN.value

    arrays = {f.name: [r.get(f.name) for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    out_table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_canonical(out_table, canonical_path, stage="reconcile")

    logger.info(
        "reconcile_dates_complete",
        canonical_total=len(rows),
        reconciled=reconciled,
        bio_narrators=len(obs_by_cid),
    )
    return canonical_path
