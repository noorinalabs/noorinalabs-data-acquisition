"""Split over-collapsed *generic-name* narrator nodes by isnad-adjacency dates (da#337).

The problem (the mirror of :mod:`src.resolve.fuzzy_cluster`)
------------------------------------------------------------
The canonical narrator id is a **pure function of the normalized name**
(:func:`src.parse.identity.make_canonical_id`): every producer that sees the same
normalized name converges on one ``nar:`` node. That cross-source collapse is what
da#99 wants, but for a *generic* name — a bare kunya (``أبو عبد الله``), a thin
two-token fragment — it **over-merges** many historically-distinct people onto one
node, which then reports a spuriously inflated betweenness centrality (a fake hub).

``disambiguate`` only ever *merges* an exact-name group; it never splits. This stage
is the missing *split* pass. It is the structural sibling of ``fuzzy_cluster`` (the
cross-source *merge* pass): it reuses that module's IO shell — read
``narrators_canonical.parquet`` + ``narrator_mentions_resolved.parquet``, rewrite the
canonical table, remap the mentions so the graph edges follow — but runs it in the
opposite direction, peeling one over-merged node into several.

Candidacy vs. decision (the load-bearing separation)
----------------------------------------------------
Candidacy is :func:`src.resolve.generic_name.is_generic_name` — a **recall-first
screen** (da#337 PR-1) that deliberately admits genuinely-single people like
``سفيان الثوري`` and ``الزهري``. **The split decision MUST NEVER be made on that
screen alone.** It is gated on independent multi-referent evidence: ≥2 well-separated,
well-supported **death-year bands** derived from the *attested* death years of each
mention's chain neighbours. A name that resolves to a single band (Sufyān al-Thawrī
d.161; al-Zuhrī d.124) yields one cluster and this stage **abstains** — the exact
over-split guard Ivana required on PR#338. Precision is bought by that evidence gate,
not by the name shape.

Algorithm (per eligible canonical id ``C``, name ``N``)
-------------------------------------------------------
1. Gather every mention with ``canonical_narrator_id == C``.
2. **Isnad-adjacency evidence** — for each such mention, look at the chain neighbours
   at ``position_in_chain`` ±1 in the same ``(hadith_id, chain_index)`` chain (da#411 —
   the composite key the graph loaders use; NOT ``hadith_id`` alone, which would
   fabricate cross-chain neighbours in a multi-isnad hadith) and map them to their
   *attested* ``death_year_ah`` (precision ≠ ``tabaqa_estimate`` — an *estimated*
   window is never used as evidence, which also keeps the pass cascade-free on
   re-run). A teacher (neighbour at position−1) dies a transmission gap *earlier*;
   a student (position+1) a gap *later* — so ``C``'s per-mention death estimate is
   ``neighbour_death ± MID_GAP`` (``MID_GAP`` the midpoint of
   :data:`~src.resolve.mononym_split._MIN_GAP`/``_MAX_GAP``), averaged over the
   mention's attested neighbours. A mention with no attested neighbour is *undatable*.
3. **Cluster** the per-mention estimates 1-D, gap-based: sort and cut wherever a
   consecutive gap exceeds :data:`SPLIT_BAND_GAP` (~one generation).
4. **ABSTAIN — leave ``C`` as ONE node — UNLESS ALL hold** (the over-split guard):
   * ≥2 clusters each with ≥ :data:`SPLIT_MIN_SUPPORT` mentions (*qualifying*);
   * ≤ :data:`SPLIT_MAX_CLUSTERS` qualifying clusters (more ⇒ the name is a generic
     bucket we cannot cleanly resolve → abstain, logged for audit);
   * adjacent qualifying-cluster midpoints separated by > :data:`SPLIT_MIN_SEPARATION`.
5. **Peel-not-partition** — undatable mentions, and any mention in a below-support
   band, STAY on the primary node. Only confidently-dated distinct clusters peel off.
6. **Id assignment** (deterministic — a pure function of the input, reproducible
   across re-runs, no running ordinals): the **largest** qualifying cluster plus the
   undatable/below-support remainder keep ``C = make_canonical_id(N)`` (empty
   discriminator ⇒ id unchanged). Each *other* qualifying cluster gets
   ``make_discriminated_canonical_id(N, discriminator)`` where the discriminator is
   the death-band bucket label (a :data:`~src.resolve.tabaqa_dates._GENERATION_DEATH_WINDOW_AH`
   window the midpoint falls in, e.g. ``d:150-250``, else a fixed 25-year bucket
   ``d:<floor(mid/25)*25>``). Two peeled clusters that collapse to the same label are
   tie-broken by appending the bare normalized **name** (not id — cascade-free) of the
   band's lexicographically-smallest attested anchor neighbour. This same-label tie-break
   IS reachable, not dead code (da#337 PR-3a review): two peeled bands are ≥
   :data:`SPLIT_BAND_GAP` (80 AH) apart by construction, which rules out a shared 25-year
   bucket, but a *generation-window* label whose unique-match sub-range is wider than that
   gap can still be hit by two distinct bands — concretely the ``LATER`` window (280-400),
   whose uniquely-matching sub-range 301-400 is 99 AH wide, so two bands e.g. d.310/d.395
   both label ``d:280-400`` and the anchor tie-break is what keeps their ids distinct.
7. Names ``mononym_split`` already resolves (:func:`is_registered_mononym`) are
   skipped — that registered-mononym splitter owns them; we do not fight it.

Outputs
-------
* ``narrators_canonical.parquet`` rewritten with the peeled rows appended (each new
  row carries its band's estimated death window; the primary's ``mention_count`` is
  reduced by the peeled total).
* ``narrator_mentions_resolved.parquet`` — the peeled mentions' ``canonical_narrator_id``
  remapped (streaming/idempotent/bounded-memory, mirroring
  ``fuzzy_cluster._remap_mention_canonical_ids`` but keyed on the per-mention
  ``mention_id`` because a split sends *different* mentions of one source id to
  *different* targets, which a global id→id remap cannot express).
* ``narrator_splits.parquet`` — a one-row-per-peel **audit report** (bare name, new
  id, discriminator, band window, mention_count, exemplar anchor neighbours): the
  owner's review artifact before any prod re-load (PR-3).

Estimated-window provenance
---------------------------
A peeled row's death window is an *isnad-adjacency estimate*, not attested and not a
ṭabaqa estimate. It carries the first-class
:attr:`~src.models.enums.DatePrecision.ISNAD_ESTIMATE` precision (da#340) — its own
"derived from isnad adjacency, not attested" bucket, no longer overloading the
semantically-wrong ``TABAQA_ESTIMATE``. This precision is deliberate and load-bearing:
it (a) marks the death as an *estimate* for display, (b) keeps the ``tabaqa_dates``
fallback from re-touching an already-estimated row (``tabaqa_dates`` only touches
undated rows), and (c) **excludes peeled nodes from the attested-neighbour evidence
pool** — the evidence gate treats *any* non-attested precision (``TABAQA_ESTIMATE`` OR
``ISNAD_ESTIMATE``, see :data:`_ESTIMATED_PRECISIONS`) as unusable — so re-running the
stage is a strict no-op (each peeled node is a single band ⇒ abstains). See the
idempotence note below.

Idempotence
-----------
A second full run reads the already-split table: the primary retains exactly one
qualifying band and every peeled node is a single band, so all candidates abstain and
no file is rewritten. Because peeled ``ISNAD_ESTIMATE`` rows (like ṭabaqa estimates)
are excluded from the attested pool, a re-run's evidence for every candidate is a
subset of the first run's — splitting only ever gets *harder* — so no candidate newly
splits. The stage returns ``None`` (no-op).

Second axis — isnad-neighbour *context* discriminator (da#337/#346 PR-B)
-----------------------------------------------------------------------
The date axis above **abstains** whenever dates cannot carve referents apart even though
multi-referent evidence clearly exists — Gate 2 (*too many* well-supported bands ⇒
"generic bucket") and Gate 3 (bands too *close* ⇒ contemporaries). PR-B adds an
orthogonal discriminator that runs *only* on those two abstains (never on Gate 1, the
genuinely-single al-Zuhrī guard): it separates the mentions by **who they keep company
with** — the identities of their specific teacher/student neighbours — instead of by
date. Two people who share a generic name but ran in different scholarly circles surface
as two dense, mutually-disjoint neighbour communities; one real hub does not.

* :func:`_collect_context` — per mention, the set of *specific* (discriminating) neighbour
  names, on the loader's **consecutive-resolved** adjacency (:func:`_chain_neighbours`,
  #439), not the date axis's exact ``position ±1``.
* :func:`_cluster_by_context` — union-find on ≥ :data:`CTX_MIN_SHARED` shared specific
  neighbours (Layer 1), then a Jaccard (:data:`CTX_MAX_JACCARD`) agglomeration guard
  folding artificially-fragmented referents back together (Layer 2).
* :func:`plan_split_by_context` — peels every non-largest qualifying community onto a
  ``"ctx:"``-discriminated id (reusing :class:`PeeledBand` / :class:`SplitPlan` / the audit
  schema unchanged); abstains outside ``[2, SPLIT_MAX_PEEL]`` communities. The
  ``SPLIT_MAX_PEEL`` ceiling is a HARD abstain, not a truncating top-K, so a re-run stays a
  strict no-op. The peeled row still carries a death window (context mentions are datable,
  §1) and ``ISNAD_ESTIMATE`` precision exactly as the date axis does.

Both axes produce the same :class:`SplitPlan` shape, so everything downstream (append
peeled rows, ``mention_id``-keyed remap, audit write, the PURE-SPLIT conservation/no-merge
asserts) is shared and unchanged.

Accepted v1 limitation (documented, NOT fixed here per da#337 scope)
-------------------------------------------------------------------
The chain-edge loaders (``load_edges._build_chain_pairs`` / ``_load_narrated``) key on
the mention ``canonical_narrator_id`` and so follow this remap correctly. But
``load_edges._studied_under_endpoint`` and ``muhaddithat_links`` mint narrator ids
from the **name only**, so a STUDIED_UNDER / muhaddithat edge naming a split narrator
attaches to the **primary** (bare-name) node, not a peeled one. That is accepted for
v1 — such an edge is never dropped and lands on the dominant referent — and is NOT
addressed in this stage (it would require re-keying those name-only endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision, NarratorGeneration
from src.parse.base import safe_str, write_parquet
from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from src.resolve._inputs import require_input
from src.resolve._run_record import write_canonical
from src.resolve.generic_name import is_generic_name
from src.resolve.mononym_split import _MAX_GAP, _MIN_GAP, is_registered_mononym
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.resolve.tabaqa_dates import _GENERATION_DEATH_WINDOW_AH
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    "SPLIT_BAND_GAP",
    "SPLIT_MIN_SUPPORT",
    "SPLIT_MIN_SEPARATION",
    "SPLIT_MAX_CLUSTERS",
    "SPLIT_MAX_PEEL",
    "CTX_MIN_SHARED",
    "CTX_MAX_JACCARD",
    "CTX_MAX_NEIGHBOUR_DF",
    "MID_GAP",
    "NARRATOR_SPLITS_SCHEMA",
    "DatableMention",
    "ContextMention",
    "PeeledBand",
    "SplitPlan",
    "plan_split",
    "plan_split_by_context",
    "split_generic_narrators",
]

# ---------------------------------------------------------------------------
# Thresholds (tunable — to be re-tuned against staging data in da#337 PR-3)
# ---------------------------------------------------------------------------
# Per-mention death estimate = attested-neighbour death ± this transmission-gap
# midpoint (teacher earlier / student later). The midpoint of the mononym_split
# plausibility window (_MIN_GAP=15, _MAX_GAP=80) so this stage and that one share
# one transmission-gap model.
MID_GAP = (_MIN_GAP + _MAX_GAP) // 2

# 1-D band cut: sort the per-mention death estimates and start a new band wherever a
# consecutive gap exceeds this many AH years (~one full generation). Two people a
# generation apart land in two bands; ordinary within-person estimate scatter does not.
SPLIT_BAND_GAP = 80

# A band must carry at least this many mentions to *qualify* as a peelable referent.
# Below it, a band is left on the primary (peel-not-partition) — a handful of
# adjacency estimates is not enough to mint a distinct historical person.
SPLIT_MIN_SUPPORT = 10

# Two qualifying bands whose midpoints are closer than this are treated as one
# referent's scatter, not two people — abstain. Secondary to SPLIT_BAND_GAP.
SPLIT_MIN_SEPARATION = 50

# More than this many *qualifying* bands means the name is a generic bucket we cannot
# cleanly resolve into a few people — abstain and log for audit rather than shatter it.
SPLIT_MAX_CLUSTERS = 6

# ---------------------------------------------------------------------------
# Context-discriminator thresholds (da#337/#346 PR-B — the isnad-neighbour axis)
# ---------------------------------------------------------------------------
# The context axis runs ONLY where the date axis abstained multi-referent-likely (Gate 2
# noise / Gate 3 contemporaries): it separates an over-merged generic name by *who its
# mentions keep company with* — the identities of their specific teacher/student
# neighbours — rather than by date. All four are tunable, to be re-tuned against staging
# in da#337 PR-3 (same status as the date thresholds above).

# Union two candidate mentions into one referent only if they share at least this many
# *specific* neighbour identities. Two independent shared narrators is a strong
# same-person signal; a single shared narrator can be coincidental co-transmission — so
# the floor is 2, which is also what keeps the Layer-2 Jaccard gate non-redundant
# (components can still incidentally share one neighbour).
CTX_MIN_SHARED = 2

# Layer-2 over-split guard (the context analogue of the date axis's Gate 3): two raw
# components whose aggregate specific-neighbour sets have Jaccard >= this are one referent
# artificially fragmented, so they agglomerate back together rather than peeling as two
# people.
CTX_MAX_JACCARD = 0.15

# Stop-neighbour cap: a neighbour appearing in more than this fraction of *this
# candidate's* mentions co-occurs with every referent, so it is non-discriminative for
# this candidate and must contribute ZERO union evidence (exclude, never down-weight —
# the أبيه "his father" lesson, project_relational_pollution_scrub_equiv). Also bounds the
# Layer-1 pair cost (no single neighbour can induce a quadratic blow-up).
CTX_MAX_NEIGHBOUR_DF = 0.50

# Context-axis ceiling (the analogue of SPLIT_MAX_CLUSTERS on the date axis, but higher —
# resolving a many-referent generic bucket is the whole point of this path). More than
# this many *qualifying* communities ⇒ a generic bucket we still cannot cleanly resolve
# ⇒ HARD abstain (a strict no-op, NOT a truncating top-K: a top-K would leave a tail that
# re-splits on the next run and break idempotence). SPLIT_MAX_CLUSTERS is unchanged — it
# stays the date-axis ceiling; the context axis gets its own, higher one.
SPLIT_MAX_PEEL = 50

# Precisions that mark an *estimated* (non-attested) death window and so must never be
# used as isnad-adjacency evidence: the ṭabaqa layer's estimate AND this stage's own
# ISNAD_ESTIMATE peel (da#340). Excluding both is what keeps the pass cascade-free and a
# re-run a strict no-op — a peeled node's own estimate can never re-seed a further split.
_ESTIMATED_PRECISIONS = frozenset(
    {DatePrecision.TABAQA_ESTIMATE.value, DatePrecision.ISNAD_ESTIMATE.value}
)

# Audit report: one row per peeled (discriminated) node — the owner's review artifact
# before any prod re-load. ``exemplar_anchors`` are a few of the attested chain
# neighbours whose death years placed the band.
NARRATOR_SPLITS_SCHEMA = pa.schema(
    [
        pa.field("name_ar_normalized", pa.string(), nullable=False),
        pa.field("primary_id", pa.string(), nullable=False),
        pa.field("new_id", pa.string(), nullable=False),
        pa.field("discriminator", pa.string(), nullable=False),
        pa.field("band_midpoint_ah", pa.int32(), nullable=False),
        pa.field("band_earliest_ah", pa.int32(), nullable=False),
        pa.field("band_latest_ah", pa.int32(), nullable=False),
        pa.field("mention_count", pa.int32(), nullable=False),
        pa.field("exemplar_anchors", pa.list_(pa.string()), nullable=True),
    ]
)


@dataclass(frozen=True)
class DatableMention:
    """One mention of a candidate with a chain-neighbour-derived death estimate.

    ``estimate`` is ``C``'s per-mention death year (AH), the mean of its attested
    neighbours' death years each shifted by ±:data:`MID_GAP`. ``anchor_names`` are
    the normalized names of those attested neighbours (the tie-break anchor pool).
    """

    mention_id: str
    estimate: int
    anchor_names: tuple[str, ...]


@dataclass(frozen=True)
class ContextMention:
    """One mention of a context candidate: its specific-neighbour set + a date estimate.

    Sibling of :class:`DatableMention` for the isnad-neighbour (context) axis (da#337/#346
    PR-B). ``specific_neighbours`` are the normalized **names** (cascade-free, matching the
    date-axis anchor tie-break) of the mention's *usable* teacher/student neighbours — the
    identities two mentions of the *same* person keep sharing. ``estimate`` is an optional
    death-year estimate (``None`` when the mention has no *attested* context neighbour, e.g.
    a mention datable only across a chain gap) used only to fill the audit row's death
    window; the split decision itself is driven entirely by ``specific_neighbours``, never
    by the estimate.
    """

    mention_id: str
    estimate: int | None
    specific_neighbours: frozenset[str]


@dataclass(frozen=True)
class PeeledBand:
    """A qualifying death-band that peels onto its own discriminated canonical id."""

    new_id: str
    discriminator: str
    midpoint_ah: int
    earliest_ah: int
    latest_ah: int
    mention_ids: tuple[str, ...]
    exemplar_anchors: tuple[str, ...]

    @property
    def mention_count(self) -> int:
        return len(self.mention_ids)


@dataclass(frozen=True)
class SplitPlan:
    """Outcome of :func:`plan_split` for one candidate: abstain (empty) or peel."""

    primary_id: str
    name_ar_normalized: str
    peeled: tuple[PeeledBand, ...] = field(default_factory=tuple)
    # Why the candidate abstained, when it did (``peeled`` empty). ``None`` on a real
    # split. The context axis (da#337/#346 PR-B) reads this to fire ONLY on the date
    # axis's Gate-2 (``"gate2_noise"``) / Gate-3 (``"gate3_unseparated"``) abstains, never
    # on Gate-1 (``"gate1_single"`` — the genuinely-single al-Zuhrī guard). A pure
    # annotation on the date path: no date behaviour changes.
    abstain_reason: str | None = None

    @property
    def is_split(self) -> bool:
        """True when at least one band peeled off (the candidate was split)."""
        return len(self.peeled) > 0

    @property
    def peeled_mention_count(self) -> int:
        return sum(b.mention_count for b in self.peeled)

    def remap(self) -> dict[str, str]:
        """``mention_id -> new discriminated id`` for every peeled mention."""
        out: dict[str, str] = {}
        for band in self.peeled:
            for mid in band.mention_ids:
                out[mid] = band.new_id
        return out


def _cut_bands(datable: list[DatableMention]) -> list[list[DatableMention]]:
    """Sort by death estimate and cut a new band on any gap > :data:`SPLIT_BAND_GAP`."""
    ordered = sorted(datable, key=lambda m: (m.estimate, m.mention_id))
    bands: list[list[DatableMention]] = []
    current: list[DatableMention] = []
    for m in ordered:
        if current and m.estimate - current[-1].estimate > SPLIT_BAND_GAP:
            bands.append(current)
            current = []
        current.append(m)
    if current:
        bands.append(current)
    return bands


def _band_midpoint(band: list[DatableMention]) -> int:
    """Representative death year (AH) of a band — the mean of its estimates."""
    return round(sum(m.estimate for m in band) / len(band))


def _generation_for_year(year: int) -> NarratorGeneration:
    """The first ṭabaqa generation whose window contains ``year`` (else UNKNOWN)."""
    for generation, (lo, hi) in _GENERATION_DEATH_WINDOW_AH.items():
        if lo <= year <= hi:
            return generation
    return NarratorGeneration.UNKNOWN


def _band_label(midpoint: int) -> str:
    """Death-band discriminator label for a band midpoint (before any tie-break).

    A generation window uniquely containing the midpoint gives ``d:<lo>-<hi>``;
    ambiguous (overlapping-window) or out-of-range midpoints fall back to a fixed
    25-year bucket ``d:<floor(mid/25)*25>`` so a label always exists and is
    deterministic.
    """
    matching = [
        (lo, hi) for (lo, hi) in _GENERATION_DEATH_WINDOW_AH.values() if lo <= midpoint <= hi
    ]
    if len(matching) == 1:
        lo, hi = matching[0]
        return f"d:{lo}-{hi}"
    return f"d:{(midpoint // 25) * 25}"


def plan_split(name_ar_normalized: str, datable: list[DatableMention]) -> SplitPlan:
    """Decide whether/how to split candidate ``name_ar_normalized``; pure + deterministic.

    ``datable`` are the candidate's mentions that carry an attested-neighbour death
    estimate (undatable mentions are handled by the caller and simply stay on the
    primary). Returns a :class:`SplitPlan` — abstaining (empty ``peeled``) unless the
    death-band evidence clears every over-split guard in the module docstring, in
    which case each non-largest qualifying band peels onto a discriminated id.
    """
    primary_id = make_canonical_id(name_ar_normalized)

    def _abstain(reason: str) -> SplitPlan:
        # ``reason`` is what the context axis (da#337/#346 PR-B) keys on: it runs only on
        # "gate2_noise"/"gate3_unseparated", never on the "gate1_single" genuinely-single guard.
        return SplitPlan(
            primary_id=primary_id, name_ar_normalized=name_ar_normalized, abstain_reason=reason
        )

    bands = _cut_bands(datable)
    qualifying = [b for b in bands if len(b) >= SPLIT_MIN_SUPPORT]

    # Gate 1: need ≥2 well-supported bands. A single-band name (a genuinely single
    # person like Sufyān al-Thawrī / al-Zuhrī) abstains here — the load-bearing guard.
    if len(qualifying) < 2:
        return _abstain("gate1_single")
    # Gate 2: too many well-supported bands ⇒ generic bucket, not a few people.
    if len(qualifying) > SPLIT_MAX_CLUSTERS:
        logger.info(
            "narrator_split_abstain_noise",
            name=name_ar_normalized,
            qualifying_bands=len(qualifying),
            max_clusters=SPLIT_MAX_CLUSTERS,
        )
        return _abstain("gate2_noise")
    # Gate 3: adjacent qualifying midpoints must be well separated.
    qsorted = sorted(qualifying, key=_band_midpoint)
    mids = [_band_midpoint(b) for b in qsorted]
    if any(b - a <= SPLIT_MIN_SEPARATION for a, b in zip(mids, mids[1:], strict=False)):
        return _abstain("gate3_unseparated")

    # The largest qualifying band (ties → lower midpoint) keeps the primary id; the
    # undatable/below-support remainder is already off the peel list by construction.
    retained = max(qsorted, key=lambda b: (len(b), -_band_midpoint(b)))
    peel_bands = [b for b in qsorted if b is not retained]

    # Assign labels, then tie-break same-label bands with the smallest anchor name so
    # every peeled id is distinct and reproducible from the data alone.
    labels = [_band_label(_band_midpoint(b)) for b in peel_bands]
    label_counts: dict[str, int] = {}
    for lab in labels:
        label_counts[lab] = label_counts.get(lab, 0) + 1

    peeled: list[PeeledBand] = []
    for band, label in zip(peel_bands, labels, strict=True):
        anchors = tuple(sorted({a for m in band for a in m.anchor_names}))
        discriminator = label
        if label_counts[label] > 1:
            anchor = anchors[0] if anchors else ""
            discriminator = f"{label}|{anchor}"
        peeled.append(
            PeeledBand(
                new_id=make_discriminated_canonical_id(name_ar_normalized, discriminator),
                discriminator=discriminator,
                midpoint_ah=_band_midpoint(band),
                earliest_ah=min(m.estimate for m in band),
                latest_ah=max(m.estimate for m in band),
                mention_ids=tuple(m.mention_id for m in band),
                exemplar_anchors=anchors[:3],
            )
        )
    return SplitPlan(
        primary_id=primary_id,
        name_ar_normalized=name_ar_normalized,
        peeled=tuple(peeled),
    )


def _as_int(value: Any) -> int | None:
    """Best-effort int (None/blank/non-numeric → None), for schema int32 columns."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _attested_death(rec: dict[str, Any]) -> int | None:
    """A record's death year iff it is *attested* (present, precision not an estimate).

    An estimated window — the ṭabaqa layer's ``tabaqa_estimate`` OR a prior isnad-split
    peel's ``isnad_estimate`` (da#340), i.e. any precision in
    :data:`_ESTIMATED_PRECISIONS` — is never used as evidence, keeping the pass
    cascade-free and re-runs a strict no-op.
    """
    year = _as_int(rec.get("death_year_ah"))
    if year is None:
        return None
    if rec.get("death_date_precision") in _ESTIMATED_PRECISIONS:
        return None
    return year


def _remap_split_mentions(mentions_path: Path, remap: dict[str, str]) -> int:
    """Rewrite ``canonical_narrator_id`` for peeled mentions, keyed on ``mention_id``.

    Mirrors ``fuzzy_cluster._remap_mention_canonical_ids`` (streaming row-group by
    row-group into a sibling ``.tmp`` then ``replace``; idempotent; bounded-memory)
    but keys on the per-mention ``mention_id``: a split sends *different* mentions of
    one canonical id to *different* targets, which the merge pass's global id→id map
    cannot express. Returns the count of rows remapped.
    """
    if not remap or not mentions_path.exists():
        return 0

    pf = pq.ParquetFile(mentions_path)
    id_idx = NARRATOR_MENTIONS_RESOLVED_SCHEMA.get_field_index("canonical_narrator_id")
    tmp_path = mentions_path.with_name(mentions_path.name + ".split.tmp")
    writer = pq.ParquetWriter(tmp_path, NARRATOR_MENTIONS_RESOLVED_SCHEMA, compression="snappy")
    remapped = 0
    try:
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx).cast(NARRATOR_MENTIONS_RESOLVED_SCHEMA)
            mention_ids = table.column("mention_id").to_pylist()
            cids = table.column("canonical_narrator_id").to_pylist()
            new_ids: list[str | None] = []
            for mid, cid in zip(mention_ids, cids, strict=True):
                target = remap.get(mid) if mid is not None else None
                if target is not None and target != cid:
                    new_ids.append(target)
                    remapped += 1
                else:
                    new_ids.append(cid)
            table = table.set_column(
                id_idx, "canonical_narrator_id", pa.array(new_ids, type=pa.string())
            )
            writer.write_table(table)
    finally:
        writer.close()

    tmp_path.replace(mentions_path)
    logger.info("narrator_split_mentions_remapped", path=str(mentions_path), remapped=remapped)
    return remapped


def _coalesce_chain_index(value: Any) -> int:
    """Per-hadith isnad-chain ordinal, coalescing a missing/None value to 0.

    Mirrors :func:`src.graph.load_edges._chain_index` (da#282) so this splitter groups
    chains on the SAME ``(hadith_id, chain_index)`` key the edge/chain loaders use. A
    ``None`` ``chain_index`` (a single-isnad producer that left it unset, or a legacy
    pre-da#282 file) coalesces to 0, so single-chain hadiths are unaffected.
    """
    return int(value) if value is not None else 0


def _build_chain_index(
    mentions_path: Path,
    candidate_ids: set[str],
) -> tuple[
    dict[tuple[str, int], list[tuple[int, str]]],
    dict[str, list[tuple[str, int, int, str]]],
]:
    """Stream the mentions (two bounded passes) → (chains, candidate_mentions).

    ``chains``: ``(hadith_id, chain_index) -> [(position, canonical_id), ...]`` for
    neighbour lookup — keyed on the composite ``(hadith_id, chain_index)`` the graph
    loaders use (:func:`src.graph.load_edges._chain_index`, da#282), NOT ``hadith_id``
    alone. A multi-isnad hadith (one ``hadith_id``, several ``chain_index`` values —
    today lk's Arabic + English isnads, each numbered from position 0) therefore keeps
    its isnads separate. Keying on ``hadith_id`` alone would flatten them into one
    position-sorted list and fabricate cross-chain ±1 neighbours that exist in no real
    chain, corrupting the death-band evidence (da#411).
    ``candidate_mentions``:
    ``candidate_id -> [(hadith_id, chain_index, position, mention_id), ...]`` — the
    ``chain_index`` is carried through so :func:`_collect_datable` can confine each
    mention's ±1 lookup to its own chain.

    Memory (da#337 PR-3a — OOM audit, cf. #723). ``chains`` is deliberately restricted
    to *only* the chains that contain ≥1 candidate mention. A candidate's per-mention
    death estimate is derived solely from its ±1 neighbours **within the same
    ``(hadith_id, chain_index)`` chain**, so a chain with no candidate mention can never
    contribute a neighbour lookup and is pure dead weight in ``chains``. The earlier
    one-pass build kept every hadith's full chain resident — i.e. the *entire* mentions
    table — which is exactly the class of full-table-resident structure that OOM'd
    resolve in #723. Restricting to candidate-touching chains bounds the resident
    footprint to O(mentions in chains that name a generic-name candidate) + O(candidate
    mentions), both a small fraction of the multi-million-row corpus (generic-name
    candidates are a curated minority), instead of O(all mentions).

    The cost is two streaming passes (2× sequential read) instead of one, each with a
    one-row-group working set — the standard time-for-memory trade, mirroring
    ``fuzzy_cluster``'s multi-pass streaming IO shell. Pass 1 collects the candidate
    mentions and the set of ``(hadith_id, chain_index)`` chains they occur in; pass 2
    fills ``chains`` for only those chains (retaining every position in them, so
    non-candidate neighbours are still present for the ±1 lookup).
    """
    chains: dict[tuple[str, int], list[tuple[int, str]]] = {}
    candidate_mentions: dict[str, list[tuple[str, int, int, str]]] = {}
    if not mentions_path.exists():
        return chains, candidate_mentions

    pf = pq.ParquetFile(mentions_path)
    cols = ["mention_id", "hadith_id", "chain_index", "position_in_chain", "canonical_narrator_id"]

    # Pass 1 — candidate mentions + the (hadith_id, chain_index) chains that contain them.
    candidate_chain_keys: set[tuple[str, int]] = set()
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=cols)
        mention_ids = table.column("mention_id").to_pylist()
        hadith_ids = table.column("hadith_id").to_pylist()
        chain_indexes = table.column("chain_index").to_pylist()
        positions = table.column("position_in_chain").to_pylist()
        cids = table.column("canonical_narrator_id").to_pylist()
        for mid, hid, cidx, pos, cid in zip(
            mention_ids, hadith_ids, chain_indexes, positions, cids, strict=True
        ):
            if cid is None or hid is None or pos is None:
                continue
            if cid in candidate_ids:
                ci = _coalesce_chain_index(cidx)
                candidate_mentions.setdefault(cid, []).append((hid, ci, int(pos), mid))
                candidate_chain_keys.add((hid, ci))

    if not candidate_chain_keys:
        return chains, candidate_mentions

    # Pass 2 — full chains, but only for candidate-touching (hadith_id, chain_index)
    # chains (bounded footprint).
    chain_cols = ["hadith_id", "chain_index", "position_in_chain", "canonical_narrator_id"]
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=chain_cols)
        hadith_ids = table.column("hadith_id").to_pylist()
        chain_indexes = table.column("chain_index").to_pylist()
        positions = table.column("position_in_chain").to_pylist()
        cids = table.column("canonical_narrator_id").to_pylist()
        for hid, cidx, pos, cid in zip(hadith_ids, chain_indexes, positions, cids, strict=True):
            if cid is None or hid is None or pos is None:
                continue
            key = (hid, _coalesce_chain_index(cidx))
            if key in candidate_chain_keys:
                chains.setdefault(key, []).append((int(pos), cid))
    return chains, candidate_mentions


def _collect_datable(
    mentions: list[tuple[str, int, int, str]],
    chains: dict[tuple[str, int], list[tuple[int, str]]],
    attested_by_id: dict[str, int],
    name_by_id: dict[str, str],
) -> list[DatableMention]:
    """Per-mention death estimates from attested chain neighbours (undatable dropped).

    A neighbour is only ever the mention's ±1 position **within the same
    ``(hadith_id, chain_index)`` chain** — never another isnad of the same hadith
    (da#411) — mirroring the graph loaders' composite chain key.
    """
    datable: list[DatableMention] = []
    for hadith_id, chain_index, position, mention_id in mentions:
        estimates: list[int] = []
        anchors: list[str] = []
        for npos, nid in chains.get((hadith_id, chain_index), ()):
            if npos == position - 1:
                sign = 1  # neighbour is the teacher (dies earlier) → C dies later
            elif npos == position + 1:
                sign = -1  # neighbour is the student (dies later) → C dies earlier
            else:
                continue
            year = attested_by_id.get(nid)
            if year is None:
                continue
            estimates.append(year + sign * MID_GAP)
            nname = name_by_id.get(nid)
            if nname:
                anchors.append(nname)
        if estimates:
            datable.append(
                DatableMention(
                    mention_id=mention_id,
                    estimate=round(sum(estimates) / len(estimates)),
                    anchor_names=tuple(sorted(set(anchors))),
                )
            )
    return datable


def _chain_neighbours(chain: list[tuple[int, str]], position: int) -> tuple[str | None, str | None]:
    """The mention's (teacher, student) using the loader's *consecutive-resolved* adjacency.

    #439 (raised in the #437 review): the graph loader's ``TRANSMITTED_TO`` edges pair
    *consecutive resolved* narrators (``load_edges._build_chain_pairs`` sorts by position,
    drops null-``canonical_narrator_id`` mentions, pairs ``resolved[i] → resolved[i+1]``),
    so on a chain with a position GAP — an intermediate mention left unresolved and dropped
    — the loader still makes an edge *across* the gap. The context axis is the split basis
    and is meaningful only insofar as it mirrors that who-taught-whom topology, so it
    matches the loader here rather than the date axis's exact ``position ±1``: it returns
    the immediate predecessor / successor of the mention **in the resolved-and-sorted list**
    (teacher, student), NOT ``position − 1`` / ``position + 1``. Exact-±1 would under-count
    real adjacencies on gappy chains and *erase* a gappy-chain-only referent (or, worse,
    falsely split a single person held together only across a gap). ``chains`` already
    excludes null-canonical mentions (built by :func:`_build_chain_index`), so the gap is
    already collapsed — sorting the survivors by position reproduces the loader's adjacency.
    Do NOT "fix" this back to exact-±1: the divergence from ``_collect_datable`` (which
    stays exact-±1 for its date bands) is intentional and #439-resolved for the context path.
    """
    ordered = sorted(chain, key=lambda pc: pc[0])
    idx = next((i for i, (pos, _) in enumerate(ordered) if pos == position), None)
    if idx is None:
        return None, None
    teacher = ordered[idx - 1][1] if idx > 0 else None
    student = ordered[idx + 1][1] if idx + 1 < len(ordered) else None
    return teacher, student


def _collect_context(
    mentions: list[tuple[str, int, int, str]],
    chains: dict[tuple[str, int], list[tuple[int, str]]],
    attested_by_id: dict[str, int],
    name_by_id: dict[str, str],
    is_specific: Callable[[str], bool],
) -> list[ContextMention]:
    """Per-mention *specific-neighbour* sets (+ an optional date estimate) for the context axis.

    Sibling of :func:`_collect_datable` sharing the same ``(hadith_id, chain_index)``-keyed
    ``chains`` (no second scan). For each candidate mention it takes the consecutive-resolved
    teacher/student (:func:`_chain_neighbours`, #439) and keeps a neighbour's normalized name
    iff ``is_specific`` accepts it (filters 1-2 of §2: not generic, not a split-candidate /
    registered mononym). It then applies the stop-neighbour document-frequency cap (filter 3,
    :data:`CTX_MAX_NEIGHBOUR_DF`): a neighbour appearing in > that fraction of *this*
    candidate's mentions is non-discriminative and is dropped from **every** mention's set
    (zero union evidence — never down-weighted).

    ``estimate`` reuses the date-estimation logic (neighbour attested death ±
    :data:`MID_GAP`, teacher earlier / student later) but on the *context* window, so a
    mention datable only across a gap still carries a window for the audit row; it is
    ``None`` when no context neighbour is attested and is used only for the audit
    death-window, never for the split decision. Every mention is yielded (even with an empty
    neighbour set) so the §5 conservation invariant holds over the candidate's full mention set.
    """
    # Pass 1 — raw specific-neighbour names + a context-window date estimate per mention.
    raw: list[tuple[str, int | None, set[str]]] = []
    df: dict[str, int] = {}
    for hadith_id, chain_index, position, mention_id in mentions:
        teacher, student = _chain_neighbours(chains.get((hadith_id, chain_index), []), position)
        names: set[str] = set()
        estimates: list[int] = []
        for nid, sign in ((teacher, 1), (student, -1)):
            if nid is None:
                continue
            year = attested_by_id.get(nid)
            if year is not None:
                estimates.append(year + sign * MID_GAP)
            if is_specific(nid):
                nname = name_by_id.get(nid)
                if nname:
                    names.add(nname)
        estimate = round(sum(estimates) / len(estimates)) if estimates else None
        for nname in names:
            df[nname] = df.get(nname, 0) + 1
        raw.append((mention_id, estimate, names))

    # Pass 2 — drop stop-neighbours (df cap) from every mention's set, then materialize.
    cap = CTX_MAX_NEIGHBOUR_DF * len(raw)
    stop = {name for name, count in df.items() if count > cap}
    return [
        ContextMention(
            mention_id=mid,
            estimate=estimate,
            specific_neighbours=frozenset(names - stop),
        )
        for mid, estimate, names in raw
    ]


def _cluster_by_context(ctx: list[ContextMention]) -> list[frozenset[str]]:
    """Two-layer, deterministic, order-independent clustering of context mentions.

    **Layer 1 (union-find):** union two mentions that share >= :data:`CTX_MIN_SHARED`
    specific neighbours — via an inverted ``neighbour_name -> {mention_id}`` index and
    pairwise shared counts (the :data:`CTX_MAX_NEIGHBOUR_DF` cap keeps every posting list
    small, bounding the pair cost). **Layer 2 (single-linkage agglomeration):** merge two
    raw components whose aggregate specific-neighbour sets have Jaccard >=
    :data:`CTX_MAX_JACCARD` — the over-split guard that folds an artificially-fragmented
    single referent back together. Returns the merged communities with >=
    :data:`SPLIT_MIN_SUPPORT` mentions (qualifying); the below-support remainder is dropped
    here and stays on the primary (peel-not-partition) in the caller.
    """
    parent: dict[str, str] = {cm.mention_id: cm.mention_id for cm in ctx}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            hi, lo = (ra, rb) if ra > rb else (rb, ra)  # smaller id as root → deterministic
            parent[hi] = lo

    # Layer 1 — inverted index → pairwise shared counts → union at the CTX_MIN_SHARED floor.
    index: dict[str, list[str]] = {}
    for cm in ctx:
        for name in cm.specific_neighbours:
            index.setdefault(name, []).append(cm.mention_id)
    shared: dict[tuple[str, str], int] = {}
    for mids in index.values():
        ordered_mids = sorted(set(mids))
        for i in range(len(ordered_mids)):
            for j in range(i + 1, len(ordered_mids)):
                key = (ordered_mids[i], ordered_mids[j])
                shared[key] = shared.get(key, 0) + 1
    for (a, b), count in shared.items():
        if count >= CTX_MIN_SHARED:
            union(a, b)

    # Raw components + their aggregate specific-neighbour sets.
    nbr_by_mid = {cm.mention_id: cm.specific_neighbours for cm in ctx}
    comp_members: dict[str, set[str]] = {}
    for cm in ctx:
        comp_members.setdefault(find(cm.mention_id), set()).add(cm.mention_id)
    comp_ids = sorted(comp_members)
    comp_nbrs: dict[str, frozenset[str]] = {}
    for c in comp_ids:
        acc: set[str] = set()
        for m in comp_members[c]:
            acc |= nbr_by_mid[m]
        comp_nbrs[c] = frozenset(acc)

    # Layer 2 — agglomerate components with Jaccard >= threshold (single-linkage, on a
    # second union-find so the merge is order-independent for a fixed threshold).
    cparent: dict[str, str] = {c: c for c in comp_ids}

    def cfind(x: str) -> str:
        while cparent[x] != x:
            cparent[x] = cparent[cparent[x]]
            x = cparent[x]
        return x

    def cunion(a: str, b: str) -> None:
        ra, rb = cfind(a), cfind(b)
        if ra != rb:
            hi, lo = (ra, rb) if ra > rb else (rb, ra)
            cparent[hi] = lo

    for i in range(len(comp_ids)):
        for j in range(i + 1, len(comp_ids)):
            na, nb = comp_nbrs[comp_ids[i]], comp_nbrs[comp_ids[j]]
            if not na or not nb:
                continue
            inter = len(na & nb)
            if inter and inter / len(na | nb) >= CTX_MAX_JACCARD:
                cunion(comp_ids[i], comp_ids[j])

    merged: dict[str, set[str]] = {}
    for c in comp_ids:
        merged.setdefault(cfind(c), set()).update(comp_members[c])

    return [
        frozenset(members)
        for _, members in sorted(merged.items())
        if len(members) >= SPLIT_MIN_SUPPORT
    ]


def _context_discriminator(anchors: tuple[str, ...], used: set[str]) -> str:
    """A deterministic ``"ctx:"`` discriminator from a community's specific anchors.

    The community's lexicographically-smallest specific anchor name (cascade-free, matching
    the date axis's anchor tie-break at ``narrator_split.py`` §6); on the
    structurally-near-impossible chance two peeled communities produce the same label,
    append the next-smallest anchor names until unique — deterministic, no running ordinals.
    """
    disc = "ctx:" + (anchors[0] if anchors else "")
    extra = 1
    while disc in used and extra < len(anchors):
        disc = disc + "|" + anchors[extra]
        extra += 1
    return disc


def plan_split_by_context(name_ar_normalized: str, ctx: list[ContextMention]) -> SplitPlan:
    """Decide whether/how to split ``name_ar_normalized`` on the isnad-neighbour context axis.

    The date-axis *fallback* (da#337/#346 PR-B): the caller runs this only where
    :func:`plan_split` abstained at Gate 2/Gate 3. Mirrors ``plan_split``'s shape and reuses
    :class:`PeeledBand` / :class:`SplitPlan` / the audit schema unchanged. Abstains (empty
    ``SplitPlan``) unless there are between 2 and :data:`SPLIT_MAX_PEEL` qualifying
    communities; otherwise the **largest** community keeps the bare id (the primary) and
    every other qualifying community peels onto a ``"ctx:"``-discriminated id. Below-support
    fragments and the un-clustered remainder stay on the primary (peel-not-partition).
    """
    primary_id = make_canonical_id(name_ar_normalized)

    communities = _cluster_by_context(ctx)
    # Gate A: < 2 communities ⇒ one referent or a diffuse hub (al-Zuhrī) → stay whole.
    if len(communities) < 2:
        return SplitPlan(
            primary_id=primary_id,
            name_ar_normalized=name_ar_normalized,
            abstain_reason="ctx_single",
        )
    # Gate B: too many well-separated communities ⇒ still-unresolvable generic bucket ⇒
    # HARD abstain (a no-op, not a truncating top-K — keeps the re-run idempotent).
    if len(communities) > SPLIT_MAX_PEEL:
        logger.info(
            "narrator_split_context_abstain_noise",
            name=name_ar_normalized,
            communities=len(communities),
            max_peel=SPLIT_MAX_PEEL,
        )
        return SplitPlan(
            primary_id=primary_id,
            name_ar_normalized=name_ar_normalized,
            abstain_reason="ctx_noise",
        )

    ctx_by_mid = {cm.mention_id: cm for cm in ctx}

    def community_anchors(members: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted({a for m in members for a in ctx_by_mid[m].specific_neighbours}))

    # Retain the largest community (ties → lexicographically-smallest specific anchor,
    # deterministic); every other qualifying community peels.
    def sort_key(members: frozenset[str]) -> tuple[int, str]:
        anchors = community_anchors(members)
        return (-len(members), anchors[0] if anchors else "")

    ordered = sorted(communities, key=sort_key)
    peel_communities = ordered[1:]

    used_disc: set[str] = set()
    peeled: list[PeeledBand] = []
    for members in peel_communities:
        anchors = community_anchors(members)
        disc = _context_discriminator(anchors, used_disc)
        used_disc.add(disc)
        # Death window (informational only — these referents are contemporaries, which is
        # WHY the date axis abstained): the mean/min/max of the community's datable estimates.
        estimates: list[int] = []
        for m in members:
            est = ctx_by_mid[m].estimate
            if est is not None:
                estimates.append(est)
        if estimates:
            midpoint = round(sum(estimates) / len(estimates))
            earliest, latest = min(estimates), max(estimates)
        else:
            midpoint = earliest = latest = 0
        peeled.append(
            PeeledBand(
                new_id=make_discriminated_canonical_id(name_ar_normalized, disc),
                discriminator=disc,
                midpoint_ah=midpoint,
                earliest_ah=earliest,
                latest_ah=latest,
                mention_ids=tuple(sorted(members)),
                exemplar_anchors=anchors[:3],
            )
        )
    return SplitPlan(
        primary_id=primary_id,
        name_ar_normalized=name_ar_normalized,
        peeled=tuple(peeled),
    )


def _peeled_record(primary_rec: dict[str, Any], band: PeeledBand, name_norm: str) -> dict[str, Any]:
    """A new canonical row for a peeled band, conforming to NARRATORS_CANONICAL_SCHEMA.

    Inherits the shared name/provenance fields from the primary record; carries the
    band's estimated death window with ``isnad_estimate`` precision (da#340 — see the
    module docstring on estimated-window provenance); birth stays unknown.
    """
    rec: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    for col in (
        "name_ar",
        "name_en",
        "aliases",
        "gender",
        "trustworthiness",
        "source_ids",
        "source_corpus",
        "source_corpora",
        "sect_affiliation",
    ):
        rec[col] = primary_rec.get(col)
    rec["canonical_id"] = band.new_id
    rec["name_ar_normalized"] = name_norm
    rec["death_year_ah"] = band.midpoint_ah
    rec["death_year_ah_earliest"] = band.earliest_ah
    rec["death_year_ah_latest"] = band.latest_ah
    rec["death_date_precision"] = DatePrecision.ISNAD_ESTIMATE.value
    rec["birth_date_precision"] = DatePrecision.UNKNOWN.value
    rec["generation"] = _generation_for_year(band.midpoint_ah).value
    rec["mention_count"] = band.mention_count
    return rec


def split_generic_narrators(output_dir: Path, *, staging_dir: Path | None = None) -> Path | None:
    """Split over-collapsed generic-name narrator nodes on death-band evidence (da#337).

    Reads ``narrators_canonical.parquet`` + ``narrator_mentions_resolved.parquet`` from
    ``output_dir`` (the curated dir). For each canonical record whose name screens in
    via :func:`~src.resolve.generic_name.is_generic_name` (and is not owned by
    ``mononym_split``), decides on death-band evidence whether to peel it into distinct
    canonical nodes (:func:`plan_split`). On a split: appends the peeled rows to the
    canonical table (reducing the primary's ``mention_count``), remaps the peeled
    mentions, and writes the ``narrator_splits.parquet`` audit report.

    Returns the canonical path when at least one node split (files rewritten), or
    ``None`` when the stage was a no-op (missing inputs, no candidates, or every
    candidate abstained — including every re-run, which is idempotent). ``staging_dir``
    is accepted for run_all call-shape parity; this stage keeps no checkpoint (it is a
    single idempotent pass, like reconcile/tabaqa_dates).
    """
    # Required input (da#361): the canonical table, produced by disambiguate/bio_promote
    # earlier THIS run. By the time narrator_split runs (step 3.55) at least one of
    # those has run in any non-empty resolve, so an absent canonical table is an
    # upstream defect — raise, not a success-shaped ``None``. A canonical table that
    # EXISTS but is empty (zero records) is the honest no-op below.
    canonical_path = output_dir / "narrators_canonical.parquet"
    mentions_path = output_dir / "narrator_mentions_resolved.parquet"
    require_input(
        stage="narrator_split",
        present=canonical_path.exists(),
        input_desc=f"{canonical_path.name} (canonical narrators)",
        produced_by="resolve disambiguate/bio_promote stages",
        remediation="re-run resolve from `disambiguate`; the canonical table is absent",
    )
    if not mentions_path.exists():
        # da#361: an absent mentions file is NOT a missing input here — NER
        # legitimately produces zero mentions for a bio-only corpus (bio_promote
        # still writes a canonical table with no chain mentions to split). A genuine
        # no-op, not a defect, so it returns ``None`` rather than raising.
        logger.warning("narrator_split_no_mentions", path=str(mentions_path))
        return None

    records = pq.read_table(canonical_path).to_pylist()
    if not records:
        # Canonical table present but empty — an honest no-op, not a missing input.
        return None

    name_by_id: dict[str, str] = {}
    attested_by_id: dict[str, int] = {}
    record_by_id: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    for rec in records:
        cid = safe_str(rec.get("canonical_id"))
        if cid is None:
            continue
        record_by_id[cid] = rec
        name_norm = safe_str(rec.get("name_ar_normalized"))
        if name_norm is not None:
            name_by_id[cid] = name_norm
        attested = _attested_death(rec)
        if attested is not None:
            attested_by_id[cid] = attested
        mention_count = _as_int(rec.get("mention_count")) or 0
        if (
            name_norm is not None
            and is_generic_name(name_norm, mention_count)
            and not is_registered_mononym(name_norm)
        ):
            candidate_ids.add(cid)

    if not candidate_ids:
        logger.info("narrator_split_no_candidates", canonical_total=len(records))
        return None

    chains, candidate_mentions = _build_chain_index(mentions_path, candidate_ids)

    def is_specific(neighbour_id: str) -> bool:
        """A context neighbour discriminates iff it is a *specific* narrator (§2 filters 1-2):
        not a split candidate, not a registered mononym, not itself a generic name."""
        if neighbour_id in candidate_ids:
            return False
        nname = name_by_id.get(neighbour_id)
        if nname is None or is_registered_mononym(nname):
            return False
        nmc = _as_int(record_by_id.get(neighbour_id, {}).get("mention_count")) or 0
        return not is_generic_name(nname, nmc)

    plans: list[SplitPlan] = []
    mentions_by_primary: dict[str, set[str]] = {}
    for cid in sorted(candidate_ids):
        name_norm = name_by_id.get(cid)
        if name_norm is None:
            continue
        cand_mentions = candidate_mentions.get(cid, [])
        datable = _collect_datable(cand_mentions, chains, attested_by_id, name_by_id)
        date_plan = plan_split(name_norm, datable)
        if date_plan.is_split:
            plans.append(date_plan)
            mentions_by_primary[cid] = {m[3] for m in cand_mentions}
            continue
        # Context fallback (da#337/#346 PR-B): runs ONLY where the DATE axis found the name
        # multi-referent-likely (≥2 datable bands) but couldn't resolve it — Gate 2 (noise)
        # / Gate 3 (contemporaries). Gate 1 (a single clean band, or undatable) is the
        # genuinely-single guard (al-Zuhrī) and is NEVER sent to context.
        if date_plan.abstain_reason in {"gate2_noise", "gate3_unseparated"}:
            ctx = _collect_context(cand_mentions, chains, attested_by_id, name_by_id, is_specific)
            ctx_plan = plan_split_by_context(name_norm, ctx)
            if ctx_plan.is_split:
                plans.append(ctx_plan)
                mentions_by_primary[cid] = {m[3] for m in cand_mentions}

    if not plans:
        logger.info(
            "narrator_split_all_abstained",
            candidates=len(candidate_ids),
            canonical_total=len(records),
        )
        return None

    # PURE-SPLIT invariant (da#337/#346 §5) — asserted for BOTH axes (a shared safety net,
    # per project_canonical_identity_invariant: "de-keying alone trades over-merge for
    # under-merge"). No-merge: every peel mints a fresh, distinct id, absent from the
    # pre-existing canonical table — never routes two distinct nodes onto one.
    # Conservation: the peeled mention_id sets are pairwise disjoint and a subset of the
    # candidate's own mentions, so peeled + retained-primary == original.
    pre_existing_ids = set(record_by_id)
    for plan in plans:
        peel_ids = [b.new_id for b in plan.peeled]
        assert len(peel_ids) == len(set(peel_ids)), "peeled ids must be distinct"
        seen_mids: set[str] = set()
        for band in plan.peeled:
            assert band.new_id != plan.primary_id
            assert band.new_id not in pre_existing_ids, (
                "a peel must mint a new node, never merge onto an existing canonical id"
            )
            mids = set(band.mention_ids)
            assert seen_mids.isdisjoint(mids), "peeled mention sets must be disjoint"
            seen_mids |= mids
        assert seen_mids <= mentions_by_primary[plan.primary_id], (
            "peeled mentions must be a subset of the candidate's own mentions"
        )

    # Apply: reduce each primary's mention_count, append peeled rows, build the remap.
    remap: dict[str, str] = {}
    new_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for plan in plans:
        primary_rec = record_by_id[plan.primary_id]
        primary_count = _as_int(primary_rec.get("mention_count")) or 0
        primary_rec["mention_count"] = max(0, primary_count - plan.peeled_mention_count)
        for band in plan.peeled:
            new_rows.append(_peeled_record(primary_rec, band, plan.name_ar_normalized))
            audit_rows.append(
                {
                    "name_ar_normalized": plan.name_ar_normalized,
                    "primary_id": plan.primary_id,
                    "new_id": band.new_id,
                    "discriminator": band.discriminator,
                    "band_midpoint_ah": band.midpoint_ah,
                    "band_earliest_ah": band.earliest_ah,
                    "band_latest_ah": band.latest_ah,
                    "mention_count": band.mention_count,
                    "exemplar_anchors": list(band.exemplar_anchors),
                }
            )
        remap.update(plan.remap())

    all_records = records + new_rows
    arrays = {f.name: [r.get(f.name) for r in all_records] for f in NARRATORS_CANONICAL_SCHEMA}
    write_canonical(
        pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        canonical_path,
        stage="narrator_split",
    )

    remapped = _remap_split_mentions(mentions_path, remap)

    audit_arrays = {f.name: [r.get(f.name) for r in audit_rows] for f in NARRATOR_SPLITS_SCHEMA}
    audit_path = output_dir / "narrator_splits.parquet"
    write_parquet(
        pa.table(audit_arrays, schema=NARRATOR_SPLITS_SCHEMA),
        audit_path,
        schema=NARRATOR_SPLITS_SCHEMA,
    )

    logger.info(
        "narrator_split_complete",
        split_nodes=len(plans),
        peeled_records=len(new_rows),
        mentions_remapped=remapped,
        canonical_total=len(all_records),
        audit=str(audit_path),
    )
    return canonical_path
