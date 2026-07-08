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
   at ``position_in_chain`` ±1 in the same ``hadith_id`` and map them to their
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
from src.resolve.generic_name import is_generic_name
from src.resolve.mononym_split import _MAX_GAP, _MIN_GAP, is_registered_mononym
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.resolve.tabaqa_dates import _GENERATION_DEATH_WINDOW_AH
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    "SPLIT_BAND_GAP",
    "SPLIT_MIN_SUPPORT",
    "SPLIT_MIN_SEPARATION",
    "SPLIT_MAX_CLUSTERS",
    "MID_GAP",
    "NARRATOR_SPLITS_SCHEMA",
    "DatableMention",
    "PeeledBand",
    "SplitPlan",
    "plan_split",
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
    abstain = SplitPlan(primary_id=primary_id, name_ar_normalized=name_ar_normalized)

    bands = _cut_bands(datable)
    qualifying = [b for b in bands if len(b) >= SPLIT_MIN_SUPPORT]

    # Gate 1: need ≥2 well-supported bands. A single-band name (a genuinely single
    # person like Sufyān al-Thawrī / al-Zuhrī) abstains here — the load-bearing guard.
    if len(qualifying) < 2:
        return abstain
    # Gate 2: too many well-supported bands ⇒ generic bucket, not a few people.
    if len(qualifying) > SPLIT_MAX_CLUSTERS:
        logger.info(
            "narrator_split_abstain_noise",
            name=name_ar_normalized,
            qualifying_bands=len(qualifying),
            max_clusters=SPLIT_MAX_CLUSTERS,
        )
        return abstain
    # Gate 3: adjacent qualifying midpoints must be well separated.
    qsorted = sorted(qualifying, key=_band_midpoint)
    mids = [_band_midpoint(b) for b in qsorted]
    if any(b - a <= SPLIT_MIN_SEPARATION for a, b in zip(mids, mids[1:], strict=False)):
        return abstain

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


def _build_chain_index(
    mentions_path: Path,
    candidate_ids: set[str],
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[str, int, str]]]]:
    """Stream the mentions (two bounded passes) → (chains, candidate_mentions).

    ``chains``: ``hadith_id -> [(position, canonical_id), ...]`` for neighbour lookup.
    ``candidate_mentions``: ``candidate_id -> [(hadith_id, position, mention_id), ...]``.

    Memory (da#337 PR-3a — OOM audit, cf. #723). ``chains`` is deliberately restricted
    to *only* the hadiths that contain ≥1 candidate mention. A candidate's per-mention
    death estimate is derived solely from its ±1 neighbours **within the same hadith**,
    so a hadith with no candidate mention can never contribute a neighbour lookup and is
    pure dead weight in ``chains``. The earlier one-pass build kept every hadith's full
    chain resident — i.e. the *entire* mentions table — which is exactly the class of
    full-table-resident structure that OOM'd resolve in #723. Restricting to
    candidate-touching hadiths bounds the resident footprint to O(mentions in hadiths
    that name a generic-name candidate) + O(candidate mentions), both a small fraction
    of the multi-million-row corpus (generic-name candidates are a curated minority),
    instead of O(all mentions).

    The cost is two streaming passes (2× sequential read) instead of one, each with a
    one-row-group working set — the standard time-for-memory trade, mirroring
    ``fuzzy_cluster``'s multi-pass streaming IO shell. Pass 1 collects the candidate
    mentions and the set of hadiths they occur in; pass 2 fills ``chains`` for only
    those hadiths (retaining every position in them, so non-candidate neighbours are
    still present for the ±1 lookup).
    """
    chains: dict[str, list[tuple[int, str]]] = {}
    candidate_mentions: dict[str, list[tuple[str, int, str]]] = {}
    if not mentions_path.exists():
        return chains, candidate_mentions

    pf = pq.ParquetFile(mentions_path)
    cols = ["mention_id", "hadith_id", "position_in_chain", "canonical_narrator_id"]

    # Pass 1 — candidate mentions + the hadiths that contain them.
    candidate_hadith_ids: set[str] = set()
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=cols)
        mention_ids = table.column("mention_id").to_pylist()
        hadith_ids = table.column("hadith_id").to_pylist()
        positions = table.column("position_in_chain").to_pylist()
        cids = table.column("canonical_narrator_id").to_pylist()
        for mid, hid, pos, cid in zip(mention_ids, hadith_ids, positions, cids, strict=True):
            if cid is None or hid is None or pos is None:
                continue
            if cid in candidate_ids:
                candidate_mentions.setdefault(cid, []).append((hid, int(pos), mid))
                candidate_hadith_ids.add(hid)

    if not candidate_hadith_ids:
        return chains, candidate_mentions

    # Pass 2 — full chains, but only for candidate-touching hadiths (bounded footprint).
    chain_cols = ["hadith_id", "position_in_chain", "canonical_narrator_id"]
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=chain_cols)
        hadith_ids = table.column("hadith_id").to_pylist()
        positions = table.column("position_in_chain").to_pylist()
        cids = table.column("canonical_narrator_id").to_pylist()
        for hid, pos, cid in zip(hadith_ids, positions, cids, strict=True):
            if cid is None or hid is None or pos is None:
                continue
            if hid in candidate_hadith_ids:
                chains.setdefault(hid, []).append((int(pos), cid))
    return chains, candidate_mentions


def _collect_datable(
    mentions: list[tuple[str, int, str]],
    chains: dict[str, list[tuple[int, str]]],
    attested_by_id: dict[str, int],
    name_by_id: dict[str, str],
) -> list[DatableMention]:
    """Per-mention death estimates from attested chain neighbours (undatable dropped)."""
    datable: list[DatableMention] = []
    for hadith_id, position, mention_id in mentions:
        estimates: list[int] = []
        anchors: list[str] = []
        for npos, nid in chains.get(hadith_id, ()):
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
    canonical_path = output_dir / "narrators_canonical.parquet"
    mentions_path = output_dir / "narrator_mentions_resolved.parquet"
    if not canonical_path.exists():
        logger.warning("narrator_split_no_canonical", path=str(canonical_path))
        return None
    if not mentions_path.exists():
        logger.warning("narrator_split_no_mentions", path=str(mentions_path))
        return None

    records = pq.read_table(canonical_path).to_pylist()
    if not records:
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

    plans: list[SplitPlan] = []
    for cid in sorted(candidate_ids):
        name_norm = name_by_id.get(cid)
        if name_norm is None:
            continue
        datable = _collect_datable(
            candidate_mentions.get(cid, []), chains, attested_by_id, name_by_id
        )
        plan = plan_split(name_norm, datable)
        if plan.is_split:
            plans.append(plan)

    if not plans:
        logger.info(
            "narrator_split_all_abstained",
            candidates=len(candidate_ids),
            canonical_total=len(records),
        )
        return None

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
    write_parquet(
        pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        canonical_path,
        schema=NARRATORS_CANONICAL_SCHEMA,
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
