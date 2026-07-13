"""Narrator disambiguation via multi-stage matching.

5-stage pipeline: exact → fuzzy (rapidfuzz) → temporal → geographic → cross-ref.
Produces canonical narrator records with deterministic UUID5 IDs,
an ambiguous-narrators report, and a merge audit log.

Performance: uses blocking indexes (first-2-char prefix) to reduce the
candidate comparison space from O(n*m) to O(n * m/k), and streams
mentions in batches to keep memory under 4GB.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.models.enums import DatePrecision
from src.parse.base import safe_str, write_parquet
from src.parse.identity import make_canonical_id
from src.parse.name_quality import is_mubham_relational
from src.resolve._checkpoint import (
    CheckpointController,
    checkpoint_dir,
    clear_checkpoint,
    hash_parquet_column_groups,
    load_checkpoint,
    resolve_cadence,
    save_checkpoint,
)
from src.resolve._inputs import require_input
from src.resolve._run_record import write_canonical
from src.resolve.attestation import derive_attestation
from src.resolve.geography import regions_plausible, resolve_region
from src.resolve.mononym_split import refine_mononym_name
from src.resolve.schemas import (
    AMBIGUOUS_NARRATORS_SCHEMA,
    NARRATOR_MENTIONS_RESOLVED_SCHEMA,
    NARRATORS_CANONICAL_SCHEMA,
)
from src.resolve.sect_affiliation import (
    derive_sect_affiliation,
    normalize_corpus,
    primary_corpus,
)
from src.utils.arabic import canonical_surface
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["run"]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_FUZZY_RATIO_THRESHOLD = 80
_LEVENSHTEIN_MAX_DIST = 2
_CONFIDENCE_THRESHOLD = 0.70
_TEMPORAL_MIN_GAP = 15
_TEMPORAL_MAX_GAP = 80

# da#356 corroboration gate. A name-similarity score alone may not authorize
# stamping a bio's metadata onto a narrator: absent chain evidence, the match must
# be *near-identical* by name. 0.90 / distance 1 is where the measured corpus
# separates OCR corruptions of one person (Ibn ʿAbbās `ابن عباس` ↔ the itqan
# `ابن عبس`: ratio 0.933, distance 1) from genuinely different people
# (al-Bāqir ↔ al-Kūfī: ratio 0.852, distance 4). These gate METADATA only —
# identity is never at stake (see `_bio_corroborated`), so a false negative costs
# a missing death year, not a chimeric node.
_CORROBORATION_STRICT_SCORE = 0.90
_CORROBORATION_STRICT_MAX_DIST = 1

# Blocking: number of prefix characters to use for candidate grouping.
_BLOCK_PREFIX_LEN = 2

# Batch size for streaming mentions from Parquet.
_MENTION_BATCH_SIZE = 50_000

# Progress log interval.
_PROGRESS_LOG_INTERVAL = 10_000

# ---------------------------------------------------------------------------
# Crash-resume checkpointing (da#268)
# ---------------------------------------------------------------------------
# Directory (under the gitignored staging tree) that holds the mid-stream
# checkpoint of run()'s accumulated state. Cleaned up after a successful run.
_CHECKPOINT_DIRNAME = ".disambiguate_checkpoint"

# Persist the accumulated state every N completed mention batches. At the prod
# streaming rate (~3.1M mentions over ~5-6 h ⇒ one 50k batch every ~5 min) this
# bounds crash-loss to ~4 batches ≈ ~20 min of recompute while keeping the
# JSON-write overhead well under 1% of wall-clock. Override for ops via
# DISAMBIGUATE_CHECKPOINT_EVERY_N_BATCHES or the run() kwarg (tests use 1-2).
_CHECKPOINT_EVERY_N_BATCHES = 4

# Bump when the persisted payload layout changes so a stale-layout checkpoint is
# rejected (cold start) rather than mis-read.
_CHECKPOINT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A known narrator from biographical sources."""

    bio_id: str
    name_ar: str | None = None
    name_en: str | None = None
    name_ar_normalized: str | None = None
    kunya: str | None = None
    nisba: str | None = None
    birth_year_ah: int | None = None
    death_year_ah: int | None = None
    birth_location: str | None = None
    death_location: str | None = None
    generation: str | None = None
    gender: str | None = None
    trustworthiness: str | None = None
    external_id: str | None = None
    source: str | None = None


@dataclass
class Match:
    """A disambiguation match between a mention and a candidate."""

    candidate: Candidate
    stage: str
    score: float


@dataclass
class ChainContext:
    """Contextual information about a narrator mention's chain."""

    hadith_id: str
    position_in_chain: int
    source_corpus: str
    adjacent_death_years: list[int] = field(default_factory=list)
    # Free-text birth/death locations of the resolved adjacent narrators, used by
    # the geographic filter. Empty when no neighbour resolved or carried a location.
    adjacent_locations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Blocking index
# ---------------------------------------------------------------------------
@dataclass
class BlockingIndex:
    """Pre-computed indexes for fast candidate lookup by name prefix.

    Reduces comparison space from O(candidates) to O(candidates / k)
    where k is the number of distinct prefix blocks.
    """

    exact_ar: dict[str, list[Candidate]]
    exact_en: dict[str, list[Candidate]]
    blocks_ar: dict[str, list[Candidate]]
    crossref_blocks: dict[str, list[Candidate]]


def _build_blocking_index(candidates: list[Candidate]) -> BlockingIndex:
    """Build blocking indexes over the candidate list."""
    exact_ar: dict[str, list[Candidate]] = defaultdict(list)
    exact_en: dict[str, list[Candidate]] = defaultdict(list)
    blocks_ar: dict[str, list[Candidate]] = defaultdict(list)
    crossref_blocks: dict[str, list[Candidate]] = defaultdict(list)

    for c in candidates:
        if c.name_ar_normalized:
            exact_ar[c.name_ar_normalized].append(c)
            prefix = c.name_ar_normalized[:_BLOCK_PREFIX_LEN]
            blocks_ar[prefix].append(c)
        if c.name_en:
            exact_en[c.name_en.strip().lower()].append(c)
        if c.external_id and c.name_ar_normalized:
            prefix = c.name_ar_normalized[:_BLOCK_PREFIX_LEN]
            crossref_blocks[prefix].append(c)

    logger.info(
        "blocking_index_built",
        exact_ar_keys=len(exact_ar),
        exact_en_keys=len(exact_en),
        block_keys=len(blocks_ar),
        crossref_keys=len(crossref_blocks),
    )
    return BlockingIndex(
        exact_ar=dict(exact_ar),
        exact_en=dict(exact_en),
        blocks_ar=dict(blocks_ar),
        crossref_blocks=dict(crossref_blocks),
    )


# ---------------------------------------------------------------------------
# Stage 1: Exact match (indexed)
# ---------------------------------------------------------------------------
def _exact_match(mention_norm: str, candidates: list[Candidate]) -> list[Match]:
    """Full normalized-name match.

    ``mention_norm`` must already be Arabic-normalized by the caller.
    """
    results: list[Match] = []
    if not mention_norm:
        return results
    for c in candidates:
        if c.name_ar_normalized and c.name_ar_normalized == mention_norm:
            results.append(Match(candidate=c, stage="exact", score=1.0))
        elif c.name_en and c.name_en.strip().lower() == mention_norm.lower():
            results.append(Match(candidate=c, stage="exact", score=1.0))
    return results


def _exact_match_indexed(mention_norm: str, index: BlockingIndex) -> list[Match]:
    """O(1) exact match using pre-built hash indexes."""
    results: list[Match] = []
    if not mention_norm:
        return results
    for c in index.exact_ar.get(mention_norm, []):
        results.append(Match(candidate=c, stage="exact", score=1.0))
    lower = mention_norm.lower()
    for c in index.exact_en.get(lower, []):
        results.append(Match(candidate=c, stage="exact", score=1.0))
    return results


# ---------------------------------------------------------------------------
# Stage 2: Fuzzy match (blocked)
# ---------------------------------------------------------------------------
def _fuzzy_match(mention_norm: str, candidates: list[Candidate]) -> list[Match]:
    """rapidfuzz ratio >= threshold AND Levenshtein distance <= max.

    ``mention_norm`` must already be Arabic-normalized by the caller.
    """
    results: list[Match] = []
    if not mention_norm:
        return results
    for c in candidates:
        cand_name = c.name_ar_normalized or ""
        if not cand_name:
            continue
        ratio = fuzz.ratio(mention_norm, cand_name)
        dist = Levenshtein.distance(mention_norm, cand_name)
        if ratio >= _FUZZY_RATIO_THRESHOLD and dist <= _LEVENSHTEIN_MAX_DIST:
            results.append(Match(candidate=c, stage="fuzzy", score=round(ratio / 100.0, 4)))
    return results


def _fuzzy_match_blocked(mention_norm: str, index: BlockingIndex) -> list[Match]:
    """Fuzzy match restricted to same-prefix block with score_cutoff pruning."""
    results: list[Match] = []
    if not mention_norm:
        return results
    prefix = mention_norm[:_BLOCK_PREFIX_LEN]
    block = index.blocks_ar.get(prefix, [])
    for c in block:
        cand_name = c.name_ar_normalized or ""
        if not cand_name:
            continue
        ratio = fuzz.ratio(mention_norm, cand_name, score_cutoff=_FUZZY_RATIO_THRESHOLD)
        if ratio == 0:
            continue
        dist = Levenshtein.distance(mention_norm, cand_name)
        if dist <= _LEVENSHTEIN_MAX_DIST:
            results.append(Match(candidate=c, stage="fuzzy", score=round(ratio / 100.0, 4)))
    return results


# ---------------------------------------------------------------------------
# Stage 3: Temporal filter
# ---------------------------------------------------------------------------
def _temporal_filter(matches: list[Match], chain_context: ChainContext) -> list[Match]:
    """Filter matches by plausible teacher-student temporal gap.

    If both the candidate and adjacent narrators in the chain have
    birth/death years, verify the gap falls within 15-80 years.
    Passes through matches when temporal data is missing.
    """
    if not chain_context.adjacent_death_years:
        return matches

    filtered: list[Match] = []
    for m in matches:
        death_year = m.candidate.death_year_ah
        if death_year is None:
            # No temporal data — keep the match (soft constraint).
            filtered.append(m)
            continue
        plausible = False
        for adj_year in chain_context.adjacent_death_years:
            gap = abs(death_year - adj_year)
            if _TEMPORAL_MIN_GAP <= gap <= _TEMPORAL_MAX_GAP:
                plausible = True
                break
        if plausible:
            filtered.append(m)

    return filtered


# ---------------------------------------------------------------------------
# Corroboration gate (da#356)
# ---------------------------------------------------------------------------
def _bio_corroborated(match: Match, mention_norm: str, adjacent_death_years: list[int]) -> bool:
    """May the matched bio's metadata be stamped onto this mention's narrator?

    This gate governs **metadata only**. Since da#356 the canonical id and display
    name are a pure function of the mention (see :func:`run`), so a rejected match
    costs a missing ``death_year_ah``, never a re-keyed or renamed node. It still
    matters: an attached year enters ``death_year_index``, which feeds
    :func:`_temporal_filter` for chain neighbours **and** the da#248 evidence in
    :func:`refine_mononym_name` — so a wrong year propagates back into *identity*
    via the mononym split. Bad metadata is not inert.

    A match is corroborated when:

    * it is an ``exact`` name match (the bio *is* this name — nothing to corroborate); or
    * chain-neighbour death-year evidence exists and **agrees** (the positive signal); or
    * no usable temporal evidence exists, and the name match is *near-identical* —
      ``score >= _CORROBORATION_STRICT_SCORE`` and Levenshtein ≤
      ``_CORROBORATION_STRICT_MAX_DIST``.

    A match is **vetoed outright** when temporal evidence exists and contradicts,
    however high the name score.

    Why not "the mention's own form must itself be bio-registered": that criterion
    is vacuous. Stage 1 (:func:`_exact_match_indexed`) returns early on any exact
    bio hit, so no mention reaching the fuzzy or crossref stage can be
    bio-registered — measured survival on the pre-fix corpus was 0 of 742,607
    fuzzy and 0 of 937,539 crossref matches. Requiring it would silently delete
    both stages.

    Why not "chain-neighbour agreement" alone: only 26.2% of the 140,174 bio
    candidates carry a ``death_year_ah`` at all, so requiring positive temporal
    agreement rejects the majority on grounds of *missing data* rather than
    evidence of wrongness. Hence the near-identity fallback.
    """
    if match.stage == "exact":
        return True

    death_year = match.candidate.death_year_ah
    if adjacent_death_years and death_year is not None:
        # Positive evidence available — it decides, in both directions.
        return any(
            _TEMPORAL_MIN_GAP <= abs(death_year - adj) <= _TEMPORAL_MAX_GAP
            for adj in adjacent_death_years
        )

    # No usable temporal evidence: fall back to near-identity of the names.
    cand_norm = match.candidate.name_ar_normalized or ""
    if not cand_norm or not mention_norm:
        return False
    return (
        match.score >= _CORROBORATION_STRICT_SCORE
        and Levenshtein.distance(mention_norm, cand_norm) <= _CORROBORATION_STRICT_MAX_DIST
    )


# ---------------------------------------------------------------------------
# Stage 4: Geographic filter
# ---------------------------------------------------------------------------
def _geographic_filter(
    matches: list[Match], chain_context: ChainContext | None = None
) -> list[Match]:
    """Soft geographic constraint — drop matches based in a region travel-implausible
    given the chain's adjacent narrators.

    Mirrors :func:`_temporal_filter`: it consumes a chain-context signal (here the
    canonical regions of the resolved adjacent narrators' locations) and prunes
    candidates that contradict it, while passing everything through when the signal
    is absent. The location normalization + travel-plausibility model lives in
    :mod:`src.resolve.geography` (da#139).

    Conservative by design — a match is dropped **only** when both its own location
    and at least one neighbour's location resolve to known regions AND the candidate
    region is travel-implausible against *every* known neighbour region. Any
    unresolved or missing location keeps the match (soft constraint), so noisy or
    sparse free-text location data can never silently drop a valid match.

    ``chain_context`` is optional: with no context (or no resolvable neighbour
    location) the stage is a pure pass-through, exactly as before location data
    became available.
    """
    if chain_context is None or not chain_context.adjacent_locations:
        return matches

    ref_regions = {
        region
        for loc in chain_context.adjacent_locations
        if (region := resolve_region(loc)) is not None
    }
    if not ref_regions:
        return matches

    filtered: list[Match] = []
    for m in matches:
        cand_location = m.candidate.death_location or m.candidate.birth_location
        cand_region = resolve_region(cand_location)
        if cand_region is None:
            # No usable location signal for this candidate — keep (soft constraint).
            filtered.append(m)
            continue
        if any(regions_plausible(cand_region, ref) for ref in ref_regions):
            filtered.append(m)
        # else: implausibly far from every known neighbour region — drop.

    return filtered


# ---------------------------------------------------------------------------
# Stage 5: Cross-reference match (blocked)
# ---------------------------------------------------------------------------
def _crossref_match(mention_norm: str, candidates: list[Candidate]) -> list[Match]:
    """Match via external IDs (e.g., muslimscholars.info).

    ``mention_norm`` must already be Arabic-normalized by the caller.
    """
    results: list[Match] = []
    if not mention_norm:
        return results
    for c in candidates:
        if c.external_id and c.name_ar_normalized:
            # If the candidate has an external ID and name is a partial match,
            # boost confidence via cross-reference.
            ratio = fuzz.ratio(mention_norm, c.name_ar_normalized)
            if ratio >= 60:
                results.append(Match(candidate=c, stage="crossref", score=round(ratio / 100.0, 4)))
    return results


def _crossref_match_blocked(mention_norm: str, index: BlockingIndex) -> list[Match]:
    """Cross-reference match restricted to same-prefix block."""
    results: list[Match] = []
    if not mention_norm:
        return results
    prefix = mention_norm[:_BLOCK_PREFIX_LEN]
    block = index.crossref_blocks.get(prefix, [])
    for c in block:
        if not (c.external_id and c.name_ar_normalized):
            continue
        ratio = fuzz.ratio(mention_norm, c.name_ar_normalized, score_cutoff=60)
        if ratio > 0:
            results.append(Match(candidate=c, stage="crossref", score=round(ratio / 100.0, 4)))
    return results


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------
def _load_candidates(staging_dir: Path) -> list[Candidate]:
    """Build candidate list from all narrators_bio_*.parquet files."""
    bio_files = sorted(staging_dir.glob("narrators_bio_*.parquet"))
    candidates: list[Candidate] = []

    for bf in bio_files:
        table = pq.read_table(bf)
        for i in range(table.num_rows):
            name_ar = safe_str(table.column("name_ar")[i].as_py())
            name_ar_norm = safe_str(table.column("name_ar_normalized")[i].as_py())
            if not name_ar_norm and name_ar:
                name_ar_norm = name_ar
            # da#376: the bio's key must live in the SAME identity surface the mention
            # key does, or Stage 1 can never exact-match across an inflection (and the
            # da#371 provenance drift leaves un-normalized bio keys nothing can match).
            name_ar_norm = canonical_surface(name_ar_norm) if name_ar_norm else None

            candidates.append(
                Candidate(
                    bio_id=table.column("bio_id")[i].as_py(),
                    name_ar=name_ar,
                    name_en=safe_str(table.column("name_en")[i].as_py()),
                    name_ar_normalized=name_ar_norm,
                    kunya=safe_str(table.column("kunya")[i].as_py()),
                    nisba=safe_str(table.column("nisba")[i].as_py()),
                    birth_year_ah=table.column("birth_year_ah")[i].as_py(),
                    death_year_ah=table.column("death_year_ah")[i].as_py(),
                    birth_location=safe_str(table.column("birth_location")[i].as_py()),
                    death_location=safe_str(table.column("death_location")[i].as_py()),
                    generation=safe_str(table.column("generation")[i].as_py()),
                    gender=safe_str(table.column("gender")[i].as_py()),
                    trustworthiness=safe_str(table.column("trustworthiness")[i].as_py()),
                    external_id=safe_str(table.column("external_id")[i].as_py()),
                    source=safe_str(table.column("source")[i].as_py()),
                )
            )

    logger.info(
        "candidates_loaded",
        bio_files=len(bio_files),
        total_candidates=len(candidates),
    )
    return candidates


def _iter_mention_batches(
    mentions_dir: Path,
    batch_size: int = _MENTION_BATCH_SIZE,
) -> Iterator[list[dict[str, str | int | float | None]]]:
    """Stream narrator mentions from Parquet in fixed-size batches.

    Reads the file using row-group-based iteration to avoid loading
    all 3.3M rows into memory at once. ``batch_size`` is the number of mentions
    per yielded batch; it is a parameter (rather than the bare module constant)
    so the checkpoint cadence and small-fixture tests can drive it deterministically
    — the batch sequence is a pure function of (file content, batch_size), which is
    what makes a resumed run's skip-then-continue byte-identical to a cold run.
    """
    path = mentions_dir / "narrator_mentions_resolved.parquet"
    if not path.exists():
        logger.warning("mentions_file_missing", path=str(path))
        return

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    logger.info("mentions_streaming_start", total_rows=total_rows, batch_size=batch_size)

    batch: list[dict[str, str | int | float | None]] = []
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        for i in range(table.num_rows):
            batch.append(
                {
                    "mention_id": table.column("mention_id")[i].as_py(),
                    "hadith_id": table.column("hadith_id")[i].as_py(),
                    "source_corpus": table.column("source_corpus")[i].as_py(),
                    "position_in_chain": table.column("position_in_chain")[i].as_py(),
                    "name_raw": safe_str(table.column("name_raw")[i].as_py()),
                    "name_normalized": safe_str(table.column("name_normalized")[i].as_py()),
                    "transmission_method": safe_str(table.column("transmission_method")[i].as_py()),
                }
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _count_mentions(mentions_dir: Path) -> int:
    """Return total mention count from Parquet metadata without reading data."""
    path = mentions_dir / "narrator_mentions_resolved.parquet"
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def _load_mentions(
    staging_dir: Path,
) -> list[dict[str, str | int | float | None]]:
    """Load narrator_mentions_resolved.parquet from NER stage.

    da#361: an absent mentions file is a missing required input (NER did not run
    or wrote elsewhere), not an empty result, so this raises rather than returning
    ``[]``. NOTE: this helper is currently uncalled — ``run`` uses
    :func:`_count_mentions` for the presence/size check — but it is kept honest so
    a future re-wiring inherits the fail-loud contract rather than the old silent
    empty. (Tracked for wire-up-or-remove; see da#361 discussion.)
    """
    path = staging_dir / "narrator_mentions_resolved.parquet"
    require_input(
        stage="disambiguate",
        present=path.exists(),
        input_desc=f"{path.name} (NER mention output)",
        produced_by="resolve NER stage",
        remediation="re-run resolve from the `ner` step; the mention output is absent",
    )

    table = pq.read_table(path)
    rows: list[dict[str, str | int | float | None]] = []
    for i in range(table.num_rows):
        rows.append(
            {
                "mention_id": table.column("mention_id")[i].as_py(),
                "hadith_id": table.column("hadith_id")[i].as_py(),
                "source_corpus": table.column("source_corpus")[i].as_py(),
                "position_in_chain": table.column("position_in_chain")[i].as_py(),
                "name_raw": safe_str(table.column("name_raw")[i].as_py()),
                "name_normalized": safe_str(table.column("name_normalized")[i].as_py()),
                "transmission_method": safe_str(table.column("transmission_method")[i].as_py()),
            }
        )
    logger.info("mentions_loaded", total=len(rows))
    return rows


def _backfill_mention_canonical_ids(
    mentions_path: Path,
    resolved: dict[str, tuple[str, float | None]],
) -> int:
    """Write resolved canonical ids + confidence back onto the mention rows.

    The NER stage writes ``narrator_mentions_resolved.parquet`` with
    ``canonical_narrator_id = None`` placeholders; disambiguation is the step
    that actually knows each mention's canonical narrator. Without rewriting
    these rows the graph edge/chain loaders — which read ``canonical_narrator_id``
    straight off the mentions — see only ``None`` and emit zero NARRATED edges
    and empty Chains (#109).

    ``resolved`` maps ``mention_id -> (canonical_id, confidence)`` and covers
    *every* mention that received a canonical narrator: both bio-matched mentions
    and the da#99 self-canonicalized (fallback) mentions, so the whole chain —
    not just the bio-backed positions — wires into the graph.

    Streams row-group by row-group and rewrites the file in place so the
    artifact stays self-contained for all downstream consumers and peak memory
    stays bounded (mirrors ``_iter_mention_batches``). Idempotent: re-running
    with the same ``resolved`` map yields the same file. Returns the number of
    mention rows backfilled.
    """
    if not mentions_path.exists():
        logger.warning("backfill_mentions_file_missing", path=str(mentions_path))
        return 0

    pf = pq.ParquetFile(mentions_path)
    total_rows = pf.metadata.num_rows
    id_idx = NARRATOR_MENTIONS_RESOLVED_SCHEMA.get_field_index("canonical_narrator_id")
    conf_idx = NARRATOR_MENTIONS_RESOLVED_SCHEMA.get_field_index("confidence")

    tmp_path = mentions_path.with_name(mentions_path.name + ".backfill.tmp")
    writer = pq.ParquetWriter(tmp_path, NARRATOR_MENTIONS_RESOLVED_SCHEMA, compression="snappy")
    backfilled = 0
    try:
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx).cast(NARRATOR_MENTIONS_RESOLVED_SCHEMA)
            mention_ids = table.column("mention_id").to_pylist()
            existing_conf = table.column("confidence").to_pylist()

            new_ids: list[str | None] = []
            new_conf: list[float | None] = []
            for mid, conf in zip(mention_ids, existing_conf, strict=True):
                hit = resolved.get(str(mid))
                if hit is None:
                    # Unresolved mention — leave the NER placeholder untouched.
                    new_ids.append(None)
                    new_conf.append(conf)
                else:
                    new_ids.append(hit[0])
                    new_conf.append(hit[1])
                    backfilled += 1

            table = table.set_column(
                id_idx, "canonical_narrator_id", pa.array(new_ids, type=pa.string())
            )
            table = table.set_column(conf_idx, "confidence", pa.array(new_conf, type=pa.float32()))
            writer.write_table(table)
    finally:
        writer.close()

    tmp_path.replace(mentions_path)
    logger.info(
        "backfill_mentions_complete",
        path=str(mentions_path),
        backfilled=backfilled,
        total=total_rows,
    )
    return backfilled


# ---------------------------------------------------------------------------
# Mononym-split evidence (da#248)
# ---------------------------------------------------------------------------
def _adjacent_death_years(
    death_year_index: dict[str, int | None], hadith_id: str, position: int
) -> list[int]:
    """Death years of the mention's immediate chain neighbours (positions ±1).

    Reads the incrementally-populated ``death_year_index`` (the same soft signal
    the temporal filter uses), returning only the neighbours already resolved to a
    dated candidate. This is the chain-context evidence the da#248 mononym split
    uses to re-resolve a bare over-merged mononym to a specific person.
    """
    years: list[int] = []
    for offset in (-1, 1):
        year = death_year_index.get(f"{hadith_id}:{position + offset}")
        if year is not None:
            years.append(year)
    return years


# ---------------------------------------------------------------------------
# Canonical ID generation
# ---------------------------------------------------------------------------
def _make_canonical_id(name_normalized: str) -> str:
    """Deterministic canonical ID — delegates to the identity contract.

    Thin wrapper over :func:`src.parse.identity.make_canonical_id` so the
    mention-driven path and the bio-direct promoter share one ``nar:<uuid5>``
    scheme and namespace (no drift). Kept as a local name for call-site brevity.
    """
    return make_canonical_id(name_normalized)


# ---------------------------------------------------------------------------
# Core disambiguation logic
# ---------------------------------------------------------------------------
def _disambiguate_mention(
    mention: dict[str, str | int | float | None],
    candidates: list[Candidate],
    death_year_index: dict[str, int | None],
    location_index: dict[str, str] | None = None,
) -> tuple[Match | None, list[Match]]:
    """Run 5-stage pipeline on a single mention. Return (best_match, all_matches)."""
    raw_name = str(mention.get("name_normalized") or mention.get("name_raw") or "")
    if not raw_name:
        return None, []

    # da#438: match on the `canonical_surface` identity surface, the SAME surface the
    # candidate index (`_load_candidates`) and the mint (`_make_canonical_id`) key on —
    # not bare `normalize_arabic`, which left Stage 1 unable to exact-match a byte-
    # identical bio across a fold the mint had already applied. All stages receive it.
    name = canonical_surface(raw_name) if raw_name else ""
    if not name:
        return None, []

    # Build chain context for temporal filtering.
    hadith_id = str(mention.get("hadith_id", ""))
    position = int(mention.get("position_in_chain") or 0)
    source_corpus = str(mention.get("source_corpus", ""))

    adjacent_years: list[int] = []
    adjacent_locations: list[str] = []
    # Look up adjacent narrators' death years + locations from the indexes.
    for offset in (-1, 1):
        key = f"{hadith_id}:{position + offset}"
        year = death_year_index.get(key)
        if year is not None:
            adjacent_years.append(year)
        if location_index is not None:
            loc = location_index.get(key)
            if loc:
                adjacent_locations.append(loc)

    chain_ctx = ChainContext(
        hadith_id=hadith_id,
        position_in_chain=position,
        source_corpus=source_corpus,
        adjacent_death_years=adjacent_years,
        adjacent_locations=adjacent_locations,
    )

    all_matches: list[Match] = []

    # Stage 1: Exact match
    exact = _exact_match(name, candidates)
    if exact:
        all_matches.extend(exact)
        return exact[0], all_matches

    # Stage 2: Fuzzy match
    fuzzy = _fuzzy_match(name, candidates)
    if fuzzy:
        # Stage 3: Temporal filter
        fuzzy = _temporal_filter(fuzzy, chain_ctx)
        # Stage 4: Geographic filter
        fuzzy = _geographic_filter(fuzzy, chain_ctx)
        all_matches.extend(fuzzy)
        if fuzzy:
            best = max(fuzzy, key=lambda m: m.score)
            return best, all_matches

    # Stage 5: Cross-reference match
    # da#356: crossref is the LOOSER stage (ratio >= 60, no Levenshtein bound) and
    # produced 92% of the pre-fix chimeric nodes, yet it alone skipped the temporal
    # and geographic filters. It is now filtered exactly as the fuzzy stage is.
    crossref = _crossref_match(name, candidates)
    if crossref:
        crossref = _temporal_filter(crossref, chain_ctx)
        crossref = _geographic_filter(crossref, chain_ctx)
        all_matches.extend(crossref)
        if crossref:
            best = max(crossref, key=lambda m: m.score)
            return best, all_matches

    return None, all_matches


def _disambiguate_mention_indexed(
    mention: dict[str, str | int | float | None],
    index: BlockingIndex,
    death_year_index: dict[str, int | None],
    location_index: dict[str, str] | None = None,
) -> tuple[Match | None, list[Match]]:
    """Run 5-stage pipeline using blocking indexes for fast lookup."""
    raw_name = str(mention.get("name_normalized") or mention.get("name_raw") or "")
    if not raw_name:
        return None, []

    name = canonical_surface(raw_name) if raw_name else ""
    if not name:
        return None, []

    hadith_id = str(mention.get("hadith_id", ""))
    position = int(mention.get("position_in_chain") or 0)
    source_corpus = str(mention.get("source_corpus", ""))

    adjacent_years: list[int] = []
    adjacent_locations: list[str] = []
    for offset in (-1, 1):
        key = f"{hadith_id}:{position + offset}"
        year = death_year_index.get(key)
        if year is not None:
            adjacent_years.append(year)
        if location_index is not None:
            loc = location_index.get(key)
            if loc:
                adjacent_locations.append(loc)

    chain_ctx = ChainContext(
        hadith_id=hadith_id,
        position_in_chain=position,
        source_corpus=source_corpus,
        adjacent_death_years=adjacent_years,
        adjacent_locations=adjacent_locations,
    )

    all_matches: list[Match] = []

    # Stage 1: Exact match (O(1) dict lookup)
    exact = _exact_match_indexed(name, index)
    if exact:
        all_matches.extend(exact)
        return exact[0], all_matches

    # Stage 2: Fuzzy match (blocked — only compare within same prefix)
    fuzzy = _fuzzy_match_blocked(name, index)
    if fuzzy:
        # Stage 3: Temporal filter
        fuzzy = _temporal_filter(fuzzy, chain_ctx)
        # Stage 4: Geographic filter
        fuzzy = _geographic_filter(fuzzy, chain_ctx)
        all_matches.extend(fuzzy)
        if fuzzy:
            best = max(fuzzy, key=lambda m: m.score)
            return best, all_matches

    # Stage 5: Cross-reference match (blocked) — filtered as of da#356; see the
    # twin in `_disambiguate_mention` for why.
    crossref = _crossref_match_blocked(name, index)
    if crossref:
        crossref = _temporal_filter(crossref, chain_ctx)
        crossref = _geographic_filter(crossref, chain_ctx)
        all_matches.extend(crossref)
        if crossref:
            best = max(crossref, key=lambda m: m.score)
            return best, all_matches

    return None, all_matches


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def _build_canonical_table(
    canonical_map: dict[str, dict[str, str | int | list[str] | None]],
) -> pa.Table:
    """Build narrators_canonical Parquet table."""

    def _corpora(r: dict[str, str | int | list[str] | None]) -> list[str]:
        v = r.get("source_corpora")
        return v if isinstance(v, list) else []

    rows = list(canonical_map.values())
    if not rows:
        return pa.table(
            {f.name: pa.array([], type=f.type) for f in NARRATORS_CANONICAL_SCHEMA},
            schema=NARRATORS_CANONICAL_SCHEMA,
        )

    arrays: dict[str, pa.Array] = {
        "canonical_id": pa.array([r["canonical_id"] for r in rows], type=pa.string()),
        "name_ar": pa.array([r.get("name_ar") for r in rows], type=pa.string()),
        "name_en": pa.array([r.get("name_en") for r in rows], type=pa.string()),
        "name_ar_normalized": pa.array(
            [r.get("name_ar_normalized") for r in rows], type=pa.string()
        ),
        "aliases": pa.array([r.get("aliases") or [] for r in rows], type=pa.list_(pa.string())),
        "birth_year_ah": pa.array([r.get("birth_year_ah") for r in rows], type=pa.int32()),
        "death_year_ah": pa.array([r.get("death_year_ah") for r in rows], type=pa.int32()),
        # Date uncertainty bounds + precision (da#162, mirrors Narrator da#161).
        # The disambiguator does not yet derive these — the date-parse/reconcile/
        # ṭabaqa-fallback stages (da#164/#165/#166) populate them downstream — so
        # bounds default null and precision defaults to UNKNOWN, matching the model.
        "birth_year_ah_earliest": pa.array(
            [r.get("birth_year_ah_earliest") for r in rows], type=pa.int32()
        ),
        "birth_year_ah_latest": pa.array(
            [r.get("birth_year_ah_latest") for r in rows], type=pa.int32()
        ),
        "birth_date_precision": pa.array(
            [r.get("birth_date_precision") or DatePrecision.UNKNOWN.value for r in rows],
            type=pa.string(),
        ),
        "death_year_ah_earliest": pa.array(
            [r.get("death_year_ah_earliest") for r in rows], type=pa.int32()
        ),
        "death_year_ah_latest": pa.array(
            [r.get("death_year_ah_latest") for r in rows], type=pa.int32()
        ),
        "death_date_precision": pa.array(
            [r.get("death_date_precision") or DatePrecision.UNKNOWN.value for r in rows],
            type=pa.string(),
        ),
        "generation": pa.array([r.get("generation") for r in rows], type=pa.string()),
        "gender": pa.array([r.get("gender") for r in rows], type=pa.string()),
        "trustworthiness": pa.array([r.get("trustworthiness") for r in rows], type=pa.string()),
        "source_ids": pa.array(
            [r.get("source_ids") or [] for r in rows], type=pa.list_(pa.string())
        ),
        "external_id": pa.array([r.get("external_id") for r in rows], type=pa.string()),
        "death_year_provenance": pa.array(
            [r.get("death_year_provenance") for r in rows], type=pa.string()
        ),
        "mention_count": pa.array([r.get("mention_count", 0) for r in rows], type=pa.int32()),
        # Attestation tag derived from the row's final mention_count (da#370). A
        # disambiguated narrator comes from real chain mentions, so this is
        # "isnad_attested" for every row here; derived (not hard-coded) so the one
        # helper stays the single source of truth across all producers.
        "attestation": pa.array(
            [derive_attestation(r.get("mention_count", 0)) for r in rows], type=pa.string()
        ),
        # Sect/corpus provenance finalized from the accumulated corpora set (da#103).
        "source_corpus": pa.array([primary_corpus(_corpora(r)) for r in rows], type=pa.string()),
        "source_corpora": pa.array(
            [sorted(set(_corpora(r))) for r in rows], type=pa.list_(pa.string())
        ),
        "sect_affiliation": pa.array(
            [derive_sect_affiliation(_corpora(r)) for r in rows], type=pa.string()
        ),
        # Over-merge flag (da#445) is set later by the ``over_merged_flag`` stage from
        # the curated seed, never here — a freshly-disambiguated node defaults to
        # unflagged (None; the graph loader reads that as False).
        "over_merged": pa.array([r.get("over_merged") for r in rows], type=pa.bool_()),
        "over_merge_note": pa.array([r.get("over_merge_note") for r in rows], type=pa.string()),
    }
    return pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)


def _build_ambiguous_csv(
    ambiguous_rows: list[dict[str, str | float | None]],
    output_path: Path,
) -> None:
    """Write ambiguous_narrators.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in AMBIGUOUS_NARRATORS_SCHEMA]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ambiguous_rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Merge log schema
# ---------------------------------------------------------------------------
_MERGE_LOG_SCHEMA = pa.schema(
    [
        pa.field("canonical_id", pa.string(), nullable=False),
        pa.field("mention_id", pa.string(), nullable=False),
        pa.field("mention_text", pa.string(), nullable=True),
        pa.field("merge_stage", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
)


def _upsert_canonical(
    canonical_map: dict[str, dict[str, str | int | list[str] | None]],
    canonical_id: str,
    *,
    norm_name: str,
    name_ar: str | None,
    name_en: str | None,
    alias: str | None,
    candidate: Candidate | None,
    corpus: str | None = None,
    bio_alias: str | None = None,
    death_year_provenance: str | None = None,
) -> None:
    """Create or merge the canonical narrator record for *canonical_id*.

    ``canonical_id`` is a pure function of ``norm_name`` (:func:`_make_canonical_id`),
    so a bio-matched mention and a bio-less mention of the *same* normalized name
    — from any number of sources — converge on **one** record. This is the
    cross-source collapse for da#99.

    A biographical *candidate*, when supplied, enriches the record by filling any
    still-empty biographical field, so resolution order never loses bio metadata
    (a bio-less mention seen first leaves a minimal record that a later
    bio-matched mention of the same name upgrades in place).

    da#356 — **the candidate may enrich, but it may never rename.** ``name_ar`` /
    ``name_en`` / ``name_ar_normalized`` are taken from the caller's mention-derived
    arguments and are never back-filled from ``candidate``: doing so is exactly how
    a mention ``عائشة`` came to be displayed as the OCR-corrupt bio spelling
    ``عائذة``. A caller that has established true name-identity (the exact stage)
    passes the bio's ``name_en`` explicitly. The bio's own spelling is preserved as
    a searchable ``bio_alias`` rather than as the record's name.
    """
    rec = canonical_map.get(canonical_id)
    if rec is None:
        rec = {
            "canonical_id": canonical_id,
            "name_ar": name_ar,
            "name_en": name_en,
            "name_ar_normalized": norm_name,
            "aliases": [],
            "birth_year_ah": None,
            "death_year_ah": None,
            "generation": None,
            "gender": None,
            "trustworthiness": None,
            "source_ids": [],
            "external_id": None,
            "death_year_provenance": None,
            "mention_count": 0,
            # Corpora this canonical narrator has been observed in (da#103). The
            # scalar ``source_corpus`` + derived ``sect_affiliation`` are finalized
            # from this set in :func:`_build_canonical_table`.
            "source_corpora": [],
        }
        canonical_map[canonical_id] = rec

    norm_corpus = normalize_corpus(corpus)
    if norm_corpus:
        corpora = rec.get("source_corpora")
        if isinstance(corpora, list) and norm_corpus not in corpora:
            corpora.append(norm_corpus)

    raw_count = rec.get("mention_count")
    rec["mention_count"] = (int(raw_count) if isinstance(raw_count, int | str) else 0) + 1

    # Display names come from the mention (or from the caller's name-identical bio),
    # never from `candidate` — see the da#356 note above.
    if not rec.get("name_ar") and name_ar:
        rec["name_ar"] = name_ar
    if not rec.get("name_en") and name_en:
        rec["name_en"] = name_en

    if candidate is not None:
        for field_name, value in (
            ("birth_year_ah", candidate.birth_year_ah),
            ("death_year_ah", candidate.death_year_ah),
            ("generation", candidate.generation),
            ("gender", candidate.gender),
            ("trustworthiness", candidate.trustworthiness),
            ("external_id", candidate.external_id),
        ):
            if rec.get(field_name) is None and value is not None:
                rec[field_name] = value
        # Provenance travels with the year it describes: set it exactly when this call
        # is the one that fills `death_year_ah`, never retroactively over another bio's.
        if rec.get("death_year_provenance") is None and rec.get("death_year_ah") is not None:
            rec["death_year_provenance"] = death_year_provenance

        src_ids = rec.get("source_ids")
        if isinstance(src_ids, list) and candidate.bio_id and candidate.bio_id not in src_ids:
            src_ids.append(candidate.bio_id)

    # The mention's own surface, and (da#356) the matched bio's spelling, both stay
    # searchable as aliases. Pre-fix the *correct* spelling was demoted to an alias
    # of a corrupt-bio-named node; now the corrupt spelling is the alias.
    # da#391: a bare relational-pronoun surface ("خالته"/"عمتي"/"امتاه") is dropped
    # here — clean_narrator_name already drops it from the NAME at NER time, but the
    # aliases list<string> is the blind spot it never reaches, so the deixis survives
    # as an alias of a real narrator. Same closed kinship lexicon, no name adjudication.
    for candidate_alias in (alias, bio_alias):
        if (
            candidate_alias
            and candidate_alias != norm_name
            and not is_mubham_relational(candidate_alias)
        ):
            aliases = rec.get("aliases")
            if isinstance(aliases, list) and candidate_alias not in aliases:
                aliases.append(candidate_alias)


# ---------------------------------------------------------------------------
# Crash-resume checkpoint plumbing (da#268, unified under src/resolve/_checkpoint
# by da#272 — dir/save/load/clear/cadence + the column hasher are shared; only
# this stage's fingerprint SPLIT and state SHAPE stay local).
# ---------------------------------------------------------------------------
# Columns hashed for the input fingerprint. Split into the STABLE disambiguation
# drivers and the mention_id join key so a resume can tell a changed corpus apart
# from an identical-content NER *rewrite* that only regenerated the random uuid4
# mention_ids (ner.py mints mention_id via uuid.uuid4()).
_FINGERPRINT_CONTENT_COLS = (
    "hadith_id",
    "source_corpus",
    "position_in_chain",
    "name_raw",
    "name_normalized",
    "transmission_method",
)


def _compute_input_fingerprint(mentions_path: Path) -> tuple[str, str, int]:
    """Content fingerprint of the disambiguation input, independent of file mtime.

    Returns ``(content_hash, mention_id_hash, total_rows)``. ``content_hash`` covers
    the STABLE disambiguation-driving columns in row order; ``mention_id_hash`` covers
    the NER-minted ``mention_id`` column. They are split so a resume can distinguish a
    genuinely changed corpus (``content_hash`` differs ⇒ cold start) from an
    identical-content NER rewrite that merely re-randomised ``mention_id`` (content
    matches, id differs ⇒ cold start *with a precise "reuse the file via --from-step
    disambiguate" hint*, since ``mention_id`` is the backfill/audit join key baked into
    the outputs). Excludes ``canonical_narrator_id``/``confidence`` (rewritten by the
    end-of-run backfill) and mtime.

    Delegates the streaming SHA-256 to the shared
    :func:`~src.resolve._checkpoint.hash_parquet_column_groups`, which hashes both
    groups in a single row-group pass (peak memory one row-group, not the whole
    3.3M-row file). The per-group digests are byte-identical to hashing each
    column set on its own, so a checkpoint written by the pre-da#272 code resumes
    unchanged.
    """
    digests, total = hash_parquet_column_groups(
        mentions_path,
        {"content": _FINGERPRINT_CONTENT_COLS, "mention_id": ("mention_id",)},
    )
    return digests["content"], digests["mention_id"], total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(
    staging_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = _MENTION_BATCH_SIZE,
    checkpoint_every_n_batches: int | None = None,
    resume: bool = True,
    stop_after: int | None = None,
) -> list[Path]:
    """Disambiguate narrator mentions to canonical narrator records.

    Multi-stage pipeline: exact match, fuzzy match, temporal filter,
    geographic filter, cross-reference match.

    Uses blocking indexes and batch streaming to handle 3M+ mentions
    within <30min and <4GB peak memory.

    Crash-resumable (da#268): the streaming loop persists its accumulated state
    (canonical registry, chain-context indexes, resolved/merge/ambiguous rows,
    processed offset) to ``staging_dir/.disambiguate_checkpoint/`` every
    ``checkpoint_every_n_batches`` batches, fingerprinting the input so a re-run
    resumes from the last checkpoint instead of restarting the multi-hour stage.
    On start, a checkpoint whose fingerprint matches the current input resumes
    (``disambiguate_resume`` event); a mismatch or absence cold-starts. The
    checkpoint dir is removed after a successful complete run.

    Because ``mention_id`` is NER-minted per run (``uuid.uuid4()``) and is baked
    into the outputs, an output-identical resume requires reading the SAME
    mentions file — so the sanctioned recovery command is
    ``resolve --from-step disambiguate`` (skips NER, reuses the existing file). A
    bare re-run that regenerates NER ids cold-starts safely (never corrupt).

    Parameters
    ----------
    batch_size:
        Mentions per streaming batch (checkpoints land on batch boundaries).
    checkpoint_every_n_batches:
        Checkpoint cadence override; ``None`` resolves from the
        ``DISAMBIGUATE_CHECKPOINT_EVERY_N_BATCHES`` env var then the default.
    resume:
        When ``False``, ignore and clear any existing checkpoint and cold-start
        (still checkpoints forward). Default ``True``.
    stop_after:
        Bounded partial-run probe (da#276): stop cleanly after this many
        checkpoint writes, leaving the checkpoint on disk (a later bare run
        resumes from it) and WITHOUT writing final output — raises
        :class:`~src.resolve._checkpoint.StopAfterReached`. ``None`` runs to
        completion.
    """
    logger.info(
        "disambiguate_run_start",
        staging_dir=str(staging_dir),
        output_dir=str(output_dir),
    )

    # Required input: candidate bio files (produced by parse). da#361 distinguishes
    # an ABSENT input (no bio shards at all — an upstream defect) from a PRESENT one
    # that is genuinely empty (shards exist, zero candidate rows). Absent → raise;
    # present-but-empty → the honest ``disambiguate_no_candidates`` empty return.
    bio_files = sorted(staging_dir.glob("narrators_bio_*.parquet"))
    require_input(
        stage="disambiguate",
        present=bool(bio_files),
        input_desc=f"narrators_bio_*.parquet under {staging_dir} (candidate narrator profiles)",
        produced_by="parse (bio adapters)",
        remediation="re-run `parse`; disambiguation has no candidate profiles to match against",
    )

    # Load candidates and build blocking index.
    candidates = _load_candidates(staging_dir)

    if not candidates:
        # Bio shards present but zero candidate rows — an honest empty, NOT a
        # missing input (distinct from the raise above), so it stays a warning.
        logger.warning("disambiguate_no_candidates")
        return []

    index = _build_blocking_index(candidates)

    # Check mention count without loading data. da#361: an absent/zero mentions file
    # is NOT a missing input here — NER legitimately produces zero mentions for a
    # bio-only corpus (e.g. muhaddithat, which NER skips) or an all-null-isnad
    # subset. That is "an input that ran and legitimately produced nothing", which
    # the da#361 defect definition explicitly excludes, so this stays an honest
    # empty return, NOT a ``require_input`` raise. (The bio candidates above ARE
    # required and DO raise when absent.)
    total_mentions = _count_mentions(output_dir)
    if total_mentions == 0:
        logger.warning("disambiguate_no_mentions")
        return []

    logger.info("disambiguate_processing", total_mentions=total_mentions)

    # Build death-year index for temporal filtering:
    # key = "hadith_id:position" → death_year_ah of the resolved candidate.
    # We populate this incrementally as we resolve mentions.
    death_year_index: dict[str, int | None] = {}

    # Parallel location index for the geographic filter (da#139):
    # key = "hadith_id:position" → free-text birth/death location of the resolved
    # candidate. Populated only for bio-matched mentions whose candidate carried a
    # location; absent neighbours simply contribute no geographic signal.
    location_index: dict[str, str] = {}

    # Canonical narrator accumulator: normalized_name → metadata dict.
    canonical_map: dict[str, dict[str, str | int | list[str] | None]] = {}
    merge_log_rows: list[dict[str, str | float | None]] = []
    ambiguous_rows: list[dict[str, str | float | None]] = []

    # mention_id -> (canonical_id, confidence) for the #109 backfill. Captured
    # for EVERY mention that received a canonical narrator — bio-matched and
    # self-canonicalized (da#99) alike — so the full chain wires into the graph.
    resolved_map: dict[str, tuple[str, float | None]] = {}

    # Cross-source collapse measurement (da#99): the naive baseline keeps one
    # node per (source, canonical) pair; the collapsed count is distinct
    # canonicals. naive > collapsed exactly when a narrator spans >1 source.
    naive_identity_pairs: set[tuple[str, str]] = set()

    # Per-source counters.
    source_resolved: dict[str, int] = {}
    source_total: dict[str, int] = {}

    processed = 0
    # Confident-by-score bio matches the da#356 corroboration gate withheld metadata
    # from. Observability only — not checkpointed, so it counts the current segment.
    gate_rejected = 0

    # --- Crash-resume (da#268): fingerprint the input and try to restore state.
    cadence = resolve_cadence(
        checkpoint_every_n_batches,
        "DISAMBIGUATE_CHECKPOINT_EVERY_N_BATCHES",
        _CHECKPOINT_EVERY_N_BATCHES,
    )
    ckpt_dir = checkpoint_dir(staging_dir, "disambiguate")
    mentions_path = output_dir / "narrator_mentions_resolved.parquet"
    content_hash, mention_id_hash, _fp_rows = _compute_input_fingerprint(mentions_path)
    mentions_to_skip = 0

    if not resume:
        clear_checkpoint(ckpt_dir)
    else:
        ckpt = load_checkpoint(ckpt_dir)
        if ckpt is not None:
            valid_layout = ckpt.get("schema_version") == _CHECKPOINT_SCHEMA_VERSION
            content_ok = ckpt.get("content_hash") == content_hash
            ids_ok = ckpt.get("mention_id_hash") == mention_id_hash
            if valid_layout and content_ok and ids_ok:
                # Restore every accumulator exactly as it stood at the checkpoint,
                # then skip the already-consumed prefix so the continuation is
                # byte-identical to a cold run.
                death_year_index = dict(ckpt["death_year_index"])
                location_index = dict(ckpt["location_index"])
                canonical_map = dict(ckpt["canonical_map"])
                merge_log_rows = list(ckpt["merge_log_rows"])
                ambiguous_rows = list(ckpt["ambiguous_rows"])
                resolved_map = {
                    str(k): (str(v[0]), v[1]) for k, v in dict(ckpt["resolved_map"]).items()
                }
                naive_identity_pairs = {
                    (str(pair[0]), str(pair[1])) for pair in ckpt["naive_identity_pairs"]
                }
                source_resolved = dict(ckpt["source_resolved"])
                source_total = dict(ckpt["source_total"])
                processed = int(ckpt["processed"])
                mentions_to_skip = processed
                logger.info(
                    "disambiguate_resume",
                    resumed_from=processed,
                    total=total_mentions,
                    pct=round(processed / max(total_mentions, 1) * 100, 1),
                    canonical=len(canonical_map),
                )
            elif valid_layout and content_ok and not ids_ok:
                # Same corpus content, but mention_ids were regenerated (an NER
                # rewrite — mention_id is a per-run uuid4). Resuming would splice
                # stale ids into the mention-keyed outputs, so cold-start and tell
                # the operator how to make the resume compose (reuse the file).
                logger.warning(
                    "disambiguate_checkpoint_stale_mention_ids",
                    hint="input content matches but mention_ids were rewritten; "
                    "rerun with `resolve --from-step disambiguate` to reuse the "
                    "existing mentions file and resume the checkpoint",
                )
                clear_checkpoint(ckpt_dir)
            else:
                logger.info("disambiguate_checkpoint_mismatch", reason="cold start")
                clear_checkpoint(ckpt_dir)

    def _snapshot() -> dict[str, Any]:
        """Serialize the current accumulated state for the checkpoint."""
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "content_hash": content_hash,
            "mention_id_hash": mention_id_hash,
            "total_rows": total_mentions,
            "batch_size": batch_size,
            "processed": processed,
            "death_year_index": death_year_index,
            "location_index": location_index,
            "canonical_map": canonical_map,
            "merge_log_rows": merge_log_rows,
            "ambiguous_rows": ambiguous_rows,
            "resolved_map": {k: [v[0], v[1]] for k, v in resolved_map.items()},
            "naive_identity_pairs": [[a, b] for (a, b) in naive_identity_pairs],
            "source_resolved": source_resolved,
            "source_total": source_total,
        }

    controller = CheckpointController(cadence, stop_after=stop_after)

    for batch in _iter_mention_batches(output_dir, batch_size=batch_size):
        # Skip the prefix already processed before the resume point. Checkpoints
        # land on batch boundaries so this normally drops whole batches; the
        # partial-slice branch is a safety net for any non-aligned offset.
        if mentions_to_skip > 0:
            if mentions_to_skip >= len(batch):
                mentions_to_skip -= len(batch)
                continue
            batch = batch[mentions_to_skip:]
            mentions_to_skip = 0

        for mention in batch:
            corpus = str(mention.get("source_corpus", "unknown"))
            source_total[corpus] = source_total.get(corpus, 0) + 1

            best, all_matches = _disambiguate_mention_indexed(
                mention, index, death_year_index, location_index
            )

            mention_id = str(mention.get("mention_id", ""))
            mention_text = str(mention.get("name_normalized") or mention.get("name_raw") or "")
            hadith_id = str(mention.get("hadith_id", ""))
            position = int(mention.get("position_in_chain") or 0)

            # da#356 — ONE identity source. The canonical id and the display name are a
            # pure function of the MENTION's surface; a bio candidate may only enrich.
            #
            # da#376 — that surface is `canonical_surface`, not bare `normalize_arabic`.
            # It is re-derived here rather than trusting the stored `name_normalized`
            # (which da#371 shows is written by a *different* function), so the id keys
            # on the very surface the matcher compared. And it folds the Arabic case
            # endings that `normalize_arabic` leaves alone: without it, de-keying from
            # the bio mints `ابي هريره` / `ابا هريره` / `ابو هريره` as three Abū
            # Hurayras, and `fuzzy_cluster` cannot repair a name with fewer than two
            # significant tokens — it generates no blocking key and is never scored.
            mention_norm = canonical_surface(mention_text)
            adjacent = _adjacent_death_years(death_year_index, hadith_id, position)

            # da#356/da#376 — corroboration gates the IDENTITY FEEDBACK LOOP, not the
            # data. A confidently-matched bio's metadata is persisted either way, tagged
            # with its provenance; only a *corroborated* year is allowed to re-enter
            # `death_year_index`, which feeds `_temporal_filter` for chain neighbours and
            # `refine_mononym_name`'s da#248 evidence.
            #
            # Dropping the year outright (the first cut of this gate) was worse than
            # keeping it: `fuzzy_cluster._death_years_conflict` returns False when EITHER
            # year is None, so a missing year makes a merge *permitted*. Withholding the
            # year therefore disarmed the very precision guard the clustering pass we now
            # depend on uses to keep two different men apart.
            attached: Candidate | None = None
            corroborated = False
            if best and best.score >= _CONFIDENCE_THRESHOLD:
                attached = best.candidate
                corroborated = _bio_corroborated(best, mention_norm, adjacent)
                if not corroborated:
                    gate_rejected += 1

            if mention_norm:
                # da#248: split an over-merged bare mononym (e.g. سفيان) into the
                # specific person the chain neighbours' generations select. This is the
                # ONE legitimate re-key away from the mention's surface — it is driven
                # by chain evidence, never by a bio name. It abstains for every
                # non-registered name and for ambiguous/absent evidence, so no
                # single-person node is fragmented. Note it now keys on the mention's
                # form: pre-fix it was handed the *bio's* name (which the resolved
                # branch had already substituted), so the registry lookup missed.
                person = refine_mononym_name(mention_norm, adjacent)
                if person is not None:
                    # The refined person is not the matched bio, so the bio's dates /
                    # external_id must not be stamped onto them.
                    norm_name = person.norm_name
                    name_ar: str | None = person.name_ar
                    name_en: str | None = None
                    attached = None
                    corroborated = False
                else:
                    norm_name = mention_norm
                    name_ar = str(mention.get("name_raw") or "") or mention_text
                    # An English name may be borrowed only from a bio whose normalized
                    # name IS this name — otherwise it renames the node in translation.
                    name_en = (
                        attached.name_en
                        if attached is not None and attached.name_ar_normalized == norm_name
                        else None
                    )

                canonical_id = _make_canonical_id(norm_name)

                # Update death-year + location indexes for chain context (da#266).
                # On a mononym split the slot must carry the REFINED person's real
                # death year, not the ambiguous pre-split mononym bio's — else a
                # chain-adjacent registered mononym reads a stale year and can
                # mis-select among genuinely-distinct persons. The registry carries no
                # per-person location, so a split contributes no geographic signal.
                # An UNCORROBORATED bio contributes neither (da#356): its year would
                # poison the temporal filter and the da#248 evidence downstream. The year
                # is still PERSISTED on the record, tagged `uncorroborated` — it just may
                # not steer the chain context that decides other narrators' identities.
                if person is not None:
                    death_year_index[f"{hadith_id}:{position}"] = person.death_year_ah
                elif attached is not None and corroborated:
                    death_year_index[f"{hadith_id}:{position}"] = attached.death_year_ah
                    cand_location = attached.death_location or attached.birth_location
                    if cand_location:
                        location_index[f"{hadith_id}:{position}"] = cand_location

                _upsert_canonical(
                    canonical_map,
                    canonical_id,
                    norm_name=norm_name,
                    name_ar=name_ar,
                    name_en=name_en,
                    alias=canonical_surface(mention_text),
                    candidate=attached,
                    corpus=corpus,
                    # da#376: `aliases` is consumed by `fuzzy_cluster._match_keys` as a
                    # NORMALIZED key space (it feeds `_significant_tokens` and
                    # `token_set_ratio` alongside `name_ar_normalized`). Passing the bio's
                    # raw `name_ar` injected an un-normalized string into it — diacritics,
                    # hamza variants and all — which scores against nothing.
                    bio_alias=(canonical_surface(attached.name_ar or "") if attached else None),
                    death_year_provenance=(
                        None
                        if attached is None or attached.death_year_ah is None
                        else ("corroborated" if corroborated else "uncorroborated")
                    ),
                )
                naive_identity_pairs.add((corpus, canonical_id))

                if attached is not None and best is not None:
                    source_resolved[corpus] = source_resolved.get(corpus, 0) + 1
                    resolved_map[mention_id] = (canonical_id, float(best.score))
                    merge_log_rows.append(
                        {
                            "canonical_id": canonical_id,
                            "mention_id": mention_id,
                            "mention_text": mention_text,
                            "merge_stage": best.stage,
                            "score": best.score,
                        }
                    )
                else:
                    # Self-canonicalized: the node exists and wires the NARRATED edge /
                    # Chain (#109), but no bio corroborated it, so confidence is unknown.
                    resolved_map[mention_id] = (canonical_id, None)

            if attached is None:
                # Record the top-3 bio candidates for the ambiguous-audit report
                # (a self-canonicalized node was still created above when named).
                top3 = sorted(all_matches, key=lambda m: m.score, reverse=True)[:3]
                row: dict[str, str | float | None] = {
                    "mention_id": mention_id,
                    "mention_text": mention_text,
                    "source_corpus": corpus,
                }
                for idx in range(3):
                    n = idx + 1
                    if idx < len(top3):
                        m = top3[idx]
                        norm = m.candidate.name_ar_normalized or ""
                        row[f"candidate_{n}_id"] = _make_canonical_id(norm)
                        row[f"candidate_{n}_name"] = (
                            m.candidate.name_ar or m.candidate.name_en or ""
                        )
                        row[f"candidate_{n}_score"] = m.score
                        row[f"candidate_{n}_stage"] = m.stage
                    else:
                        row[f"candidate_{n}_id"] = None
                        row[f"candidate_{n}_name"] = None
                        row[f"candidate_{n}_score"] = None
                        row[f"candidate_{n}_stage"] = None
                ambiguous_rows.append(row)

            processed += 1
            if processed % _PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    "disambiguate_progress",
                    processed=processed,
                    total=total_mentions,
                    pct=round(processed / total_mentions * 100, 1),
                    resolved=sum(source_resolved.values()),
                    canonical=len(canonical_map),
                    gate_rejected=gate_rejected,
                )

        # Checkpoint on batch boundaries (da#268). ``processed`` is batch-aligned
        # here, so a resume skips whole batches and continues identically.
        if controller.batch_complete():
            save_checkpoint(ckpt_dir, _snapshot())
            logger.info(
                "disambiguate_checkpoint_saved",
                processed=processed,
                total=total_mentions,
                canonical=len(canonical_map),
            )
            if controller.checkpoint_written():
                # --stop-after budget hit: checkpoint is on disk, no output written
                # yet — emit perf summary and halt (da#276).
                controller.stop(
                    "disambiguate",
                    processed=processed,
                    total=total_mentions,
                    canonical=len(canonical_map),
                )

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    total_resolved = sum(source_resolved.values())
    total_canonical = len(canonical_map)
    total_ambiguous = len(ambiguous_rows)

    for corpus in sorted(source_total):
        resolved = source_resolved.get(corpus, 0)
        total = source_total[corpus]
        rate = round(resolved / total * 100, 1) if total else 0.0
        logger.info(
            "disambiguate_source_rate",
            source_corpus=corpus,
            resolved=resolved,
            total=total,
            rate_pct=rate,
        )

    # Bio coverage: fraction of candidates that got at least one mention match.
    matched_bios = {r["canonical_id"] for r in merge_log_rows if r.get("canonical_id")}
    bio_coverage = round(len(matched_bios) / max(len(candidates), 1) * 100, 1)

    # Cross-source collapse (da#99): how many per-source identities merged into a
    # shared canonical narrator. naive = one node per (source, canonical) pair.
    naive_identity_count = len(naive_identity_pairs)
    cross_source_merged = naive_identity_count - total_canonical

    logger.info(
        "disambiguate_summary",
        total_mentions=total_mentions,
        total_resolved=total_resolved,
        total_canonical=total_canonical,
        total_ambiguous=total_ambiguous,
        resolution_rate_pct=round(total_resolved / max(total_mentions, 1) * 100, 1),
        bio_coverage_pct=bio_coverage,
        naive_identity_count=naive_identity_count,
        cross_source_merged=cross_source_merged,
    )

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    # 1. narrators_canonical.parquet
    canonical_table = _build_canonical_table(canonical_map)
    canonical_path = output_dir / "narrators_canonical.parquet"
    write_canonical(canonical_table, canonical_path, stage="disambiguate")
    output_paths.append(canonical_path)

    # 2. ambiguous_narrators.csv
    if ambiguous_rows:
        ambiguous_path = output_dir / "ambiguous_narrators.csv"
        _build_ambiguous_csv(ambiguous_rows, ambiguous_path)
        output_paths.append(ambiguous_path)

    # 3. merge_log.parquet
    if merge_log_rows:
        log_arrays: dict[str, pa.Array] = {
            "canonical_id": pa.array([r["canonical_id"] for r in merge_log_rows], type=pa.string()),
            "mention_id": pa.array([r["mention_id"] for r in merge_log_rows], type=pa.string()),
            "mention_text": pa.array([r["mention_text"] for r in merge_log_rows], type=pa.string()),
            "merge_stage": pa.array([r["merge_stage"] for r in merge_log_rows], type=pa.string()),
            "score": pa.array([r["score"] for r in merge_log_rows], type=pa.float32()),
        }
        log_table = pa.table(log_arrays, schema=_MERGE_LOG_SCHEMA)
        log_path = output_dir / "merge_log.parquet"
        write_parquet(log_table, log_path, schema=_MERGE_LOG_SCHEMA)
        output_paths.append(log_path)

    # 4. Backfill canonical ids onto the resolved-mention rows (#109).
    # The NER stage left canonical_narrator_id=None on every mention; rewrite
    # them with what disambiguation resolved so the downstream graph loaders
    # (which key NARRATED edges and Chains off canonical_narrator_id) actually
    # materialize narrator->hadith wiring instead of producing zero edges.
    if resolved_map:
        _backfill_mention_canonical_ids(
            output_dir / "narrator_mentions_resolved.parquet", resolved_map
        )

    # The run completed and all outputs are on disk — drop the checkpoint so the
    # next cold run doesn't spuriously resume against stale state (da#268).
    clear_checkpoint(ckpt_dir)

    logger.info(
        "disambiguate_run_complete",
        output_files=[str(p) for p in output_paths],
    )
    return output_paths
