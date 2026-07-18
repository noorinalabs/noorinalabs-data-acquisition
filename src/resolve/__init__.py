"""Phase 2: Entity Resolution pipeline.

Dependency-aware orchestrator:
``NER -> disambiguate -> bio_promote -> fuzzy_cluster -> (dedup + detect_parallels)``.

Ordering invariants (da#117):
- ``disambiguate`` OVERWRITES ``narrators_canonical.parquet`` while
  ``bio_promote`` MERGEs into it, so ``bio_promote`` MUST run *after*
  ``disambiguate`` or promoted bio-only narrators get clobbered (da#99).
- ``fuzzy_cluster`` (da#118) runs *after* both: it clusters cross-source name
  variants the exact-name pass under-merged, operating on the canonical set
  those two produce. It re-keys nothing (merges route through
  ``make_canonical_id``) and is idempotent, so adding it never perturbs the
  disambiguate→bio_promote invariant above.
- ``dedup`` (semantic embeddings, degrades to empty without the model) and
  ``detect_parallels`` (deterministic lexical, offline/CI) both write the shared
  ``staging/parallel_links.parquet``. run_all runs both and *composes* their
  outputs (union, deduped by canonical hadith pair) so an orchestrated run in a
  no-model environment still emits cross-sect PARALLEL_OF edges (da#100/da#114).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.exit_codes import EXIT_STAGE_FAILED
from src.resolve._checkpoint import EXIT_STOPPED_AT_LIMIT, StopAfterReached
from src.resolve._deps import EXIT_MISSING_DEPENDENCY, MissingDependencyError
from src.resolve._inputs import MissingInputError
from src.resolve._provenance import DetectorProvenance, DetectorStatus, read_provenance
from src.resolve._run_record import finalize_run
from src.resolve.ner import EXIT_UNROUTED_CORPUS, UnroutedCorpusError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pyarrow as pa

logger = get_logger(__name__)

__all__ = [
    "EXIT_MISSING_DEPENDENCY",
    "EXIT_STAGE_FAILED",
    "EXIT_STOPPED_AT_LIMIT",
    "EXIT_UNROUTED_CORPUS",
    "RESOLVE_STEP_ORDER",
    "RESUMABLE_STEPS",
    "MissingDependencyError",
    "MissingInputError",
    "ResolveMetrics",
    "ResolveStageError",
    "StageErrored",
    "StageOutcome",
    "StageRan",
    "StageSkipped",
    "StopAfterReached",
    "UnroutedCorpusError",
    "run_all",
]


@dataclass(frozen=True)
class StageRan:
    """A stage executed and produced ``files`` (possibly an empty list).

    An empty ``files`` here is an honest "ran, produced no output file" — it is
    NOT the ``[]`` that pre-da#360 ``run_all`` also left behind for a stage that
    RAISED. Those two were indistinguishable in the old ``dict[str, list[Path]]``;
    a raised stage now yields :class:`StageErrored`, never a bare empty list.
    """

    step: str
    files: list[Path]


@dataclass(frozen=True)
class StageSkipped:
    """A stage was not run this invocation, by design.

    ``reason`` is ``"precomputed"`` (skipped by ``--from-step`` past it, outputs
    assumed on disk) or ``"dependency_unmet"`` (e.g. ``disambiguate`` when NER
    produced no mentions). A skip is a success, not a failure.
    """

    step: str
    reason: str


@dataclass(frozen=True)
class StageErrored:
    """A stage raised. Its output is absent or partial.

    Recording this — rather than swallowing the exception and leaving an empty
    file list that reads as success — is the da#360 fix: any ``StageErrored`` in
    the outcome makes ``run_all`` raise :class:`ResolveStageError` at the end, so
    the process can never exit 0 having skipped a stage.
    """

    step: str
    exc_type: str
    traceback_str: str


# The discriminated stage-result union. Every step of a ``run_all`` maps to
# exactly one of these — there is no fourth "empty list that might mean either"
# state, which is the representation defect da#360 closes.
StageOutcome = StageRan | StageSkipped | StageErrored


class ResolveStageError(Exception):
    """One or more resolve stages raised; ``run_all`` refuses to report success.

    Raised at the END of ``run_all`` (after the best-effort, dependency-aware
    continue-past-failure sweep) when any stage produced a :class:`StageErrored`.
    The CLI maps it to :data:`~src.exit_codes.EXIT_STAGE_FAILED`, so a run that
    silently skipped a stage can no longer exit 0 (da#360). It aggregates *every*
    failed stage, not just the first, so one 7.5-hour run surfaces them all.
    """

    def __init__(self, errored: list[StageErrored]) -> None:
        self.errored = errored
        steps = ", ".join(e.step for e in errored)
        super().__init__(f"resolve stage(s) failed: {steps}")


def _outcome_files(outcome: StageOutcome | None) -> list[Path]:
    """Files a stage produced, or ``[]`` for a skipped/errored/absent stage."""
    return outcome.files if isinstance(outcome, StageRan) else []


def _stage_provenance(outcomes: dict[str, StageOutcome]) -> tuple[str, ...]:
    """Render each stage's outcome as a ``RESOLVE_STAGE`` body for the run record.

    Forensics, not a gate — the consumer keys only on ``canonical_ids`` and
    ``run_status``. But a counts-only record cannot be told apart from one an
    operator hand-authored in a single line straight off the parquet, and the
    pressure to hand-mint is maximal exactly when the gate bites. Recording which
    stages ran, skipped and why makes hand-minting an act of forgery rather than
    one of convenience (da#428, Jean-Claude Habimana). Deliberately unsigned:
    pre-launch, no adversary, not worth the key management.
    """
    rendered: list[str] = []
    for step in RESOLVE_STEP_ORDER:
        outcome = outcomes.get(step)
        match outcome:
            case StageRan():
                detail = f"outcome=ran files={len(outcome.files)}"
            case StageSkipped():
                detail = f"outcome=skipped reason={outcome.reason}"
            case StageErrored():
                detail = f"outcome=errored exc={outcome.exc_type}"
            case _:
                detail = "outcome=absent"
        rendered.append(f"step={step} {detail}")
    return tuple(rendered)


# Canonical execution order of the resolve steps. ``run_all(from_step=...)`` skips
# every step BEFORE the named one (their outputs must already exist on disk) and
# starts from it — used to resume a crashed pipeline without redoing completed
# stages (da#268). The order is load-bearing: ``disambiguate`` OVERWRITES
# ``narrators_canonical.parquet`` while ``bio_promote`` MERGEs into it, so nothing
# may reorder them (da#99/da#117). ``dedup`` and ``parallels`` are the two
# PARALLEL_OF detectors composed at the end; ``--from-step dedup`` re-runs both.
RESOLVE_STEP_ORDER = (
    "ner",
    "disambiguate",
    "bio_promote",
    "cluster",
    "narrator_split",
    "contextual_disambiguation",
    "reconcile",
    "tabaqa_dates",
    "narrator_unify",
    "over_merged_flag",
    "muhaddithat_links",
    "dedup",
    "parallels",
)

# The stages that keep an intra-stage checkpoint and therefore honour
# ``--stop-after`` (da#276) / intra-stage ``resume``. ``cluster`` (fuzzy_cluster)
# is the multi-day block-scoring pass and is checkpointed by da#272 PR2. The rest
# are exempt (``ner`` is cold-by-design; ``bio_promote``/date stages are
# sub-minute idempotent re-runs), so ``--stop-after`` targeting them is a CLI
# error rather than a silent no-op.
RESUMABLE_STEPS = frozenset({"disambiguate", "cluster", "dedup", "parallels"})


def _resolve_start_index(from_step: str | None) -> int:
    """Index into :data:`RESOLVE_STEP_ORDER` to start at; 0 when ``from_step`` is None."""
    if from_step is None:
        return 0
    if from_step not in RESOLVE_STEP_ORDER:
        raise ValueError(
            f"unknown resolve step {from_step!r}; expected one of {', '.join(RESOLVE_STEP_ORDER)}"
        )
    return RESOLVE_STEP_ORDER.index(from_step)


@dataclass
class ResolveMetrics:
    """Typed metrics returned by the resolve pipeline."""

    ner_mention_count: int = 0
    canonical_narrator_count: int = 0
    ambiguous_count: int = 0
    resolution_rate_pct: float = 0.0
    ambiguous_pct: float = 0.0
    parallel_links_count: int = 0
    parallel_verbatim: int = 0
    parallel_close_paraphrase: int = 0
    parallel_thematic: int = 0
    parallel_cross_sect: int = 0
    ner_files: list[Path] = field(default_factory=list)
    disambiguate_files: list[Path] = field(default_factory=list)
    bio_promote_files: list[Path] = field(default_factory=list)
    dedup_files: list[Path] = field(default_factory=list)
    parallels_files: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=== Phase 2: Entity Resolution Summary ===",
            f"  NER mentions extracted   : {self.ner_mention_count}",
            f"  Canonical narrators      : {self.canonical_narrator_count}",
            f"  Ambiguous mentions       : {self.ambiguous_count}",
            f"  Resolution rate          : {self.resolution_rate_pct:.1f}%",
            f"  Ambiguous %              : {self.ambiguous_pct:.1f}%",
            f"  Parallel links           : {self.parallel_links_count}",
        ]
        if self.parallel_links_count > 0:
            lines.extend(
                [
                    f"    verbatim               : {self.parallel_verbatim}",
                    f"    close paraphrase       : {self.parallel_close_paraphrase}",
                    f"    thematic               : {self.parallel_thematic}",
                    f"    cross-sect             : {self.parallel_cross_sect}",
                ]
            )
        total_files = (
            len(self.ner_files)
            + len(self.disambiguate_files)
            + len(self.bio_promote_files)
            + len(self.dedup_files)
            + len(self.parallels_files)
        )
        lines.append(f"  Output files             : {total_files}")
        return "\n".join(lines)


def _has_staging_parquets(staging_dir: Path) -> bool:
    """Check that staging directory contains at least one Parquet file."""
    return any(staging_dir.glob("**/*.parquet"))


def _collect_dedup_metrics(metrics: ResolveMetrics, staging_dir: Path) -> None:
    """Read parallel_links.parquet to populate dedup metrics."""
    path = staging_dir / "parallel_links.parquet"
    if not path.exists():
        return
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        metrics.parallel_links_count = table.num_rows
        if table.num_rows > 0:
            vt_col = table.column("variant_type")
            cs_col = table.column("cross_sect")
            metrics.parallel_verbatim = pc.sum(pc.equal(vt_col, "verbatim")).as_py()
            metrics.parallel_close_paraphrase = pc.sum(pc.equal(vt_col, "close_paraphrase")).as_py()
            metrics.parallel_thematic = pc.sum(pc.equal(vt_col, "thematic")).as_py()
            metrics.parallel_cross_sect = pc.sum(cs_col).as_py()
    except Exception:  # noqa: BLE001
        logger.warning("dedup_metrics_read_failed", path=str(path))


def _read_parallel_links(staging_dir: Path) -> pa.Table | None:
    """Read ``parallel_links.parquet`` into a table, or ``None`` if absent/unreadable.

    Used to capture a detector's output before the next detector overwrites the
    shared artifact, so the two can be composed.
    """
    path = staging_dir / "parallel_links.parquet"
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path)
    except Exception:  # noqa: BLE001
        logger.warning("parallel_links_read_failed", path=str(path))
        return None


def _compose_parallel_links(
    staging_dir: Path,
    semantic: pa.Table | None,
    deterministic: pa.Table | None,
    provenance: DetectorProvenance,
) -> Path | None:
    """Union the two PARALLEL_OF detectors' links into the shared artifact.

    ``dedup`` (semantic embeddings) and ``detect_parallels`` (deterministic
    lexical) both materialize ``staging/parallel_links.parquet`` under the
    identical ``PARALLEL_LINKS_SCHEMA``. We union them, deduping by the canonical
    ``(hadith_id_a, hadith_id_b)`` pair; the **semantic** row wins on a key
    collision (the production embedding signal supersedes the lexical fallback).
    Composition (rather than last-writer-wins) guarantees an orchestrated run in
    a no-model environment — where ``dedup`` degrades to an empty table — still
    emits the deterministic cross-sect edges (da#117).

    ``provenance`` (da#378) is stamped into the composed artifact's parquet
    metadata so a zero-row result records WHICH detectors produced it and in what
    state — a ``DEDUP_REQUIRE_ML=false`` degraded composition is no longer
    byte-identical to a true negative.

    Returns the written path, or ``None`` when neither detector produced a table
    (the shared artifact is then left untouched).
    """
    if semantic is None and deterministic is None:
        return None

    import pyarrow as pa

    from src.resolve._provenance import write_parallel_links
    from src.resolve.schemas import PARALLEL_LINKS_SCHEMA

    # Insert deterministic first, then semantic, so a semantic row overwrites a
    # deterministic one for the same canonical pair.
    merged: dict[tuple[object, object], dict[str, object]] = {}
    for table in (deterministic, semantic):
        if table is None:
            continue
        for row in table.to_pylist():
            merged[(row["hadith_id_a"], row["hadith_id_b"])] = row

    rows = list(merged.values())
    composed = pa.table(
        {
            "hadith_id_a": pa.array([r["hadith_id_a"] for r in rows], type=pa.string()),
            "hadith_id_b": pa.array([r["hadith_id_b"] for r in rows], type=pa.string()),
            "similarity_score": pa.array([r["similarity_score"] for r in rows], type=pa.float32()),
            "variant_type": pa.array([r["variant_type"] for r in rows], type=pa.string()),
            "cross_sect": pa.array([r["cross_sect"] for r in rows], type=pa.bool_()),
        },
        schema=PARALLEL_LINKS_SCHEMA,
    )
    output_path = write_parallel_links(composed, staging_dir / "parallel_links.parquet", provenance)
    logger.info(
        "parallel_links_composed",
        semantic=0 if semantic is None else semantic.num_rows,
        deterministic=0 if deterministic is None else deterministic.num_rows,
        composed=composed.num_rows,
        semantic_status=provenance.semantic.value,
        deterministic_status=provenance.deterministic.value,
        path=str(output_path),
    )
    return output_path


def _collect_disambig_metrics(metrics: ResolveMetrics, output_dir: Path) -> None:
    """Read disambiguation outputs to populate narrator metrics."""
    canonical_path = output_dir / "narrators_canonical.parquet"
    ambiguous_path = output_dir / "ambiguous_narrators.parquet"

    try:
        if canonical_path.exists():
            import pyarrow.parquet as pq

            table = pq.read_table(canonical_path)
            metrics.canonical_narrator_count = table.num_rows
    except Exception:  # noqa: BLE001
        logger.warning("canonical_metrics_read_failed", path=str(canonical_path))

    try:
        if ambiguous_path.exists():
            import pyarrow.parquet as pq

            meta = pq.read_metadata(ambiguous_path)
            metrics.ambiguous_count = meta.num_rows
    except Exception:  # noqa: BLE001
        logger.warning("ambiguous_metrics_read_failed", path=str(ambiguous_path))


def _collect_ner_metrics(metrics: ResolveMetrics, output_dir: Path) -> None:
    """Read NER output to populate mention count."""
    path = output_dir / "narrator_mentions_resolved.parquet"
    if not path.exists():
        return
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        metrics.ner_mention_count = table.num_rows
    except Exception:  # noqa: BLE001
        logger.warning("ner_metrics_read_failed", path=str(path))


def run_all(
    raw_dir: Path,
    staging_dir: Path,
    output_dir: Path,
    *,
    from_step: str | None = None,
    resume: bool = True,
    stop_after: int | None = None,
) -> dict[str, StageOutcome]:
    """Run full entity resolution pipeline.

    Returns a discriminated ``{step: StageOutcome}`` map (da#360): each step is a
    :class:`StageRan`, :class:`StageSkipped` or :class:`StageErrored`, so a stage
    that produced no file (``StageRan`` with an empty ``files``) is no longer
    indistinguishable from a stage that raised. If ANY stage raised, this does not
    return a map at all — it raises :class:`ResolveStageError` after the sweep, so
    the process can never exit 0 having silently skipped a stage.

    Order: ``NER -> disambiguate -> bio_promote -> (dedup + detect_parallels)``.

    Dependency-aware: if NER fails, skip disambiguation. ``bio_promote`` runs
    after disambiguation regardless (it merges bios into the canonical table and
    must never precede the overwriting ``disambiguate``, da#99/da#117). ``dedup``
    and ``detect_parallels`` only need hadith text; both write the shared
    ``parallel_links.parquet`` and their outputs are composed.

    ``output_dir`` is the curated dir — the same location the graph loader reads
    ``narrators_canonical.parquet`` from; it is passed to both ``disambiguate``
    and ``bio_promote`` so they reconcile against one canonical artifact (da#112).

    ``from_step`` (da#268) skips every step BEFORE the named one in
    :data:`RESOLVE_STEP_ORDER`, resuming a crashed run without redoing completed
    stages. Skipped steps' outputs must already be on disk; a skipped ``ner`` is
    treated as complete iff ``narrator_mentions_resolved.parquet`` exists (so
    ``disambiguate`` then reads that file and resumes its own mid-stream
    checkpoint). Recovering a mid-``disambiguate`` crash is
    ``run_all(..., from_step="disambiguate")``: NER is skipped so the existing
    mentions file — and disambiguate's checkpoint keyed off it — is reused.

    ``resume`` is the uniform crash-resume switch (da#272): when ``True`` (default)
    every stage that keeps an intra-stage checkpoint — ``disambiguate`` (da#268),
    ``fuzzy_cluster``'s block-scoring pass (PR2), ``dedup``'s FAISS/collection
    phase, and ``detect_parallels``'s anchor scan — restores it and continues;
    ``False`` (CLI ``--no-resume``) forces each of those stages to cold-start. It is
    orthogonal to ``from_step`` (which step to start at) and does not affect the
    exempt stages (``ner`` is cold-by-design because it re-mints uuid4 mention_ids;
    ``bio_promote``/date stages are sub-minute and simply re-run).

    ``stop_after`` (da#276, CLI ``--stop-after``) is the bounded partial-run probe:
    the first resumable stage to reach ``stop_after`` checkpoint writes stops
    cleanly (checkpoint left on disk, final output NOT written) by raising
    :class:`~src.resolve._checkpoint.StopAfterReached`, which propagates through
    this orchestrator (it is a ``BaseException``, so the per-step ``except
    Exception`` guards do not swallow it) and **halts the pipeline** — no later
    stage runs on the partial output. The CLI catches it and exits
    :data:`~src.resolve.EXIT_STOPPED_AT_LIMIT`. It threads only to the resumable
    stages; the CLI rejects ``--stop-after`` aimed at an exempt ``from_step``.
    """
    start_idx = _resolve_start_index(from_step)

    def _do(step: str) -> bool:
        """True when ``step`` should run for this invocation's ``from_step``."""
        return RESOLVE_STEP_ORDER.index(step) >= start_idx

    logger.info("resolve_pipeline_start", from_step=from_step or "ner")

    from src.resolve import (
        bio_promote,
        contextual_disambiguation,
        date_reconcile,
        dedup,
        disambiguate,
        fuzzy_cluster,
        muhaddithat_links,
        narrator_split,
        narrator_unify,
        ner,
        over_merged_flag,
        parallels,
        tabaqa_dates,
    )

    # Discriminated per-stage outcomes (da#360). Every step maps to exactly one
    # StageRan/StageSkipped/StageErrored — there is no "empty list that might mean
    # either ran-empty or raised" state, which is the conflation that let a run
    # exit 0 having skipped a stage. ``_stage_files`` mirrors StageRan.files into
    # the metrics/compose bookkeeping below.
    outcomes: dict[str, StageOutcome] = {}

    def _stage_files(step: str) -> list[Path]:
        return _outcome_files(outcomes.get(step))

    # Pre-flight check: verify staging has Parquet files.
    if not staging_dir.exists() or not _has_staging_parquets(staging_dir):
        logger.warning(
            "resolve_preflight_failed",
            staging_dir=str(staging_dir),
            msg="No Parquet files found in staging directory",
        )
        logger.warning("resolution_skipped", reason="no staging Parquet files found")
        return {s: StageSkipped(s, "preflight_no_staging") for s in RESOLVE_STEP_ORDER}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: NER
    ner_ok = False
    if _do("ner"):
        try:
            logger.info("resolve_step", step="ner", status="running")
            ner_files = ner.run(staging_dir, output_dir)
            outcomes["ner"] = StageRan("ner", ner_files)
            ner_ok = True
            logger.info("resolve_step", step="ner", status="complete", files=len(ner_files))
        except Exception as exc:  # noqa: BLE001
            outcomes["ner"] = StageErrored("ner", type(exc).__name__, traceback.format_exc())
            logger.error("resolve_step_failed", step="ner", traceback=traceback.format_exc())
    else:
        # Precomputed by an earlier run (--from-step past ner): its mention output
        # must already exist for disambiguate to consume + resume against.
        mentions_present = (output_dir / "narrator_mentions_resolved.parquet").exists()
        ner_ok = mentions_present
        outcomes["ner"] = StageSkipped("ner", "precomputed")
        logger.info(
            "resolve_step_skipped_precomputed",
            step="ner",
            outputs_present=mentions_present,
        )

    # Step 2: Disambiguation (skip if NER failed — needs mention output).
    if _do("disambiguate") and ner_ok:
        try:
            logger.info("resolve_step", step="disambiguate", status="running")
            disambiguate_files = disambiguate.run(
                staging_dir, output_dir, resume=resume, stop_after=stop_after
            )
            outcomes["disambiguate"] = StageRan("disambiguate", disambiguate_files)
            logger.info(
                "resolve_step",
                step="disambiguate",
                status="complete",
                files=len(disambiguate_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["disambiguate"] = StageErrored(
                "disambiguate", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed",
                step="disambiguate",
                traceback=traceback.format_exc(),
            )
    elif not _do("disambiguate"):
        outcomes["disambiguate"] = StageSkipped("disambiguate", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="disambiguate")
    else:
        outcomes["disambiguate"] = StageSkipped("disambiguate", "dependency_unmet")
        logger.warning(
            "resolve_step_skipped",
            step="disambiguate",
            reason="NER failed — no mention data available",
        )

    # Step 3: Bio promotion — MERGE bios into narrators_canonical.parquet.
    # MUST run after disambiguate (which OVERWRITES the same file); reversing the
    # order clobbers promoted bio-only narrators (da#99/da#117). Runs regardless
    # of NER/disambiguation status — it is merge-safe against whatever canonical
    # table already exists (or none).
    if _do("bio_promote"):
        try:
            logger.info("resolve_step", step="bio_promote", status="running")
            promoted = bio_promote.promote_bios_to_canonical(staging_dir, output_dir)
            bio_files = [promoted] if promoted is not None else []
            outcomes["bio_promote"] = StageRan("bio_promote", bio_files)
            logger.info(
                "resolve_step",
                step="bio_promote",
                status="complete",
                files=len(bio_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["bio_promote"] = StageErrored(
                "bio_promote", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="bio_promote", traceback=traceback.format_exc()
            )
    else:
        outcomes["bio_promote"] = StageSkipped("bio_promote", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="bio_promote")

    # Step 3.5: Fuzzy cross-source clustering (da#118) — recall increment on the
    # exact-name pass. Merges high-confidence cross-source name variants the
    # byte-exact collapse left split, operating ON the canonical set the prior two
    # steps produced. Routes every merge through make_canonical_id and is
    # idempotent, so it never perturbs the disambiguate→bio_promote ordering. The
    # mentions file is remapped so graph NARRATED edges follow the merge (#109).
    if _do("cluster"):
        try:
            logger.info("resolve_step", step="cluster", status="running")
            canonical_path = output_dir / "narrators_canonical.parquet"
            mentions_path = output_dir / "narrator_mentions_resolved.parquet"
            merge_log_path = output_dir / "merge_log.parquet"
            cluster_metrics = fuzzy_cluster.cluster_canonical_narrators(
                canonical_path,
                mentions_path=mentions_path if mentions_path.exists() else None,
                merge_log_path=merge_log_path if merge_log_path.exists() else None,
                staging_dir=staging_dir,
                resume=resume,
                stop_after=stop_after,
            )
            cluster_files = [canonical_path] if cluster_metrics.merged_records else []
            outcomes["cluster"] = StageRan("cluster", cluster_files)
            logger.info(
                "resolve_step",
                step="cluster",
                status="complete",
                merged=cluster_metrics.merged_records,
                clusters=cluster_metrics.multi_member_clusters,
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["cluster"] = StageErrored(
                "cluster", type(exc).__name__, traceback.format_exc()
            )
            logger.error("resolve_step_failed", step="cluster", traceback=traceback.format_exc())
    else:
        outcomes["cluster"] = StageSkipped("cluster", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="cluster")

    # Step 3.55: Same-name split (da#337) — the *split* mirror of fuzzy_cluster's
    # merge. For each over-collapsed generic-name node (recall-first screen
    # generic_name.is_generic_name), peel it into distinct canonical nodes ONLY when
    # ≥2 well-separated, well-supported death-year bands (from attested chain-neighbour
    # dates) prove multiple referents; a genuinely single person (Sufyān al-Thawrī,
    # al-Zuhrī) clusters to one band and abstains. Runs AFTER cluster so it operates on
    # the final merged canonical set, and BEFORE reconcile/tabaqa_dates/dedup so the
    # peeled ids and remapped mentions flow through the date + parallel stages. Rewrites
    # the canonical table + remaps mentions + emits narrator_splits.parquet (audit).
    # Idempotent: a re-run re-reads the split table, every node is a single band, all
    # abstain, and nothing is rewritten.
    if _do("narrator_split"):
        try:
            logger.info("resolve_step", step="narrator_split", status="running")
            split_path = narrator_split.split_generic_narrators(output_dir, staging_dir=staging_dir)
            split_files = [split_path] if split_path is not None else []
            outcomes["narrator_split"] = StageRan("narrator_split", split_files)
            logger.info(
                "resolve_step",
                step="narrator_split",
                status="complete",
                files=len(split_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["narrator_split"] = StageErrored(
                "narrator_split", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="narrator_split", traceback=traceback.format_exc()
            )
    else:
        outcomes["narrator_split"] = StageSkipped("narrator_split", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="narrator_split")

    # Step 3.57: Contextual (isnad-neighbour) disambiguation (da#346). The date-axis
    # narrator_split abstains on a BARE name (no attested death-band to cut), but the
    # isnad POSITION still identifies a referent: a bare ʿAbd Allāh narrating ⟵Prophet
    # ⟶Nāfiʿ is Ibn ʿUmar. For each curated, hand-verified neighbour-pair signature,
    # peel the matching mentions onto a distinct discriminated node; leave every unknown
    # or ambiguous mention on the bare primary (which over_merged_flag keeps flagged) and
    # to da#443's external-rijāl split. Runs AFTER narrator_split (operates on the settled
    # canonical set) and BEFORE reconcile/tabaqa_dates/over_merged_flag so the peeled ids
    # + remapped mentions flow through the date stages and the residual flag counts the
    # reduced primary. Empty seed ⇒ pure no-op. Idempotent (peeled ids no longer carry the
    # bare name; the residual still matches no signature). Emits contextual_splits.parquet
    # (audit) + contextual_coverage.parquet (the da#346 blast-radius report).
    if _do("contextual_disambiguation"):
        try:
            logger.info("resolve_step", step="contextual_disambiguation", status="running")
            ctx_path = contextual_disambiguation.apply_contextual_disambiguation(
                output_dir, staging_dir=staging_dir
            )
            ctx_files = [ctx_path] if ctx_path is not None else []
            outcomes["contextual_disambiguation"] = StageRan("contextual_disambiguation", ctx_files)
            logger.info(
                "resolve_step",
                step="contextual_disambiguation",
                status="complete",
                files=len(ctx_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["contextual_disambiguation"] = StageErrored(
                "contextual_disambiguation", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed",
                step="contextual_disambiguation",
                traceback=traceback.format_exc(),
            )
    else:
        outcomes["contextual_disambiguation"] = StageSkipped(
            "contextual_disambiguation", "precomputed"
        )
        logger.info("resolve_step_skipped_precomputed", step="contextual_disambiguation")

    # Step 3.6: Multi-source date reconciliation (da#165). After bio_promote and
    # cluster have built the final canonical set, fold each narrator's per-source
    # parsed life-dates (da#164) into one canonical birth/death envelope + a
    # concrete precision, written onto the canonical date columns. Runs AFTER
    # cluster so it keys on final canonical ids and its always-concrete-precision
    # invariant holds on the emitted table; a no-op if no canonical table exists.
    if _do("reconcile"):
        try:
            logger.info("resolve_step", step="reconcile_dates", status="running")
            reconciled = date_reconcile.reconcile_canonical_dates(staging_dir, output_dir)
            reconcile_files = [reconciled] if reconciled is not None else []
            outcomes["reconcile"] = StageRan("reconcile", reconcile_files)
            logger.info(
                "resolve_step",
                step="reconcile_dates",
                status="complete",
                files=len(reconcile_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["reconcile"] = StageErrored(
                "reconcile", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="reconcile_dates", traceback=traceback.format_exc()
            )
    else:
        outcomes["reconcile"] = StageSkipped("reconcile", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="reconcile")

    # Step 3.65: ṭabaqa → estimated-window fallback (da#166). The LAST date stage:
    # after reconciliation has folded every *attested* dating into the canonical
    # envelope, some narrators still have no death date. For exactly those — death
    # still undated AND a known ṭabaqa class (``generation``) — derive an estimated
    # death window tagged ``tabaqa_estimate`` so the timeline isn't blank where the
    # rijāl sources are silent. Runs AFTER reconcile so it only ever fills the gaps
    # reconcile left; never overwrites a reconciled/parsed date; idempotent.
    if _do("tabaqa_dates"):
        try:
            logger.info("resolve_step", step="tabaqa_dates", status="running")
            estimated = tabaqa_dates.apply_tabaqa_fallback(output_dir)
            tabaqa_files = [estimated] if estimated is not None else []
            outcomes["tabaqa_dates"] = StageRan("tabaqa_dates", tabaqa_files)
            logger.info(
                "resolve_step",
                step="tabaqa_dates",
                status="complete",
                files=len(tabaqa_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["tabaqa_dates"] = StageErrored(
                "tabaqa_dates", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="tabaqa_dates", traceback=traceback.format_exc()
            )
    else:
        outcomes["tabaqa_dates"] = StageSkipped("tabaqa_dates", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="tabaqa_dates")

    # Step 3.66: Curated under-merge unification (da#431 / da#347) — the merge mirror of
    # narrator_split. Where fuzzy_cluster's precision guard structurally cannot bridge two
    # references to ONE person that share too few significant tokens (kunya↔ism, bare
    # ism↔qualified), this stage merges a hand-verified, corroboration-gated curated set
    # (narrator_unify.yaml) onto one survivor and remaps the absorbed mentions — reusing
    # fuzzy_cluster's own _merge_cluster/_remap. Runs after tabaqa_dates so the survivor
    # inherits final dates + mention_count, and BEFORE over_merged_flag so that stage stays
    # the last writer of narrators_canonical.parquet. Every group is behind the da#423
    # bidirectional acceptance fixture; a refused group is logged, never merged. A no-op
    # (None) when the seed matches < 2 distinct nodes per group (already unified).
    if _do("narrator_unify"):
        try:
            logger.info("resolve_step", step="narrator_unify", status="running")
            unified = narrator_unify.apply_narrator_unification(output_dir)
            unify_files = [unified] if unified is not None else []
            outcomes["narrator_unify"] = StageRan("narrator_unify", unify_files)
            logger.info(
                "resolve_step",
                step="narrator_unify",
                status="complete",
                files=len(unify_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["narrator_unify"] = StageErrored(
                "narrator_unify", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="narrator_unify", traceback=traceback.format_exc()
            )
    else:
        outcomes["narrator_unify"] = StageSkipped("narrator_unify", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="narrator_unify")

    # Step 3.67: Over-merged bare-generic flag (da#445 / #337 flag-now). The LAST writer
    # of narrators_canonical.parquet: after every date stage has finalized the canonical
    # table, set the ``over_merged`` boolean (+ note) on the curated bare-generic hubs
    # (bare ʿAbd Allāh, Sufyān, …) so the honest leaderboard can disclose their inflated
    # betweenness. PURE annotation — no node is split or minted; the flagged set is a
    # hand-verified curated list under a bidirectional acceptance fixture (no
    # corpus-internal threshold separates chimeras from genuine hubs). Goes through
    # write_canonical like every canonical writer, so the da#428 completeness tally is
    # re-minted; a no-op (None) when the seed matches no row.
    if _do("over_merged_flag"):
        try:
            logger.info("resolve_step", step="over_merged_flag", status="running")
            flagged = over_merged_flag.apply_over_merged_flags(output_dir)
            flag_files = [flagged] if flagged is not None else []
            outcomes["over_merged_flag"] = StageRan("over_merged_flag", flag_files)
            logger.info(
                "resolve_step",
                step="over_merged_flag",
                status="complete",
                files=len(flag_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["over_merged_flag"] = StageErrored(
                "over_merged_flag", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="over_merged_flag", traceback=traceback.format_exc()
            )
    else:
        outcomes["over_merged_flag"] = StageSkipped("over_merged_flag", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="over_merged_flag")

    # Step 3.7: Curated muhaddithat orphan mention-links (da#228 / ADR-004 item #3).
    # The 8 bio-only muhaddithat narrators promoted by bio_promote carry no chain
    # mention, so they sit orphaned with no graph edges. This emits a curated,
    # provenance-bearing NARRATED mention-link for exactly those 8 (no bulk-link),
    # resolving each to the SAME canonical id bio_promote minted and verifying it
    # exists in the canonical master first (no link to a non-promoted narrator).
    # Runs after bio_promote + cluster so the canonical master is final; the output
    # rides the existing resolved-mentions glob into the NARRATED loader.
    if _do("muhaddithat_links"):
        try:
            logger.info("resolve_step", step="muhaddithat_links", status="running")
            canonical_path = output_dir / "narrators_canonical.parquet"
            link_path = muhaddithat_links.build_muhaddithat_mention_links(
                output_dir,
                canonical_path=canonical_path if canonical_path.exists() else None,
            )
            link_files = [link_path] if link_path is not None else []
            outcomes["muhaddithat_links"] = StageRan("muhaddithat_links", link_files)
            logger.info(
                "resolve_step",
                step="muhaddithat_links",
                status="complete",
                files=len(link_files),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["muhaddithat_links"] = StageErrored(
                "muhaddithat_links", type(exc).__name__, traceback.format_exc()
            )
            logger.error(
                "resolve_step_failed", step="muhaddithat_links", traceback=traceback.format_exc()
            )
    else:
        outcomes["muhaddithat_links"] = StageSkipped("muhaddithat_links", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="muhaddithat_links")

    # Step 4: Dedup (semantic; degrades to an empty table without the embedding
    # model). Runs independently of NER/disambiguation. Capture its output before
    # detect_parallels overwrites the shared artifact, so the two can be composed.
    # ``semantic_status`` (da#378) records WHY the semantic side is what it is —
    # read back from the provenance dedup stamped on its own artifact — so the
    # composed artifact can distinguish a degraded run from a true negative.
    semantic_links: pa.Table | None = None
    semantic_status = DetectorStatus.NOT_RUN
    if _do("dedup"):
        try:
            logger.info("resolve_step", step="dedup", status="running")
            dedup_files = dedup.run(staging_dir, output_dir, resume=resume, stop_after=stop_after)
            outcomes["dedup"] = StageRan("dedup", dedup_files)
            semantic_links = _read_parallel_links(staging_dir)
            dedup_prov = read_provenance(staging_dir / "parallel_links.parquet")
            semantic_status = dedup_prov.semantic if dedup_prov else DetectorStatus.RAN
            logger.info("resolve_step", step="dedup", status="complete", files=len(dedup_files))
        except Exception as exc:  # noqa: BLE001
            outcomes["dedup"] = StageErrored("dedup", type(exc).__name__, traceback.format_exc())
            semantic_status = DetectorStatus.ERRORED
            logger.error("resolve_step_failed", step="dedup", traceback=traceback.format_exc())
    else:
        outcomes["dedup"] = StageSkipped("dedup", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="dedup")
        # dedup skipped but parallels will re-run and overwrite the shared artifact;
        # preserve the already-composed links (which include the semantic side) so
        # the recompose below doesn't drop them, and carry the prior run's semantic
        # provenance forward so the recomposed artifact stays honest (da#378).
        if _do("parallels"):
            semantic_links = _read_parallel_links(staging_dir)
            prior_prov = read_provenance(staging_dir / "parallel_links.parquet")
            semantic_status = prior_prov.semantic if prior_prov else DetectorStatus.NOT_RUN

    # Step 5: Deterministic lexical parallels (offline/CI complement + no-model
    # fallback). Overwrites parallel_links.parquet — capture its output too.
    deterministic_links: pa.Table | None = None
    deterministic_status = DetectorStatus.NOT_RUN
    if _do("parallels"):
        try:
            logger.info("resolve_step", step="parallels", status="running")
            parallels_files = parallels.run(
                staging_dir, output_dir, resume=resume, stop_after=stop_after
            )
            outcomes["parallels"] = StageRan("parallels", parallels_files)
            deterministic_links = _read_parallel_links(staging_dir)
            deterministic_status = DetectorStatus.RAN
            logger.info(
                "resolve_step", step="parallels", status="complete", files=len(parallels_files)
            )
        except Exception as exc:  # noqa: BLE001
            outcomes["parallels"] = StageErrored(
                "parallels", type(exc).__name__, traceback.format_exc()
            )
            deterministic_status = DetectorStatus.ERRORED
            logger.error("resolve_step_failed", step="parallels", traceback=traceback.format_exc())
    else:
        outcomes["parallels"] = StageSkipped("parallels", "precomputed")
        logger.info("resolve_step_skipped_precomputed", step="parallels")

    # Step 6: Compose both detectors' links into the single shared artifact so a
    # no-model run still emits the deterministic cross-sect edges (da#117). Only
    # recompose when at least one detector re-ran this invocation; otherwise the
    # existing composed artifact from the prior run is left untouched. The composed
    # artifact carries the combined detector provenance (da#378) so a zero-row
    # result records which detectors produced it and in what state.
    if _do("dedup") or _do("parallels"):
        provenance = DetectorProvenance(
            semantic=semantic_status,
            semantic_rows=semantic_links.num_rows if semantic_links is not None else 0,
            deterministic=deterministic_status,
            deterministic_rows=deterministic_links.num_rows
            if deterministic_links is not None
            else 0,
        )
        composed = _compose_parallel_links(
            staging_dir, semantic_links, deterministic_links, provenance
        )
        # Do NOT let a successful compose overwrite a StageErrored: on the common
        # path (dedup produced links, ``parallels.run`` raised) ``composed`` is
        # non-None because the semantic side alone composes, so an unconditional
        # ``outcomes["parallels"] = StageRan(...)`` would clobber the StageErrored
        # recorded by the parallels step — recreating the exact da#360 swallow one
        # block down, letting the ``errored`` scan below find nothing and run_all
        # exit 0. The composed artifact is still written (its provenance already
        # records ``deterministic=ERRORED``); we just refuse to relabel the stage
        # as succeeded (Nikolaos Papadopoulos, PR#404 review).
        if composed is not None and not isinstance(outcomes.get("parallels"), StageErrored):
            outcomes["parallels"] = StageRan("parallels", [composed])

    # Collect metrics from output files.
    metrics = ResolveMetrics(
        ner_files=_stage_files("ner"),
        disambiguate_files=_stage_files("disambiguate"),
        bio_promote_files=_stage_files("bio_promote"),
        dedup_files=_stage_files("dedup"),
        parallels_files=_stage_files("parallels"),
    )
    _collect_ner_metrics(metrics, output_dir)
    _collect_disambig_metrics(metrics, output_dir)
    _collect_dedup_metrics(metrics, staging_dir)

    # Compute derived rates.
    if metrics.ner_mention_count > 0:
        resolved = metrics.ner_mention_count - metrics.ambiguous_count
        metrics.resolution_rate_pct = resolved / metrics.ner_mention_count * 100
        metrics.ambiguous_pct = metrics.ambiguous_count / metrics.ner_mention_count * 100

    logger.info(
        "resolve_pipeline_complete",
        ner_mentions=metrics.ner_mention_count,
        canonical_narrators=metrics.canonical_narrator_count,
        ambiguous=metrics.ambiguous_count,
        resolution_rate_pct=round(metrics.resolution_rate_pct, 1),
        parallel_links=metrics.parallel_links_count,
    )

    logger.info("resolve_metrics_summary", summary=metrics.summary())

    # da#360: a stage that raised must never let the process report success. The
    # sweep above is best-effort and dependency-aware (a failed NER still lets
    # bio_promote run), so failures are collected rather than fail-fast — but any
    # StageErrored now forces a non-zero exit via the CLI's ResolveStageError
    # handler, aggregating every failed stage from this one (possibly 7.5-hour) run.
    errored = [o for o in outcomes.values() if isinstance(o, StageErrored)]

    # da#428: stamp the run's terminal status onto curated/_resolve_run.txt. This is
    # the ONLY place run_status may be upgraded to `complete`; every write_canonical
    # stamps `incomplete`, so a killed process, a raised stage or a --stop-after probe
    # all leave `incomplete` on disk and the publish refuses. `complete` is exactly
    # "reached this line with no stage in error" — the one fact no count can supply,
    # because a run that halts after writing a coherent partial file is internally
    # consistent and its tally matches what it managed to write.
    finalize_run(output_dir, _stage_provenance(outcomes), complete=not errored)

    if errored:
        for e in errored:
            logger.error("resolve_stage_errored", step=e.step, exc_type=e.exc_type)
        raise ResolveStageError(errored)

    return outcomes
