"""Phase 2: Entity Resolution pipeline.

Dependency-aware orchestrator:
``NER -> disambiguate -> bio_promote -> (dedup + detect_parallels)``.

Ordering invariants (da#117):
- ``disambiguate`` OVERWRITES ``narrators_canonical.parquet`` while
  ``bio_promote`` MERGEs into it, so ``bio_promote`` MUST run *after*
  ``disambiguate`` or promoted bio-only narrators get clobbered (da#99).
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

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pyarrow as pa

logger = get_logger(__name__)

__all__ = ["ResolveMetrics", "run_all"]


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

    Returns the written path, or ``None`` when neither detector produced a table
    (the shared artifact is then left untouched).
    """
    if semantic is None and deterministic is None:
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

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
    output_path = staging_dir / "parallel_links.parquet"
    pq.write_table(composed, output_path)
    logger.info(
        "parallel_links_composed",
        semantic=0 if semantic is None else semantic.num_rows,
        deterministic=0 if deterministic is None else deterministic.num_rows,
        composed=composed.num_rows,
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


def run_all(raw_dir: Path, staging_dir: Path, output_dir: Path) -> dict[str, list[Path]]:
    """Run full entity resolution pipeline.

    Order: ``NER -> disambiguate -> bio_promote -> (dedup + detect_parallels)``.

    Dependency-aware: if NER fails, skip disambiguation. ``bio_promote`` runs
    after disambiguation regardless (it merges bios into the canonical table and
    must never precede the overwriting ``disambiguate``, da#99/da#117). ``dedup``
    and ``detect_parallels`` only need hadith text; both write the shared
    ``parallel_links.parquet`` and their outputs are composed.

    ``output_dir`` is the curated dir — the same location the graph loader reads
    ``narrators_canonical.parquet`` from; it is passed to both ``disambiguate``
    and ``bio_promote`` so they reconcile against one canonical artifact (da#112).
    """
    logger.info("resolve_pipeline_start")

    from src.resolve import bio_promote, dedup, disambiguate, ner, parallels

    results: dict[str, list[Path]] = {
        "ner": [],
        "disambiguate": [],
        "bio_promote": [],
        "dedup": [],
        "parallels": [],
    }

    # Pre-flight check: verify staging has Parquet files.
    if not staging_dir.exists() or not _has_staging_parquets(staging_dir):
        logger.warning(
            "resolve_preflight_failed",
            staging_dir=str(staging_dir),
            msg="No Parquet files found in staging directory",
        )
        logger.warning("resolution_skipped", reason="no staging Parquet files found")
        return results

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: NER
    ner_ok = False
    try:
        logger.info("resolve_step", step="ner", status="running")
        results["ner"] = ner.run(staging_dir, output_dir)
        ner_ok = True
        logger.info("resolve_step", step="ner", status="complete", files=len(results["ner"]))
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="ner", traceback=traceback.format_exc())

    # Step 2: Disambiguation (skip if NER failed — needs mention output).
    if ner_ok:
        try:
            logger.info("resolve_step", step="disambiguate", status="running")
            results["disambiguate"] = disambiguate.run(staging_dir, output_dir)
            logger.info(
                "resolve_step",
                step="disambiguate",
                status="complete",
                files=len(results["disambiguate"]),
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "resolve_step_failed",
                step="disambiguate",
                traceback=traceback.format_exc(),
            )
    else:
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
    try:
        logger.info("resolve_step", step="bio_promote", status="running")
        promoted = bio_promote.promote_bios_to_canonical(staging_dir, output_dir)
        results["bio_promote"] = [promoted] if promoted is not None else []
        logger.info(
            "resolve_step",
            step="bio_promote",
            status="complete",
            files=len(results["bio_promote"]),
        )
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="bio_promote", traceback=traceback.format_exc())

    # Step 4: Dedup (semantic; degrades to an empty table without the embedding
    # model). Runs independently of NER/disambiguation. Capture its output before
    # detect_parallels overwrites the shared artifact, so the two can be composed.
    semantic_links: pa.Table | None = None
    try:
        logger.info("resolve_step", step="dedup", status="running")
        results["dedup"] = dedup.run(staging_dir, output_dir)
        semantic_links = _read_parallel_links(staging_dir)
        logger.info("resolve_step", step="dedup", status="complete", files=len(results["dedup"]))
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="dedup", traceback=traceback.format_exc())

    # Step 5: Deterministic lexical parallels (offline/CI complement + no-model
    # fallback). Overwrites parallel_links.parquet — capture its output too.
    deterministic_links: pa.Table | None = None
    try:
        logger.info("resolve_step", step="parallels", status="running")
        results["parallels"] = parallels.run(staging_dir, output_dir)
        deterministic_links = _read_parallel_links(staging_dir)
        logger.info(
            "resolve_step", step="parallels", status="complete", files=len(results["parallels"])
        )
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="parallels", traceback=traceback.format_exc())

    # Step 6: Compose both detectors' links into the single shared artifact so a
    # no-model run still emits the deterministic cross-sect edges (da#117).
    composed = _compose_parallel_links(staging_dir, semantic_links, deterministic_links)
    if composed is not None:
        results["parallels"] = [composed]

    # Collect metrics from output files.
    metrics = ResolveMetrics(
        ner_files=results["ner"],
        disambiguate_files=results["disambiguate"],
        bio_promote_files=results["bio_promote"],
        dedup_files=results["dedup"],
        parallels_files=results["parallels"],
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

    return results
