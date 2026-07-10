"""Phase 3: Neo4j node/edge loaders and validation queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.graph.load_edges import EdgeLoadResult, load_all_edges
from src.graph.load_nodes import LoadResult, load_all_nodes
from src.graph.validate import (
    _DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    ValidationResult,
    run_validation,
)
from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

# --- Why quarantine, not abort (da#359) --------------------------------------
# A malformed id is quarantined, never aborted: these loaders commit per batch
# with no spanning transaction, so an escaping exception would strand batches
# 1..N-1 in Neo4j hours into a load. But a load that ran to completion having
# REFUSED input is not a success, and it must not exit 0 -- that is how 650,986
# sanadset hadiths (76.3% of staging) could be absent from the graph while
# parse, load and validation all reported success.
#
# The exit code that says so is `ExitCode.REFUSED_ROWS`, declared -- like every
# other exit code this pipeline emits -- in `src/exit_codes.py` (da#384). It is
# keyed on `LoadSummary.total_refused` below. This module deliberately declares
# no exit constant of its own: the value it once hard-coded here was `5`, which
# da#372 subsequently assigned to `ExitCode.VALIDATION_FINDINGS`. A comment in
# this file claiming to track who owns which integer was exactly the artifact
# the registry exists to abolish.

__all__ = [
    "EdgeLoadResult",
    "LoadResult",
    "LoadSummary",
    "ValidationResult",
    "load_all",
    "load_all_edges",
    "load_all_nodes",
    "run_validation",
]


@dataclass(frozen=True)
class LoadSummary:
    """Full pipeline outcome: nodes + edges + validation."""

    node_results: list[LoadResult]
    edge_results: list[EdgeLoadResult] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    validation_passed: bool = True

    @property
    def total_malformed_ids(self) -> int:
        """Rows and edges refused across every loader because their id was malformed.

        Summed over BOTH node and edge results: a doubled hadith id is refused by
        the Hadith/Chain/Grading node loaders *and* by the APPEARS_IN, PARALLEL_OF,
        NARRATED, TRANSMITTED_TO and GRADED_BY edge loaders, and a total that
        counted only one side would under-report a load that dropped the other.
        """
        return sum(r.malformed_ids for r in self.node_results) + sum(
            r.malformed_ids for r in self.edge_results
        )

    @property
    def total_refused(self) -> int:
        """Rows and edges the loaders declined because the input was defective.

        The answer to "did every loader accept every input row" -- and therefore
        the predicate both the exit code and the last-loaded manifest (da#374)
        key on. It is deliberately keyed on the *concept* rather than on today's
        instance (``total_malformed_ids``): a doubled id is one refusal class,
        an absent/blank ``source_id`` is another, and a load that silently drops
        rows for the second reason is the same silent data loss as the first.

        ``skipped`` cannot serve here -- it also counts the loader's *deliberate*
        drops (non-canonical editions, cross-edition dedup), which are the loader
        working correctly.
        """
        return sum(r.refused for r in self.node_results) + sum(r.refused for r in self.edge_results)


def load_all(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,
    queries_dir: Path,
    *,
    strict: bool = True,
    skip_validation: bool = False,
    nodes_only: bool = False,
    skip_files: list[str] | None = None,
    validation_timeout_seconds: float = _DEFAULT_VALIDATION_TIMEOUT_SECONDS,
) -> LoadSummary:
    """Full graph loading pipeline: nodes -> edges -> validate.

    Parameters
    ----------
    client:
        Connected Neo4j client.
    staging_dir:
        Directory containing staging Parquet files.
    curated_dir:
        Directory containing curated reference data (YAML).
    queries_dir:
        Root queries directory (contains ``validation/`` subdirectory).
    strict:
        If ``True``, raise on missing required data files.
    skip_validation:
        Explicit opt-out of the post-load validation queries. This is now a
        genuine opt-out, not a workaround for a hang: validation is bounded and
        a slow query downgrades to a non-fatal warning (da#259).
    nodes_only:
        If ``True``, load only nodes (skip edges and validation).
    skip_files:
        List of manifest keys (e.g. ``staging/hadiths_bukhari.parquet``) to skip
        during incremental loading. Files not in this list are loaded normally.
    validation_timeout_seconds:
        Per-query time budget for post-load validation. A query that overruns is
        recorded as a warning and does not fail the load.
    """
    logger.info(
        "load_all_start", strict=strict, skip_validation=skip_validation, nodes_only=nodes_only
    )

    # 1. Load nodes
    node_results = load_all_nodes(
        client, staging_dir, curated_dir, strict=strict, skip_files=skip_files
    )
    total_nodes = sum(r.created + r.merged for r in node_results)
    logger.info("load_all_nodes_done", total_nodes=total_nodes)

    # 2. Load edges (unless nodes_only)
    edge_results: list[EdgeLoadResult] = []
    total_edges = 0
    if not nodes_only:
        edge_results = load_all_edges(client, staging_dir, curated_dir, strict=strict)
        total_edges = sum(r.created for r in edge_results)
        logger.info("load_all_edges_done", total_edges=total_edges)

    # 3. Validate (unless skipped or nodes_only)
    validation_results: list[ValidationResult] = []
    validation_passed = True
    if not skip_validation and not nodes_only:
        validation_results = run_validation(
            client, queries_dir, timeout_seconds=validation_timeout_seconds
        )
        # A timed-out/downgraded check is a WARN, not a FAIL: it must not fail the
        # load (da#259). Only a hard failure (is_fatal) trips validation_passed.
        validation_passed = not any(v.is_fatal for v in validation_results)
        warnings = sum(1 for v in validation_results if v.warning)
        status = "PASS" if validation_passed else "FAIL"
        logger.info(
            "load_all_validation_done",
            status=status,
            checks=len(validation_results),
            warnings=warnings,
        )

    summary = LoadSummary(
        node_results=node_results,
        edge_results=edge_results,
        validation_results=validation_results,
        total_nodes=total_nodes,
        total_edges=total_edges,
        validation_passed=validation_passed,
    )
    logger.info(
        "load_all_complete",
        total_nodes=total_nodes,
        total_edges=total_edges,
        validation_passed=validation_passed,
    )
    return summary
