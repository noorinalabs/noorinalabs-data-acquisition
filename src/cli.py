"""CLI entry point for the isnad-graph-ingestion pipeline."""

from __future__ import annotations

import argparse
import sys

from src.exit_codes import ExitCode


def _mask_password(value: str) -> str:
    """Replace all but first and last character with asterisks."""
    if len(value) <= 2:
        return "*" * len(value)
    return value[0] + "*" * (len(value) - 2) + value[-1]


def _check_neo4j() -> None:
    """Pre-flight Neo4j connectivity check. Exits ``ExitCode.DB_UNREACHABLE`` on failure."""
    from neo4j import GraphDatabase

    from src.config import get_settings

    settings = get_settings()
    print("Checking Neo4j connectivity...")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
        )
        driver.verify_connectivity()
        driver.close()
    except Exception as exc:
        # This helper has FOUR callers (_cmd_load, _cmd_validate,
        # _cmd_migrate_transmitted_hadith_ids, _cmd_enrich) and knows exactly one
        # fact, identical at every one of them: the database is not there. It
        # must not name a stage it cannot know it is in -- it exited LOAD_FAILED,
        # which was true for one caller of four, and under `pipeline` that meant
        # rc=1 from a fully committed load (da#384 Amendment R).
        #
        # Taking the code as a parameter would not fix this: it would turn the
        # call sites into a hand-maintained mapping from caller to exit code.
        print(f"ERROR: Cannot connect to Neo4j at {settings.neo4j.uri}: {exc}")
        sys.exit(ExitCode.DB_UNREACHABLE)
    print("  Neo4j is reachable.")


def _cmd_info() -> None:
    """Print configuration (masked passwords) and check DB connectivity."""
    from src.config import get_settings

    settings = get_settings()

    print("=== isnad-ingest configuration ===")
    print(f"  neo4j.uri      : {settings.neo4j.uri}")
    print(f"  neo4j.user     : {settings.neo4j.user}")
    print(f"  neo4j.password : {_mask_password(settings.neo4j.password)}")
    print(f"  postgres.dsn   : {settings.postgres.dsn}")
    print(f"  data_raw_dir   : {settings.data_raw_dir}")
    print(f"  data_staging_dir: {settings.data_staging_dir}")
    print(f"  data_curated_dir: {settings.data_curated_dir}")
    print(f"  log_level      : {settings.log_level}")
    print()

    print("=== connectivity ===")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
        )
        driver.verify_connectivity()
        driver.close()
        print("  neo4j    : connected")
    except Exception:  # noqa: BLE001
        print("  neo4j    : unavailable")

    try:
        import psycopg

        conn = psycopg.connect(settings.postgres.dsn)
        conn.close()
        print("  postgres : connected")
    except Exception:  # noqa: BLE001
        print("  postgres : unavailable")


def _cmd_acquire() -> None:
    """Run all data acquisition downloaders."""
    from pathlib import Path

    from src.acquire import run_all as acquire_all
    from src.config import get_settings

    settings = get_settings()
    results = acquire_all(Path(settings.data_raw_dir))
    ok = sum(1 for v in results.values() if v)
    print(f"Acquisition complete. {ok}/{len(results)} sources downloaded.")


def _cmd_parse() -> None:
    """Run all parsers to produce staging Parquet files."""
    from pathlib import Path

    from src.config import get_settings
    from src.parse import run_all as parse_all

    settings = get_settings()
    results = parse_all(Path(settings.data_raw_dir), Path(settings.data_staging_dir))
    total_files = sum(len(v) for v in results.values())
    print(f"Parsing complete. {total_files} staging files produced.")


def _cmd_resolve(
    *, from_step: str | None = None, resume: bool = True, stop_after: int | None = None
) -> None:
    """Run the Phase 2 entity resolution pipeline."""
    from pathlib import Path

    from src.config import get_settings
    from src.resolve import (
        EXIT_MISSING_DEPENDENCY,
        EXIT_STOPPED_AT_LIMIT,
        MissingDependencyError,
        StopAfterReached,
    )
    from src.resolve import run_all as resolve_all

    settings = get_settings()
    if from_step:
        print(f"Resuming resolve from step: {from_step}")
    if not resume:
        print("Crash-resume disabled (--no-resume): every stage cold-starts.")
    if stop_after is not None:
        print(f"Bounded probe: will stop after {stop_after} checkpoint(s) (--stop-after).")
    try:
        results = resolve_all(
            Path(settings.data_raw_dir),
            Path(settings.data_staging_dir),
            Path(settings.data_curated_dir),
            from_step=from_step,
            resume=resume,
            stop_after=stop_after,
        )
    except StopAfterReached as stopped:
        # A stage hit its --stop-after budget: the pipeline halted with the
        # checkpoint on disk and no partial output written. Surface the perf
        # summary and exit with a status distinct from completion/crash (da#276).
        print(f"\n{stopped.summary()}")
        sys.exit(EXIT_STOPPED_AT_LIMIT)
    except MissingDependencyError as missing:
        # An enabled stage is missing a declared dependency. Halt loudly rather
        # than emit a zero-row output that downstream reads as a true negative
        # (da#309).
        print(f"\nERROR: {missing}", file=sys.stderr)
        sys.exit(EXIT_MISSING_DEPENDENCY)
    total = sum(len(v) for v in results.values())
    print(f"\nResolution complete. {total} output files.")


def _cmd_load(
    *,
    skip_validation: bool = False,
    nodes_only: bool = False,
    incremental: bool = False,
    validation_timeout: float | None = None,
) -> None:
    """Run the Phase 3 graph loading pipeline."""
    import time
    import traceback
    from pathlib import Path

    from src.config import get_settings

    settings = get_settings()

    _check_neo4j()

    from src.graph import EXIT_MALFORMED_IDS, load_all
    from src.graph.validate import _DEFAULT_VALIDATION_TIMEOUT_SECONDS
    from src.pipeline.audit import create_audit_entry, write_audit_entry
    from src.pipeline.manifest import (
        LAST_LOADED_MANIFEST_FILENAME,
        MANIFEST_FILENAME,
        compare_manifests,
        generate_manifest,
        load_manifest,
        save_manifest,
    )
    from src.utils.neo4j_client import Neo4jClient

    staging_dir = Path(settings.data_staging_dir)
    curated_dir = Path(settings.data_curated_dir)
    data_dir = staging_dir.parent
    queries_dir = Path("queries")

    start = time.monotonic()

    current_manifest = generate_manifest(data_dir)
    save_manifest(current_manifest, data_dir / MANIFEST_FILENAME)

    skipped_files: list[str] = []
    changed_file_details: list[dict[str, str]] = []

    if incremental:
        previous_manifest = load_manifest(data_dir / LAST_LOADED_MANIFEST_FILENAME)
        diff = compare_manifests(current_manifest, previous_manifest)
        if not diff.has_changes:
            print("No changes detected. Skipping incremental load.")
            return
        print(
            f"Incremental load: {len(diff.changed_files)} changed, "
            f"{len(diff.unchanged)} unchanged, {len(diff.removed)} removed."
        )
        skipped_files = diff.unchanged

        for f in diff.changed_files:
            entry: dict[str, str] = {"file": f, "md5_after": current_manifest[f]["md5"]}
            if f in previous_manifest:
                entry["md5_before"] = previous_manifest[f]["md5"]
            changed_file_details.append(entry)

    timeout_seconds = (
        validation_timeout
        if validation_timeout is not None
        else _DEFAULT_VALIDATION_TIMEOUT_SECONDS
    )

    try:
        with Neo4jClient() as client:
            summary = load_all(
                client,
                staging_dir,
                curated_dir,
                queries_dir,
                strict=False,
                skip_validation=skip_validation,
                nodes_only=nodes_only,
                skip_files=skipped_files if incremental else None,
                validation_timeout_seconds=timeout_seconds,
            )
    except Exception:
        # A genuine load failure (da#354). Exits LOAD_FAILED, distinct from the
        # code used for validation findings, and reaches neither the manifest
        # write nor the audit entry below -- a load that raised did not happen.
        #
        # That invariant is scoped to `isnad-ingest load`. It does NOT hold for
        # `isnad-ingest pipeline`, which goes on to run `_cmd_enrich`. Enrich's
        # own audit write sits outside any handler, and neither `main()` nor
        # `_cmd_pipeline` has a top-level one, so an escaping exception reaches
        # CPython's default handler and exits 1 -- long after this load committed
        # and recorded its manifest. Read rc=1 as "the load did not happen" only
        # for the `load` command.
        #
        # The mechanism, not the number: enrich once exited a bare 1 here, and
        # now exits ENRICH_FAILED (da#384 Amendment I). The leak outlived its
        # cause because it was never about enrich's exit code (da#394).
        traceback.print_exc()
        print(
            "\nLOAD FAILED: the graph was not fully written. See traceback above.",
            file=sys.stderr,
        )
        sys.exit(ExitCode.LOAD_FAILED)

    duration = time.monotonic() - start

    # Post-load bookkeeping. `load_all` returned, so the graph is COMMITTED --
    # the loaders write per batch with no spanning transaction. A failure to
    # record the manifest or the audit entry is therefore not a failed load, and
    # must not exit EXIT_LOAD_FAILED: an rc of 1 *before* the commit is safe to
    # retry, and an rc of 1 *after* it sends the operator to re-run a load whose
    # data is already in the graph. Nothing below may raise past this handler --
    # `main()` has none, so an escaping exception reaches Python's default
    # handler and exits 1, which is exactly the ambiguity this PR exists to end.
    #
    # Bookkeeping failure leaves the rc alone (0, or findings) rather than
    # claiming a code of its own; the exit-code space is being consolidated into
    # one registry (da#384) and a new value does not get minted here.
    try:
        save_manifest(current_manifest, data_dir / LAST_LOADED_MANIFEST_FILENAME)

        audit = create_audit_entry(
            "load",
            duration_seconds=round(duration, 2),
            files_changed=changed_file_details,
            rows_affected=summary.total_nodes + summary.total_edges,
            summary={
                "total_nodes": summary.total_nodes,
                "total_edges": summary.total_edges,
                "incremental": incremental,
                "files_skipped": len(skipped_files),
            },
        )
        write_audit_entry(data_dir, audit)
    except Exception:
        traceback.print_exc()
        print(
            "\nWARNING: the load SUCCEEDED and is committed to the graph, but "
            "post-load bookkeeping failed (traceback above). The last-loaded "
            "manifest and/or the audit entry may be missing. Do NOT read this as "
            "a load failure. A missing last-loaded manifest makes the next "
            "--incremental load re-load every input, which is safe: every loader "
            "is an idempotent MERGE.",
            file=sys.stderr,
        )

    print("\n=== Load Summary ===")
    print(f"  Nodes loaded : {summary.total_nodes}")
    print(f"  Edges loaded : {summary.total_edges}")
    if incremental:
        print(f"  Files skipped: {len(skipped_files)}")

    for nr in summary.node_results:
        print(
            f"    {nr.node_type}: created={nr.created} merged={nr.merged} skipped={nr.skipped}"
            f" refused={nr.refused} (malformed_ids={nr.malformed_ids}"
            f" invalid_source_ids={nr.invalid_source_ids})"
        )
    for er in summary.edge_results:
        print(
            f"    {er.edge_type}: created={er.created} skipped={er.skipped}"
            f" missing_endpoints={er.missing_endpoints} malformed_ids={er.malformed_ids}"
        )

    if summary.validation_results:
        print("\n=== Validation ===")
        for vr in summary.validation_results:
            print(f"  [{vr.status}] {vr.query_name}: {vr.details}")
        warned = [vr for vr in summary.validation_results if vr.warning]
        if summary.validation_passed and warned:
            # Non-fatal: the load succeeded and no check hard-failed; one or more
            # checks were downgraded (e.g. timed out). da#259.
            print(f"\nLoad OK. {len(warned)} validation check(s) downgraded to warning.")
        elif summary.validation_passed:
            print("\nAll validation checks passed.")

    # da#359. Deliberately OUTSIDE the `if summary.validation_results:` block: a
    # --nodes-only or --skip-validation load produces no validation results, and
    # would otherwise fall off the end with rc=0 having refused every row.
    #
    # Checked BEFORE the validation verdict because absent input is the stronger
    # statement. Validation reports on what landed; it cannot report on what was
    # never offered to it, which is why `collection_coverage` called a graph
    # missing 650,986 hadiths "all within threshold" (da#382).
    refused = summary.total_refused
    if refused:
        malformed = summary.total_malformed_ids
        print(
            f"\nREFUSED INPUT: {refused} row(s)/edge(s) were quarantined because their "
            f"source_id was defective ({malformed} violate the id grammar, "
            f"{refused - malformed} are absent/blank/non-string).\n"
            "The load ran to completion and committed what it accepted, but the graph is "
            "INCOMPLETE. This is a producer defect: re-run `parse` to regenerate the "
            f"staging artifact, then re-load. Exiting {EXIT_MALFORMED_IDS}.",
            file=sys.stderr,
        )
        sys.exit(EXIT_MALFORMED_IDS)

    if summary.validation_results and not summary.validation_passed:
        fatal = [vr for vr in summary.validation_results if vr.is_fatal]
        # The load is already committed to the graph. Findings are a report
        # ABOUT that graph, not evidence the write failed -- so they get an
        # exit code of their own (da#354). main#723 read this rc as a data
        # defect when the load had succeeded.
        print(
            f"\nLOAD SUCCEEDED ({summary.total_nodes} nodes, {summary.total_edges} edges "
            "written). Validation reported "
            f"{len(fatal)} finding(s): {', '.join(vr.query_name for vr in fatal)}.\n"
            f"The load did not fail. Exiting {ExitCode.VALIDATION_FINDINGS.value} "
            f"(validation findings), not {ExitCode.LOAD_FAILED.value} (load failure)."
        )
        sys.exit(ExitCode.VALIDATION_FINDINGS)


def _cmd_validate(*, validation_timeout: float | None = None) -> None:
    """Run graph validation queries against an existing Neo4j database."""
    from pathlib import Path

    _check_neo4j()

    from src.graph.validate import _DEFAULT_VALIDATION_TIMEOUT_SECONDS, run_validation
    from src.utils.neo4j_client import Neo4jClient

    queries_dir = Path("queries")
    timeout_seconds = (
        validation_timeout
        if validation_timeout is not None
        else _DEFAULT_VALIDATION_TIMEOUT_SECONDS
    )

    with Neo4jClient() as client:
        results = run_validation(client, queries_dir, timeout_seconds=timeout_seconds)

    if not results:
        print("No validation queries found.")
        sys.exit(0)

    print("=== Validation Results ===")
    any_fatal = False
    warned = 0
    for vr in results:
        print(f"  [{vr.status}] {vr.query_name}: {vr.details}")
        if vr.is_fatal:
            any_fatal = True
        elif vr.warning:
            warned += 1

    if any_fatal:
        fatal_names = ", ".join(vr.query_name for vr in results if vr.is_fatal)
        # `validate` only reads the graph. A finding describes the data; it is
        # not a failure to run (da#384 §4 / Amendment G, da#354).
        print(
            f"\nVALIDATION FINDINGS: {fatal_names}.\n"
            f"This is a report about the graph as it stands, not a load failure — "
            f"`validate` writes nothing. Exiting {ExitCode.VALIDATION_FINDINGS.value}."
        )
        sys.exit(ExitCode.VALIDATION_FINDINGS)
    elif warned:
        # Timed-out/downgraded checks are non-fatal (da#259).
        print(f"\nValidation OK. {warned} check(s) downgraded to warning.")
    else:
        print("\nAll validation checks passed.")


def _cmd_migrate_transmitted_hadith_ids(*, batch_size: int = 1000) -> None:
    """Canonicalize existing TRANSMITTED_TO.hadith_id edge ids in place (da#325).

    Remediates a live graph whose ``TRANSMITTED_TO`` edges carry the raw staging
    ``hadith_id`` (``sanadset:sanadset:...``) instead of the canonical
    ``Hadith.id`` (``hdt:sanadset:...``) — the mismatch that made the isnad-chain
    endpoint empty for every hadith. The loader fix only helps NEW loads; this
    rewrites the ~2.68M existing edges in place. Idempotent and safe to re-run.
    """
    _check_neo4j()

    from src.graph.migrate import migrate_transmitted_hadith_ids
    from src.utils.neo4j_client import Neo4jClient

    with Neo4jClient() as client:
        result = migrate_transmitted_hadith_ids(client, batch_size=batch_size)

    print("=== TRANSMITTED_TO.hadith_id migration (da#325) ===")
    print(f"  Distinct hadith_id values seen  : {result.distinct_ids_seen}")
    print(f"  Distinct values needing rewrite : {result.distinct_ids_rewritten}")
    print(f"  Edges updated                   : {result.edges_updated}")
    if result.distinct_ids_rewritten == 0:
        print("\nAll TRANSMITTED_TO.hadith_id values already canonical — nothing to do.")
    else:
        print("\nMigration complete.")


def _cmd_enrich(
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    incremental: bool = False,
    betweenness_sampling_size: int | None = None,
) -> None:
    """Run the Phase 4 enrichment pipeline."""
    import time
    from pathlib import Path

    from src.config import get_settings
    from src.enrich import run_all as enrich_all
    from src.pipeline.audit import create_audit_entry, write_audit_entry
    from src.pipeline.manifest import (
        LAST_LOADED_MANIFEST_FILENAME,
        MANIFEST_FILENAME,
        compare_manifests,
        generate_manifest,
        load_manifest,
    )
    from src.utils.neo4j_client import Neo4jClient

    settings = get_settings()
    _check_neo4j()

    # CLI --betweenness-sampling-size overrides the configured default (da#326).
    sampling_size = (
        betweenness_sampling_size
        if betweenness_sampling_size is not None
        else settings.betweenness_sampling_size
    )

    staging_dir = Path(settings.data_staging_dir)
    data_dir = staging_dir.parent
    start = time.monotonic()

    affected_corpora: set[str] | None = None
    changed_file_details: list[dict[str, str]] = []

    if incremental:
        current_manifest = generate_manifest(data_dir)
        previous_manifest = load_manifest(data_dir / LAST_LOADED_MANIFEST_FILENAME)
        if not previous_manifest:
            previous_manifest = load_manifest(data_dir / MANIFEST_FILENAME)
        diff = compare_manifests(current_manifest, previous_manifest)
        if not diff.has_changes:
            print("No changes detected. Skipping incremental enrich.")
            return

        affected_corpora = set()
        for f in diff.changed_files:
            basename = f.rsplit("/", 1)[-1]
            parts = basename.replace(".parquet", "").split("_", 1)
            if len(parts) > 1:
                affected_corpora.add(parts[1])

            entry: dict[str, str] = {"file": f, "md5_after": current_manifest[f]["md5"]}
            if f in previous_manifest:
                entry["md5_before"] = previous_manifest[f]["md5"]
            changed_file_details.append(entry)

        print(
            f"Incremental enrich: {len(diff.changed_files)} files changed, "
            f"affected corpora: {', '.join(sorted(affected_corpora)) or 'none'}"
        )

    with Neo4jClient() as client:
        summary = enrich_all(
            client,
            staging_dir,
            only=only,
            skip=skip,
            affected_corpora=affected_corpora,
            betweenness_sampling_size=sampling_size,
            betweenness_sampling_seed=settings.betweenness_sampling_seed,
        )

    duration = time.monotonic() - start

    audit = create_audit_entry(
        "enrich",
        duration_seconds=round(duration, 2),
        files_changed=changed_file_details,
        summary={
            "steps_completed": summary.steps_completed,
            "steps_failed": summary.steps_failed,
            "incremental": incremental,
            "affected_corpora": sorted(affected_corpora) if affected_corpora else [],
        },
    )
    write_audit_entry(data_dir, audit)

    print("\n=== Enrich Summary ===")
    print(f"  Steps completed: {', '.join(summary.steps_completed) or 'none'}")
    print(f"  Steps failed   : {', '.join(summary.steps_failed) or 'none'}")

    if summary.metrics:
        m = summary.metrics
        print(
            f"\n  Metrics: {m.narrators_enriched} narrators enriched, "
            f"{m.communities_found} communities"
        )
    if summary.topics:
        t = summary.topics
        print(f"  Topics: {t.hadiths_classified} classified, {t.hadiths_skipped} skipped")
    if summary.historical:
        h = summary.historical
        print(
            f"  Historical: {h.edges_created} edges, "
            f"{h.narrators_linked} narrators, {h.events_linked} events"
        )

    if summary.steps_failed:
        # The graph, the manifest and the audit entry are all written; the

        # ENRICH stage failed. Not LOAD_FAILED -- a failed enrich is not a

        # failed load, and routing it through the registry would make the

        # registry assert that falsehood authoritatively. da#384 Amendment I.

        sys.exit(ExitCode.ENRICH_FAILED)


def _cmd_audit(*, last_n: int = 10) -> None:
    """Display recent pipeline audit entries."""
    from pathlib import Path

    from src.config import get_settings
    from src.pipeline.audit import list_recent_entries

    settings = get_settings()
    data_dir = Path(settings.data_staging_dir).parent

    entries = list_recent_entries(data_dir, last_n=last_n)
    if not entries:
        print("No audit entries found.")
        return

    print(f"=== Last {len(entries)} Audit Entries ===\n")
    for entry in entries:
        print(f"  [{entry.stage}] {entry.timestamp}")
        print(f"    Duration: {entry.duration_seconds}s | Operator: {entry.operator}")
        if entry.files_changed:
            print(f"    Files changed: {len(entry.files_changed)}")
        if entry.rows_affected:
            print(f"    Rows affected: {entry.rows_affected}")
        if entry.summary:
            for k, v in entry.summary.items():
                print(f"    {k}: {v}")
        print()


def _cmd_pipeline() -> None:
    """Run the full pipeline: acquire -> parse -> resolve -> load -> enrich."""
    print("=== Full Pipeline Run ===\n")

    print("--- Stage 1: Acquire ---")
    _cmd_acquire()
    print()

    print("--- Stage 2: Parse ---")
    _cmd_parse()
    print()

    print("--- Stage 3: Resolve ---")
    _cmd_resolve()
    print()

    print("--- Stage 4: Load ---")
    _cmd_load()
    print()

    print("--- Stage 5: Enrich ---")
    _cmd_enrich()
    print()

    print("=== Full Pipeline Complete ===")


def main() -> None:
    """Run the isnad-ingest CLI."""
    from src.graph.validate import _DEFAULT_VALIDATION_TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(description="isnad-ingest: Hadith Data Ingestion Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    from src.resolve import RESOLVE_STEP_ORDER, RESUMABLE_STEPS

    subparsers.add_parser("info", help="Show configuration and database status")
    subparsers.add_parser("acquire", help="Download data sources")
    subparsers.add_parser("parse", help="Parse raw data to staging")
    resolve_parser = subparsers.add_parser("resolve", help="Entity resolution")
    resolve_parser.add_argument(
        "--from-step",
        choices=list(RESOLVE_STEP_ORDER),
        default=None,
        help=(
            "Resume the pipeline from this step, skipping earlier steps whose "
            "outputs already exist (da#268). Use --from-step disambiguate to "
            "recover a mid-disambiguate crash: NER is skipped so the existing "
            "mentions file and disambiguate's checkpoint are reused."
        ),
    )
    resolve_parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Force every crash-resumable stage (disambiguate, fuzzy_cluster's "
            "block-scoring pass, dedup's FAISS/collection phase, parallels' anchor "
            "scan) to discard its checkpoint and cold-start (da#272). Independent of "
            "--from-step, which selects WHICH step to begin at; --no-resume selects "
            "HOW that step starts."
        ),
    )
    resolve_parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Bounded partial-run probe (da#276): stop the resumable stage cleanly "
            "after its Nth checkpoint write, leaving the checkpoint on disk (a "
            "later bare run resumes from it) and writing NO final output, then halt "
            "the pipeline with a perf summary and a distinct exit status. Only "
            "applies to resumable stages (disambiguate, cluster, dedup, parallels); "
            "pairing it with --from-step on an exempt stage is an error. Combine "
            "with --from-step <stage> --no-resume for a cold bounded probe of one "
            "stage."
        ),
    )

    load_parser = subparsers.add_parser("load", help="Load graph database")
    load_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Explicitly skip post-load validation queries. Not required for a full "
            "graph load: validation is time-bounded and a slow query downgrades to "
            "a non-fatal warning (da#259)."
        ),
    )
    load_parser.add_argument(
        "--nodes-only", action="store_true", help="Load only nodes (skip edges and validation)"
    )
    load_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only load Parquet files whose hash changed since last load",
    )
    load_parser.add_argument(
        "--validation-timeout",
        type=float,
        default=None,
        help=(
            "Per-query validation time budget in seconds (default: "
            f"{int(_DEFAULT_VALIDATION_TIMEOUT_SECONDS)}). On overrun the check "
            "is reported as a WARN and does not fail the load."
        ),
    )

    enrich_parser = subparsers.add_parser("enrich", help="Compute metrics and enrichment")
    enrich_parser.add_argument(
        "--only",
        nargs="+",
        choices=["metrics", "topics", "historical"],
        help="Run only these enrichment steps",
    )
    enrich_parser.add_argument(
        "--skip",
        nargs="+",
        choices=["metrics", "topics", "historical"],
        help="Skip these enrichment steps",
    )
    enrich_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only re-enrich data affected by changed Parquet files",
    )
    enrich_parser.add_argument(
        "--betweenness-sampling-size",
        type=int,
        default=None,
        help=(
            "Pivot count for sampled (approximate) betweenness centrality "
            "(da#326). Overrides the configured default. 0 = exact betweenness "
            "(intractable at prod scale; use only for small graphs)."
        ),
    )

    validate_parser = subparsers.add_parser("validate", help="Run graph validation queries")
    validate_parser.add_argument(
        "--validation-timeout",
        type=float,
        default=None,
        help=(
            "Per-query validation time budget in seconds (default: "
            f"{int(_DEFAULT_VALIDATION_TIMEOUT_SECONDS)}). On overrun the check "
            "is reported as a WARN and is non-fatal."
        ),
    )

    vs_parser = subparsers.add_parser("validate-staging", help="Validate staging Parquet files")
    vs_parser.add_argument(
        "--strict",
        action="store_true",
        help="Halt on any validation failure (default: warn mode)",
    )
    vs_parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write JSON validation report to this path",
    )
    vs_parser.add_argument(
        "--drift-tolerance",
        type=float,
        default=30.0,
        help="Maximum drift percentage from baseline (default: 30.0)",
    )

    migrate_parser = subparsers.add_parser(
        "migrate-transmitted-hadith-ids",
        help="Canonicalize existing TRANSMITTED_TO.hadith_id edge ids in place (da#325)",
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Edges rewritten per write batch (default: 1000)",
    )

    audit_parser = subparsers.add_parser("audit", help="View recent pipeline audit entries")
    audit_parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of recent audit entries to display (default: 10)",
    )

    subparsers.add_parser("pipeline", help="Run the full pipeline end-to-end")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "resolve" and args.stop_after is not None:
        # Validate --stop-after up front (argparse exit code 2) so the bounded
        # probe can never silently no-op (da#276).
        if args.stop_after < 1:
            parser.error("--stop-after N must be a positive integer")
        if args.from_step is not None and args.from_step not in RESUMABLE_STEPS:
            resumable = ", ".join(sorted(RESUMABLE_STEPS))
            parser.error(
                f"--stop-after cannot bound '{args.from_step}': it keeps no checkpoint. "
                f"Target a resumable stage with --from-step ({resumable}), or omit "
                f"--from-step to bound the first resumable stage (disambiguate)."
            )

    if args.command == "info":
        _cmd_info()
    elif args.command == "acquire":
        _cmd_acquire()
    elif args.command == "parse":
        _cmd_parse()
    elif args.command == "resolve":
        _cmd_resolve(
            from_step=args.from_step, resume=not args.no_resume, stop_after=args.stop_after
        )
    elif args.command == "load":
        _cmd_load(
            skip_validation=args.skip_validation,
            nodes_only=args.nodes_only,
            incremental=args.incremental,
            validation_timeout=args.validation_timeout,
        )
    elif args.command == "enrich":
        _cmd_enrich(
            only=args.only,
            skip=args.skip,
            incremental=args.incremental,
            betweenness_sampling_size=args.betweenness_sampling_size,
        )
    elif args.command == "validate":
        _cmd_validate(validation_timeout=args.validation_timeout)
    elif args.command == "validate-staging":
        from pathlib import Path

        from src.config import get_settings
        from src.parse.validate import Strictness, validate_staging

        settings = get_settings()
        strictness = Strictness.STRICT if args.strict else Strictness.WARN
        output_json = Path(args.output_json) if args.output_json else None
        report = validate_staging(
            Path(settings.data_staging_dir),
            strictness=strictness,
            drift_tolerance_pct=args.drift_tolerance,
            output_json=output_json,
        )
        if not report.passed and strictness == Strictness.STRICT:
            # FINDINGS, not a failure to run: `validate-staging` reads the staging
            # parquets and writes nothing. It exited a bare 1 while `validate`
            # reports findings with its own code -- da#354's conflation, one
            # subcommand over (da#384 §4).
            print("\nValidation FAILED in strict mode. Pipeline halted.")
            sys.exit(ExitCode.VALIDATION_FINDINGS)
    elif args.command == "migrate-transmitted-hadith-ids":
        _cmd_migrate_transmitted_hadith_ids(batch_size=args.batch_size)
    elif args.command == "audit":
        _cmd_audit(last_n=args.last)
    elif args.command == "pipeline":
        _cmd_pipeline()


if __name__ == "__main__":
    main()
