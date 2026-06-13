#!/usr/bin/env python3
"""Produce + load the REAL narrator graph — da#141 (main#601 criterion #1).

The data half of the Phase-4 #601 end-state. Where the da#73 first-light slice
(``scripts/first_light/run_slice.py``) proved only the hadith + ``APPEARS_IN``
path on one collection, this driver produces and loads the **full narrator
graph** — ``Narrator`` + ``NARRATED`` + ``STUDIED_UNDER`` + ``APPEARS_IN`` +
``TRANSMITTED_TO``, **both sects** — end-to-end from REAL sources (not toy
fixtures):

    acquire -> parse -> resolve.run_all -> graph.load_all -> verify

Default source set (bounded but REAL, so the whole path runs locally against a
``neo4j:5`` container in ~10 min):

* ``muhaddithat`` — cross-tradition (both-sects) female narrators. No isnad
  text, so its narrators reach the canonical table via ``bio_promote``
  (**bio-only promoted** narrators, tagged ``neutral``) and its studentship
  pairs become ``STUDIED_UNDER`` edges. Open-licensed GitHub CSV; tiny.
* ``lk`` (LK-Hadith-Corpus) — 6 Sunni books with isnad chains. Produces hadiths
  + narrator mentions, which ``NER`` -> ``disambiguate`` resolve to canonical
  narrators and the loader turns into ``NARRATED`` + ``TRANSMITTED_TO`` +
  ``APPEARS_IN``. Open GitHub CSV.

Together they exercise every narrator-graph edge type AND both sects (``lk``
sunni mention-derived narrators + ``muhaddithat`` neutral bio-only narrators),
the da#120 disambiguate-before-bio_promote ordering, and the da#133 per-row
``relation`` STUDIED_UNDER routing.

Widen the set with ``--sources`` for the full staging load — e.g. add
``thaqalayn`` (Shia Four Books, for Shia hadith coverage + cross-sect
``PARALLEL_OF``) or ``itqan`` (the 100k-narrator rijal DB). See
``docs/narrator-graph-load.md`` for the gated staging-load runbook.

Modes
-----
* default            : acquire + parse + resolve + load + verify.
* ``--produce-only`` : acquire + parse + resolve; skip Neo4j (no DB connection).
* ``--skip-resolve`` : reuse an existing ``<data>/curated`` (skip the slow
  resolve) and go straight to load + verify.

The Neo4j target is whatever ``get_settings().neo4j`` resolves to
(``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``). Loading **staging** is a
GATED run (it pairs with the ingest-platform worker bring-up) — do it from the
runbook, never casually from a dev worktree.

Examples
--------
    # Local: produce from the default real sources, load a local neo4j, verify.
    NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
        uv run python scripts/narrator_graph/run.py

    # Produce only (no DB); inspect the curated parquet.
    uv run python scripts/narrator_graph/run.py --produce-only

    # Re-load already-produced data without re-resolving.
    uv run python scripts/narrator_graph/run.py --skip-resolve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bounded-but-real default set: both sects, every narrator-graph edge type.
DEFAULT_SOURCES = ("muhaddithat", "lk")

# Verification queries, run (and printed) after a load. Each returns one row.
VERIFY_QUERIES: dict[str, str] = {
    "narrators_total": "MATCH (n:Narrator) RETURN count(n) AS v",
    "narrators_by_sect": (
        "MATCH (n:Narrator) RETURN n.sect_affiliation AS sect, count(n) AS v ORDER BY v DESC"
    ),
    "narrators_by_corpus": (
        "MATCH (n:Narrator) RETURN n.source_corpus AS corpus, count(n) AS v ORDER BY v DESC"
    ),
    # A bio-only promoted narrator: muhaddithat has no isnad text, so any
    # muhaddithat Narrator node came through bio_promote, not a mention.
    "bio_only_promoted_sample": (
        "MATCH (n:Narrator {source_corpus: 'muhaddithat'}) "
        "RETURN n.id AS id, n.name_ar AS name_ar, n.sect_affiliation AS sect LIMIT 1"
    ),
    "studied_under_edges": "MATCH (:Narrator)-[r:STUDIED_UNDER]->(:Narrator) RETURN count(r) AS v",
    "narrated_edges": "MATCH (:Narrator)-[r:NARRATED]->(:Hadith) RETURN count(r) AS v",
    "transmitted_to_edges": (
        "MATCH (:Narrator)-[r:TRANSMITTED_TO]->(:Narrator) RETURN count(r) AS v"
    ),
    "appears_in_edges": "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS v",
    # The ip#84 / da#77 null-safe MERGE: APPEARS_IN edges whose in-book ordinal
    # is null must still be created (a null property in the MERGE pattern would
    # abort the load). lk supplies the positional ordinal, so this is reported
    # rather than asserted; it is asserted when a null-ordinal source (e.g.
    # sunnah_scraped) is in the set.
    "appears_in_null_ordinal": (
        "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) "
        "WHERE r.hadith_number_in_book IS NULL RETURN count(r) AS v"
    ),
    "parallel_of_edges": "MATCH (:Hadith)-[r:PARALLEL_OF]->(:Hadith) RETURN count(r) AS v",
}


def _produce(sources: tuple[str, ...], raw: Path, staging: Path, curated: Path) -> None:
    """Acquire + parse each source, then run the full resolve pipeline."""
    from src.adapters import get_adapter
    from src.resolve import run_all as resolve_all

    for slug in sources:
        adapter = get_adapter(slug)
        print(f"[acquire] {slug} ({adapter.license_note.split('.')[0]})")
        adapter.acquire(raw)
        print(f"[parse]   {slug}")
        adapter.parse(raw, staging)

    print("[resolve] NER -> disambiguate -> bio_promote -> cluster -> dedup + parallels")
    resolve_all(raw, staging, curated)

    canonical = curated / "narrators_canonical.parquet"
    if not canonical.exists():
        print(f"ERROR: resolve produced no {canonical}", file=sys.stderr)
        sys.exit(1)

    import pyarrow.parquet as pq

    rows = pq.read_table(canonical).num_rows
    print(f"[produce] {rows} canonical narrators in {canonical}")


def _load_and_verify(staging: Path, curated: Path, *, strict_assert: bool) -> int:
    """Load the produced graph into the configured Neo4j and verify counts.

    Returns a process exit code (0 = all required edge/node types non-empty).
    """
    from src.graph import load_all
    from src.utils.neo4j_client import Neo4jClient

    queries_dir = REPO_ROOT / "queries"
    with Neo4jClient() as client:
        summary = load_all(
            client,
            staging,
            curated,
            queries_dir,
            strict=False,
            # The validation queries assume a fuller multi-source graph; the
            # produce+load+verify here reports its own counts below instead.
            skip_validation=True,
        )

        print("\n=== Load Summary ===")
        print(f"  Nodes loaded : {summary.total_nodes}")
        print(f"  Edges loaded : {summary.total_edges}")
        for nr in summary.node_results:
            print(f"    node {nr.node_type:12s}: created={nr.created} merged={nr.merged}")
        for er in summary.edge_results:
            print(
                f"    edge {er.edge_type:14s}: created={er.created} "
                f"missing_endpoints={er.missing_endpoints}"
            )

        print("\n=== Verification (Cypher counts) ===")
        results: dict[str, list[dict[str, object]]] = {}
        for name, cypher in VERIFY_QUERIES.items():
            rows = client.execute_read(cypher)
            results[name] = rows
            if len(rows) == 1 and "v" in rows[0]:
                print(f"  {name:26s}: {rows[0]['v']}")
            else:
                print(f"  {name:26s}: {rows}")

    # Required non-empty invariants for the narrator graph (both sects).
    def _scalar(name: str) -> int:
        rows = results.get(name) or []
        return int(rows[0]["v"]) if rows and "v" in rows[0] else 0

    checks = {
        "Narrator nodes > 0": _scalar("narrators_total") > 0,
        "STUDIED_UNDER edges > 0": _scalar("studied_under_edges") > 0,
        "NARRATED edges > 0": _scalar("narrated_edges") > 0,
        "APPEARS_IN edges > 0": _scalar("appears_in_edges") > 0,
        "bio-only promoted narrator present": bool(results.get("bio_only_promoted_sample")),
        "both sects present (>= 2 sect tags)": len(results.get("narrators_by_sect") or []) >= 2,
    }
    print("\n=== Required invariants ===")
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    if failed and strict_assert:
        print(f"\nFAILED invariants: {failed}", file=sys.stderr)
        return 1
    print("\n=== narrator-graph slice complete ===")
    return 0


def main() -> None:
    """Run the da#141 produce + load + verify driver."""
    parser = argparse.ArgumentParser(description="da#141 produce + load the real narrator graph")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help=f"Adapter slugs to produce from (default: {' '.join(DEFAULT_SOURCES)})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root data dir holding raw/ staging/ curated/ (default: ./data)",
    )
    parser.add_argument(
        "--produce-only",
        action="store_true",
        help="Acquire + parse + resolve only; skip the Neo4j load (no DB connection)",
    )
    parser.add_argument(
        "--skip-resolve",
        action="store_true",
        help="Reuse an existing curated/ (skip acquire/parse/resolve); load + verify only",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Report verification counts but do not exit non-zero on an empty required type",
    )
    args = parser.parse_args()

    raw = args.data_dir / "raw"
    staging = args.data_dir / "staging"
    curated = args.data_dir / "curated"
    for d in (raw, staging, curated):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== narrator-graph slice (da#141) — data_dir={args.data_dir} ===")
    print(f"    sources: {', '.join(args.sources)}")

    if not args.skip_resolve:
        _produce(tuple(args.sources), raw, staging, curated)
    else:
        print("[skip-resolve] reusing existing curated/ + staging/")

    if args.produce_only:
        print("\n[produce-only] skipping Neo4j load.")
        return

    exit_code = _load_and_verify(staging, curated, strict_assert=not args.no_assert)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
