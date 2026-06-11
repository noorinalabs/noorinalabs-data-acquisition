#!/usr/bin/env python3
"""Data-first vertical slice ("first light") — da#73.

Take the verified keyless ``sunnah_scraper`` sample (riyadussalihin), run it
through parse (``parse.sunnah_scraped``) and the Phase-3 graph loader into the
configured Neo4j, then report node/edge counts and a verification Cypher query.

This is a *thin* slice: it proves the acquire -> parse -> load -> query path
end-to-end on one collection, ahead of the full multi-source pipeline
(main#139). It does NOT scale to all 8 sources.

Two acquisition modes:

* ``--sample`` (default): replay the committed real sample
  (``scripts/first_light/sample/sunnah_scraped/``). Offline, deterministic,
  reproducible by reviewers and CI without network.
* ``--live``: live-scrape ``--collection``/``--book`` from sunnah.com via the
  real ``sunnah_scraper`` functions (rate-limited; needs network).

The Neo4j target is whatever ``get_settings().neo4j`` resolves to (``NEO4J_URI``
etc.). To load **staging**, run this from inside the cluster (or with a tunnel)
where ``bolt://neo4j:7687`` resolves, with ``NEO4J_PASSWORD`` set. Use
``--dry-run`` to run acquire+parse only and skip the Neo4j connection.

Examples
--------
    # Offline: replay the committed sample, parse only (no Neo4j)
    uv run python scripts/first_light/run_slice.py --dry-run

    # Staging load (run from inside the cluster, NEO4J_* pointed at staging)
    uv run python scripts/first_light/run_slice.py

    # Live re-scrape riyadussalihin book 1, then load
    uv run python scripts/first_light/run_slice.py --live --collection riyadussalihin --book 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = Path(__file__).resolve().parent / "sample" / "sunnah_scraped"

# Verification query reported (and runnable) after a load. Counts the loaded
# hadiths and their APPEARS_IN edges into the collection, with a few samples.
VERIFY_CYPHER = """\
MATCH (c:Collection {id: 'col:sunnah:riyadussalihin'})
OPTIONAL MATCH (h:Hadith)-[:APPEARS_IN]->(c)
RETURN c.name_en AS collection,
       count(h) AS hadith_count,
       collect(h.matn_en)[0..3] AS sample_matn_en"""


def _acquire_live(raw_dir: Path, collection: str, book: int) -> Path:
    """Live-scrape one book of one collection into ``raw_dir/sunnah_scraped``."""
    import httpx

    from src.acquire.base import ensure_dir, write_manifest
    from src.acquire.sunnah_scraper import (
        REQUEST_TIMEOUT,
        USER_AGENT,
        _scrape_book_page,
    )

    dest = ensure_dir(raw_dir / "sunnah_scraped")
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True
    )
    try:
        hadiths = _scrape_book_page(client, collection, book)
    finally:
        client.close()

    out = dest / f"{collection}.json"
    out.write_text(json.dumps(hadiths, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(dest, [out])
    print(f"[acquire:live] scraped {len(hadiths)} hadiths from {collection} book {book} -> {out}")
    return dest


def _acquire_sample(raw_dir: Path) -> Path:
    """Copy the committed real sample into ``raw_dir/sunnah_scraped``."""
    from src.acquire.base import ensure_dir, write_manifest

    if not SAMPLE_DIR.exists():
        print(f"ERROR: committed sample missing at {SAMPLE_DIR}", file=sys.stderr)
        sys.exit(1)

    dest = ensure_dir(raw_dir / "sunnah_scraped")
    copied: list[Path] = []
    for src in sorted(SAMPLE_DIR.glob("*.json")):
        if src.name == "manifest.json":
            continue
        target = dest / src.name
        shutil.copyfile(src, target)
        copied.append(target)
    write_manifest(dest, copied)
    total = sum(len(json.loads(p.read_text(encoding="utf-8"))) for p in copied)
    print(f"[acquire:sample] replayed {len(copied)} file(s), {total} hadiths -> {dest}")
    return dest


def _parse(raw_dir: Path, staging_dir: Path) -> int:
    """Parse scraped JSON into staging Parquet. Returns hadith row count."""
    import pyarrow.parquet as pq

    from src.parse import sunnah_scraped

    files = sunnah_scraped.run(raw_dir, staging_dir)
    print(f"[parse] wrote {len(files)} parquet file(s): {[f.name for f in files]}")

    hadith_path = staging_dir / "hadiths_sunnah_scraped.parquet"
    rows = pq.read_table(hadith_path).to_pylist()
    distinct = len({r["source_id"] for r in rows})
    collisions = len(rows) - distinct
    print(f"[parse] hadith rows={len(rows)} distinct source_id={distinct} collisions={collisions}")
    if collisions:
        # da#72: hadith_number is not extracted, so chapter-grouped collections
        # can collide on source_id. Surface it loudly; the load MERGEs on id so
        # collisions silently coalesce rows rather than erroring.
        print(
            f"[parse] WARNING: {collisions} source_id collision(s) — see da#72 "
            "(hadith_number not extracted). Colliding hadiths will MERGE into one node.",
            file=sys.stderr,
        )
    return len(rows)


def _load(staging_dir: Path, curated_dir: Path) -> None:
    """Load staging Parquet into the configured Neo4j and run verification."""
    from src.graph import load_all
    from src.utils.neo4j_client import Neo4jClient

    queries_dir = REPO_ROOT / "queries"
    with Neo4jClient() as client:
        summary = load_all(
            client,
            staging_dir,
            curated_dir,
            queries_dir,
            strict=False,  # thin slice: narrators/chains/parallel files are absent
            skip_validation=True,  # validation queries assume a fuller graph
        )

        print("\n=== Load Summary ===")
        print(f"  Nodes loaded : {summary.total_nodes}")
        print(f"  Edges loaded : {summary.total_edges}")
        for nr in summary.node_results:
            if nr.created or nr.merged:
                print(f"    {nr.node_type}: created={nr.created} merged={nr.merged}")
        for er in summary.edge_results:
            if er.created:
                print(f"    {er.edge_type}: created={er.created}")

        print("\n=== Verification (Cypher) ===")
        print(VERIFY_CYPHER)
        result = client.execute_read(VERIFY_CYPHER)
        print("\n  Result:")
        for rec in result:
            samples = rec.get("sample_matn_en") or []
            print(f"    collection   : {rec.get('collection')}")
            print(f"    hadith_count : {rec.get('hadith_count')}")
            for i, s in enumerate(samples):
                snippet = (s or "")[:80]
                print(f"    sample[{i}]    : {snippet}")


def main() -> None:
    """Run the first-light vertical slice."""
    parser = argparse.ArgumentParser(description="da#73 data-first vertical slice")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Replay the committed real sample (default; offline)",
    )
    mode.add_argument("--live", action="store_true", help="Live-scrape from sunnah.com")
    parser.add_argument("--collection", default="riyadussalihin", help="Collection (live mode)")
    parser.add_argument("--book", type=int, default=1, help="Book number (live mode)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Acquire + parse only; skip the Neo4j load (no DB connection)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch dir for raw/staging (default: a temp dir)",
    )
    args = parser.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="first_light_"))
    raw_dir = work_dir / "raw"
    staging_dir = work_dir / "staging"
    curated_dir = work_dir / "curated"
    for d in (raw_dir, staging_dir, curated_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== first-light slice (da#73) — work_dir={work_dir} ===")

    if args.live:
        _acquire_live(raw_dir, args.collection, args.book)
    else:
        _acquire_sample(raw_dir)

    row_count = _parse(raw_dir, staging_dir)

    if args.dry_run:
        print(f"\n[dry-run] parsed {row_count} hadiths; skipping Neo4j load.")
        return

    _load(staging_dir, curated_dir)
    print("\n=== first-light slice complete ===")


if __name__ == "__main__":
    main()
