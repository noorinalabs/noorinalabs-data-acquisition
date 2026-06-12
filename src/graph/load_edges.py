"""Neo4j edge/relationship loading.

Batch UNWIND loaders for 6 relationship types: TRANSMITTED_TO, NARRATED,
APPEARS_IN, PARALLEL_OF, STUDIED_UNDER, and GRADED_BY.  Each loader uses
MATCH (not MERGE) for endpoints, logging and counting missing endpoints
rather than silently creating dangling references.  Edge creation uses MERGE
for idempotent re-runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.parse.identity import (
    collection_node_id,
    grading_node_id,
    hadith_node_id,
    make_canonical_id,
    narrator_node_id,
)
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = ["load_all_edges", "EdgeLoadResult"]

DEFAULT_BATCH_SIZE = 1000


@dataclass(frozen=True)
class EdgeLoadResult:
    """Outcome of loading a single edge/relationship type."""

    edge_type: str
    created: int
    skipped: int
    missing_endpoints: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked_read(
    client: Neo4jClient,
    query: str,
    batch: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Execute a read query in chunks to avoid memory issues at scale."""
    results: list[dict[str, Any]] = []
    for i in range(0, len(batch), batch_size):
        chunk = batch[i : i + batch_size]
        results.extend(client.execute_read(query, {"batch": chunk}))
    return results


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file and return row dicts."""
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _parquet_files(directory: Path, prefix: str) -> list[Path]:
    """Return sorted parquet files matching *prefix* in *directory*."""
    return sorted(directory.glob(f"{prefix}*.parquet"))


def _val(row: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from *row*, returning *default* for ``None``."""
    v = row.get(key)
    return default if v is None else v


# ---------------------------------------------------------------------------
# 1. TRANSMITTED_TO — consecutive narrator pairs in each chain
# ---------------------------------------------------------------------------

_TRANSMITTED_TO_QUERY = """\
UNWIND $batch AS row
MATCH (n1:Narrator {id: row.from_id})
MATCH (n2:Narrator {id: row.to_id})
MERGE (n1)-[:TRANSMITTED_TO {
    position_in_chain: row.position,
    hadith_id: row.hadith_id
}]->(n2)
"""

_TRANSMITTED_TO_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (n1:Narrator {id: row.from_id})
OPTIONAL MATCH (n2:Narrator {id: row.to_id})
RETURN row.from_id AS from_id,
       row.to_id AS to_id,
       n1 IS NOT NULL AS from_exists,
       n2 IS NOT NULL AS to_exists
"""


def _build_chain_pairs(
    mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build consecutive narrator pairs from sorted chain mentions.

    Each mention must have ``canonical_narrator_id``, ``position_in_chain``,
    and ``hadith_id``.  Returns dicts with ``from_id``, ``to_id``,
    ``position``, and ``hadith_id``.
    """
    sorted_mentions = sorted(mentions, key=lambda r: r.get("position_in_chain", 0))
    # Filter to mentions with resolved narrator IDs
    resolved = [m for m in sorted_mentions if m.get("canonical_narrator_id")]
    pairs: list[dict[str, Any]] = []
    for i in range(len(resolved) - 1):
        pairs.append(
            {
                "from_id": resolved[i]["canonical_narrator_id"],
                "to_id": resolved[i + 1]["canonical_narrator_id"],
                "position": resolved[i].get("position_in_chain", i),
                "hadith_id": resolved[i].get("hadith_id", ""),
            }
        )
    return pairs


def _load_transmitted_to(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load TRANSMITTED_TO edges from narrator_mentions_resolved.parquet."""
    files = _parquet_files(staging_dir, "narrator_mentions_resolved")
    if not files:
        files = _parquet_files(staging_dir, "narrator_mentions_")
    if not files:
        if strict:
            msg = f"No narrator_mentions files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("transmitted_to_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("TRANSMITTED_TO", 0, 0, 0)

    # Group mentions by hadith
    by_hadith: dict[str, list[dict[str, Any]]] = {}
    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            hid = row.get("hadith_id") or row.get("source_hadith_id")
            if not hid:
                continue
            by_hadith.setdefault(hid, []).append(row)

    # Build all chain pairs
    all_pairs: list[dict[str, Any]] = []
    for hid, mentions in by_hadith.items():
        all_pairs.extend(_build_chain_pairs(mentions))

    if not all_pairs:
        logger.info("transmitted_to_no_pairs")
        return EdgeLoadResult("TRANSMITTED_TO", 0, 0, 0)

    # Check for missing endpoints
    check_results = _chunked_read(client, _TRANSMITTED_TO_CHECK, all_pairs, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for pair, check in zip(all_pairs, check_results):
        if check.get("from_exists") and check.get("to_exists"):
            valid_batch.append(pair)
        else:
            missing += 1
            if not check.get("from_exists"):
                logger.debug("transmitted_to_missing_from", id=pair["from_id"])
            if not check.get("to_exists"):
                logger.debug("transmitted_to_missing_to", id=pair["to_id"])

    created = (
        client.execute_write_batch(_TRANSMITTED_TO_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info(
        "transmitted_to_loaded",
        created=created,
        missing_endpoints=missing,
        total_pairs=len(all_pairs),
    )
    return EdgeLoadResult("TRANSMITTED_TO", created, 0, missing)


# ---------------------------------------------------------------------------
# 2. NARRATED — first narrator in each chain -> hadith
# ---------------------------------------------------------------------------

_NARRATED_QUERY = """\
UNWIND $batch AS row
MATCH (n:Narrator {id: row.narrator_id})
MATCH (h:Hadith {id: row.hadith_id})
MERGE (n)-[:NARRATED]->(h)
"""

_NARRATED_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (n:Narrator {id: row.narrator_id})
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
RETURN row.narrator_id AS narrator_id,
       row.hadith_id AS hadith_id,
       n IS NOT NULL AS narrator_exists,
       h IS NOT NULL AS hadith_exists
"""


def _load_narrated(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load NARRATED edges — first narrator (position 0) in each chain -> hadith."""
    files = _parquet_files(staging_dir, "narrator_mentions_resolved")
    if not files:
        files = _parquet_files(staging_dir, "narrator_mentions_")
    if not files:
        if strict:
            msg = f"No narrator_mentions files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("narrated_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("NARRATED", 0, 0, 0)

    # Find position-0 narrator per hadith (lowest position_in_chain)
    first_narrators: dict[str, tuple[int, str]] = {}  # hid -> (pos, narrator_id)
    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            hid = row.get("hadith_id") or row.get("source_hadith_id")
            nid = row.get("canonical_narrator_id")
            pos = row.get("position_in_chain", 0)
            if not hid or not nid:
                continue
            if hid not in first_narrators or pos < first_narrators[hid][0]:
                first_narrators[hid] = (pos, nid)

    batch: list[dict[str, Any]] = []
    for hid, (_pos, nid) in first_narrators.items():
        full_hid = hadith_node_id(hid)
        batch.append({"narrator_id": nid, "hadith_id": full_hid})

    if not batch:
        logger.info("narrated_no_edges")
        return EdgeLoadResult("NARRATED", 0, 0, 0)

    # Check endpoints
    check_results = _chunked_read(client, _NARRATED_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("narrator_exists") and check.get("hadith_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_NARRATED_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("narrated_loaded", created=created, missing_endpoints=missing)
    return EdgeLoadResult("NARRATED", created, 0, missing)


# ---------------------------------------------------------------------------
# 3. APPEARS_IN — hadith -> collection
# ---------------------------------------------------------------------------

# Identity of an APPEARS_IN edge is the (hadith, collection) PAIR — a hadith
# appears in a given collection at exactly one position, so the positional
# values (book / chapter / hadith number) are *attributes* of that single edge,
# not part of its key. They are therefore SET after the MERGE, never inside the
# MERGE pattern (da#77): Neo4j refuses to MERGE a relationship with a null
# property value, and the scraper leaves ``hadith_number`` null until da#72, so a
# keyed-on-properties MERGE aborts the whole load on real scraped data.
#
# The SET uses the streaming ingest path's coalesce-preserve contract
# (ingest-platform ``workers/ingest/processor.py _build_edge_cypher``:
# ``r.<f> = coalesce(row.<f>, r.<f>)``) so the two ingest paths converge byte-for-
# byte on idempotency: a row with an explicit-null positional value preserves any
# value already on the edge rather than clearing it. This keeps dedup correct
# (one APPEARS_IN per hadith->collection) and is null-safe.
_APPEARS_IN_QUERY = """\
UNWIND $batch AS row
MATCH (h:Hadith {id: row.hadith_id})
MATCH (c:Collection {id: row.collection_id})
MERGE (h)-[r:APPEARS_IN]->(c)
SET r.book_number = coalesce(row.book_number, r.book_number),
    r.chapter_number = coalesce(row.chapter_number, r.chapter_number),
    r.hadith_number_in_book = coalesce(row.hadith_number, r.hadith_number_in_book)
"""

_APPEARS_IN_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
OPTIONAL MATCH (c:Collection {id: row.collection_id})
RETURN row.hadith_id AS hadith_id,
       row.collection_id AS collection_id,
       h IS NOT NULL AS hadith_exists,
       c IS NOT NULL AS collection_exists
"""


def _load_appears_in(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load APPEARS_IN edges — hadith -> collection with positional properties."""
    files = _parquet_files(staging_dir, "hadiths_")
    if not files:
        if strict:
            msg = f"No hadiths_*.parquet files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("appears_in_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("APPEARS_IN", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    skipped = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            sid = row.get("source_id")
            cname = row.get("collection_name")
            if not sid or not cname:
                skipped += 1
                continue
            hid = hadith_node_id(sid)
            # Collection IDs in staging use "{corpus}:{name}" format (e.g. "lk:bukhari").
            # Build the same key so we match the Collection nodes that were loaded.
            corpus = row.get("source_corpus", "")
            raw_cid = f"{corpus}:{cname}" if corpus else cname
            cid = collection_node_id(raw_cid)
            batch.append(
                {
                    "hadith_id": hid,
                    "collection_id": cid,
                    "book_number": _val(row, "book_number"),
                    "chapter_number": _val(row, "chapter_number"),
                    "hadith_number": _val(row, "hadith_number"),
                }
            )

    if not batch:
        logger.info("appears_in_no_edges")
        return EdgeLoadResult("APPEARS_IN", 0, skipped, 0)

    # Check endpoints
    check_results = _chunked_read(client, _APPEARS_IN_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("hadith_exists") and check.get("collection_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_APPEARS_IN_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("appears_in_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("APPEARS_IN", created, skipped, missing)


# ---------------------------------------------------------------------------
# 4. PARALLEL_OF — from parallel_links.parquet
# ---------------------------------------------------------------------------

# Identity of a PARALLEL_OF edge is the (hadith_a, hadith_b) PAIR — the similarity
# score / variant tier / cross_sect flag are *attributes* of that one edge, not
# part of its key. They are therefore SET after a bare-edge MERGE, never inside
# the MERGE pattern (da#77 / main#139): Neo4j refuses to MERGE a relationship with
# a null property in the pattern, and keying on the properties would mint a SECOND
# edge between the same pair whenever a re-run produced a slightly different score
# (dedup is re-run as more corpora land). coalesce-preserve keeps re-runs
# idempotent and null-safe — exactly the APPEARS_IN contract.
_PARALLEL_OF_QUERY = """\
UNWIND $batch AS row
MATCH (h1:Hadith {id: row.id_a})
MATCH (h2:Hadith {id: row.id_b})
MERGE (h1)-[r:PARALLEL_OF]->(h2)
SET r.similarity_score = coalesce(row.score, r.similarity_score),
    r.variant_type = coalesce(row.variant_type, r.variant_type),
    r.cross_sect = coalesce(row.cross_sect, r.cross_sect)
"""

_PARALLEL_OF_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h1:Hadith {id: row.id_a})
OPTIONAL MATCH (h2:Hadith {id: row.id_b})
RETURN row.id_a AS id_a,
       row.id_b AS id_b,
       h1 IS NOT NULL AS a_exists,
       h2 IS NOT NULL AS b_exists
"""


def _load_parallel_of(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load PARALLEL_OF edges from parallel_links.parquet."""
    path = staging_dir / "parallel_links.parquet"
    if not path.exists():
        if strict:
            msg = f"Missing required file: {path}"
            raise FileNotFoundError(msg)
        logger.warning("parallel_links_missing", path=str(path))
        return EdgeLoadResult("PARALLEL_OF", 0, 0, 0)

    rows = _read_parquet_rows(path)
    batch: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        id_a = row.get("hadith_id_a")
        id_b = row.get("hadith_id_b")
        if not id_a or not id_b:
            skipped += 1
            continue
        # Ensure lower ID -> higher ID for consistent directionality
        full_a = hadith_node_id(id_a)
        full_b = hadith_node_id(id_b)
        if full_a > full_b:
            full_a, full_b = full_b, full_a
        batch.append(
            {
                "id_a": full_a,
                "id_b": full_b,
                "score": _val(row, "similarity_score", 0.0),
                "variant_type": _val(row, "variant_type", "unknown"),
                "cross_sect": _val(row, "cross_sect", False),
            }
        )

    if not batch:
        logger.info("parallel_of_no_edges")
        return EdgeLoadResult("PARALLEL_OF", 0, skipped, 0)

    # Check endpoints
    check_results = _chunked_read(client, _PARALLEL_OF_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("a_exists") and check.get("b_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_PARALLEL_OF_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("parallel_of_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("PARALLEL_OF", created, skipped, missing)


# ---------------------------------------------------------------------------
# 5. STUDIED_UNDER — from the studentship network_edges_*.parquet (muhaddithat, itqan)
# ---------------------------------------------------------------------------

# NETWORK_EDGE_SCHEMA is reused by producers whose edges mean DIFFERENT relations.
# muhaddithat + itqan emit teacher↔student (studentship) pairs that ARE
# STUDIED_UNDER. Others — e.g. `mis`, whose rows are *isnad transmission* pairs (a
# different relation type AND the opposite direction) — must NOT be globbed in
# here, or their edges would load as wrong-type, wrong-direction STUDIED_UNDER.
# This filename allowlist is the cheap, safe interim; the durable fix is an
# explicit edge-relation field on NETWORK_EDGE_SCHEMA so the loader keys on the
# data, not on a slug — tracked in da#133.
_STUDIED_UNDER_SOURCES: frozenset[str] = frozenset({"muhaddithat", "itqan"})


def _is_studied_under_file(path: Path) -> bool:
    """True if a ``network_edges_<slug>.parquet`` belongs to a studentship source.

    Matches the exact slug or a chunked ``<slug>_NNN`` variant, so a future
    sharded write still resolves but a different corpus (``mis``) never does.
    """
    slug = path.stem.removeprefix("network_edges_")
    return any(slug == src or slug.startswith(f"{src}_") for src in _STUDIED_UNDER_SOURCES)


_STUDIED_UNDER_QUERY = """\
UNWIND $batch AS row
MATCH (s:Narrator {id: row.from_id})
MATCH (t:Narrator {id: row.to_id})
MERGE (s)-[:STUDIED_UNDER]->(t)
"""

_STUDIED_UNDER_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (s:Narrator {id: row.from_id})
OPTIONAL MATCH (t:Narrator {id: row.to_id})
RETURN row.from_id AS from_id,
       row.to_id AS to_id,
       s IS NOT NULL AS from_exists,
       t IS NOT NULL AS to_exists
"""


def _studied_under_endpoint(name: Any, external_id: Any) -> str | None:
    """Resolve a NETWORK_EDGE endpoint to a canonical Narrator node id.

    Canonical Narrator nodes are keyed ``nar:<uuid5(normalized-name)>``
    (``identity.make_canonical_id``) — by the narrator's *name*, not by the
    source's external id. An edge endpoint must therefore be resolved the same
    way the bio that created the node was: through the name. We key on the
    narrator name (the value that actually matches a node) and only fall back to
    prefixing a pre-canonical id when the name is absent. Resolving by
    ``narrator_node_id(external_id)`` — as an earlier revision did — produced
    ``nar:<source-id>`` ids that match no canonical node, so every edge was
    silently dropped as a missing endpoint.
    """
    if isinstance(name, str) and name.strip():
        return make_canonical_id(normalize_arabic(name))
    if isinstance(external_id, str) and external_id.strip():
        return narrator_node_id(external_id)
    return None


def _load_studied_under(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load STUDIED_UNDER edges from the studentship ``network_edges_*.parquet``.

    Globs ``network_edges_*`` but keeps only the studentship producers
    (:data:`_STUDIED_UNDER_SOURCES` — muhaddithat, itqan); a NETWORK_EDGE producer
    whose edges are a different relation (e.g. ``mis`` isnad transmission) is
    deliberately skipped here (see da#133). Gracefully skips when no such file
    exists.
    """
    edge_files = [
        p for p in _parquet_files(staging_dir, "network_edges_") if _is_studied_under_file(p)
    ]
    if not edge_files:
        logger.info("studied_under_skipped", reason="file_not_found", staging_dir=str(staging_dir))
        return EdgeLoadResult("STUDIED_UNDER", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0

    for path in edge_files:
        for row in _read_parquet_rows(path):
            from_id = _studied_under_endpoint(
                row.get("from_narrator_name"), row.get("from_external_id")
            )
            to_id = _studied_under_endpoint(row.get("to_narrator_name"), row.get("to_external_id"))
            if not from_id or not to_id:
                skipped += 1
                continue
            # Dedup the same student->teacher pair across source files.
            key = (from_id, to_id)
            if key in seen:
                continue
            seen.add(key)
            batch.append({"from_id": from_id, "to_id": to_id})

    if not batch:
        logger.info("studied_under_no_edges")
        return EdgeLoadResult("STUDIED_UNDER", 0, skipped, 0)

    # Check endpoints
    check_results = _chunked_read(client, _STUDIED_UNDER_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("from_exists") and check.get("to_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_STUDIED_UNDER_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("studied_under_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("STUDIED_UNDER", created, skipped, missing)


# ---------------------------------------------------------------------------
# 6. GRADED_BY — hadith -> grading
# ---------------------------------------------------------------------------

_GRADED_BY_QUERY = """\
UNWIND $batch AS row
MATCH (h:Hadith {id: row.hadith_id})
MATCH (g:Grading {id: row.grading_id})
MERGE (h)-[:GRADED_BY]->(g)
"""

_GRADED_BY_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
OPTIONAL MATCH (g:Grading {id: row.grading_id})
RETURN row.hadith_id AS hadith_id,
       row.grading_id AS grading_id,
       h IS NOT NULL AS hadith_exists,
       g IS NOT NULL AS grading_exists
"""


def _load_graded_by(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load GRADED_BY edges from hadith staging data.

    Gracefully skips if no hadiths with grades exist.
    """
    files = _parquet_files(staging_dir, "hadiths_")
    if not files:
        logger.info("graded_by_skipped", reason="no_hadith_files")
        return EdgeLoadResult("GRADED_BY", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    skipped = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            grade = row.get("grade")
            if not grade:
                continue
            sid = row.get("source_id")
            if not sid:
                skipped += 1
                continue
            hid = hadith_node_id(sid)
            gid = grading_node_id(sid)
            batch.append({"hadith_id": hid, "grading_id": gid})

    if not batch:
        logger.info("graded_by_no_edges")
        return EdgeLoadResult("GRADED_BY", 0, skipped, 0)

    # Check endpoints
    check_results = _chunked_read(client, _GRADED_BY_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("hadith_exists") and check.get("grading_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_GRADED_BY_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("graded_by_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("GRADED_BY", created, skipped, missing)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_all_edges(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,  # noqa: ARG001
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EdgeLoadResult]:
    """Load all edge/relationship types into Neo4j.

    Parameters
    ----------
    client:
        Connected Neo4j client.
    staging_dir:
        Directory containing staging Parquet files.
    curated_dir:
        Directory containing curated reference data (unused for edges
        currently but kept for API symmetry with ``load_all_nodes``).
    strict:
        If ``True``, raise on missing required files. If ``False``,
        skip gracefully.
    batch_size:
        Number of edges per batch for both read checks and writes.
    """
    results: list[EdgeLoadResult] = []

    results.append(_load_transmitted_to(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_narrated(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_appears_in(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_parallel_of(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_studied_under(client, staging_dir, batch_size=batch_size))
    results.append(_load_graded_by(client, staging_dir, batch_size=batch_size))

    total_created = sum(r.created for r in results)
    total_missing = sum(r.missing_endpoints for r in results)
    logger.info(
        "all_edges_loaded",
        total_created=total_created,
        total_missing_endpoints=total_missing,
        edge_types=len(results),
    )
    return results
