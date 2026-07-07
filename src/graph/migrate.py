"""In-place remediation of ``TRANSMITTED_TO.hadith_id`` → canonical ids (da#325).

The chain loader historically stored the RAW staging ``hadith_id`` on every
``TRANSMITTED_TO`` edge (``sanadset:sanadset:0:0:2326`` — double-prefixed, no
``hdt:``) while ``Hadith.id`` nodes are canonical (``hdt:sanadset:0:0:2326``, via
:func:`~src.parse.identity.hadith_node_id`). ``GET /graph/hadith/{id}/chain``
reconstructs the chain by joining ``TRANSMITTED_TO.hadith_id`` against
``Hadith.id``, so the two disjoint id namespaces made that join return nothing —
an empty chain for every hadith.

:mod:`src.graph.load_edges` is fixed at the source so NEW loads carry canonical
edge ids, but a live graph already holds ~2.68M edges with the raw id, and a
re-load from parquet is not possible against a live box. This module migrates
those edges **in place** (property rewrite only — no structure change, no
delete/reload).

Idempotent by construction: the canonical id is re-derived with
:func:`~src.parse.identity.hadith_node_id`, which is itself idempotent
(``hdt:``-prefix-safe and double-corpus-collapsing), and only the distinct values
that actually change are written. A second run over an already-canonical graph
finds nothing to rewrite and updates zero edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.parse.identity import hadith_node_id
from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = ["MigrationResult", "migrate_transmitted_hadith_ids", "compute_rewrites"]

DEFAULT_BATCH_SIZE = 1000

# Distinct edge ids only: the rewrite keys on the raw value, so we compute the
# canonical mapping once per distinct id (~523k) rather than once per edge (~2.68M).
_DISTINCT_HADITH_IDS = """\
MATCH ()-[t:TRANSMITTED_TO]->()
WHERE t.hadith_id IS NOT NULL AND t.hadith_id <> ""
RETURN DISTINCT t.hadith_id AS hadith_id
"""

# Rewrite every edge whose raw id matches, returning the count actually touched.
# ``SET`` is a property write, so the driver's node/relationship-created counters
# stay 0 — we RETURN ``count(t)`` and sum it instead of trusting those counters.
_REWRITE_HADITH_ID = """\
UNWIND $batch AS row
MATCH ()-[t:TRANSMITTED_TO {hadith_id: row.raw}]->()
SET t.hadith_id = row.canon
RETURN count(t) AS updated
"""


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a :func:`migrate_transmitted_hadith_ids` run."""

    distinct_ids_seen: int
    distinct_ids_rewritten: int
    edges_updated: int


def compute_rewrites(distinct_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map each distinct raw ``hadith_id`` to its canonical form, dropping no-ops.

    Skips empty/null ids and any value already canonical (``hadith_node_id`` is a
    no-op on it) so the returned list holds only the ids that must change — this
    is what makes a re-run idempotent (an already-migrated graph yields ``[]``).
    """
    rewrites: list[dict[str, str]] = []
    for row in distinct_rows:
        raw = row.get("hadith_id")
        if not raw or not isinstance(raw, str):
            continue
        canon = hadith_node_id(raw)
        if canon != raw:
            rewrites.append({"raw": raw, "canon": canon})
    return rewrites


def migrate_transmitted_hadith_ids(
    client: Neo4jClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> MigrationResult:
    """Canonicalize existing ``TRANSMITTED_TO.hadith_id`` edge ids in place.

    Reads the distinct raw ids from Neo4j, computes the canonical mapping in
    Python (:func:`compute_rewrites`), then rewrites the edges in batches. Empty
    ids are skipped and already-canonical ids are no-ops, so the run is idempotent
    and safe to repeat. Returns a :class:`MigrationResult` summary.
    """
    distinct_rows = client.execute_read(_DISTINCT_HADITH_IDS)
    seen = sum(1 for r in distinct_rows if r.get("hadith_id"))
    rewrites = compute_rewrites(distinct_rows)

    edges_updated = 0
    for i in range(0, len(rewrites), batch_size):
        chunk = rewrites[i : i + batch_size]
        result = client.execute_write(_REWRITE_HADITH_ID, {"batch": chunk})
        if result:
            value = result[0].get("updated", 0)
            edges_updated += value if isinstance(value, int) else 0

    logger.info(
        "transmitted_hadith_id_migration",
        distinct_ids_seen=seen,
        distinct_ids_rewritten=len(rewrites),
        edges_updated=edges_updated,
    )
    return MigrationResult(
        distinct_ids_seen=seen,
        distinct_ids_rewritten=len(rewrites),
        edges_updated=edges_updated,
    )
