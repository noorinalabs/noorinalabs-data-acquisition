"""Invalidate topology-derived enrich properties on load (da#351).

`load_all` is MERGE-only: it never deletes, so properties an earlier `enrich`
wrote onto a Narrator survive a reload untouched. `betweenness_centrality` from
a previous enrich therefore outlives the topology it was computed from and
**reads as current** -- there is nothing on the node to say otherwise.

Measured on stg (2026-07-09): all five metrics below sit on exactly 150,187
Narrator nodes while the graph holds 160,614. The 150,187 carry centrality
computed against the *pre-reload* topology; the 10,427 nodes created by the last
load carry none. Nothing distinguishes the two.

Why removal rather than a generation stamp
------------------------------------------
A stamp (``enrich_generation``, ``enrich_valid``) only protects consumers that
opt in to checking it. Every existing consumer reads ``n.betweenness_centrality``
directly, so a stamp would leave them serving stale numbers -- documenting the
bug rather than fixing it. Cypher has no strict-property mode: a missing property
and a misspelled one both yield NULL and neither errors (the same trap that made
``n.betweenness`` return 0 across all 160,614 rows and read as "the enrich was
wiped"). Absence is consequently the *only* value a reader cannot mistake for a
measurement. So the loader removes what it has invalidated, and `make enrich`
recomputes it.

That trade is deliberate: after a load without a follow-up enrich, centrality is
visibly absent rather than confidently wrong. A visible gap is recoverable; a
silent wrong number is not.

Scope: **topology-derived** properties only. Topic labels (``topic_1`` … on
Hadith) are computed from the matn text, not from the graph structure, so a
reload does not invalidate them and they are deliberately left alone.
"""

from __future__ import annotations

from typing import Final

from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = [
    "TOPOLOGY_DERIVED_NARRATOR_PROPERTIES",
    "invalidate_topology_derived_properties",
]

# Exactly the properties `src/enrich/metrics.py` writes via GDS
# (`writeProperty:`). Every one is a function of the graph's structure, so every
# one is falsified by a load that changes that structure.
TOPOLOGY_DERIVED_NARRATOR_PROPERTIES: Final[tuple[str, ...]] = (
    "betweenness_centrality",
    "community_id",
    "in_degree",
    "out_degree",
    "pagerank",
)

_DEFAULT_BATCH_SIZE: Final = 10_000


def _validate_property_names(names: tuple[str, ...]) -> None:
    """Property names are interpolated into Cypher; keep them inert."""
    for name in names:
        if not name.isidentifier():
            raise ValueError(f"unsafe property name for Cypher interpolation: {name!r}")


_validate_property_names(TOPOLOGY_DERIVED_NARRATOR_PROPERTIES)

# The match predicate and the REMOVE clause are generated from the SAME tuple.
# If they could drift, a predicate matching a property the REMOVE did not clear
# would select the same rows forever and the batch loop below would not
# terminate. One source of truth makes that unrepresentable.
_STALE_PREDICATE: Final = " OR ".join(
    f"n.{prop} IS NOT NULL" for prop in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES
)
_REMOVE_CLAUSE: Final = ", ".join(f"n.{prop}" for prop in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES)

_INVALIDATE_BATCH_QUERY: Final = f"""
MATCH (n:Narrator)
WHERE {_STALE_PREDICATE}
WITH n LIMIT $batch_size
REMOVE {_REMOVE_CLAUSE}
RETURN count(n) AS invalidated
"""


def invalidate_topology_derived_properties(
    client: Neo4jClient,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Remove stale enrich metrics from every Narrator. Returns the count.

    Batched in Python rather than with ``CALL { … } IN TRANSACTIONS``, which
    requires an implicit (auto-commit) transaction and is rejected inside the
    explicit write transaction ``Neo4jClient.execute_write`` opens.

    Idempotent: once the properties are gone the predicate matches nothing, the
    batch returns 0, and the loop exits. Running it against an un-enriched graph
    removes nothing and costs one query.
    """
    total = 0
    while True:
        rows = client.execute_write(_INVALIDATE_BATCH_QUERY, {"batch_size": batch_size})
        removed = int(rows[0]["invalidated"]) if rows else 0
        if removed == 0:
            break
        total += removed
        logger.info("enrich_metrics_invalidated_batch", removed=removed, running_total=total)

    if total:
        logger.warning(
            "enrich_metrics_invalidated",
            narrators=total,
            properties=list(TOPOLOGY_DERIVED_NARRATOR_PROPERTIES),
            remediation="re-run `make enrich` to recompute against the new topology",
        )
    return total
