"""Canonical-set SHRINK: DETACH DELETE Narrator nodes not in the canonical master (da#413).

Why this cannot be pure cypher (deploy#557, Aisha Idrissi's measure-first)
--------------------------------------------------------------------------
The graph accumulates ``Narrator`` nodes across loads, but a re-resolve can collapse
two ids into one (over-merge fixes, de-keying, fuzzy folds). The nodes left behind —
present in the graph, absent from the freshly written ``narrators_canonical.parquet``
— are orphans that must go, edges and all.

**The only correct orphan predicate is canonical-set membership**, and the canonical
set lives in the parquet, unreadable by a cypher-only graph job. A graph-side proxy
gets it wrong in both directions, measured on the real data:

* A zero-degree / "unreferenced" predicate MISSES 18,518 of the 31,380 orphans, which
  still have edges (a collapsed id keeps its ``TRANSMITTED_TO`` / ``NARRATED`` edges).
* It also WRONGLY DELETES 57,993 zero-degree canonical narrators, which are
  legitimate (owner da#352: a narrator attested only in a deduped edition is a real
  narrator, edge-count zero yet canonical). A degree prune wipes exactly these.

No in-graph stamp separates orphan from legit — only canonical-set membership does.
So this reads the authoritative id set from the parquet in Python (like
:mod:`src.graph.migrate` reads its distinct ids), computes the complement against the
graph's ``Narrator`` ids, and DETACH-DELETEs that complement in batches.

The guard is the point
----------------------
``DETACH DELETE`` is irreversible against a live box. The keep-set — the canonical id
set — is the ONLY thing standing between this command and deleting a node, so
:func:`read_canonical_ids` refuses (raises :class:`EmptyCanonicalSetError`, exit
:attr:`~src.exit_codes.ExitCode.EMPTY_CANONICAL_SET`) BEFORE the graph is read or
written if that set cannot be trusted — an **empty** set (which would delete every
narrator), a **missing** parquet, or an **unreadable/malformed** one. A bad read must
never wipe the graph. This is the da#309 / da#361 fail-loud-on-missing-input
discipline (see :mod:`src.resolve._inputs`) applied to a destructive op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EmptyCanonicalSetError",
    "PruneResult",
    "prune_narrators",
    "read_canonical_ids",
]

DEFAULT_BATCH_SIZE = 1000
# How many orphan ids to carry in the summary/dry-run sample. Enough to eyeball, not
# so many that a 31k-orphan run prints a wall of ids.
SAMPLE_SIZE = 20

# Every Narrator id currently in the graph. The complement against the canonical set
# (computed in Python) is the orphan set — see the module docstring for why the set
# difference cannot be a graph-side degree/reference predicate.
_ALL_NARRATOR_IDS = "MATCH (n:Narrator) RETURN n.id AS id"

# Delete one batch of orphans by id, edges and all. ``count(n)`` is an aggregation
# over the matched rows, computed as the rows are deleted, so it returns the number
# actually removed without ever returning a deleted node object. We do not trust the
# driver's node-deleted counter (execute_write surfaces record data, not a summary);
# the integration test verifies the graph by reading it back regardless.
_DETACH_DELETE_BY_ID = """\
UNWIND $batch AS nid
MATCH (n:Narrator {id: nid})
DETACH DELETE n
RETURN count(n) AS deleted
"""


class EmptyCanonicalSetError(Exception):
    """The canonical keep-set is unusable, so ``prune-narrators`` deletes nothing.

    Raised by :func:`read_canonical_ids` BEFORE any graph read or write. One
    exception for three doors — a missing parquet, an unreadable/malformed one, and a
    present-but-empty set — because each yields the same fact: there is no trustworthy
    set of ids to keep, and a destructive op with no keep-set would delete the whole
    graph. The message names which door fired so an operator can act without the
    traceback. Maps to :attr:`~src.exit_codes.ExitCode.EMPTY_CANONICAL_SET`.
    """

    def __init__(self, *, reason: str, path: Path) -> None:
        self.reason = reason
        self.path = path
        super().__init__(
            f"refusing to prune narrators: {reason} (canonical master: {path}). "
            "No node was deleted; the graph is exactly as it was. A destructive prune "
            "against an empty/unreadable keep-set would delete every narrator, so it is "
            "refused. Fix the canonical parquet and re-run."
        )


@dataclass(frozen=True)
class PruneResult:
    """Outcome of a :func:`prune_narrators` run.

    ``deleted`` is 0 on a ``--dry-run`` (nothing was deleted) and on a real run that
    found no orphans. ``orphans_identified`` is the count the run WOULD delete (dry
    run) or DID delete (real run) — they are equal by construction, the sample is the
    same set, and a dry run is exactly a real run with the delete batches suppressed.
    """

    canonical_ids_seen: int
    graph_narrators_seen: int
    orphans_identified: int
    deleted: int
    dry_run: bool
    sample: list[str] = field(default_factory=list)


def read_canonical_ids(canonical_path: Path) -> set[str]:
    """Return the authoritative canonical-id keep-set, or raise the destructive-op guard.

    Reads only the ``canonical_id`` column of ``narrators_canonical.parquet``. Raises
    :class:`EmptyCanonicalSetError` — deleting nothing, touching nothing — when the
    file is missing, unreadable/malformed, or present but empty. That refusal is the
    load-bearing safety property of the whole subcommand: it is the only thing that
    stands between a bad read and an emptied graph, so it happens here, before the
    caller has a chance to touch Neo4j.
    """
    if not canonical_path.exists():
        raise EmptyCanonicalSetError(reason="canonical parquet is missing", path=canonical_path)
    try:
        table = pq.read_table(canonical_path, columns=["canonical_id"])
    except Exception as exc:  # noqa: BLE001 — a corrupt file, absent column, non-parquet: all "unreadable"
        raise EmptyCanonicalSetError(
            reason=f"canonical parquet is unreadable ({exc})", path=canonical_path
        ) from exc
    ids = {cid for cid in table.column("canonical_id").to_pylist() if cid}
    if not ids:
        raise EmptyCanonicalSetError(
            reason="canonical parquet holds zero canonical_id values", path=canonical_path
        )
    return ids


def prune_narrators(
    client: Neo4jClient,
    canonical_path: Path,
    *,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PruneResult:
    """DETACH DELETE every ``Narrator`` whose ``id`` is not in the canonical master.

    Order is load-bearing: :func:`read_canonical_ids` runs FIRST and raises the guard
    on an unusable keep-set, so a bad read cannot reach the graph. Only once a
    non-empty canonical set is in hand does this read the graph's Narrator ids, compute
    the orphan complement in Python, and — unless ``dry_run`` — DETACH-DELETE it in
    batches (removing each orphan's edges with it). A ``--dry-run`` computes the same
    orphan set and reports it but issues no write.

    The complement is a set difference, not a degree/reference test, which is the whole
    reason this exists rather than a cypher one-liner (see the module docstring): a
    canonical narrator with zero edges is kept, and an orphan with edges is deleted.
    """
    # Guard first, before ANY graph access. An unusable keep-set raises here.
    canonical_ids = read_canonical_ids(canonical_path)

    graph_rows = client.execute_read(_ALL_NARRATOR_IDS)
    graph_ids = [row["id"] for row in graph_rows if row.get("id")]
    orphans = [gid for gid in graph_ids if gid not in canonical_ids]
    sample = orphans[:SAMPLE_SIZE]

    if dry_run:
        logger.info(
            "prune_narrators_dry_run",
            canonical_ids_seen=len(canonical_ids),
            graph_narrators_seen=len(graph_ids),
            orphans_identified=len(orphans),
        )
        return PruneResult(
            canonical_ids_seen=len(canonical_ids),
            graph_narrators_seen=len(graph_ids),
            orphans_identified=len(orphans),
            deleted=0,
            dry_run=True,
            sample=sample,
        )

    deleted = 0
    for i in range(0, len(orphans), batch_size):
        chunk = orphans[i : i + batch_size]
        result = client.execute_write(_DETACH_DELETE_BY_ID, {"batch": chunk})
        if result:
            value = result[0].get("deleted", 0)
            deleted += value if isinstance(value, int) else 0

    logger.info(
        "prune_narrators_complete",
        canonical_ids_seen=len(canonical_ids),
        graph_narrators_seen=len(graph_ids),
        orphans_identified=len(orphans),
        deleted=deleted,
    )
    return PruneResult(
        canonical_ids_seen=len(canonical_ids),
        graph_narrators_seen=len(graph_ids),
        orphans_identified=len(orphans),
        deleted=deleted,
        dry_run=False,
        sample=sample,
    )
