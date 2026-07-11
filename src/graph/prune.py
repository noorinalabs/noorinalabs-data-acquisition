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
:attr:`~src.exit_codes.ExitCode.UNUSABLE_CANONICAL_SET`) BEFORE the graph is read or
written if that set cannot be trusted — an **empty** set (which would delete every
narrator), a **missing** parquet, or an **unreadable/malformed** one. A bad read must
never wipe the graph. This is the da#309 / da#361 fail-loud-on-missing-input
discipline (see :mod:`src.resolve._inputs`) applied to a destructive op.

Binding the keep-set to the graph it prunes (da#413 review)
-----------------------------------------------------------
Those three doors all catch an **unusable** keep-set. None catches a **wrong-but-valid**
one, and that is the more dangerous failure. ``parquet_ref`` is free text in the
consumer (deploy#557/#574): a stale parquet from a *previous* resolve run is perfectly
readable, perfectly non-empty, and names tens of thousands of ids that no longer exist —
so pruning against it DETACH-DELETEs the live generation's legitimate narrators.

Worse, it does so silently. Every post-op gate — pre minus orphans equals post, deleted
equals predicted, zero orphans remain — is **derived from the same keep-set being
validated**. They all prove the prune faithfully obeyed the parquet; not one of them can
ask whether it was the *right* parquet. They pass for any parquet
(``feedback_silent_zero_is_not_a_measurement``: a detector that cannot separate the two
classes it exists to separate is not producing a measurement).

The discriminator the gates lack is the **reverse** difference, at zero extra queries:

    ``missing`` = ``|canonical − graph|`` — keep-set ids the graph has never seen.

For the parquet the graph was actually loaded from, this is ~0. The da#352 join measures
it exactly: 71,241 edged + 57,993 zero-degree = 129,234 = the whole canonical count, so
today **every** canonical id is present in the graph. For a stale or foreign parquet it
is large, in proportion to how wrong the parquet is. That is the provenance signal.

Two rules govern it, and they pull in opposite directions on purpose:

* **It is reported, not asserted to zero.** A legitimately partial load leaves canonical
  ids unloaded by design — :attr:`~src.exit_codes.ExitCode.REFUSED_ROWS`,
  :attr:`~src.exit_codes.ExitCode.STOPPED_AT_LIMIT`, a nodes-only load. A hard
  ``missing == 0`` here would refuse a valid prune after any of them. So ``missing=`` is
  surfaced as the fifth key of :func:`summary_line` and the **consumer owns the
  refuse-or-proceed policy** (deploy#574 gates on ``missing == 0`` unless an explicit
  ``allow_partial_load`` is set).
* **It is refused on implausible magnitude**, as a backstop, before any write — see
  :class:`ForeignCanonicalSetError` and :data:`DEFAULT_MAX_MISSING_FRACTION`.

Note what this guard does and does not buy. It catches the *catastrophic* wrong parquet —
one whose ids are mostly strangers to this graph. A *subtly* stale parquet (one generation
back, most ids still present) slips under any plausible magnitude ceiling; catching that
one is the consumer's ``missing == 0`` gate. The two are layered deliberately, and neither
alone is sufficient.

The delete side needs its own bound (da#413 second review)
----------------------------------------------------------
``missing`` bounds the direction that **cannot delete anything**. ``canonical - graph`` is
a set of ids that are not in the graph, so no element of it can be DETACH DELETEd. The
quantity that actually deletes is the *other* difference, ``orphans = graph - canonical``,
and ``missing`` says nothing whatever about its size.

That gap is not theoretical. Consider a **truncated but same-generation** keep-set: a
resolve run stopped early (:attr:`~src.exit_codes.ExitCode.STOPPED_AT_LIMIT`), a partial
upload, a half-written parquet, a null-heavy ``canonical_id`` column. It holds 5,000 ids
that are all **genuinely this graph's** — so ``missing`` is exactly **0**, and:

* the empty/unreadable guard passes (it is readable, and 5,000 ids is not zero);
* the ``missing`` ceiling passes (``0 > 0.5 * 5,000`` is false);
* the consumer's whole-graph-wipe check passes (155,614 orphans is not >= 160,614);
* and the consumer's ``missing == 0`` policy **actively blesses it**;

while the prune DETACH DELETEs 155,614 of 160,614 narrators — **96.9% of the graph** — and
every post-op gate greens, because they are all still derived from the keep-set being
validated. The set is not *foreign*. It is merely *incomplete*, and incompleteness is
invisible to every instrument that asks whether the keep-set's ids belong to this graph.

So the delete side is bounded too, symmetrically:

    ``orphan_fraction`` = ``|orphans| / |graph|`` — the share of the graph this would delete.

:class:`ExcessiveDeletionError` refuses above :data:`DEFAULT_MAX_ORPHAN_FRACTION`, before
any write, overridable with ``--max-orphan-fraction``.

**Why that ceiling is 0.9 and emphatically not 0.5.** A ceiling near one half looks
prudent and would be wrong, because it refuses this command's own primary use case. The
graph is the union of every load ever run against it, so a re-resolve that re-mints ids
*en masse* — exactly the da#356/da#376 event this subcommand was built to clean up after —
legitimately orphans the entire previous generation. Working it in the measured numbers:
an accumulated graph holding the old 160,614-node generation plus a freshly loaded,
fully re-minted 129,234-node one is 289,848 nodes, of which 160,614 are legitimate
orphans — **55.4%**, with ``missing == 0``, and it is a *correct* prune. A 0.5 ceiling
refuses it; the operator overrides; the override becomes reflex; the guard is dead.

The honest separation is wide and it is not centred on one half:

===========================================  ==================  ==============
class                                        ``orphan_fraction``  verdict
===========================================  ==================  ==============
single-generation prune (da#352 measured)    19.5%                proceed
accumulated graph, full re-mint (worst legit) 55.4%               proceed
truncated 5,000-id keep-set                  96.9%                REFUSE
truncated 500-id keep-set                    99.7%                REFUSE
===========================================  ==================  ==============

0.9 sits in the empty band between ~55% and ~97% with margin on both sides. It is the
"no legitimate prune looks like this" line, not a tuned parameter — and, like the
``missing`` ceiling, it is a **backstop**, not a policy. Surviving fewer than one node in
ten is not a prune; it is a wipe wearing a prune's clothes.
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
    "DEFAULT_MAX_MISSING_FRACTION",
    "DEFAULT_MAX_ORPHAN_FRACTION",
    "EmptyCanonicalSetError",
    "ExcessiveDeletionError",
    "ForeignCanonicalSetError",
    "SUMMARY_KEYS",
    "PruneResult",
    "UnusableCanonicalSetError",
    "prune_narrators",
    "read_canonical_ids",
    "summary_line",
]

DEFAULT_BATCH_SIZE = 1000
# How many orphan ids to carry in the summary/dry-run sample. Enough to eyeball, not
# so many that a 31k-orphan run prints a wall of ids.
SAMPLE_SIZE = 20

# The magnitude ceiling on ``missing`` (= |canonical - graph|), as a fraction of the
# keep-set. Above it the keep-set is refused as not-this-graph's, before any write.
#
# Why one half, and why a fraction rather than a count. For the parquet the graph was
# loaded from, ``missing`` is 0 — da#352 measures the join exactly (129,234 canonical =
# 71,241 edged + 57,993 zero-degree, all present). A partial load pushes it up, but the
# keep-set still describes THIS graph. At one half it no longer does: more of the keep-set
# is unknown to the graph than known, so the parquet has lost majority authority over the
# thing it is being used to prune. That is not a partial load; that is a different graph.
#
# It is deliberately NOT a tuned number pretending to precision — it is the "obviously
# wrong" line. The subtle stale parquet is the consumer's ``missing == 0`` gate to catch,
# not this one's (see the module docstring). Raise it with --max-missing-fraction when a
# genuinely large partial load is intended; 1.0 disables the ceiling entirely.
DEFAULT_MAX_MISSING_FRACTION = 0.5

# The magnitude ceiling on the DELETE side: |orphans| / |graph|, the share of the narrator
# graph this prune would DETACH DELETE. Refused above it, before any write.
#
# 0.9, NOT 0.5 — see the module docstring for the full argument, because the intuitive
# number is the wrong one here. A ceiling near one half would refuse a legitimate prune:
# after a mass id re-mint (da#356/da#376), the accumulated graph legitimately orphans its
# entire previous generation — 160,614 of 289,848 nodes = 55.4%, with missing == 0. That is
# precisely the cleanup this subcommand exists to perform. Refuse it and the operator
# overrides by reflex, which kills the guard.
#
# The threat class sits far above the legitimate one, and the band between them is empty:
#   19.5%  single-generation prune (da#352 measured)   legit
#   55.4%  accumulated graph, full re-mint             legit, worst case
#   96.9%  truncated 5,000-id keep-set                 THREAT (missing == 0; all other guards pass)
#   99.7%  truncated 500-id keep-set                   THREAT
# 0.9 separates them with margin on both sides. Surviving fewer than one node in ten is not
# a prune. Override with --max-orphan-fraction when a wipe is genuinely intended.
DEFAULT_MAX_ORPHAN_FRACTION = 0.9

# Every Narrator id currently in the graph. The complement against the canonical set
# (computed in Python) is the orphan set — see the module docstring for why the set
# difference cannot be a graph-side degree/reference predicate.
_ALL_NARRATOR_IDS = "MATCH (n:Narrator) RETURN n.id AS id"

# Delete one batch of orphans by id, edges and all. No RETURN: we do NOT trust a
# driver-side deleted counter. ``deleted`` and the post-count of orphans are measured
# by reading the graph back after the deletes (the count>=0 discipline: prove the
# post-state, don't infer it from a write's own report).
_DETACH_DELETE_BY_ID = """\
UNWIND $batch AS nid
MATCH (n:Narrator {id: nid})
DETACH DELETE n
"""


class UnusableCanonicalSetError(Exception):
    """``prune-narrators`` refused: the keep-set cannot be trusted. Nothing was deleted.

    The common supertype of every refusal door, and the thing ``_cmd_prune_narrators``
    catches. All of them share the one fact that defines
    :attr:`~src.exit_codes.ExitCode.UNUSABLE_CANONICAL_SET` — **the graph is exactly as it
    was**, zero nodes deleted, zero edges touched — which is why they share its exit
    code rather than minting a second one for an identical on-disk state. The subclass
    (and the message) says which door fired; the exit code says the graph is safe.
    """

    def __init__(self, *, reason: str, path: Path, message: str) -> None:
        self.reason = reason
        self.path = path
        super().__init__(message)


class EmptyCanonicalSetError(UnusableCanonicalSetError):
    """The canonical keep-set is unusable *on its face*, so nothing is deleted.

    Raised by :func:`read_canonical_ids` BEFORE any graph read or write. One
    exception for three doors — a missing parquet, an unreadable/malformed one, and a
    present-but-empty set — because each yields the same fact: there is no trustworthy
    set of ids to keep, and a destructive op with no keep-set would delete the whole
    graph. The message names which door fired so an operator can act without the
    traceback. Maps to :attr:`~src.exit_codes.ExitCode.UNUSABLE_CANONICAL_SET`.

    This door tests the keep-set **in isolation**. It cannot see the fourth door — a
    keep-set that is perfectly well-formed but belongs to a *different graph* — because
    that is a relation between the parquet and the graph, not a property of the parquet.
    See :class:`ForeignCanonicalSetError`.
    """

    def __init__(self, *, reason: str, path: Path) -> None:
        super().__init__(
            reason=reason,
            path=path,
            message=(
                f"refusing to prune narrators: {reason} (canonical master: {path}). "
                "No node was deleted; the graph is exactly as it was. A destructive prune "
                "against an empty/unreadable keep-set would delete every narrator, so it is "
                "refused. Fix the canonical parquet and re-run."
            ),
        )


class ForeignCanonicalSetError(UnusableCanonicalSetError):
    """The keep-set is well-formed but is not **this graph's**, so nothing is deleted.

    The fourth door, and the one no post-op gate can see (module docstring). Raised when
    ``missing`` — the count of canonical ids the graph has never seen — exceeds
    ``max_missing_fraction`` of the keep-set. Raised after the graph is *read* (the
    signal is a relation between the two sets, so it cannot be computed without reading
    both) but strictly BEFORE any write: the refusal costs the graph nothing.

    A stale ``parquet_ref`` is the most plausible operator error there is, and it is not
    self-announcing: it is readable, non-empty, and every derived gate downstream would
    green. Maps to :attr:`~src.exit_codes.ExitCode.UNUSABLE_CANONICAL_SET` — the on-disk
    state is identical to the other doors (the graph, untouched), and the consumer
    propagates the rc verbatim, so a distinct code would buy nothing and would re-open
    the three-way collision hazard :mod:`src.exit_codes` exists to document.
    """

    def __init__(
        self,
        *,
        path: Path,
        missing: int,
        canonical_total: int,
        graph_total: int,
        max_missing_fraction: float,
    ) -> None:
        self.missing = missing
        self.canonical_total = canonical_total
        self.graph_total = graph_total
        self.max_missing_fraction = max_missing_fraction
        pct = 100.0 * missing / canonical_total
        ceiling_pct = 100.0 * max_missing_fraction
        super().__init__(
            reason="canonical keep-set does not belong to this graph",
            path=path,
            message=(
                f"refusing to prune narrators: the canonical keep-set does not belong to "
                f"this graph — {missing} of its {canonical_total} canonical ids "
                f"({pct:.1f}%) are absent from the graph, above the {ceiling_pct:.1f}% "
                f"ceiling. The graph holds {graph_total} Narrator node(s). "
                f"(canonical master: {path}). "
                "No node was deleted; the graph is exactly as it was. A keep-set the graph "
                "has never seen is a keep-set for a DIFFERENT graph: pruning against it "
                "would DETACH DELETE legitimate narrators, and every post-op check would "
                "still pass, because they are all derived from this same keep-set. Point "
                "--canonical at the parquet the live graph was loaded from. If this graph "
                "really is a partial load of this keep-set, raise the ceiling deliberately "
                "with --max-missing-fraction."
            ),
        )


class ExcessiveDeletionError(UnusableCanonicalSetError):
    """The keep-set is this graph's, but keeps so little of it that the prune is a wipe.

    The **fifth** door, and the one :class:`ForeignCanonicalSetError` is structurally
    incapable of seeing. ``missing`` bounds ``canonical - graph`` — a set whose elements
    are *not in the graph* and therefore cannot be deleted. It says nothing about
    ``orphans = graph - canonical``, which is the set that actually gets DETACH DELETEd.

    A **truncated same-generation** keep-set exploits exactly that blind spot: a resolve
    run stopped at :attr:`~src.exit_codes.ExitCode.STOPPED_AT_LIMIT`, a half-written
    parquet, a partial upload, a null-heavy ``canonical_id`` column. Every id in it is
    real and current, so ``missing`` is **0** and every id-provenance instrument — here
    and in the consumer — reports the keep-set as perfectly healthy, while the prune
    deletes all but the truncated remnant.

    So this refuses on the delete magnitude itself: ``|orphans| / |graph|`` above
    ``max_orphan_fraction`` (:data:`DEFAULT_MAX_ORPHAN_FRACTION`). Raised after the graph
    read it depends on and strictly BEFORE any write, in dry-run mode too. Shares
    :attr:`~src.exit_codes.ExitCode.UNUSABLE_CANONICAL_SET` with its siblings: the on-disk
    state is the same one that defines the code — the graph, exactly as it was.
    """

    def __init__(
        self,
        *,
        path: Path,
        orphans: int,
        graph_total: int,
        canonical_total: int,
        missing: int,
        max_orphan_fraction: float,
    ) -> None:
        self.orphans = orphans
        self.graph_total = graph_total
        self.canonical_total = canonical_total
        self.missing = missing
        self.max_orphan_fraction = max_orphan_fraction
        pct = 100.0 * orphans / graph_total
        ceiling_pct = 100.0 * max_orphan_fraction
        survivors = graph_total - orphans
        super().__init__(
            reason="prune would delete an implausible share of the narrator graph",
            path=path,
            message=(
                f"refusing to prune narrators: this would DETACH DELETE {orphans} of "
                f"{graph_total} Narrator node(s) ({pct:.1f}%), above the {ceiling_pct:.1f}% "
                f"ceiling, leaving only {survivors}. The keep-set holds {canonical_total} "
                f"canonical id(s), of which {missing} are absent from the graph "
                f"(canonical master: {path}). "
                "No node was deleted; the graph is exactly as it was. Note that a low "
                "'missing' does NOT clear this: a TRUNCATED keep-set (a resolve stopped "
                "early, a half-written or partially uploaded parquet, a null-heavy "
                "canonical_id column) holds ids that are all genuinely this graph's — so "
                "every id-provenance check passes — while keeping only a remnant of it. "
                "Check that the canonical parquet is COMPLETE, not merely valid. If a "
                "prune this large is genuinely intended, say so with --max-orphan-fraction."
            ),
        )


@dataclass(frozen=True)
class PruneResult:
    """Outcome of a :func:`prune_narrators` run, shaped to feed deploy#557's verify.

    The fields map onto the machine-readable :func:`summary_line` its workflow greps:

    * ``canonical_ids_seen`` → ``canonical=`` — size of the keep-set.
    * ``graph_total`` → ``graph_total=`` — Narrator nodes in the graph AFTER the op
      (for a dry run nothing is deleted, so this is the current total).
    * ``orphans`` → ``orphans=`` — **its meaning differs by mode, by contract**: on a
      dry run it is the count that WOULD be deleted (pre-count); on a real run it is
      the orphans that REMAIN after the delete, re-measured by reading the graph back,
      and it MUST be 0 (that is what proves exactly the orphans went and every
      referenced/canonical narrator survived, not merely that the graph was not wiped).
    * ``deleted`` → ``deleted=`` — nodes removed, measured as pre-total minus
      post-total (readback truth), 0 on a dry run.
    * ``missing`` → ``missing=`` — canonical ids the graph has never seen
      (``|canonical - graph|``), the keep-set provenance signal. **Unlike ``orphans``,
      its meaning does NOT vary by mode**: it is measured against the PRE-state in both,
      and it is the same number either way, because deleting an orphan (by definition a
      non-canonical id) can never remove a canonical id from the graph. ~0 for the
      parquet the graph was loaded from; large for a stale or foreign one.

    ``orphans_identified`` is always the pre-count (what a real run would delete); it
    drives the human-readable line and the ``sample``. A dry run is exactly a real run
    with the delete batches and the post-readback suppressed.
    """

    canonical_ids_seen: int
    graph_total: int
    orphans: int
    deleted: int
    dry_run: bool
    orphans_identified: int
    missing: int
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
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    max_orphan_fraction: float = DEFAULT_MAX_ORPHAN_FRACTION,
) -> PruneResult:
    """DETACH DELETE every ``Narrator`` whose ``id`` is not in the canonical master.

    Order is load-bearing, and there are now two guards in it:

    1. :func:`read_canonical_ids` runs FIRST and raises :class:`EmptyCanonicalSetError`
       on a keep-set that is unusable on its face, so a bad read cannot reach the graph
       at all.
    2. The graph's Narrator ids are read, and the keep-set's **provenance** is checked
       against them — ``missing`` = ``|canonical - graph|`` — raising
       :class:`ForeignCanonicalSetError` if the keep-set is mostly strangers to this
       graph. This one needs the graph, so it cannot precede the read; it does precede
       every **write**, which is the property that matters.
    3. The orphan complement is computed, and its **magnitude** is checked —
       ``|orphans| / |graph|`` — raising :class:`ExcessiveDeletionError` if the prune
       would delete an implausible share of the graph. Guard 2 cannot cover this: it
       bounds the difference that deletes *nothing*, while this bounds the one that
       deletes *everything it names*. A truncated same-generation keep-set has
       ``missing == 0`` and would still wipe the graph.

    All three fire in dry-run mode too: the operator's mandatory dry run is exactly where
    a bad keep-set needs to surface.

    Only then does this — unless ``dry_run`` — DETACH-DELETE the complement in batches
    (removing each orphan's edges with it). A ``--dry-run`` computes the same orphan set
    and reports it but issues no write.

    The complement is a set difference, not a degree/reference test, which is the whole
    reason this exists rather than a cypher one-liner (see the module docstring): a
    canonical narrator with zero edges is kept, and an orphan with edges is deleted.
    """
    if not 0.0 <= max_missing_fraction <= 1.0:
        msg = f"max_missing_fraction must be in [0.0, 1.0], got {max_missing_fraction}"
        raise ValueError(msg)
    if not 0.0 <= max_orphan_fraction <= 1.0:
        msg = f"max_orphan_fraction must be in [0.0, 1.0], got {max_orphan_fraction}"
        raise ValueError(msg)

    # Guard 1, before ANY graph access. An unusable keep-set raises here.
    canonical_ids = read_canonical_ids(canonical_path)

    graph_ids_pre = [row["id"] for row in client.execute_read(_ALL_NARRATOR_IDS) if row.get("id")]
    graph_id_set = set(graph_ids_pre)

    # The provenance signal: keep-set ids this graph has never seen. Measured against the
    # PRE-state, and mode-invariant (a deleted orphan is by definition not canonical, so
    # the delete cannot move this number). Guard 2 refuses on it BEFORE any write.
    missing = sum(1 for cid in canonical_ids if cid not in graph_id_set)
    if missing > max_missing_fraction * len(canonical_ids):
        raise ForeignCanonicalSetError(
            path=canonical_path,
            missing=missing,
            canonical_total=len(canonical_ids),
            graph_total=len(graph_ids_pre),
            max_missing_fraction=max_missing_fraction,
        )

    orphans = [gid for gid in graph_ids_pre if gid not in canonical_ids]
    sample = orphans[:SAMPLE_SIZE]

    # Guard 3: the DELETE-side magnitude. `missing` above is structurally blind to this —
    # it bounds `canonical - graph`, none of whose elements can be deleted. A truncated
    # same-generation keep-set has missing == 0 and would still wipe the graph. Refuses
    # BEFORE any write, in both modes. (An empty graph has nothing to delete and no
    # fraction to speak of; guard 2 has already refused it unless deliberately overridden.)
    if graph_ids_pre and len(orphans) > max_orphan_fraction * len(graph_ids_pre):
        raise ExcessiveDeletionError(
            path=canonical_path,
            orphans=len(orphans),
            graph_total=len(graph_ids_pre),
            canonical_total=len(canonical_ids),
            missing=missing,
            max_orphan_fraction=max_orphan_fraction,
        )

    if dry_run:
        logger.info(
            "prune_narrators_dry_run",
            canonical_ids_seen=len(canonical_ids),
            graph_total=len(graph_ids_pre),
            orphans_identified=len(orphans),
            missing=missing,
        )
        return PruneResult(
            canonical_ids_seen=len(canonical_ids),
            graph_total=len(graph_ids_pre),
            orphans=len(orphans),  # dry run: the WOULD-delete count
            deleted=0,
            dry_run=True,
            orphans_identified=len(orphans),
            missing=missing,
            sample=sample,
        )

    for i in range(0, len(orphans), batch_size):
        chunk = orphans[i : i + batch_size]
        client.execute_write(_DETACH_DELETE_BY_ID, {"batch": chunk})

    # Read the graph BACK to measure the outcome — never trust a write's self-report.
    # ``orphans_remaining`` must be 0: that is the proof that exactly the orphans went
    # and every canonical narrator (edged or zero-degree) survived. ``deleted`` is the
    # pre/post total difference, the honest count of what left the graph.
    graph_ids_post = [row["id"] for row in client.execute_read(_ALL_NARRATOR_IDS) if row.get("id")]
    orphans_remaining = [gid for gid in graph_ids_post if gid not in canonical_ids]
    deleted = len(graph_ids_pre) - len(graph_ids_post)

    logger.info(
        "prune_narrators_complete",
        canonical_ids_seen=len(canonical_ids),
        graph_total=len(graph_ids_post),
        orphans_identified=len(orphans),
        orphans_remaining=len(orphans_remaining),
        deleted=deleted,
        missing=missing,
    )
    return PruneResult(
        canonical_ids_seen=len(canonical_ids),
        graph_total=len(graph_ids_post),
        orphans=len(orphans_remaining),  # real run: post-count, MUST be 0
        deleted=deleted,
        dry_run=False,
        orphans_identified=len(orphans),
        missing=missing,
        sample=sample,
    )


# The summary line's key set, in emission order. An APPEND-ONLY namespace: the consumer
# extracts a key with a greedy, unanchored `.*<key>=` (deploy#557 `summary_field`), so a
# NEW key that ends with an OLD key's name silently hijacks it — `edges_deleted=` would
# be read as `deleted=` (da#419). `missing` was chosen partly because it collides with
# nothing here, and `test_summary_line` pins that invariant mechanically rather than
# trusting anyone to remember it. Append; never rename, never insert a suffix-collider.
SUMMARY_KEYS = ("canonical", "graph_total", "orphans", "deleted", "missing")


def summary_line(result: PruneResult) -> str:
    """The one machine-readable line deploy#557's graph-side verify greps from stdout.

    Emitted on BOTH a dry run and a real run. Its absence is a hard failure in the
    workflow — a missing instrument reading is a stop, not a silent pass — so it is
    printed unconditionally by the CLI. ``orphans=`` carries the by-mode meaning
    documented on :class:`PruneResult`: the would-delete count on a dry run, the
    post-delete remaining count (``0`` on success) on a real run.

    ``missing=`` is appended LAST and is the fifth key, so the consumer's existing
    four-key extraction is untouched by its arrival. It is **reported, not asserted** —
    a legitimately partial load (``REFUSED_ROWS``, ``STOPPED_AT_LIMIT``, nodes-only)
    leaves canonical ids unloaded, so the refuse-or-proceed policy on a non-zero
    ``missing`` belongs to the workflow, not here. This command refuses only on a
    magnitude no partial load could produce (:class:`ForeignCanonicalSetError`).
    """
    return (
        f"PRUNE_NARRATORS_SUMMARY canonical={result.canonical_ids_seen} "
        f"graph_total={result.graph_total} orphans={result.orphans} "
        f"deleted={result.deleted} missing={result.missing}"
    )
