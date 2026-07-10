"""The one owner of this pipeline's process exit-code space (da#384).

Why this module exists
----------------------
Two open PRs each declared ``= 4`` in a file the other did not contain --
``EXIT_MISSING_DEPENDENCY`` (da#309) and ``EXIT_VALIDATION_FINDINGS`` (da#354).
Both suites were green, because neither could see the other's constant. The only
guard was a hand-written reserved tuple, which reddens on the collision somebody
already thought of and is blind to every sibling nobody remembered to add.

``_cmd_pipeline`` runs acquire -> parse -> resolve -> load -> enrich in ONE
process, so the exit-code space is global. A double-booking is a real ambiguity,
not a theoretical one: ``isnad-ingest pipeline`` exiting ``4`` would have meant
either "resolve is missing its ML dependency group" or "load succeeded and
validation has findings."

The fix is not to renumber. It is to make a duplicate **inexpressible**:
:func:`enum.unique` raises ``ValueError`` when this module is imported, and
every ``EXIT_*`` consumer imports this module. A test that *checks* uniqueness
is the fallback; the decorator is the mechanism.

What each code means
--------------------
Each code is defined by **what is on disk when the process exits**. There is no
"band" claim here, deliberately -- see the warning below.

==========  ================================================================
code        meaning
==========  ================================================================
``0``       success
``1``       ``LOAD_FAILED`` -- the load raised; the graph was not fully written
``2``       argparse's usage error
``3``       ``STOPPED_AT_LIMIT`` -- a stage stopped at its ``--stop-after`` budget
``4``       ``MISSING_DEPENDENCY`` -- ``resolve`` aborted at ``run_dedup``'s
            entry; Steps 1-3.7 ran and their artifacts ARE ON DISK
``5``       ``VALIDATION_FINDINGS`` -- the stage completed; validation has findings
``6``       ``REFUSED_ROWS`` -- the load completed having REFUSED input; the
            graph is incomplete
==========  ================================================================

``4`` sits nearer ``6`` than to argparse's ``2``. ``MissingDependencyError`` is
raised at ``run_dedup``'s entry, and dedup is **Step 4** of ``run_all``: NER,
disambiguation, bio-promotion, fuzzy clustering, same-name split, date
reconciliation and the ṭabaqa fallback have all already run, and Step 3 has
already written ``narrators_canonical.parquet``. The error subclasses
``BaseException`` precisely so it escapes those stages' handlers. The stage-entry
comment saying "before any input is read" is true of *dedup*, not of *resolve*.

.. warning::

   **``1`` is NOT a "nothing ran" band, and an earlier draft of this table said
   it was.** The claim was generalized from ``_check_neo4j``'s call site, where
   *"the load did not happen; nothing was written"* is locally true. It is not
   true of the code. ``_cmd_enrich`` (``src/cli.py``) still exits a bare ``1``
   **after** ``_cmd_load`` has committed the graph, written the manifest and
   written the audit entry -- so ``isnad-ingest pipeline`` exiting ``1``
   routinely means a fully-loaded graph plus a failed enrich.

   This is the same error as the "before any input is read" one above, one code
   over: **a comment that is true at one call site is not a specification of the
   code it returns.** Both are recorded rather than quietly corrected, in the
   artifact whose entire purpose is to stop people reasoning from a neighbouring
   constant.

Call sites still emitting a bare integer
----------------------------------------
The registry is only the whole truth if nothing exits a bare literal. One
remains:

* ``src/cli.py`` ``_cmd_enrich`` -- ``sys.exit(1)`` when ``summary.steps_failed``.
  It runs *after* the load committed, so it is neither :attr:`ExitCode.LOAD_FAILED`
  ("the load raised; nothing was written") nor :attr:`ExitCode.VALIDATION_FINDINGS`
  ("the stage completed; validation has findings"). It needs a member of its own
  (an ``ENRICH_FAILED``), and **assigning a value is the da#384 ruling's call, not
  the implementer's** -- taking one here is precisely the "reasoned from a
  neighbouring constant" move this module exists to prevent. It is also half of
  the ``rc=1`` invariant leak, which is deliberately NOT in this issue's scope
  (da#384 §5) and remains a blocker on da#372.

``0`` and ``2`` are NOT members: ``0`` is the absence of an error and ``2``
belongs to ``argparse``. Neither is ours to declare. :data:`RESERVED_BY_RUNTIME`
holds them, and :meth:`ExitCode.__new__` raises if a member claims one -- a test
would be the fallback, not the mechanism.
"""

from __future__ import annotations

import enum

__all__ = [
    "EXIT_LOAD_FAILED",
    "EXIT_MISSING_DEPENDENCY",
    "EXIT_REFUSED_ROWS",
    "EXIT_STOPPED_AT_LIMIT",
    "EXIT_VALIDATION_FINDINGS",
    "ExitCode",
    "RESERVED_BY_RUNTIME",
]

# Values this process does not get to assign: 0 is success (the absence of an
# error) and 2 is argparse's usage error, emitted before any of our code runs.
RESERVED_BY_RUNTIME: frozenset[int] = frozenset({0, 2})


@enum.unique
class ExitCode(enum.IntEnum):
    """Every non-zero exit status the pipeline may emit.

    ``@enum.unique`` is load-bearing: ``IntEnum`` silently accepts a duplicate
    value as an *alias*, so without the decorator a second claim on ``4`` would
    be a no-op rather than an error. With it, the duplicate raises ``ValueError``
    **when this module is imported** -- and every ``EXIT_*`` consumer imports it.

    Note the precise claim. ``src/cli.py`` imports ``src.resolve`` and
    ``src.graph`` lazily, inside command functions, so absent an eager import a
    duplicate would surface on first use of the offending command rather than at
    process start. Pytest collection imports this module, so CI catches it either
    way. "At import time" is the same imprecision this registry exists to remove.

    ``__new__`` is the second half of the same posture. ``@enum.unique`` rejects a
    duplicate value *among members*; it has no opinion whatever about ``0`` or
    ``2``, so a member claiming argparse's ``2`` would be **expressible**, with
    only a test standing between it and production. That is the exact posture this
    registry abolishes for duplicates, and a docstring calling a test "the
    fallback" cannot then rely on one. ``__new__`` makes it inexpressible too.
    """

    def __new__(cls, value: int) -> ExitCode:
        # A member may not claim a value the runtime already owns (0 success,
        # 2 argparse). Raised when the registry is imported, exactly like
        # @enum.unique's ValueError -- not deferred to a test.
        if value in RESERVED_BY_RUNTIME:
            msg = (
                f"exit code {value} is reserved by the runtime "
                f"({sorted(RESERVED_BY_RUNTIME)}) and cannot be claimed by a member"
            )
            raise ValueError(msg)
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    LOAD_FAILED = 1
    """The load raised. The graph was not fully written.

    NOT a general "nothing ran" band: ``_cmd_enrich`` still exits a bare ``1``
    after the load has committed its graph, manifest and audit entry. See the
    module docstring, "Call sites still emitting a bare integer".
    """

    # 2 is argparse's usage error. See RESERVED_BY_RUNTIME.

    STOPPED_AT_LIMIT = 3
    """A stage stopped cleanly at its ``--stop-after`` budget (da#276)."""

    MISSING_DEPENDENCY = 4
    """``resolve`` aborted at ``run_dedup``'s entry; Steps 1-3.7 already ran (da#309)."""

    # --- the stage completed; its output is suspect or incomplete ----------
    VALIDATION_FINDINGS = 5
    """The load succeeded and is committed; post-load validation has findings (da#354)."""

    REFUSED_ROWS = 6
    """The load ran to completion having REFUSED input; the graph is incomplete (da#355/da#359).

    Named for the CONCEPT, not for today's instance. The loaders refuse a row for
    more than one reason -- a doubled corpus prefix and a blank ``source_id`` are
    the same silent data loss through different doors -- and the predicate that
    raises this code is ``LoadSummary.total_refused``. Naming it after the
    malformed-id class alone would re-arm the trap for the next refusal class,
    the same mistake as ``total_skipped`` blending refusals with deliberate skips.
    """


# Module-level aliases. Consumers may import either the enum or these names; the
# alias form keeps a migrating call site to an *import-site* change rather than a
# value change, which is what lets an already-approved PR rebase without
# invalidating its approvals.
EXIT_LOAD_FAILED = ExitCode.LOAD_FAILED
EXIT_STOPPED_AT_LIMIT = ExitCode.STOPPED_AT_LIMIT
EXIT_MISSING_DEPENDENCY = ExitCode.MISSING_DEPENDENCY
EXIT_VALIDATION_FINDINGS = ExitCode.VALIDATION_FINDINGS
EXIT_REFUSED_ROWS = ExitCode.REFUSED_ROWS
