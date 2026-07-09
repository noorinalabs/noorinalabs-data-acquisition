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

What the bands mean
-------------------
The ordering is by **what is on disk when the process exits**, not by severity:

===========  ==========================================================
band         meaning
===========  ==========================================================
``0``        success
``1``, ``2`` nothing ran -- no stage produced output
``4``        a stage aborted mid-pipeline, PRIOR ARTIFACTS ARE ON DISK
``5``, ``6`` the stage completed; its output is suspect or incomplete
===========  ==========================================================

``4`` sits nearer ``6`` than to argparse's ``2``, which is the opposite of what
the da#384 ruling first asserted. ``MissingDependencyError`` is raised at
``run_dedup``'s entry, and dedup is **Step 4** of ``run_all``: NER,
disambiguation, bio-promotion, fuzzy clustering, same-name split, date
reconciliation and the ṭabaqa fallback have all already run, and Step 3 has
already written ``narrators_canonical.parquet``. The error subclasses
``BaseException`` precisely so it escapes those stages' handlers. The stage-entry
comment saying "before any input is read" is true of *dedup*, not of *resolve*.

``0`` and ``2`` are NOT members here: ``0`` is the absence of an error and ``2``
belongs to ``argparse``. Neither is ours to declare. They are reserved against,
in :data:`RESERVED_BY_RUNTIME`, so a future member cannot claim one.
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
    """

    # --- nothing ran -------------------------------------------------------
    LOAD_FAILED = 1
    """The load raised. The graph was not fully written."""

    # 2 is argparse's usage error. See RESERVED_BY_RUNTIME.

    # --- a stage stopped or aborted, artifacts may be on disk --------------
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
