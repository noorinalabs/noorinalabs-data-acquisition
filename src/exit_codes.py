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

What each code means -- a monotone on how much is on disk
---------------------------------------------------------
The ordering is **how much is on disk when the process exits, ascending**. Not
"nothing ran / something ran": that framing produced two false claims in this
registry's own ruling, one about ``4`` and one about ``1``. A monotone has a fact
behind it at every value, rather than a story that must hold for every emitter.

======  =====================  ===================================================
code    name                   on disk when it fires
======  =====================  ===================================================
``1``   ``LOAD_FAILED``        nothing the load would have written
``2``   *(argparse)*           nothing
``3``   ``STOPPED_AT_LIMIT``   partial, by request (``--stop-after``)
``4``   ``MISSING_DEPENDENCY`` prior stages' artifacts (dedup is Step 4)
``5``   ``VALIDATION_FINDINGS`` the graph, fully written
``6``   ``REFUSED_ROWS``       the graph, minus the refused rows
``7``   ``ENRICH_FAILED``      the graph and the manifest, plus a partial enrich
======  =====================  ===================================================

``4`` sits nearer ``6`` than to argparse's ``2``. ``MissingDependencyError`` is
raised at ``run_dedup``'s entry, and dedup is **Step 4** of ``run_all``: NER,
disambiguation, bio-promotion, fuzzy clustering, same-name split, date
reconciliation and the ṭabaqa fallback have all already run, and Step 3 has
already written ``narrators_canonical.parquet``. The error subclasses
``BaseException`` precisely so it escapes those stages' handlers.

``7`` exists because **a failed enrich is not a failed load.** ``_cmd_enrich``
runs after ``_cmd_load`` has committed the graph, the manifest and the audit
entry. Routing it to :attr:`ExitCode.LOAD_FAILED` would have made this module
*assert* that falsehood — and a bare ``1`` claims nothing, while a registry
constant claims something and invites citation. Laundering a wrong code through
the registry does not make it right; it makes it citable.

.. note::

   **Two false claims are recorded here rather than quietly corrected**, because
   both were made in the artifact whose purpose is to stop people reasoning from
   a neighbouring constant, and both had the same mechanism:

   * ``run_dedup``'s *"checked at stage entry, before any input is read"* is true
     of **dedup**; it was generalized to **resolve**, making ``4`` look like a
     "nothing ran" code.
   * ``_check_neo4j``'s *"the load did not happen"* is true of **that call
     site**; it was generalized into a band table row, making ``1`` look like a
     "nothing ran" band while ``_cmd_enrich`` still exited ``1`` from a fully
     committed graph.

   **A comment true at one call site is not a specification of the code it
   returns.** ``1`` is true again now — not by rewording the row, but by giving
   ``_cmd_enrich`` a code of its own.

Call sites still emitting a bare integer
----------------------------------------
One remains, and it is deliberately not this PR's to fix:

* ``src/cli.py`` ``_cmd_load``, on ``not summary.validation_passed`` -- a findings
  condition mis-emitting ``1``. It belongs to da#372, which owns ``_cmd_load``'s
  exit codes and whose reviewers' verdicts hang on that diff (da#384 Amendment I).

Once it moves to :attr:`ExitCode.VALIDATION_FINDINGS`, the only emitters of ``1``
are ``_check_neo4j`` (fires before the load; nothing written) and ``_cmd_load``'s
genuine failure path. Both mean the load did not happen, which is what makes
``rc=1 implies an unwritten manifest`` true for ``load`` *and* for ``pipeline``.

**This registry cannot fix a control-flow defect.** ``rc=1`` remains reachable
from a *committed* load because ``save_manifest``/``create_audit_entry``/
``write_audit_entry`` sit outside ``_cmd_load``'s ``try`` and ``main()`` has no
top-level handler. That is Ivana Horvat's blocker on da#372, not a numbering
question, and da#384 does not close it.

``0`` and ``2`` are NOT members: ``0`` is the absence of an error and ``2``
belongs to ``argparse``. Neither is ours to declare. :data:`RESERVED_BY_RUNTIME`
holds them, and :meth:`ExitCode.__new__` raises if a member claims one -- a test
would be the fallback, not the mechanism.

Scope of the sole-declarer guard
--------------------------------
The scan walks for an ``ast.Name`` in ``Store`` context rather than enumerating
the statement types that bind a name. **A hand-maintained list of node types is
the same defect as a hand-maintained list of reserved values** -- it was the last
one hiding inside the fix for hand-maintained lists. The ``Store``-context walk
enumerates *the language's* notion of a binding; the ``enum`` API enumerates
*the enum's* notion of a member. Same move, twice.

A companion guard reds on any ``sys.exit(<int literal != 0>)``: ``@enum.unique``
makes a duplicate *declaration* inexpressible, and nothing else would make a
duplicate *emission* inexpressible. ``sys.exit(7)`` names no constant and is
invisible to every name-keyed check.

``tests/test_exit_codes.py`` proves no module outside this one *declares* an
exit-code name. It cannot see a **second** ``IntEnum`` whose members carry
neither an ``EXIT_`` prefix nor a registry member name::

    class OtherExit(enum.IntEnum):
        SOMETHING_ELSE = 4          # invisible to the scan AND to @enum.unique

The registry owns the space by convention there, not by mechanism. Cheap to say;
expensive to discover.
"""

from __future__ import annotations

import enum

__all__ = [
    "EXIT_ENRICH_FAILED",
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

    Note the precise claim, and note that it CHANGED in the commit that added it.
    ``src/cli.py`` imports this module eagerly, at module scope, so **every CLI
    invocation raises on a duplicate** -- ``isnad-ingest --help`` included. An
    earlier draft of this docstring hedged that "absent an eager import a
    duplicate would surface on first use of the offending command"; the same
    commit added that import and falsified the hedge. It understated the
    guarantee, which is the safe direction and therefore the one that survives
    review unchallenged. Pytest collection imports the module too, so CI catches
    a duplicate independently of the CLI.

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

    ENRICH_FAILED = 7
    """The enrich stage failed. The graph, manifest and audit entry are written (da#384).

    A failed enrich is NOT a failed load. `_cmd_enrich` runs after `_cmd_load`
    committed everything it writes, so this cannot be `LOAD_FAILED` without the
    registry asserting a falsehood about on-disk state.
    """


# Module-level aliases. Consumers may import either the enum or these names; the
# alias form keeps a migrating call site to an *import-site* change rather than a
# value change, which is what lets an already-approved PR rebase without
# invalidating its approvals.
EXIT_ENRICH_FAILED = ExitCode.ENRICH_FAILED
EXIT_LOAD_FAILED = ExitCode.LOAD_FAILED
EXIT_STOPPED_AT_LIMIT = ExitCode.STOPPED_AT_LIMIT
EXIT_MISSING_DEPENDENCY = ExitCode.MISSING_DEPENDENCY
EXIT_VALIDATION_FINDINGS = ExitCode.VALIDATION_FINDINGS
EXIT_REFUSED_ROWS = ExitCode.REFUSED_ROWS
