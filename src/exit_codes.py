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
Each code is defined by **what is on disk when it fires**. There is no ordering
claim: an earlier draft called this "a monotone, how much is on disk, ascending"
and that is FALSE -- ``5`` is a fully written graph while ``6`` is an incomplete
one, so ``6`` wrote *less* than ``5``. Nor does it ascend on how far the process
got: ``REFUSED_ROWS`` fires before the validation verdict that produces
``VALIDATION_FINDINGS``.

The ordering is not the specification. **The per-member docstrings are.** A
monotone offered as the specification reinstates reasoning-from-a-neighbouring-
constant as the specification -- the very thing this module exists to abolish.

======  =====================  ===================================================
code    name                   what is on disk when it fires
======  =====================  ===================================================
``0``   *(success)*            everything the command meant to write
``1``   ``LOAD_FAILED``        nothing the load would have written
``2``   *(argparse)*           nothing
``3``   ``STOPPED_AT_LIMIT``   partial, by request (``--stop-after``)
``4``   ``MISSING_DEPENDENCY`` prior stages' artifacts (dedup is Step 4)
``5``   ``VALIDATION_FINDINGS`` the graph, fully written
``6``   ``REFUSED_ROWS``       the graph, minus the refused rows
``7``   ``ENRICH_FAILED``      the graph and the manifest; enrich failed
``8``   ``DB_UNREACHABLE``     nothing written by the command that raised it
``9``   ``UNROUTED_CORPUS``    nothing resolve would have written (NER Step-1 abort)
``10``  ``PARSE_PRODUCER_DEFECT`` the clean sources' staging; the defective source's is absent
``11``  ``RESOLVE_STAGE_FAILED`` prior stages' artifacts; a stage raised (da#360)
``12``  ``UNUSABLE_CANONICAL_SET`` the graph, untouched; ``prune-narrators`` refused (da#413)
======  =====================  ===================================================

da#360, da#369 and da#386 all branched from a main whose next free code was ``9``
— a three-way collision resolved centrally: da#369 kept ``9`` (``UNROUTED_CORPUS``),
da#386 took ``10`` (``PARSE_PRODUCER_DEFECT``), this took ``11``. All three have
since merged, so the table now carries all three.

``4`` is not a "nothing ran" code. ``MissingDependencyError`` is raised at
``run_dedup``'s entry, and dedup is **Step 4** of ``run_all``: NER,
disambiguation, bio-promotion, fuzzy clustering, same-name split, date
reconciliation and the ṭabaqa fallback have all already run, and Step 3 has
already written ``narrators_canonical.parquet``. The error subclasses
``BaseException`` precisely so it escapes those stages' handlers.

.. note::

   **Three false claims were made about this table and are recorded rather than
   quietly corrected**, because each was made in the artifact whose purpose is to
   stop people reasoning from a neighbouring constant:

   * ``run_dedup``'s *"before any input is read"* is true of **dedup**; it was
     generalized to **resolve**, making ``4`` look like "nothing ran".
   * ``_check_neo4j``'s *"the load did not happen"* is true of **that helper's
     own knowledge**; it was generalized into a band row, making ``1`` look like
     a "nothing ran" band while ``_cmd_enrich`` still exited ``1``.
   * The **monotone** itself: a story that fitted the values already chosen,
     pinned by a test that compared a list to its own sort and so could never
     fail. It passed on a deliberately scrambled enum.

   **A comment true at one call site is not a specification of the code it
   returns**, and an ordering that fits the values you happened to pick is not a
   property of them.

Call sites still emitting a bare integer
----------------------------------------
None. da#372 routed ``_cmd_load``'s findings condition to
:attr:`ExitCode.VALIDATION_FINDINGS`, and ``TestNoUnnamedExit`` now asserts the
set is empty rather than pinning a survivor.

Who emits ``1``
---------------
Exactly one call site, read from the tree rather than reasoned about::

    src/cli.py   sys.exit(ExitCode.LOAD_FAILED)   # _cmd_load, in the except around load_all

Named by function and enclosing block, never by line number. The first version of
this paragraph wrote ``src/cli.py:258`` -- true of the tree the walk ran against,
false by the time that same commit landed, because rewriting the comment block
directly above the call moved it six lines down. The call has sat at ``252``,
``258`` and ``264`` across the three commits of da#372: it drifted at every one.

A line number in prose is a hand-maintained fact about a file, unguarded, in the
module whose whole argument is that hand-maintained facts rot and that the
mechanism must be the guard rather than the test. Nothing here can red on a stale
line number. Nothing should have to.

``_check_neo4j`` is **not** among them; it exits :attr:`ExitCode.DB_UNREACHABLE`
(da#384 Amendment R). An earlier version of this paragraph named it, which is the
defect da#392 exists to catch: prose describing a call site it no longer matches.

That single emitter fires from inside ``_cmd_load``'s ``except`` around
``load_all`` -- before any manifest is written -- so for ``isnad-ingest load``,
``rc=1 implies an unwritten manifest`` holds.

It does **not** hold for ``isnad-ingest pipeline``. ``pipeline`` runs
``_cmd_enrich`` after the load has committed, and enrich's audit write sits
outside any handler while neither ``main()`` nor ``_cmd_pipeline`` has a
top-level one -- so an escaping exception reaches CPython's default handler and
exits ``1`` (da#394). ``sys.exit`` is not the only route to a process status, and
enumerating ``sys.exit`` call sites cannot see the other one.

**This registry cannot fix a control-flow defect.** Ivana Horvat's blocker on
da#372 -- that ``save_manifest``/``create_audit_entry``/``write_audit_entry`` sat
outside ``_cmd_load``'s ``try``, so a committed load could exit ``1`` -- was a
control-flow question, not a numbering one, and da#384 did not close it. da#372
closes it for ``load`` by wrapping that bookkeeping. da#394 tracks the same shape
in ``_cmd_enrich``.

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
    "EXIT_DB_UNREACHABLE",
    "EXIT_UNUSABLE_CANONICAL_SET",
    "EXIT_ENRICH_FAILED",
    "EXIT_LOAD_FAILED",
    "EXIT_MISSING_DEPENDENCY",
    "EXIT_PARSE_PRODUCER_DEFECT",
    "EXIT_REFUSED_ROWS",
    "EXIT_STAGE_FAILED",
    "EXIT_STOPPED_AT_LIMIT",
    "EXIT_UNROUTED_CORPUS",
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

    Says nothing about any other command. ``_cmd_enrich`` exits
    :attr:`ENRICH_FAILED`; the connectivity pre-check exits :attr:`DB_UNREACHABLE`.
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

    Fires only on ``summary.steps_failed`` -- a real enrich failure after a
    committed load. **A failed enrich is not a failed load.** That is the whole
    reason this member exists; no argument from where it sits relative to any
    other value is doing any work here.
    """

    RESOLVE_STAGE_FAILED = 11
    """A ``resolve`` stage raised. Prior stages' artifacts are on disk; the failed
    stage's output is absent or partial (da#360).

    Value ``11``, not ``9``: da#360, da#369 (``UNROUTED_CORPUS``) and da#386
    (``PARSE_PRODUCER_DEFECT``) all branched from a main whose next free code was
    ``9``, a three-way collision ``@enum.unique`` would have caught only pairwise
    on merge. Deconflicted centrally — da#369 keeps ``9``, da#386 takes ``10``,
    this takes ``11``. The gap between ``8`` and ``11`` in the table below is those
    two siblings' rows, which land in their own PRs; a member is not documented
    here until it exists here.

    Distinct from :attr:`MISSING_DEPENDENCY` (4), which is ``dedup``'s declared-
    dependency guard — a ``BaseException`` raised at ``run_dedup``'s entry before
    it reads input. This code is for a stage that raised a *plain* ``Exception``
    mid-run: ``run_all`` used to wrap every stage in ``except Exception`` and
    log-and-continue, reach ``resolve_pipeline_complete`` and exit ``0`` — a
    7.5-hour run could report success having skipped a stage. ``run_all`` now
    records each raise as a ``StageErrored`` outcome and raises
    ``ResolveStageError`` at the end; ``_cmd_resolve`` maps that here. It fires
    only after the best-effort continue-past-failure sweep, so it means "at least
    one stage failed", aggregating every failed stage, not just the first.
    """

    DB_UNREACHABLE = 8
    """The connectivity pre-check could not reach Neo4j (da#384 Amendment R).

    No stage ran; nothing was written by the command that raised it.
    **Stage-agnostic by construction.** ``_check_neo4j`` has four callers --
    ``_cmd_load``, ``_cmd_validate``, ``_cmd_migrate_transmitted_hadith_ids``,
    ``_cmd_enrich`` -- and cannot know which one it is serving. It knows exactly
    one fact, identical at every site: the database is not there. This names that.

    It previously exited :attr:`LOAD_FAILED`, which was true for one caller of
    four. Passing the code in as a parameter would not have fixed it: it would
    have turned four call sites into a hand-maintained mapping from caller to
    exit code -- the same object as a hand-maintained list of reserved values, of
    AST node types, or of permitted bare exits. Each was proposed as the fix for
    the one before it.
    """

    UNROUTED_CORPUS = 9
    """``resolve`` aborted in NER (Step 1) on a staged corpus with no route (da#369).

    Raised by :func:`src.resolve.ner.run` **before** any mention output is written,
    when a corpus is present in staging but is opted in to no NER extraction route
    (phase1 / arabic / english) and is not on the skip list. Nothing this
    ``resolve`` invocation would have written is on disk: NER raises before
    ``narrator_mentions_resolved.parquet``, so disambiguation, bio-promotion and
    every later stage have not run. It replaces the removed matn fallback, which
    silently mined ``full_text_ar`` for any unrouted or isnad-null corpus instead
    of failing. Distinct from :attr:`MISSING_DEPENDENCY` (an absent *dependency* at
    dedup's Step-4 entry) -- this is an unroutable *input* at NER's Step-1 entry.
    """

    PARSE_PRODUCER_DEFECT = 10
    """``parse`` aborted: an adapter hit the da#355 producer gate (da#386).

    ``generate_source_id`` asserts the id grammar and raises
    :exc:`~src.parse.identity.DoubledCorpusPrefixError` when ``collection ==
    corpus``. ``src.parse.run_all`` used to catch that in its broad ``except
    Exception`` (it subclasses :exc:`ValueError`), log ``parse_failed`` and
    ``return`` -- so ``parse`` exited ``0`` with the **stale** staging parquet in
    place, and da#359's ``_cmd_load`` remediation ("re-run ``parse``") silently
    changed nothing.

    What is on disk when it fires: the offending source's partial writes from this
    run are purged and its stale output was NOT regenerated; every OTHER source
    that parsed cleanly has its staging parquet written. Raised from ``_cmd_parse``
    (and so halts ``pipeline`` before ``resolve``/``load`` reads an incomplete
    staging set). A data-defect assertion is not a transient parse failure, so it
    fails the whole invocation rather than being logged and skipped.
    """

    UNUSABLE_CANONICAL_SET = 12
    """``prune-narrators`` refused: its authoritative keep-set was untrustworthy (da#413).

    What is on disk when it fires: **the graph, exactly as it was** — zero nodes
    deleted, zero edges touched. This is the load-bearing guarantee of a destructive
    op: a bad read must never wipe the graph.

    ``prune-narrators`` DETACH-DELETEs every ``Narrator`` whose ``id`` is not in
    ``narrators_canonical.parquet``. That keep-set is the ONLY thing standing between
    the command and deleting a node, so the command refuses — before it writes to the
    graph at all — if the keep-set cannot be trusted to name the survivors.

    Named for the CONCEPT — an untrustworthy keep-set — not for any one door, and there
    are **five**. Three ask whether the parquet is usable *on its face* and fire before
    the graph is even read: an **empty** keep-set (which would delete every narrator), a
    **missing** parquet, an **unreadable/malformed** one. The last two ask questions those
    three structurally cannot, because they are *relations between the parquet and the
    graph* rather than properties of the parquet, and so they fire after the graph is
    **read** and before any **write**:

    * **not this graph's** — a stale ``--canonical`` from a previous resolve run,
      readable and non-empty and catastrophic. Detected by ``missing``
      (= ``|canonical - graph|``); raised as ``ForeignCanonicalSetError``.
    * **this graph's, but a remnant of it** — a *truncated* keep-set (a resolve stopped
      early, a half-written or partially uploaded parquet, a null-heavy ``canonical_id``
      column). Every id in it is real and current, so ``missing`` is **0** and every
      id-provenance instrument reports it healthy, while the prune deletes all but the
      remnant. Detected by the delete magnitude ``|orphans| / |graph|``; raised as
      ``ExcessiveDeletionError``.

    The fifth door is the one worth dwelling on, because it is what the *name* of this
    member used to hide. It was called ``EMPTY_CANONICAL_SET`` — and its own docstring
    argued that "naming the member after only the empty case would re-arm the trap for
    the other doors" while doing precisely that. The value stays ``12``; the name now
    says what it has always meant. (Renamed on Jean-Claude Habimana's read of da#413,
    which caught the contradiction.)

    All five leave the graph untouched, name the same remedy (fix the input, re-run) and
    are propagated verbatim by a consumer that only asks whether the rc is zero. Sharing
    one code is therefore the deliberate reading of this registry's own rule — a code is
    defined by **what is on disk when it fires**. A second member would encode a
    distinction nothing acts on, while re-opening the three-way collision hazard this
    module was written to record. The exception subclasses carry the distinction where it
    is actually consumed: in-process, and on stderr.

    This is the da#309 / da#361 fail-loud-on-missing-input discipline applied to a
    *destructive* op. It gets its own code — not :attr:`LOAD_FAILED` — because a
    prune is not a load, and because the safe "deleted nothing" state must be
    unambiguous: a distinct code lets deploy#557's workflow tell "refused, graph
    safe, fix the parquet and retry" apart from any partial-deletion failure.
    """


# Module-level aliases. Consumers may import either the enum or these names; the
# alias form keeps a migrating call site to an *import-site* change rather than a
# value change, which is what lets an already-approved PR rebase without
# invalidating its approvals.
EXIT_DB_UNREACHABLE = ExitCode.DB_UNREACHABLE
EXIT_ENRICH_FAILED = ExitCode.ENRICH_FAILED
EXIT_LOAD_FAILED = ExitCode.LOAD_FAILED
EXIT_STOPPED_AT_LIMIT = ExitCode.STOPPED_AT_LIMIT
EXIT_MISSING_DEPENDENCY = ExitCode.MISSING_DEPENDENCY
EXIT_VALIDATION_FINDINGS = ExitCode.VALIDATION_FINDINGS
EXIT_REFUSED_ROWS = ExitCode.REFUSED_ROWS
EXIT_UNROUTED_CORPUS = ExitCode.UNROUTED_CORPUS
EXIT_PARSE_PRODUCER_DEFECT = ExitCode.PARSE_PRODUCER_DEFECT
EXIT_STAGE_FAILED = ExitCode.RESOLVE_STAGE_FAILED
EXIT_UNUSABLE_CANONICAL_SET = ExitCode.UNUSABLE_CANONICAL_SET
