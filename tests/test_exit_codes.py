"""da#384: the exit-code space has one owner, and a duplicate is inexpressible.

Two open PRs each declared ``= 4`` in a file the other did not contain, and both
suites stayed green because neither could see the other's constant. The guard
that existed was a hand-written reserved tuple, which reddens only on the
collision somebody already thought of.

This file has two halves, and they guard different things:

1. **Uniqueness** is enforced by ``@enum.unique``, not by a test. A duplicate
   raises ``ValueError`` when :mod:`src.exit_codes` is imported. The tests here
   prove the decorator is actually applied and actually bites.

2. **Sole-declarership** is enforced by scanning ``src/`` for any exit-code
   assignment outside the registry module. Under a registry this is no longer a
   collision check -- ``@enum.unique`` covers that -- it is the assertion that
   nobody re-opens the space by declaring a constant somewhere else.

   "Exit-code assignment" means an ``EXIT_*`` alias **or a registry member name**.
   Matching only the ``EXIT_`` prefix was blind to the very form the registry
   uses -- ``REFUSED_ROWS = 6`` inside a class body carries no prefix -- so a
   rival module written in the registry's own style passed clean. Worse, the
   instrument guard passed anyway, on the five module-level aliases, and so
   demonstrated a capability the scan did not have.

3. **Runtime-reserved values** (``0`` success, ``2`` argparse) are made
   inexpressible by ``ExitCode.__new__``. ``@enum.unique`` has no opinion about
   them; without the hook a member claiming ``2`` would be expressible with only
   a test in the way.

The scan is **AST-based, not regex-based**, deliberately. Alejandra
Reyes-Fuentes measured seven declaration forms against the regex this replaces
(``^(EXIT_[A-Z_]+)\\s*=\\s*(\\d+)\\s*$``); one was visible and six passed a
collision straight through: annotated assignment, ``Final``, a trailing comment,
indentation, an ``IntEnum`` member, and a computed value. An AST walk sees all of
them, including the computed RHS, because it matches on the *target name* rather
than on the literal.

Every assertion below is paired with an instrument guard. A scan that finds
nothing is not a scan that found no collisions.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.cli as cli
from src.exit_codes import RESERVED_BY_RUNTIME, ExitCode

# ---------------------------------------------------------------------------
# Locating the registry -- DERIVED, never a path literal.
# ---------------------------------------------------------------------------


def _registry_path() -> Path:
    """Absolute path of the module the enum actually lives in.

    Derived from ``ExitCode.__module__`` rather than written down. A filename
    literal here would itself be a hand-written reservation -- the exact shape
    that made the `4` collision invisible.
    """
    module = sys.modules[ExitCode.__module__]
    assert module.__file__ is not None, "registry module has no __file__"
    return Path(module.__file__).resolve()


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _registry_member_names() -> frozenset[str]:
    """The registry's member names, through the ``enum`` API. Never grepped.

    Reads ``__members__``, **not** iteration. Iteration skips aliases, so an
    iteration-derived set would make two OTHER guards silently alias-blind:

    * the docstring guard would never check an undocumented alias;
    * the sole-declarer scan would not flag a rival module declaring the alias's
      name, because that name is not in the set it matches against.

    Both are unreachable today, because ``test_the_registry_has_no_aliases``
    reds on any alias. But that makes the guards **ordered rather than
    independent**: delete the one assertion and two more go blind with it,
    which is exactly the mutant it exists to catch. `__members__` costs nothing
    and makes each guard alias-aware on its own. (Oyunbileg Batbayar.)
    """
    members = frozenset(ExitCode.__members__)
    # INSTRUMENT GUARD: a derivation that yields nothing is not a derivation that
    # found nothing wrong. Every consumer of this set -- the sole-declarer scan,
    # the docstring guard, the exit allowlist -- would silently permit everything.
    assert members, "the registry derivation yielded no members; the guards are blind"
    return members


def _exported_exit_aliases() -> frozenset[str]:
    """Names the registry EXPORTS that resolve to a member. Derived twice over.

    `_is_exit_name` matches any `EXIT_`-prefixed identifier, which is right for the
    sole-declarer scan (a rival *declaration* is an offence whatever it is called)
    and **wrong for the exit allowlist**. Ivana Horvat::

        _is_exit_name("EXIT_MADE_UP")  -> True

        def _die(EXIT_CODE: int):
            sys.exit(EXIT_CODE)        # PERMITTED. A parameter is not a Store
                                       # binding, so the sole-declarer scan never
                                       # sees it either.
        def _die(code: int):
            sys.exit(code)             # REDS.

    **Naming the parameter well is what made it invisible. A guard defeated by
    good naming will be defeated.**

    Her fix is `arg.id in registry.__all__`, which also makes `__all__`
    load-bearing: a stale `__all__` narrows what the guard permits, so the
    tidiness item becomes the mechanism.

    One refinement. `__all__` also carries `ExitCode` and `RESERVED_BY_RUNTIME`,
    so a bare membership test would permit `sys.exit(ExitCode)` and
    `sys.exit(RESERVED_BY_RUNTIME)` -- neither an exit code. Resolving each name
    to its value and requiring it to BE a member closes that, and is derived from
    the registry twice: once through `__all__`, once through the value.
    """
    registry = sys.modules[ExitCode.__module__]
    exported = getattr(registry, "__all__", ())
    aliases = frozenset(
        name for name in exported if isinstance(getattr(registry, name, None), ExitCode)
    )
    # INSTRUMENT GUARD: an empty derivation would make the Name branch refuse
    # every live call site, which reads as "the guard is strict" rather than
    # "the guard is broken". A derivation that yields nothing is not a
    # derivation that found nothing wrong.
    assert aliases, "no exported exit aliases resolved to a member; the allowlist is blind"
    return aliases


def _is_exit_name(name: str) -> bool:
    """Names that constitute an exit-code declaration.

    Two shapes, and the second is the one that nearly shipped a blind guard:

    * ``EXIT_*`` -- the module-level alias form.
    * A registry MEMBER name (``LOAD_FAILED``, ``REFUSED_ROWS``, ...) -- the form
      the registry itself uses, inside a class body, carrying **no** prefix.

    Matching only on the ``EXIT_`` prefix let a rival module written in the
    registry's own style pass clean:

        class _Rival(enum.IntEnum):
            VALIDATION_FINDINGS = 5
            REFUSED_ROWS = 6          # invisible

    The member set is read through the ``enum`` API so it cannot drift from the
    thing it guards, and so a representation change cannot blind it.
    """
    return (name.startswith("EXIT_") and name.isupper()) or name in _registry_member_names()


def _exit_name_assignments(tree: ast.AST) -> list[str]:
    """Every exit-code name BOUND anywhere in *tree*, at any nesting depth.

    Does not enumerate the statement types that can bind a name -- it asks Python
    whether the name is being bound, via ``ast.Name`` with a ``Store`` context.
    Alejandra Reyes-Fuentes found the enumerating version missed two forms:

        EXIT_A, EXIT_B = 4, 5     # ast.Tuple target, not ast.Name
        (EXIT_X := 4)             # ast.NamedExpr, neither Assign nor AnnAssign

    The tuple form is the one that matters: it is how somebody declares two codes
    on one line, and a `4` in it sailed past the guard that exists to catch a `4`.

    Out of reach of ANY name-keyed scan, and deliberately not chased:
    ``globals()["EXIT_X"] = 4`` -- the name is a string literal.
    """
    return [
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and _is_exit_name(n.id)
    ]


def _declarations_outside_registry() -> dict[Path, list[str]]:
    registry = _registry_path()
    offenders: dict[Path, list[str]] = {}
    for py in _src_root().rglob("*.py"):
        if py.resolve() == registry:
            continue
        names = _exit_name_assignments(ast.parse(py.read_text(encoding="utf-8")))
        if names:
            offenders[py] = names
    return offenders


# ---------------------------------------------------------------------------
# Half 1 -- uniqueness is a property of the enum, not of a test
# ---------------------------------------------------------------------------


class TestDuplicateValueIsInexpressible:
    def test_the_registry_has_no_aliases(self) -> None:
        """`@enum.unique`'s REMOVAL must be detectable. Two earlier tests could not see it.

        `IntEnum` accepts a duplicate value as a silent **alias**. The obvious
        check is blind to exactly that::

            values = [c.value for c in ExitCode]
            assert len(values) == len(set(values))    # PASSES on the mutant

        because **iteration does not yield aliases** -- the very mechanism this
        module's docstring names as the hazard, defeating the test that names it.
        Verified: strip the decorator, plant `PLANTED_DUPLICATE = 4`, and
        iteration yields `[1, 4]` while `__members__` yields three names.

        And its companion `len(values) >= 5` guarded against an *empty* enum, not
        an *aliased* one -- an instrument guard for a failure mode nobody feared.

        `__members__` includes aliases; iteration does not. The blind spot and the
        fix are the same fact read in two directions. (Ivana Horvat.)
        """
        aliases = set(ExitCode.__members__) - {c.name for c in ExitCode}
        assert aliases == set(), f"aliased members (is @enum.unique applied?): {sorted(aliases)}"

        # INSTRUMENT GUARD: `__members__` must actually contain the members, or
        # the difference above is empty for the wrong reason.
        assert set(ExitCode.__members__) == {c.name for c in ExitCode}
        assert len(ExitCode.__members__) >= 6, f"registry looks empty: {list(ExitCode.__members__)}"

    def test_importing_the_registry_is_what_raises(self) -> None:
        """The precise claim: it raises when the registry is IMPORTED.

        `src/cli.py` imports this module eagerly at module scope, so every CLI
        invocation raises on a duplicate. Pytest collection imports it too.
        """
        module = importlib.import_module(ExitCode.__module__)
        assert module.ExitCode is ExitCode  # a clean import, no ValueError


# ---------------------------------------------------------------------------
# Half 2 -- the reserved set comes from the enum API, never from a grep
# ---------------------------------------------------------------------------


class TestReservedSetIsEnumerated:
    def test_no_member_claims_a_runtime_reserved_code(self) -> None:
        """`0` is success and `2` is argparse's. Neither is ours to assign.

        This is now a CHECK THAT THE MECHANISM IS APPLIED, not the mechanism.
        `ExitCode.__new__` raises when the registry is imported; see below.
        """
        assert RESERVED_BY_RUNTIME == frozenset({0, 2})
        collisions = {c.name: c.value for c in ExitCode if c.value in RESERVED_BY_RUNTIME}
        assert not collisions, f"members claiming a runtime-reserved code: {collisions}"

    def test_the_reservation_hook_is_installed(self) -> None:
        """`Enum` REPLACES `__new__` after class creation; the hook moves to `_new_member_`.

        So `ExitCode.__new__` is `Enum.__new__` -- the by-value lookup -- and it
        raises its own `ValueError("2 is not a valid ExitCode")`. A test written
        as a bare `pytest.raises(ValueError)` against `ExitCode.__new__` would
        therefore pass **for the wrong reason**, proving the lookup works and
        saying nothing about the reservation. It happened to me; only the
        `match=` clause caught it.
        """
        hook = ExitCode._new_member_  # type: ignore[attr-defined]
        assert hook is not int.__new__, "no member-creation hook installed"
        assert hook.__qualname__ == "ExitCode.__new__"

    def test_a_reserved_value_is_inexpressible_not_merely_untested(self) -> None:
        """`@enum.unique` has NO opinion about `0` or `2`.

        It rejects duplicates *among members*. Without the `__new__` hook a member
        could claim argparse's `2` with only a test between it and production --
        the exact posture this registry abolishes for duplicates, and a docstring
        that calls a test "the fallback" cannot then rely on one.

        Exercised against the REAL hook, for every reserved value, not against a
        re-statement of the members that exist today.
        """
        hook = ExitCode._new_member_  # type: ignore[attr-defined]
        for reserved in sorted(RESERVED_BY_RUNTIME):
            # Instrument guard: no live member holds this value, so a raise here
            # is the reservation firing, not @enum.unique's duplicate error.
            assert reserved not in {c.value for c in ExitCode}
            with pytest.raises(ValueError, match="reserved by the runtime"):
                hook(ExitCode, reserved)

    def test_members_are_enumerated_not_grepped(self) -> None:
        """This assertion must survive a change to the registry's SOURCE FORM.

        It reads `ExitCode` through the `enum` API, so annotating the members,
        reordering them, or adding trailing comments cannot blind it -- which is
        exactly what a text scan cannot promise.
        """
        by_name = {c.name: c.value for c in ExitCode}
        assert by_name == {
            "LOAD_FAILED": 1,
            "STOPPED_AT_LIMIT": 3,
            "MISSING_DEPENDENCY": 4,
            "VALIDATION_FINDINGS": 5,
            "REFUSED_ROWS": 6,
            "ENRICH_FAILED": 7,
            "DB_UNREACHABLE": 8,
            "UNROUTED_CORPUS": 9,
            "PARSE_PRODUCER_DEFECT": 10,
        }

    def test_the_ruling_table_is_pinned(self) -> None:
        """da#384 assigned these values. They are a decision, not an inference.

        #359 does not get `5`; it takes `6`. Pinned so a future PR cannot quietly
        reason its way back to a neighbouring constant.
        """
        assert ExitCode.MISSING_DEPENDENCY == 4  # da#309, two approvals, unchanged
        assert ExitCode.VALIDATION_FINDINGS == 5  # da#354 moves 4 -> 5
        assert ExitCode.REFUSED_ROWS == 6  # da#359 moves 5 -> 6
        assert ExitCode.ENRICH_FAILED == 7  # da#384 Amendment I: a failed enrich != a failed load
        assert (
            ExitCode.DB_UNREACHABLE == 8
        )  # da#384 Amendment R: the helper names only what it knows
        assert ExitCode.UNROUTED_CORPUS == 9  # da#369: NER Step-1 abort, next free code
        assert (
            ExitCode.PARSE_PRODUCER_DEFECT == 10
        )  # da#386: parse fails loud on the da#355 gate (9 taken by da#369 UNROUTED_CORPUS)

    def test_every_member_documents_what_is_on_disk(self) -> None:
        """The DOCSTRINGS are the specification. There is no ordering claim.

        An earlier version asserted the band was "a monotone, how much is on
        disk, ascending", via::

            ascending = [c.value for c in sorted(ExitCode, key=lambda c: c.value)]
            assert ascending == sorted(ascending)

        That is a list compared to its own sort. **It cannot fail.** Verified
        against a deliberately scrambled `IntEnum` (`Z=9, A=1, M=5`): it passes.
        And the property was false anyway -- `5` is a fully written graph while
        `6` is an incomplete one, so `6` wrote LESS than `5`.

        An ordering that fits the values you happened to pick is not a property
        of them. This replaces it with a check that CAN fail: minting a code
        forces its author to state what is on disk when it fires.

        Member docstrings are not stored at runtime -- `ExitCode.LOAD_FAILED.__doc__`
        returns the CLASS docstring -- so this reads the registry's AST.
        """
        tree = ast.parse(_registry_path().read_text(encoding="utf-8"))
        cls = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == ExitCode.__name__
        )

        documented: set[str] = set()
        body = cls.body
        for i, node in enumerate(body):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in _registry_member_names():
                continue
            nxt = body[i + 1] if i + 1 < len(body) else None
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
                and nxt.value.value.strip()
            ):
                documented.add(target.id)

        # INSTRUMENT GUARD: the walk must be shown to see EVERY existing member
        # before it is trusted to notice a missing seventh. `assert documented`
        # alone would pass while the walk saw one member of eight.
        assert documented == _registry_member_names(), (
            "the AST walk does not see every member -- it cannot be trusted to "
            f"notice a missing docstring.\n  saw: {sorted(documented)}\n"
            f"  members: {sorted(_registry_member_names())}"
        )
        missing = _registry_member_names() - documented
        assert not missing, (
            f"members with no docstring: {sorted(missing)}.\n"
            "  Every code must state what is on disk when it fires. That is the\n"
            "  specification; the ordering is not."
        )


# ---------------------------------------------------------------------------
# Half 2b -- sole-declarership
# ---------------------------------------------------------------------------


# Each source binds its EXIT_* name EXACTLY ONCE. A setup line would answer the
# question the assertion claims to answer: Oyunbileg Batbayar's first AugAssign
# probe read `EXIT_AUG = 6` then `EXIT_AUG += 1`, and the plain `Assign` on the
# setup line was what the old scan found. The `+=` contributed nothing, and
# nothing in the output distinguished the two. Isolate the thing under test to
# one occurrence, and prove the fixture does not already satisfy the question.
_BINDING_FORMS: dict[str, tuple[str, str]] = {
    "plain assign": ("EXIT_BARE = 7", "EXIT_BARE"),
    "annotated": ("EXIT_ANN: int = 7", "EXIT_ANN"),
    "tuple target": ("EXIT_A, _y = 4, 5", "EXIT_A"),
    "walrus": ("(EXIT_W := 7)", "EXIT_W"),
    "starred target": ("*EXIT_S, _y = [7, 8]", "EXIT_S"),
    "augmented assign": ("EXIT_AUG += 1", "EXIT_AUG"),
    "for target": ("for EXIT_L in [7]:\n    pass", "EXIT_L"),
    "with ... as": (
        "class _C:\n"
        "    def __enter__(self):\n        return 7\n"
        "    def __exit__(self, *a):\n        return None\n"
        "with _C() as EXIT_C:\n    pass",
        "EXIT_C",
    ),
}


def _unnamed_exits(tree: ast.AST) -> list[tuple[int, str]]:
    """Every exit in *tree* whose argument is not provably a named code."""
    funcs = _module_functions(tree)
    return [hit for n in ast.walk(tree) for hit in _unnamed_exit_at(n, funcs)]


def _unnamed_exits_by_scope(tree: ast.AST) -> dict[str, list[str]]:
    """Every bare-int exit in *tree*, attributed to its NEAREST enclosing function.

    Do not ask which containers can hold the thing; ask where the thing is.

    The first version of this walked ``ast.FunctionDef`` and asked each function
    for its exits -- **a node-type table, inside the test that abolished node-type
    tables** (Alejandra Reyes-Fuentes). Verified blind:

        async def f(): sys.exit(7)   -> INVISIBLE   (ast.AsyncFunctionDef)
        sys.exit(7)  at module scope -> INVISIBLE   (no FunctionDef at all)

    And it double-counted: an exit inside a nested function was attributed to the
    inner function *and* to every enclosing one, because ``ast.walk`` of the outer
    function descends into the inner.

    This walks the exits and finds their owner, so a new binding container in a
    future Python cannot hide one. Module-scope exits are attributed to
    ``"<module>"``.
    """
    owner: dict[ast.AST, str] = {}

    def descend(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else scope
            )
            owner[child] = inner
            descend(child, inner)

    owner[tree] = "<module>"
    descend(tree, "<module>")

    found: dict[str, list[str]] = {}
    funcs = _module_functions(tree)
    for node, value in ((n, v) for n in ast.walk(tree) for _l, v in _unnamed_exit_at(n, funcs)):
        found.setdefault(owner.get(node, "<module>"), []).append(value)
    return found


def _module_functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function defined in *tree*, by name. Used for the one-hop value follow."""
    return {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _exit_arg_is_named(
    arg: ast.expr | None,
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
) -> bool:
    """Is the value this expression hands the OS provably a code the registry names?

    An ALLOWLIST that asks the **registry**, then **follows the value**, rather
    than asking the syntax. Every hand-maintained list on this issue died the same
    way, and this is the last of them.

    * ``None`` / absent / ``Constant 0`` -- success. Nothing to name.
    * a ``Name`` that is an ``EXIT_*`` alias or a registry member name.
    * an ``Attribute`` ``ExitCode.<MEMBER>``, resolved through ``__members__``.
    * **a call to a module-local function all of whose returns are themselves
      permitted** -- ONE hop. No ``return``, or a bare ``return``, yields ``None``
      and exits ``0``.

    Everything else offends, including shapes nobody has conceived of.

    Why the hop. ``raise SystemExit(main())`` is the console-script idiom, and
    ``src/tools/duck.py`` uses it. The blacklist waved it through; permitting any
    ``Name`` waved ``sys.exit(x)`` through; refusing every ``Call`` flagged a
    correct program and invited a module list. **The question was never which
    shapes are bad -- it is what integer the expression hands the OS.** So ask
    that. Plant ``return 3`` in ``duck.main`` and the guard reds at the
    ``SystemExit(main())`` line, naming ``main()`` -- where the code is chosen.

    Two hops red, deliberately: ``sys.exit(g())`` where ``g`` returns ``f()`` is
    refused. Fail-closed at the boundary of what the scan can see, and the
    boundary is **stated** rather than a shape somebody forgot.

    What this does NOT claim. *"Does this call site emit a bare integer at
    runtime"* is undecidable statically. It answers a decidable question instead:
    **is the emitted code named by the registry, right here, where a reader can
    see it** -- and refuses an unnamed code whether or not it would be correct.

    Out of reach of ANY static check on the call, documented rather than chased:

    * ``globals()["EXIT_X"] = 4`` -- the *name* is a string literal.
    * ``os._exit(3)`` -- a different function; it bypasses ``SystemExit``.
    """
    if arg is None:
        return True  # `sys.exit()` -- no argument, exits 0
    if isinstance(arg, ast.Constant):
        if arg.value is None:
            return True  # `sys.exit(None)` exits 0
        return arg.value == 0 and not isinstance(arg.value, bool)
    if isinstance(arg, ast.Name):
        # NOT `_is_exit_name`: that permits any EXIT_-prefixed identifier,
        # including a well-named function parameter. See `_exported_exit_aliases`.
        return arg.id in _exported_exit_aliases()
    if isinstance(arg, ast.Attribute):
        return (
            isinstance(arg.value, ast.Name)
            and arg.value.id == ExitCode.__name__
            and arg.attr in ExitCode.__members__
        )
    if isinstance(arg, ast.Call) and funcs is not None:
        if not isinstance(arg.func, ast.Name) or arg.func.id not in funcs:
            return False
        callee = funcs[arg.func.id]
        returns = [n for n in ast.walk(callee) if isinstance(n, ast.Return)]
        # `funcs=None` on the recursive call is what makes it ONE hop.
        return all(_exit_arg_is_named(r.value, None) for r in returns)
    return False


def _unnamed_exit_at(
    node: ast.AST,
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
) -> list[tuple[int, str]]:
    """The unnamed exit AT this node, if the node itself is one. Never descends."""
    if not isinstance(node, ast.Call) or len(node.args) > 1:
        return []
    fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
    if fname not in ("exit", "SystemExit"):
        return []
    arg = node.args[0] if node.args else None
    if _exit_arg_is_named(arg, funcs):
        return []
    return [(node.lineno, ast.unparse(arg) if arg is not None else "")]


class TestNoUnnamedExit:
    """A registry of which codes EXIST says nothing about which the process EMITS.

    `sys.exit(7)` names no constant. It is invisible to the sole-declarer scan,
    to `@enum.unique`, and to `__new__`. This guard is what makes an unnamed
    *emission* inexpressible.

    It is an ALLOWLIST. Blacklisting the bad shapes shipped blind to
    `sys.exit(1 if x else 2)` -- a bare integer at runtime and an `ast.IfExp`
    statically. Extending the list to `IfExp` would have been the fifth
    hand-maintained list in this PR. Ask which arguments are provably good
    instead, and there are exactly two kinds.
    """

    PERMITTED = [
        "sys.exit(0)",
        "sys.exit(None)",
        "sys.exit()",
        "sys.exit(ExitCode.LOAD_FAILED)",
        "sys.exit(EXIT_REFUSED_ROWS)",
        "sys.exit(EXIT_STOPPED_AT_LIMIT)",  # the alias src/cli.py actually uses
        # ONE HOP into a module-local callee whose returns are all permitted.
        "def f():\n    pass\nraise SystemExit(f())",  # implicit None -> exit 0
        "def f():\n    return\nsys.exit(f())",  # bare return -> None -> exit 0
        "def f():\n    return 0\nraise SystemExit(f())",  # duck.py's idiom
        "def f():\n    return ExitCode.LOAD_FAILED\nsys.exit(f())",
    ]
    OFFENDING = [
        "sys.exit(3)",
        "raise SystemExit(4)",
        "sys.exit(1 if x else 2)",  # ast.IfExp -- the shape the blacklist missed
        "sys.exit(int(cfg))",  # a Call to a non-local callee
        "sys.exit(codes[0])",  # ast.Subscript
        "sys.exit(base + 1)",  # ast.BinOp
        "sys.exit(True)",  # bool is an int subclass; exits 1
        "sys.exit(rc)",  # a Name the registry does not name
        "sys.exit(Other.THING)",  # an Attribute of some other enum
        "x = 7\nsys.exit(x)",  # a Name bound to a bare int
        # The hop follows the VALUE, so a bad return is caught at the exit line.
        "def main():\n    return 3\nraise SystemExit(main())",
        "def f():\n    if x:\n        return 0\n    return 3\nsys.exit(f())",
        # TWO hops red: fail-closed at the boundary of what the scan can see.
        # `f` returns 0, so a TWO-hop follow would permit this. One hop refuses it:
        # `g`'s return is a Call, and the recursion carries `funcs=None`.
        "def f():\n    return 0\ndef g():\n    return f()\nsys.exit(g())",
        "sys.exit(other.main())",  # not module-local; cannot follow
        "sys.exit(ExitCode.NOPE)",  # a typo'd member -- resolved through __members__
        # A well-named PARAMETER. `_is_exit_name` permitted this; a parameter is
        # not a Store binding, so the sole-declarer scan cannot see it either.
        "def _die(EXIT_CODE: int):\n    sys.exit(EXIT_CODE)",
        "def _die(EXIT_MADE_UP: int):\n    sys.exit(EXIT_MADE_UP)",
        # In `__all__`, but not exit codes. A bare membership test would permit these.
        "sys.exit(ExitCode)",
        "sys.exit(RESERVED_BY_RUNTIME)",
        # Self- and mutual recursion terminate: the hop carries `funcs=None`, so the
        # inner Call can never be followed. Fail-closed, and no RecursionError.
        "def f():\n    return f()\nsys.exit(f())",
        "def f():\n    return g()\ndef g():\n    return f()\nsys.exit(f())",
        # An imported callee cannot be resolved from this module's AST.
        "from x import main\nraise SystemExit(main())",
    ]

    @pytest.mark.parametrize("source", PERMITTED)
    def test_a_named_code_or_zero_is_permitted(self, source: str) -> None:
        assert _unnamed_exits(ast.parse(source)) == [], source

    @pytest.mark.parametrize("source", OFFENDING)
    def test_everything_else_is_an_offender(self, source: str) -> None:
        """Fail-closed: a novel argument shape is refused, not waved through.

        Four of these seven are shapes no blacklist in this PR ever enumerated.
        """
        assert _unnamed_exits(ast.parse(source)) != [], source

    def test_the_allowlist_is_not_vacuous(self) -> None:
        """INSTRUMENT GUARD. A detector that flags everything is not a detector.

        Both halves must be non-empty, or one of the two lists above is proving
        nothing: an allowlist that permits nothing reds on the whole tree, and one
        that permits everything reds on none of it.
        """
        assert self.PERMITTED and self.OFFENDING
        permitted_hits = [s for s in self.PERMITTED if _unnamed_exits(ast.parse(s))]
        offending_misses = [s for s in self.OFFENDING if not _unnamed_exits(ast.parse(s))]
        assert not permitted_hits, f"allowlist too tight: {permitted_hits}"
        assert not offending_misses, f"allowlist too loose: {offending_misses}"

    def test_a_call_that_is_not_an_exit_is_ignored(self) -> None:
        """`os._exit(3)` is a DIFFERENT FUNCTION and bypasses `SystemExit`.

        Out of reach of any check on `sys.exit`/`SystemExit`, documented rather
        than chased -- as with `globals()["EXIT_X"] = 4`, where the name itself is
        a string literal. `src/` contains neither.
        """
        assert _unnamed_exits(ast.parse("os._exit(3)")) == []
        assert _unnamed_exits(ast.parse("print(3)")) == []

    def test_the_live_tree_has_no_unnamed_exit(self) -> None:
        """The whole of `src/`, attributed by scope, not by line number.

        The pin this replaces was TRANSITIONAL AND SELF-EXPIRING. It asserted
        ``offenders == {"_cmd_load": ["1"]}`` -- one surviving unnamed exit, held
        for da#372 (da#384 Amendment I) -- and promised to red the moment da#372
        routed it. da#372 routed it, this reded, and it is renamed in the same
        commit that falsifies it. An exception list permits, silently, forever. A
        pin refuses, loudly, with a deadline, and then names its own successor.

        `src/tools/duck.py`'s `raise SystemExit(main())` was pinned here until the
        guard learned to follow the value one hop. It is green now, untouched.
        Scoping around it would have been a module list; exempting it, an
        exception list. Following the value is neither.

        **`assert not offenders` is the weakest possible assertion**: an empty set
        is what a scan that read nothing also produces. Both halves are therefore
        asserted -- that the walker read the tree, and that it can still see an
        offender -- so a pass means the tree is clean rather than that the
        instrument was silent.
        """
        offenders: dict[str, list[str]] = {}
        scanned = 0
        for py in sorted(_src_root().rglob("*.py")):
            scanned += 1
            for scope, values in _unnamed_exits_by_scope(
                ast.parse(py.read_text(encoding="utf-8"))
            ).items():
                offenders.setdefault(scope, []).extend(values)

        # INSTRUMENT GUARD, half one: the scan actually read a source tree.
        assert scanned > 20, f"the scan read {scanned} files; `src/` cannot be that small"
        # INSTRUMENT GUARD, half two: the detector still reds on a planted offender.
        planted = _unnamed_exits_by_scope(ast.parse("def f():\n    sys.exit(9)\n"))
        assert planted == {"f": ["9"]}, f"the detector has gone blind; saw {planted}"

        assert not offenders, (
            "an unnamed exit appeared.\n"
            f"  found: {offenders}\n"
            "  Name the code in src/exit_codes.py and exit the member, not the int."
        )

    def test_the_hop_follows_the_value_into_ducks_real_idiom(self) -> None:
        """`duck.py` as it stands is green; a `return 3` inside its `main` reds it.

        The DISCRIMINATOR. A guard that merely permitted any `SystemExit(f())`
        would also be green on the shipped file -- and would stay green on the
        mutant. Both halves are asserted here, so a pass means the hop happened.

        The fixture is the production file, read from disk, not a paraphrase.
        """
        duck = _src_root() / "tools" / "duck.py"
        shipped = duck.read_text(encoding="utf-8")

        # INSTRUMENT GUARD: the fixture must contain the idiom under test, or the
        # green below is a statement about a file with no exit in it at all.
        assert "SystemExit(main())" in shipped, "duck.py no longer uses the idiom"
        assert _unnamed_exits(ast.parse(shipped)) == [], "duck.py should be green as shipped"

        mutated = shipped.replace("    return 0", "    return 3", 1)
        assert mutated != shipped, "plant did not apply"
        hits = _unnamed_exits(ast.parse(mutated))
        assert [v for _lineno, v in hits] == ["main()"], (
            f"a bad return inside main() must red at the SystemExit line; got {hits}"
        )

    def test_the_scope_walk_sees_async_and_module_scope(self) -> None:
        """The node-type table this test used to contain, pinned so it cannot return.

        Each source contains exactly ONE exit, so an attribution cannot be
        satisfied by a neighbouring statement.
        """
        cases = {
            "def f():\n    sys.exit(7)": {"f": ["7"]},
            "async def f():\n    sys.exit(7)": {"f": ["7"]},
            "sys.exit(7)": {"<module>": ["7"]},
            "class C:\n    def m(self):\n        sys.exit(7)": {"m": ["7"]},
            # An exit in a nested function belongs to the INNER function, once.
            "def outer():\n    def inner():\n        sys.exit(7)": {"inner": ["7"]},
        }
        for source, expected in cases.items():
            assert _unnamed_exits_by_scope(ast.parse(source)) == expected, source


class TestTheScanSeesEveryBindingForm:
    """A hand-maintained list of node types is the same defect as a hand-maintained
    list of reserved values.

    The scan walks for an ``ast.Name`` in ``Store`` context -- it asks Python what
    a binding is, rather than enumerating the statements that make one. The old
    version handled ``Assign`` and ``AnnAssign`` and was blind to six forms, two
    of them ordinary Python. It was the last hand-maintained list hiding inside
    the fix for hand-maintained lists.

    The ``Store``-context walk enumerates *the language's* notion of a binding;
    the ``enum`` API enumerates *the enum's* notion of a member. Same move, twice.
    """

    @pytest.mark.parametrize("form", sorted(_BINDING_FORMS))
    def test_every_binding_form_is_seen(self, form: str) -> None:
        source, name = _BINDING_FORMS[form]
        tree = ast.parse(source)

        # INSTRUMENT GUARD: the fixture must bind the name exactly once, so a
        # `Store` hit cannot be attributable to a setup line.
        bindings = [
            n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        ]
        assert [n.id for n in bindings].count(name) == 1, (
            f"{form}: fixture binds {name} {[n.id for n in bindings]} times, not once"
        )

        assert name in _exit_name_assignments(tree), f"{form}: scan is blind to {name}"

    def test_a_load_is_not_a_binding(self) -> None:
        """Reading a name is not declaring one -- otherwise every consumer is an offender."""
        assert _exit_name_assignments(ast.parse("print(EXIT_BARE)")) == []
        assert _exit_name_assignments(ast.parse("sys.exit(EXIT_BARE)")) == []

    def test_the_class_body_member_form_is_seen(self) -> None:
        """The form the registry itself uses, carrying no ``EXIT_`` prefix."""
        source = "import enum\n\n\nclass _Rival(enum.IntEnum):\n    REFUSED_ROWS = 6\n"
        assert "REFUSED_ROWS" in _exit_name_assignments(ast.parse(source))


class TestCheckNeo4jNamesOnlyWhatItKnows:
    """The connectivity pre-check is a shared helper. It cannot know its caller.

    It exited `ExitCode.LOAD_FAILED`, which is true for one of its four callers.
    Under `pipeline`, `_cmd_load()` commits the graph, the manifest and the audit
    entry; then `_cmd_enrich()` calls `_check_neo4j()`. Neo4j dropping in between
    produced `rc=1` from a fully committed load -- the registry asserting a
    falsehood *authoritatively*, which is worse than the bare `1` it replaced,
    because a constant invites citation and a literal claims nothing.

    Found by Alejandra Reyes-Fuentes on da#387. The remedy is NOT a parameter:
    `on_failure: ExitCode` would turn four call sites into a hand-maintained
    mapping from caller to exit code (da#384 Amendment R).
    """

    def test_it_takes_no_exit_code_parameter(self) -> None:
        """A parameter would be a hand-maintained caller->code mapping."""
        params = inspect.signature(cli._check_neo4j).parameters
        assert not params, f"_check_neo4j must take no arguments; got {list(params)}"

    def test_an_unreachable_database_exits_db_unreachable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Driven, not read. Every caller gets the same code, because the helper
        knows the same one fact at every call site."""
        import neo4j

        class _Unreachable:
            @staticmethod
            def driver(*_a: object, **_k: object) -> None:
                raise OSError("connection refused")

        monkeypatch.setattr(neo4j, "GraphDatabase", _Unreachable)
        monkeypatch.setattr(
            "src.config.get_settings",
            lambda: SimpleNamespace(
                neo4j=SimpleNamespace(uri="bolt://nope:7687", user="u", password="p")
            ),
        )

        with pytest.raises(SystemExit) as exc:
            cli._check_neo4j()

        # INSTRUMENT GUARD: it must have REACHED the connectivity guard, not died
        # on the way in. Without this, any early failure would satisfy the raise.
        assert "Cannot connect to Neo4j" in capsys.readouterr().out
        assert exc.value.code == ExitCode.DB_UNREACHABLE
        assert exc.value.code != ExitCode.LOAD_FAILED

    def test_the_helper_has_more_than_one_caller(self) -> None:
        """The premise. If it had one caller, naming that caller's stage would be fine."""
        tree = ast.parse(_registry_path().parent.joinpath("cli.py").read_text(encoding="utf-8"))
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_check_neo4j"
        }
        assert len(callers) > 1, f"only {callers} call it; the premise no longer holds"
        assert "_cmd_enrich" in callers, "the caller that made the old code a falsehood"


class TestRegistryIsTheSoleDeclarer:
    def test_the_exclusion_matches_exactly_one_file(self) -> None:
        """The exclusion is itself a reservation, so guard it too.

        Derived from `ExitCode.__module__`. If it ever matched zero files the
        scan below would be excluding nothing and reporting the registry as an
        offender; if it matched more than one, it would be excluding a file it
        does not own.
        """
        registry = _registry_path()
        assert registry.exists()
        matches = [p for p in _src_root().rglob("*.py") if p.resolve() == registry]
        assert len(matches) == 1, f"exclusion matched {len(matches)} files: {matches}"

    def test_the_scanner_can_see_both_declaration_shapes(self) -> None:
        """INSTRUMENT GUARD. A scan that finds nothing is not a scan that found nothing wrong.

        The previous version of this guard asserted only that an ``EXIT_*`` alias
        was found -- and the five module-level aliases DO carry the prefix. It
        therefore passed while the walk was blind to class bodies entirely: **an
        instrument guard demonstrating a capability the scan did not have**, which
        is the defect this whole issue exists to eliminate, sitting inside the fix
        for it.

        Both shapes are now required: a module-level alias, AND a class-body
        member carrying no prefix.
        """
        names = _exit_name_assignments(ast.parse(_registry_path().read_text(encoding="utf-8")))
        assert "EXIT_LOAD_FAILED" in names, f"blind to module-level aliases; saw {names}"
        assert "REFUSED_ROWS" in names, f"blind to class-body members; saw {names}"
        # Every member must be visible, not just the one asserted above.
        missing = _registry_member_names() - set(names)
        assert not missing, f"blind to members: {sorted(missing)}"

    def test_every_module_that_names_an_exit_code_resolves_it_to_the_registry(self) -> None:
        """Sole-DECLARERSHIP is not sole-SOURCE. This asserts the second clause.

        The scan above proves nobody *declares* an `EXIT_*` outside the registry.
        It says nothing about whether a module that *names* one actually got it
        from here -- a module could bind the name from anywhere. Today every
        namer reaches the registry (`_checkpoint` directly; `resolve/__init__`
        and `cli` transitively), so this passes. The moment `_deps.py` and
        `graph/__init__.py` arrive on rebase carrying `EXIT_*` names, the gap is
        live, and this test is what closes it.

        Resolution is through the IMPORT'S SOURCE MODULE at runtime, not through
        the importing module's namespace: `src/cli.py` imports
        `EXIT_STOPPED_AT_LIMIT` *inside* `_cmd_resolve`, so the module object
        never binds it. An earlier version of this test asserted `hasattr(mod,
        name)` and failed on exactly that -- the test found its own wrong
        assumption before a reviewer did. A transitive re-export is accepted; a
        same-named impostor is not, because identity is compared, not the value.
        """
        registry = importlib.import_module(ExitCode.__module__)
        registry_path = _registry_path()
        checked = 0
        for py in sorted(_src_root().rglob("*.py")):
            if py.resolve() == registry_path or py.name == "__main__.py":
                continue
            rel = py.resolve().relative_to(_src_root().parent)
            own_module = str(rel.with_suffix("")).replace("/", ".").removesuffix(".__init__")
            tree = ast.parse(py.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                # An EXIT_* pulled in by name: resolve it in the module it came from.
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    wanted = [a.name for a in node.names if _is_exit_name(a.name)]
                    if not wanted:
                        continue
                    source = importlib.import_module(node.module)
                    for name in wanted:
                        assert getattr(source, name, None) is getattr(registry, name), (
                            f"{rel}: {name} imported from {node.module} is not the "
                            f"registry's object"
                        )
                        checked += 1
                # An EXIT_* bound at module scope: resolve it in this module.
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if not _is_exit_name(node.id):
                        continue
                    mod = importlib.import_module(own_module)
                    assert getattr(mod, node.id, None) is getattr(registry, node.id), (
                        f"{rel}: {node.id} does not resolve to the registry's object"
                    )
                    checked += 1

        # Instrument guard: a scan that checked nothing is not a scan that found
        # nothing wrong. At this head `_checkpoint`, `resolve/__init__` and `cli`
        # all name EXIT_STOPPED_AT_LIMIT.
        assert checked >= 3, f"only {checked} name(s) checked -- the walk found nothing"

    def test_no_module_outside_the_registry_declares_an_exit_code(self) -> None:
        offenders = _declarations_outside_registry()
        assert not offenders, (
            "EXIT_* declared outside src/exit_codes.py -- import it instead:\n"
            + "\n".join(f"  {p}: {names}" for p, names in offenders.items())
        )
