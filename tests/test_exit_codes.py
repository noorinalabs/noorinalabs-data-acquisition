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
import enum
import importlib
import sys
from pathlib import Path

import pytest

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
    """The registry's member names, through the ``enum`` API. Never grepped."""
    return frozenset(c.name for c in ExitCode)


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
    def test_registry_has_no_aliases(self) -> None:
        """`IntEnum` accepts a duplicate as a silent ALIAS. `@enum.unique` does not."""
        values = [c.value for c in ExitCode]
        assert len(values) == len(set(values))
        # Instrument guard: an empty enum would satisfy the above vacuously.
        assert len(values) >= 5, f"registry looks empty: {values}"

    def test_the_decorator_is_actually_applied(self) -> None:
        """Without `@enum.unique` the aliasing check above passes for a DIFFERENT reason.

        `@enum.unique` sets no public marker, so prove it by construction: build
        an enum carrying the registry's real members plus a planted duplicate,
        apply the same decorator, and require it to raise.
        """
        members = {c.name: c.value for c in ExitCode}
        victim = ExitCode.MISSING_DEPENDENCY
        # Proof the plant applied: the planted value collides with a real member.
        assert victim.value in members.values()
        members["PLANTED_DUPLICATE"] = victim.value

        with pytest.raises(ValueError, match="duplicate values"):
            enum.unique(enum.IntEnum("_Planted", members))  # type: ignore[arg-type]

    def test_importing_the_registry_is_what_raises(self) -> None:
        """The precise claim: it raises when the registry is IMPORTED.

        Not "at process start" -- `src/cli.py` imports `src.resolve` and
        `src.graph` lazily inside command functions. Pytest collection imports
        this module, which is why CI catches a duplicate either way.
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

    def test_the_band_is_a_monotone_on_how_much_is_on_disk(self) -> None:
        """The codes ascend with what the operator will find written.

        Pinned because the ordering IS the specification: `1` nothing, `4` prior
        stages' artifacts, `5` the full graph, `6` the graph minus refused rows,
        `7` the graph plus a partial enrich. The earlier "nothing ran / something
        ran" framing produced two false rows.
        """
        ascending = [c.value for c in sorted(ExitCode, key=lambda c: c.value)]
        assert ascending == sorted(ascending)
        assert ExitCode.LOAD_FAILED < ExitCode.MISSING_DEPENDENCY < ExitCode.VALIDATION_FINDINGS
        assert ExitCode.VALIDATION_FINDINGS < ExitCode.REFUSED_ROWS < ExitCode.ENRICH_FAILED


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


def _bare_int_exits(tree: ast.AST) -> list[tuple[int, int]]:
    """``(lineno, value)`` for every ``sys.exit(<int literal>)`` where the literal is not ``0``.

    ``@enum.unique`` makes a duplicate *declaration* inexpressible. Nothing makes a
    duplicate *emission* inexpressible: ``sys.exit(7)`` names no constant, so it is
    invisible to any name-keyed scan, and the registry would be the whole truth
    about which codes EXIST while saying nothing about which the process EMITS.

    ``0`` is exempt: it is success, it is not a member (see
    :data:`~src.exit_codes.RESERVED_BY_RUNTIME`), and there is nothing to name.
    ``True``/``False`` are ``int`` subclasses and are not exit codes.
    """
    found: list[tuple[int, int]] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or len(n.args) != 1:
            continue
        fname = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
        if fname not in ("exit", "SystemExit"):
            continue
        arg = n.args[0]
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
            and arg.value != 0
        ):
            found.append((n.lineno, arg.value))
    return found


class TestNoBareIntegerExit:
    """A registry of which codes EXIST says nothing about which the process EMITS.

    `sys.exit(7)` names no constant. It is invisible to the sole-declarer scan,
    to `@enum.unique`, and to `__new__`. This guard is what makes a duplicate
    *emission* inexpressible, and it is what retires a documented bare literal by
    construction rather than by comment.
    """

    def test_a_bare_nonzero_int_exit_is_detected(self) -> None:
        found = _bare_int_exits(ast.parse("import sys\n\nsys.exit(7)"))
        assert found == [(3, 7)]

    def test_exit_zero_is_allowed(self) -> None:
        """`0` is success; it is not a member and there is nothing to name."""
        assert _bare_int_exits(ast.parse("import sys\n\nsys.exit(0)")) == []

    def test_a_registry_member_is_allowed(self) -> None:
        """The whole point: name the code, do not spell its value."""
        assert _bare_int_exits(ast.parse("sys.exit(ExitCode.LOAD_FAILED)")) == []
        assert _bare_int_exits(ast.parse("sys.exit(EXIT_REFUSED_ROWS)")) == []

    def test_raise_systemexit_with_a_literal_is_detected(self) -> None:
        """`raise SystemExit(1)` is the same emission through another door."""
        assert _bare_int_exits(ast.parse("raise SystemExit(1)")) == [(1, 1)]

    def test_a_bool_is_not_an_exit_code(self) -> None:
        """`bool` is an `int` subclass. `sys.exit(True)` is not a code claim."""
        assert _bare_int_exits(ast.parse("sys.exit(True)")) == []

    def test_the_live_tree_has_exactly_one_bare_exit_and_da372_owns_it(self) -> None:
        """The whole of `src/`, and the one offender is named by FUNCTION, not line.

        `@enum.unique` makes a duplicate DECLARATION inexpressible; this makes a
        duplicate EMISSION inexpressible. A new `sys.exit(7)` anywhere reds here
        immediately -- which is the coverage da#384 Amendment J asks for.

        TRANSITIONAL, AND SELF-EXPIRING. One bare literal survives at this head:
        `_cmd_load`'s findings exit. It is da#372's diff (da#384 Amendment I) and
        must not be touched here. So this test pins it rather than exempting it:

        * a NEW bare-int exit anywhere -> RED, because the set grows;
        * da#372 routing `_cmd_load` -> RED, because the set empties.

        The second is the point. This assertion cannot be left behind: when da#372
        lands it fails, and whoever rebases must replace it with `assert not
        offenders`. A guard that quietly tolerates its exception forever is the
        hand-maintained list this registry exists to abolish. This one deletes
        itself.
        """
        offenders: dict[str, list[int]] = {}
        for py in sorted(_src_root().rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            # Attribute each bare exit to its enclosing function, so the pin
            # survives any edit that moves a line number.
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                values = [v for _lineno, v in _bare_int_exits(fn)]
                if values:
                    offenders.setdefault(fn.name, []).extend(values)

        assert offenders == {"_cmd_load": [1]}, (
            "the bare-integer-exit set changed.\n"
            f"  found: {offenders}\n"
            "  If da#372 has landed and _cmd_load now exits VALIDATION_FINDINGS, this\n"
            "  test has done its job: replace it with `assert not offenders`.\n"
            "  If something NEW exits a bare int: name the code in src/exit_codes.py."
        )

    def test_the_detector_is_not_vacuous(self) -> None:
        """INSTRUMENT GUARD. A detector that finds nothing is not a clean tree.

        Each source below binds exactly one exit call, so a hit cannot be
        attributed to a neighbouring statement.
        """
        for src, expected in (("sys.exit(1)", 1), ("sys.exit(6)", 6), ("sys.exit(255)", 255)):
            got = _bare_int_exits(ast.parse(src))
            assert [v for _, v in got] == [expected], f"{src!r} -> {got}"


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
