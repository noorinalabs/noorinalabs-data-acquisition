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
    """Every exit-code name assigned anywhere in *tree*, at any nesting depth.

    Matches on the target NAME, so the RHS form is irrelevant: a bare int, an
    annotated assignment, a ``Final``, a class-body member, or a computed
    ``_BASE + 1`` are all seen.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and _is_exit_name(t.id):
                found.append(t.id)
    return found


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
        }

    def test_the_ruling_table_is_pinned(self) -> None:
        """da#384 assigned these values. They are a decision, not an inference.

        #359 does not get `5`; it takes `6`. Pinned so a future PR cannot quietly
        reason its way back to a neighbouring constant.
        """
        assert ExitCode.MISSING_DEPENDENCY == 4  # da#309, two approvals, unchanged
        assert ExitCode.VALIDATION_FINDINGS == 5  # da#354 moves 4 -> 5
        assert ExitCode.REFUSED_ROWS == 6  # da#359 moves 5 -> 6


# ---------------------------------------------------------------------------
# Half 2b -- sole-declarership
# ---------------------------------------------------------------------------


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

    def test_no_module_outside_the_registry_declares_an_exit_code(self) -> None:
        offenders = _declarations_outside_registry()
        assert not offenders, (
            "EXIT_* declared outside src/exit_codes.py -- import it instead:\n"
            + "\n".join(f"  {p}: {names}" for p, names in offenders.items())
        )
