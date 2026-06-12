#!/usr/bin/env python3
"""Static guard for ``generate_source_id`` call-site corpus arguments.

``src.parse.base.generate_source_id(corpus, collection, *parts)`` fails fast
(raises ``ValueError``) when *corpus* is not a known :class:`SourceCorpus`
value. That check only fires **at runtime**, when the real acquisition dir is
present — so a caller that passes a non-corpus first arg (e.g. a free-text bio
``source`` label like ``"kaggle_narrators"``) aborts the whole parse and is
masked in unit tests that build no raw dir. That was the da#89 sanadset crash.

This guard catches the same class of bug **at CI time**: it AST-scans every
``generate_source_id`` call-site and asserts the first positional argument
statically resolves to a :class:`SourceCorpus` value. Free-text provenance keys
(bio ``source`` values, etc.) must build their ids via the direct
``ID_DELIMITER.join(...)`` path instead — see ``sanadset.py`` ``bio_id``.

Run standalone (defaults to scanning ``src/parse``)::

    python scripts/check_source_id_corpus.py
    python scripts/check_source_id_corpus.py src/parse/sanadset.py

Exit code is 1 if any violation is found, 0 otherwise. The same scan functions
are exercised by ``tests/test_parse/test_source_id_corpus_guard.py``, which runs
in CI's existing pytest job (so no new CI job / pre-commit ⇄ CI sync entry is
required — the gate rides the ``Test`` workflow already mirrored in
``.pre-commit-config.yaml``).

Access forms covered (per the lint-gate-cover-all-syntactic-forms lesson):

* bare call after ``from src.parse.base import generate_source_id`` —
  ``generate_source_id(...)``
* dotted/attribute call — ``base.generate_source_id(...)`` / any
  ``<module>.generate_source_id(...)``

First-arg forms accepted as valid:

* a string literal whose value is a ``SourceCorpus`` value (``"lk"``)
* a *module-level* string constant that resolves to one (``SOURCE_CORPUS = "fawaz"``)
* an enum member access ``SourceCorpus.LK`` or ``SourceCorpus.LK.value``

Anything else (an unknown literal/constant, an unknown enum member, or a
non-static runtime expression such as a bare parameter) is a violation:
it cannot be statically proven corpus-safe, which is exactly the reintroduction
risk this guard exists to block.

Form deliberately **not** chased: a corpus value rebound through a local
(non-module-level) variable or imported from another module under an alias.
Current call-sites never do this, and resolving arbitrary dataflow is out of
scope for a line-of-defence static gate; such a first arg simply reads as a
non-static expression and is flagged, which is the safe default.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.enums import SourceCorpus  # noqa: E402  (after sys.path bootstrap)

FUNC_NAME = "generate_source_id"
ENUM_NAME = "SourceCorpus"

#: Valid first-arg string values (the ``SourceCorpus`` *values*, e.g. ``"lk"``).
VALID_CORPORA: frozenset[str] = frozenset(c.value for c in SourceCorpus)
#: Valid enum *member* names (e.g. ``LK`` in ``SourceCorpus.LK``).
VALID_MEMBERS: frozenset[str] = frozenset(c.name for c in SourceCorpus)


@dataclass(frozen=True)
class Violation:
    """A ``generate_source_id`` call-site whose corpus arg is not provably valid."""

    file: str
    line: int
    col: int
    arg: str
    reason: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}: corpus={self.arg} — {self.reason}"


def _is_target_call(node: ast.Call) -> bool:
    """True for both ``generate_source_id(...)`` and ``x.generate_source_id(...)``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == FUNC_NAME
    if isinstance(func, ast.Attribute):
        return func.attr == FUNC_NAME
    return False


def _module_str_consts(tree: ast.Module) -> dict[str, str]:
    """Map module-level ``NAME = "literal"`` assignments to their string value."""
    consts: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets: list[ast.expr] = stmt.targets
            value: ast.expr | None = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
            value = stmt.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = value.value
    return consts


def _enum_member_attr(arg: ast.expr) -> ast.Attribute | None:
    """Return the ``SourceCorpus.MEMBER`` attribute node, unwrapping a trailing ``.value``.

    Handles both ``SourceCorpus.LK`` and ``SourceCorpus.LK.value``; returns None
    if *arg* is not an access on the ``SourceCorpus`` name.
    """
    node = arg
    if isinstance(node, ast.Attribute) and node.attr == "value":
        node = node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == ENUM_NAME
    ):
        return node
    return None


def _check_first_arg(arg: ast.expr, consts: dict[str, str]) -> tuple[bool, str, str]:
    """Return ``(ok, rendered_arg, reason)`` for a call's first positional arg."""
    rendered = ast.unparse(arg)

    # 1. String literal — must be a SourceCorpus value.
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        if arg.value in VALID_CORPORA:
            return True, rendered, ""
        return False, rendered, f"string literal {arg.value!r} is not a SourceCorpus value"

    # 2. Module-level string constant (e.g. SOURCE_CORPUS = "fawaz").
    if isinstance(arg, ast.Name):
        if arg.id in consts:
            resolved = consts[arg.id]
            if resolved in VALID_CORPORA:
                return True, f"{arg.id} (= {resolved!r})", ""
            return (
                False,
                f"{arg.id} (= {resolved!r})",
                f"module constant {arg.id} = {resolved!r} is not a SourceCorpus value",
            )
        return (
            False,
            rendered,
            f"name {arg.id!r} does not resolve to a module-level SourceCorpus string constant",
        )

    # 3. Enum member access: SourceCorpus.LK / SourceCorpus.LK.value
    member = _enum_member_attr(arg)
    if member is not None:
        if member.attr in VALID_MEMBERS:
            return True, rendered, ""
        return False, rendered, f"{ENUM_NAME} has no member {member.attr!r}"

    # 4. Anything else — not statically provable as corpus-safe.
    return (
        False,
        rendered,
        "first arg is not a statically-verifiable SourceCorpus literal, "
        "module constant, or enum member",
    )


def scan_source(source: str, filename: str) -> list[Violation]:
    """Scan one module's source text for unsafe ``generate_source_id`` call-sites."""
    tree = ast.parse(source, filename=filename)
    consts = _module_str_consts(tree)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_target_call(node)):
            continue
        if not node.args:
            violations.append(
                Violation(
                    filename,
                    node.lineno,
                    node.col_offset,
                    "<no positional args>",
                    "call has no positional corpus argument to verify",
                )
            )
            continue
        first = node.args[0]
        ok, rendered, reason = _check_first_arg(first, consts)
        if not ok:
            violations.append(Violation(filename, first.lineno, first.col_offset, rendered, reason))
    return violations


def scan_path(path: Path) -> list[Violation]:
    """Scan a single ``.py`` file."""
    return scan_source(path.read_text(encoding="utf-8"), str(path))


def scan_tree(root: Path) -> list[Violation]:
    """Recursively scan every ``.py`` file under *root* (sorted for stable output)."""
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        violations.extend(scan_path(path))
    return violations


def collect(paths: list[Path]) -> list[Violation]:
    """Scan a mix of files and directories."""
    violations: list[Violation] = []
    for path in paths:
        if path.is_dir():
            violations.extend(scan_tree(path))
        else:
            violations.extend(scan_path(path))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/parse"],
        help="files or directories to scan (default: src/parse)",
    )
    args = parser.parse_args(argv)
    paths = [Path(p) for p in (args.paths or ["src/parse"])]

    violations = collect(paths)
    scanned = ", ".join(str(p) for p in paths)
    if violations:
        print("generate_source_id call-site corpus guard: FAIL", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s) under [{scanned}]. "
            "Pass a SourceCorpus value as the first arg, or — for a free-text "
            "provenance key (a bio `source` label, not a hadith corpus) — build "
            "the id via ID_DELIMITER.join(...) directly instead of "
            "generate_source_id().",
            file=sys.stderr,
        )
        return 1
    print(f"generate_source_id call-site corpus guard: OK ([{scanned}] clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
