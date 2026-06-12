"""Tests for the ``generate_source_id`` call-site corpus static guard.

The guard (``scripts/check_source_id_corpus.py``) AST-scans call-sites and fails
if a first ("corpus") argument cannot be statically proven to be a
``SourceCorpus`` value. This catches the da#89-class crash (a free-text bio
``source`` label passed where a corpus is required) at CI time instead of at
runtime parse.

The real-tree test below is the actual regression gate: it asserts the live
``src/parse`` package is clean. The synthetic cases lock in the discrimination
logic across both access forms (bare + dotted) and every accepted first-arg
shape.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD_PATH = REPO_ROOT / "scripts" / "check_source_id_corpus.py"

_spec = importlib.util.spec_from_file_location("check_source_id_corpus", _GUARD_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
# Register before exec: the module's frozen dataclass resolves its annotations
# via sys.modules[__module__] at class-creation time (Python 3.14).
sys.modules[_spec.name] = guard
_spec.loader.exec_module(guard)


def _scan(code: str) -> list:
    return guard.scan_source(code, "<test>")


# --- Real-tree regression gate ------------------------------------------------


def test_live_parse_package_is_clean() -> None:
    """Every current generate_source_id call-site in src/parse passes the guard."""
    violations = guard.scan_tree(REPO_ROOT / "src" / "parse")
    assert violations == [], "\n".join(str(v) for v in violations)


def test_guard_actually_finds_the_call_sites() -> None:
    """Sanity: the live scan is exercising real calls, not silently matching none.

    Guards against a vacuous green (e.g. a refactor that renames the function and
    makes the scan match nothing — which would also report zero violations).
    """
    found = False
    for path in (REPO_ROOT / "src" / "parse").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if f"{guard.FUNC_NAME}(" in source and "def " + guard.FUNC_NAME not in source:
            found = True
            break
    assert found, "expected at least one generate_source_id call-site in src/parse"


# --- Accepted first-arg forms (no violation) ----------------------------------


def test_valid_string_literal() -> None:
    assert _scan('generate_source_id("lk", "bukhari", 1)') == []


def test_valid_module_constant() -> None:
    code = 'SOURCE_CORPUS = "fawaz"\ngenerate_source_id(SOURCE_CORPUS, "coll", 1)'
    assert _scan(code) == []


def test_valid_underscore_module_constant() -> None:
    # sanadset.py uses a leading-underscore constant name.
    code = '_SOURCE_CORPUS = "sanadset"\ngenerate_source_id(_SOURCE_CORPUS, "m", 1)'
    assert _scan(code) == []


def test_valid_enum_member_access() -> None:
    assert _scan('generate_source_id(SourceCorpus.LK, "bukhari")') == []


def test_valid_enum_member_value_access() -> None:
    assert _scan('generate_source_id(SourceCorpus.SUNNAH.value, "muslim")') == []


def test_valid_dotted_access_form() -> None:
    # `base.generate_source_id(...)` — the dotted/attribute call form.
    assert _scan('base.generate_source_id("thaqalayn", "kafi")') == []


# --- Rejected first-arg forms (violation) -------------------------------------


def test_invalid_string_literal_is_flagged() -> None:
    # The exact da#89 shape: a bio source label, not a corpus.
    violations = _scan('generate_source_id("kaggle_narrators", "x", 1)')
    assert len(violations) == 1
    assert "kaggle_narrators" in violations[0].reason


def test_invalid_module_constant_is_flagged() -> None:
    code = 'BIO_SOURCE = "kaggle_narrators"\ngenerate_source_id(BIO_SOURCE, "x")'
    violations = _scan(code)
    assert len(violations) == 1
    assert "BIO_SOURCE" in violations[0].arg


def test_unknown_enum_member_is_flagged() -> None:
    violations = _scan('generate_source_id(SourceCorpus.BOGUS, "x")')
    assert len(violations) == 1
    assert "BOGUS" in violations[0].reason


def test_dotted_access_form_invalid_corpus_is_flagged() -> None:
    # The dotted form must be policed too, not only the bare form.
    violations = _scan('base.generate_source_id("not_a_corpus", "x")')
    assert len(violations) == 1


def test_non_static_runtime_expression_is_flagged() -> None:
    # A runtime expression (function call) cannot be statically proven corpus-safe.
    code = "generate_source_id(pick_corpus(), 'x', 1)"
    violations = _scan(code)
    assert len(violations) == 1
    assert "statically-verifiable" in violations[0].reason


def test_undeclared_name_is_flagged() -> None:
    # A Name with no module-level string-constant assignment.
    violations = _scan('generate_source_id(MYSTERY, "x")')
    assert len(violations) == 1
    assert "does not resolve" in violations[0].reason


def test_local_constant_does_not_count_as_module_constant() -> None:
    # Deliberately not chased: a corpus rebound through a local (function-scope)
    # variable reads as a non-static expression and is flagged (safe default).
    code = "def f():\n    c = 'lk'\n    return generate_source_id(c, 'x')"
    violations = _scan(code)
    assert len(violations) == 1


# --- CLI entry point ----------------------------------------------------------


def test_main_returns_zero_on_clean_tree(capsys) -> None:
    rc = guard.main(["src/parse"])
    capsys.readouterr()
    assert rc == 0


def test_main_returns_one_on_dirty_file(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('generate_source_id("kaggle_narrators", "x")\n', encoding="utf-8")
    rc = guard.main([str(bad)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "FAIL" in err
