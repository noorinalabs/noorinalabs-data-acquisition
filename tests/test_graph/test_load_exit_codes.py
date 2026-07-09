"""da#354: ``load`` must not conflate validation findings with load failure.

Before this, ``load`` exited ``1`` both when the load genuinely failed and when
the load succeeded but a post-load validation check reported a finding. An
operator staring at ``rc=1`` could not tell which had happened. That is exactly
how main#723's ``rc=1`` was read as a data defect when the load had in fact
succeeded and the non-zero came from a validation-harness ``CypherSyntaxError``
(da#319) plus a benign orphan tail.

The load result and the validation verdict are independent facts and now have
independent exit codes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli as cli
import src.graph as graph
import src.graph.validate as validate_mod
from src.exit_codes import ExitCode
from src.graph import LoadSummary
from src.graph.validate import ValidationResult
from src.pipeline.manifest import LAST_LOADED_MANIFEST_FILENAME


class _StubClient:
    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _install_load_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    summary: LoadSummary,
) -> None:
    """Neutralise everything in ``_cmd_load`` except the exit-code decision."""
    (tmp_path / "staging").mkdir(exist_ok=True)
    (tmp_path / "curated").mkdir(exist_ok=True)

    monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
    monkeypatch.setattr(
        "src.config.get_settings",
        lambda: SimpleNamespace(
            data_raw_dir=tmp_path / "raw",
            data_staging_dir=tmp_path / "staging",
            data_curated_dir=tmp_path / "curated",
        ),
    )
    monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
    monkeypatch.setattr(graph, "load_all", lambda *a, **k: summary)


def _summary(*, validation: list[ValidationResult], nodes: int = 10, edges: int = 5) -> LoadSummary:
    return LoadSummary(
        node_results=[],
        edge_results=[],
        validation_results=validation,
        total_nodes=nodes,
        total_edges=edges,
        validation_passed=not any(v.is_fatal for v in validation),
    )


def _fatal(name: str = "chain_integrity") -> ValidationResult:
    return ValidationResult(name, passed=False, details="a finding", row_count=3)


def _passing(name: str = "orphan_narrators") -> ValidationResult:
    return ValidationResult(name, passed=True, details="clean", row_count=0)


def _warned(name: str = "chain_integrity") -> ValidationResult:
    return ValidationResult(name, passed=False, details="timed out", row_count=0, warning=True)


class TestExitCodesAreDistinct:
    """da#354's claim, and only that claim.

    A previous version of this class also asserted
    ``EXIT_VALIDATION_FINDINGS not in (0, 1, 2, EXIT_STOPPED_AT_LIMIT)``.
    That hand-written exclusion list stayed green for the entire life of this
    branch's own ``EXIT_VALIDATION_FINDINGS = 4``, which collided with
    ``MISSING_DEPENDENCY`` (da#309) -- because ``4`` was never on the list. A
    reserved set maintained by hand cannot track a registry that grows.

    The collision is now structurally impossible rather than untested:
    ``@enum.unique`` rejects a duplicate value at import, and the sole-declarer
    guard forbids the out-of-registry constant that made a duplicate expressible
    in the first place. Those invariants are asserted once, in
    ``tests/test_exit_codes.py`` (``test_the_registry_has_no_aliases``,
    ``test_no_member_claims_a_runtime_reserved_code``). Restating them here would
    duplicate their coverage without adding a failure mode.
    """

    def test_codes_do_not_collide(self) -> None:
        """The whole point: an operator can tell these apart."""
        assert ExitCode.LOAD_FAILED != ExitCode.VALIDATION_FINDINGS
        assert ExitCode.LOAD_FAILED == 1


class TestLoadExitCode:
    def test_validation_findings_exit_with_findings_code_not_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_fatal()]))

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load()

        assert exc.value.code == ExitCode.VALIDATION_FINDINGS
        out = capsys.readouterr().out
        # The operator must be told the data IS in the graph.
        assert "LOAD SUCCEEDED" in out

    def test_clean_load_exits_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_passing()]))
        cli._cmd_load()  # must not raise SystemExit
        # Positive control for the negative assertion in the load-failure test
        # below: this path *does* write the manifest, so "not written" there is
        # a real observation rather than an inert assertion.
        assert (tmp_path / LAST_LOADED_MANIFEST_FILENAME).exists()

    def test_downgraded_warning_is_not_a_finding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """da#259: a timed-out check is inconclusive, not a finding."""
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_warned()]))
        cli._cmd_load()  # must not raise SystemExit

    def test_genuine_load_failure_is_exit_one_and_writes_no_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A raising loader is a real failure: code 1, distinct from findings."""
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[]))

        def _boom(*a: Any, **k: Any) -> LoadSummary:
            raise RuntimeError("neo4j went away mid-load")

        monkeypatch.setattr(graph, "load_all", _boom)

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load()
        assert exc.value.code == ExitCode.LOAD_FAILED
        # A load that raised did not happen: it must not be recorded as loaded.
        assert not (tmp_path / LAST_LOADED_MANIFEST_FILENAME).exists()


class TestValidateCommandExitCode:
    def test_findings_exit_with_findings_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
        monkeypatch.setattr(validate_mod, "run_validation", lambda *a, **k: [_fatal()])

        with pytest.raises(SystemExit) as exc:
            cli._cmd_validate()
        assert exc.value.code == ExitCode.VALIDATION_FINDINGS
        assert "not a load failure" in capsys.readouterr().out.lower()

    def test_clean_validate_does_not_exit_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
        monkeypatch.setattr(validate_mod, "run_validation", lambda *a, **k: [_passing()])

        # Falls off the end without SystemExit -- rc 0.
        cli._cmd_validate()
