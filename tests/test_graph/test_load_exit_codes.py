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

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli as cli
import src.graph as graph
import src.graph.validate as validate_mod
import src.pipeline.manifest as manifest_mod
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


class TestPostLoadBookkeepingCannotExitOne:
    """The `rc=1 => the load did not happen` binding must not leak (da#354).

    `save_manifest`, `create_audit_entry` and `write_audit_entry` run *after*
    `load_all` returns, so the graph is already committed when they execute.
    `main()` installs no top-level handler, so before this guard an exception in
    any of them reached Python's default handler and exited `1` -- the same code
    as a load that never wrote a node. `rc=1` before the commit is safe to retry;
    `rc=1` after it re-loads a graph that already holds the data, and nothing
    separated the two.
    """

    def _order(self, monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
        """Record load_all's position relative to the bookkeeping calls."""
        summary = _summary(validation=[_passing()])

        def _load_all(*a: Any, **k: Any) -> LoadSummary:
            calls.append("load_all")
            return summary

        monkeypatch.setattr(graph, "load_all", _load_all)

    def test_audit_write_failure_does_not_exit_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_passing()]))

        def _boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system: 'data'")

        monkeypatch.setattr("src.pipeline.audit.write_audit_entry", _boom)

        cli._cmd_load()  # must not raise SystemExit at all

        # The load committed and IS recorded as loaded -- the audit is a side
        # note, not the load.
        assert (tmp_path / LAST_LOADED_MANIFEST_FILENAME).exists()
        err = capsys.readouterr().err
        assert "the load SUCCEEDED" in err

    def test_last_loaded_manifest_failure_does_not_exit_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Raise on the POST-load `save_manifest` only, and prove it was that one.

        `save_manifest` is called twice: once at the top of `_cmd_load` for
        `MANIFEST_FILENAME`, and once after the load for
        `LAST_LOADED_MANIFEST_FILENAME`. Patching it globally raises on the first
        call, *before* `load_all` -- which leaves the graph unwritten and makes an
        `rc=1` entirely correct. Such a probe passes for the wrong reason and says
        nothing about the post-load path. The recorded call order is what makes
        this a measurement: `load_all` must appear before the raise.
        """
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_passing()]))
        calls: list[str] = []
        self._order(monkeypatch, calls)

        real_save = manifest_mod.save_manifest

        def _save(manifest: Any, path: Path) -> Any:
            if path.name == LAST_LOADED_MANIFEST_FILENAME:
                calls.append("save_manifest(LAST_LOADED)")
                raise OSError("read-only file system: 'data'")
            calls.append("save_manifest(MANIFEST)")
            return real_save(manifest, path)

        monkeypatch.setattr("src.pipeline.manifest.save_manifest", _save)

        cli._cmd_load()  # must not raise SystemExit

        assert calls == ["save_manifest(MANIFEST)", "load_all", "save_manifest(LAST_LOADED)"], calls
        # The raise happened after the commit, and the operator is told so.
        err = capsys.readouterr().err
        assert "the load SUCCEEDED" in err
        assert "re-load every input, which is safe" in err

    def test_bookkeeping_failure_still_yields_findings_code_not_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A findings load whose audit write fails is still findings, never `1`."""
        _install_load_stubs(monkeypatch, tmp_path, _summary(validation=[_fatal()]))

        def _boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system: 'data'")

        monkeypatch.setattr("src.pipeline.audit.write_audit_entry", _boom)

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load()
        assert exc.value.code == ExitCode.VALIDATION_FINDINGS
        assert exc.value.code != ExitCode.LOAD_FAILED

    def test_enrich_leaks_rc_one_after_the_load_has_committed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Why the invariant still names `load` and not `pipeline` -- for a NEW reason.

        This test was named ``test_enrich_exits_bare_one_after_the_load_has_committed``
        and asserted that ``_cmd_enrich`` exits a bare ``1`` when a step fails. Its
        docstring promised: *"If enrich grows an exit code of its own, this test reds
        and that comment must change with it."* da#384 Amendment I gave enrich
        ``ExitCode.ENRICH_FAILED`` (7). It reded. Renamed here, in the commit that
        falsified it, together with the comment it guards.

        The leak survives its own cause. ``_cmd_enrich`` writes its audit entry
        OUTSIDE any handler, and neither ``main()`` nor ``_cmd_pipeline`` has a
        top-level one -- so an escaping exception reaches Python's default handler
        and exits ``1``, long after ``_cmd_load`` committed the graph and recorded
        its manifest. Under ``isnad-ingest pipeline`` an rc of ``1`` therefore still
        does NOT mean the load did not happen.

        That is the same unguarded-bookkeeping defect this PR fixes in ``_cmd_load``,
        sitting in ``_cmd_enrich``. Fixing it is not da#372's (this PR owns
        ``_cmd_load``); it is filed as da#394. Scoping the claim is what this PR
        can honestly do, and the comment in ``_cmd_load`` now names the mechanism
        (no top-level handler) rather than an exit code someone can renumber.
        """
        (tmp_path / "staging").mkdir(exist_ok=True)
        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr(
            "src.config.get_settings",
            lambda: SimpleNamespace(
                data_staging_dir=tmp_path / "staging",
                betweenness_sampling_size=None,
                betweenness_sampling_seed=42,
            ),
        )
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
        monkeypatch.setattr(
            "src.enrich.run_all",
            lambda *a, **k: SimpleNamespace(
                steps_completed=["centrality"],
                steps_failed=[],
                metrics=None,
                topics=None,
                historical=None,
            ),
        )

        def _audit_boom(*a: Any, **k: Any) -> None:
            raise OSError("read-only file system: 'data'")

        monkeypatch.setattr("src.pipeline.audit.write_audit_entry", _audit_boom)

        # NOT a SystemExit: the exception escapes _cmd_enrich entirely. Nothing in
        # cli.py catches it, so CPython's default handler prints it and exits 1.
        with pytest.raises(OSError, match="read-only file system"):
            cli._cmd_enrich()

        # POSITIVE CONTROL for the claim above -- that an escaping exception really
        # does yield rc=1, rather than this test merely observing an exception.
        # Asserted from a real process, because `sys.exit(m)` and an uncaught raise
        # reach the OS by different routes and only $? sees both.
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "raise OSError('read-only file system')"],
            capture_output=True,
        )
        assert proc.returncode == 1, f"CPython's default handler exited {proc.returncode}"

        # SCOPE CONTROL: a *failed step* is a named code, not the leak. If this
        # stops being 7, the comment in _cmd_load must be re-read, not re-numbered.
        monkeypatch.setattr("src.pipeline.audit.write_audit_entry", lambda *a, **k: None)
        monkeypatch.setattr(
            "src.enrich.run_all",
            lambda *a, **k: SimpleNamespace(
                steps_completed=["centrality"],
                steps_failed=["topics"],
                metrics=None,
                topics=None,
                historical=None,
            ),
        )
        with pytest.raises(SystemExit) as exc:
            cli._cmd_enrich()
        assert exc.value.code == ExitCode.ENRICH_FAILED
        assert exc.value.code != ExitCode.LOAD_FAILED


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
