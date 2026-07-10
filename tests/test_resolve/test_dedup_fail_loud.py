"""dedup must fail loud when an enabled stage is missing a declared dependency (da#309).

Before da#309 ``run_dedup`` caught ``ImportError`` on sentence-transformers / faiss
and returned a schema-valid, **zero-row** ``parallel_links.parquet``. The pipeline
continued, ``run_all`` logged ``resolve_pipeline_complete``, and the CLI exited 0 —
so a run on a box without the ``ml`` dependency group produced zero ``PARALLEL``
edges (production carries ~4.49M) presented as a genuine negative.

Against ``origin/main`` (2cf5968) the two behavioural tests here fail with:

    Failed: DID NOT RAISE <class 'src.resolve._deps.MissingDependencyError'>
    AssertionError: SILENT SKIP: run_dedup wrote parallel_links.parquet with 0 rows

and ``test_missing_dependency_error_propagates_through_run_all`` guards the other
half of the defect: ``run_all`` wraps every stage in ``except Exception`` and
logs-and-continues, so a guard raising a plain ``Exception`` would be swallowed and
this fix would be inert.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.resolve as resolve_pkg
from src.config import Settings, get_settings
from src.resolve._deps import (
    EXIT_MISSING_DEPENDENCY,
    MissingDependencyError,
    missing_dependencies,
)
from src.resolve.dedup import run_dedup

from .test_dedup import _make_hadith, write_hadiths


def _staging_with_hadiths(tmp_path: Path) -> Path:
    """A staging dir holding two near-duplicate hadiths — a non-empty, valid input."""
    write_hadiths(
        tmp_path / "hadiths_test.parquet",
        [
            _make_hadith(
                "h-1", "Actions are judged by intentions and every person gets his intent"
            ),
            _make_hadith(
                "h-2", "Actions are judged by intentions and every man shall have his intent"
            ),
        ],
    )
    return tmp_path


# ---------------------------------------------------------------------------
# The production default.
#
# `tests/conftest.py` sets DEDUP_REQUIRE_ML=false suite-wide (CI cannot import the
# embedder). Both tests below therefore `delenv` it first — verified load-bearing:
# with the delenv removed, the suite's env var reaches `Settings()` and the
# assertion reads False. Both go red if the default is flipped to False or the
# field is deleted, so neither is inert.
#
# The first pins the flag's *value*; the second pins the *behaviour* the flag is
# supposed to buy, so deleting or flipping the default cannot leave the pipeline
# silently degrading with only a constant-comparison test to notice.
# ---------------------------------------------------------------------------
def test_dedup_requires_ml_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped default is fail-loud, regardless of the suite's opt-out fixture."""
    monkeypatch.delenv("DEDUP_REQUIRE_ML", raising=False)
    get_settings.cache_clear()
    assert Settings(_env_file=None).dedup_require_ml is True


def test_default_settings_make_run_dedup_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: with the suite's opt-out removed and no `require_ml` argument,
    a missing declared dep raises. This is the production call shape — `run_all`
    passes no `require_ml`, so `run_dedup` reads the setting.
    """
    monkeypatch.delenv("DEDUP_REQUIRE_ML", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    staging = _staging_with_hadiths(tmp_path)

    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(MissingDependencyError):
            run_dedup(staging, threshold=0.70)

    assert not (staging / "parallel_links.parquet").exists()


# ---------------------------------------------------------------------------
# The raise itself — the leg that was red against origin/main.
# ---------------------------------------------------------------------------
class TestEnabledStageMissingDeclaredDep:
    def test_run_dedup_raises_when_embedder_missing(self, tmp_path: Path) -> None:
        staging = _staging_with_hadiths(tmp_path)

        # sys.modules[name] = None makes `import name` raise ImportError.
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(MissingDependencyError) as excinfo:
                run_dedup(staging, threshold=0.70, require_ml=True)

        err = excinfo.value
        assert "sentence_transformers" in err.missing
        assert err.stage == "dedup"
        # The operator must be able to act on the message alone.
        assert "dedup" in str(err)
        assert "sentence_transformers" in str(err)
        assert "uv sync --group ml" in str(err)

    def test_run_dedup_raises_when_faiss_missing(self, tmp_path: Path) -> None:
        staging = _staging_with_hadiths(tmp_path)

        with patch.dict(sys.modules, {"faiss": None}):
            with pytest.raises(MissingDependencyError) as excinfo:
                run_dedup(staging, threshold=0.70, require_ml=True)

        assert "faiss" in excinfo.value.missing

    def test_no_empty_output_is_written_when_dep_missing(self, tmp_path: Path) -> None:
        """The silent-skip signature: a zero-row parquet standing in for a negative."""
        staging = _staging_with_hadiths(tmp_path)

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(MissingDependencyError):
                run_dedup(staging, threshold=0.70, require_ml=True)

        output = staging / "parallel_links.parquet"
        if output.exists():
            rows = pq.read_table(output).num_rows
            raise AssertionError(f"SILENT SKIP: run_dedup wrote {output.name} with {rows} rows")


# ---------------------------------------------------------------------------
# The one legitimate skip: an explicit, auditable opt-out.
# ---------------------------------------------------------------------------
class TestExplicitDegradedSkip:
    def test_degraded_skip_writes_empty_output_without_raising(self, tmp_path: Path) -> None:
        staging = _staging_with_hadiths(tmp_path)

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            output = run_dedup(staging, threshold=0.70, require_ml=False)

        assert output.exists()
        assert pq.read_table(output).num_rows == 0

    def test_settings_opt_out_is_honoured(self, tmp_path: Path) -> None:
        """`require_ml=None` defers to DEDUP_REQUIRE_ML (set false by conftest)."""
        staging = _staging_with_hadiths(tmp_path)

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            output = run_dedup(staging, threshold=0.70)

        assert pq.read_table(output).num_rows == 0


# ---------------------------------------------------------------------------
# Anti-inert-guard: run_all must not swallow the error.
# ---------------------------------------------------------------------------
class TestNotSwallowedByOrchestrator:
    def test_missing_dependency_error_is_not_an_exception_subclass(self) -> None:
        """`run_all` wraps each stage in `except Exception`. Subclassing Exception
        here would make the guard inert — the pipeline would log `resolve_step_failed`
        and still report `resolve_pipeline_complete`.
        """
        assert issubclass(MissingDependencyError, BaseException)
        assert not issubclass(MissingDependencyError, Exception)

    def test_missing_dependency_error_propagates_through_run_all(self, tmp_path: Path) -> None:
        raw, staging, out = tmp_path / "raw", tmp_path / "staging", tmp_path / "out"
        for directory in (raw, staging, out):
            directory.mkdir()
        # Satisfy run_all's staging pre-flight.
        pq.write_table(pa.table({"a": pa.array([1])}), staging / "hadiths_test.parquet")

        boom = MissingDependencyError(
            stage="dedup",
            missing=["sentence_transformers"],
            dependency_group="ml",
            remediation="uv sync --group ml",
        )
        with patch("src.resolve.dedup.run", side_effect=boom):
            with pytest.raises(MissingDependencyError):
                resolve_pkg.run_all(raw, staging, out, from_step="dedup")


# ---------------------------------------------------------------------------
# CLI surface: a distinct, non-zero exit status.
# ---------------------------------------------------------------------------
def test_cli_maps_missing_dependency_to_distinct_exit_code() -> None:
    from src.cli import _cmd_resolve

    boom = MissingDependencyError(
        stage="dedup",
        missing=["faiss"],
        dependency_group="ml",
        remediation="uv sync --group ml",
    )
    with patch("src.resolve.run_all", side_effect=boom):
        with pytest.raises(SystemExit) as excinfo:
            _cmd_resolve()

    assert excinfo.value.code == EXIT_MISSING_DEPENDENCY
    assert EXIT_MISSING_DEPENDENCY != 0


# ---------------------------------------------------------------------------
# The probe helper itself.
# ---------------------------------------------------------------------------
def test_missing_dependencies_reports_only_absent_modules() -> None:
    assert missing_dependencies(["sys", "pathlib"]) == []
    assert missing_dependencies(["sys", "definitely_not_a_real_module_xyz"]) == [
        "definitely_not_a_real_module_xyz"
    ]
