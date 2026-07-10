"""da#386: ``src.parse.run_all`` must not swallow the da#355 producer gate.

``generate_source_id`` raises :exc:`DoubledCorpusPrefixError` (a :exc:`ValueError`)
when ``collection == corpus``. ``run_all`` used to catch that in its broad
``except Exception``, log ``parse_failed`` and ``return`` -- so ``parse`` exited
``0`` with the stale staging parquet in place, and da#359's ``_cmd_load``
remediation ("re-run ``parse``") silently changed nothing.

Every test here is red on the pre-fix orchestrator:

* the swallow tests would see ``run_all`` return normally instead of raising;
* the purge test would see the partial parquet survive;
* the CLI test would see ``_cmd_parse`` exit ``0`` instead of
  :attr:`ExitCode.PARSE_PRODUCER_DEFECT`.

The narrowing tests pin the *other* half: a genuinely transient adapter failure is
still logged and skipped, so the fix does not turn every parse hiccup fatal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.cli as cli
import src.config as config_mod
import src.parse as parse_pkg
from src.exit_codes import ExitCode
from src.parse import ParseProducerError, run_all
from src.parse.identity import DoubledCorpusPrefixError

ParseFn = Callable[[Path, Path], object]


@dataclass
class _StubAdapter:
    """Minimal stand-in for a :class:`~src.adapters.SourceAdapter` row.

    ``run_all`` only touches ``slug``, ``active`` and ``parse(raw_dir,
    staging_dir)``, so a registry of these exercises the orchestrator without any
    real source data.
    """

    slug: str
    fn: ParseFn
    active: bool = True

    def parse(self, raw_dir: Path, staging_dir: Path) -> object:
        return self.fn(raw_dir, staging_dir)


def _write_parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), path)
    return path


def _good_parse(hadith_name: str) -> ParseFn:
    def _fn(_raw: Path, staging: Path) -> dict[str, Path]:
        return {"hadiths": _write_parquet(staging / hadith_name)}

    return _fn


def _producer_defect_parse(_raw: Path, _staging: Path) -> object:
    # Mirrors what generate_source_id raises on a collection == corpus id.
    raise DoubledCorpusPrefixError("source_id has a doubled leading corpus: 'stub:stub:1'")


def _partial_then_defect_parse(_raw: Path, staging: Path) -> object:
    # Writes one output, THEN the gate fires on a later one -- a partial set.
    _write_parquet(staging / "hadiths_partial.parquet")
    raise DoubledCorpusPrefixError("source_id has a doubled leading corpus: 'partial:partial:1'")


def _transient_failure_parse(_raw: Path, _staging: Path) -> object:
    raise RuntimeError("network flaked")  # NOT a producer defect


def _patch_registry(monkeypatch: pytest.MonkeyPatch, adapters: list[_StubAdapter]) -> None:
    monkeypatch.setattr(parse_pkg, "SOURCE_REGISTRY", tuple(adapters))


class TestProducerDefectFailsLoud:
    def test_run_all_raises_on_producer_defect(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A da#355 gate makes ``run_all`` raise ParseProducerError, not return."""
        _patch_registry(
            monkeypatch,
            [
                _StubAdapter("good", _good_parse("hadiths_good.parquet")),
                _StubAdapter("sanadset", _producer_defect_parse),
            ],
        )
        with pytest.raises(ParseProducerError) as err:
            run_all(tmp_path / "raw", tmp_path / "staging")
        # Names every defective source; the clean one is not listed.
        assert set(err.value.defects) == {"sanadset"}
        assert isinstance(err.value.defects["sanadset"], DoubledCorpusPrefixError)

    def test_summary_prints_before_the_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The per-source summary is still emitted, so the abort is diagnosable."""
        _patch_registry(monkeypatch, [_StubAdapter("sanadset", _producer_defect_parse)])
        with pytest.raises(ParseProducerError):
            run_all(tmp_path / "raw", tmp_path / "staging")
        out = capsys.readouterr().out
        assert "=== Parse Summary ===" in out
        assert "FAIL" in out


class TestPartialWritesPurged:
    def test_partial_output_is_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A parquet the failing adapter wrote THIS run is purged (da#386)."""
        staging = tmp_path / "staging"
        _patch_registry(monkeypatch, [_StubAdapter("partial", _partial_then_defect_parse)])
        with pytest.raises(ParseProducerError):
            run_all(tmp_path / "raw", staging)
        assert not (staging / "hadiths_partial.parquet").exists()

    def test_pre_existing_untouched_file_is_left_in_place(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The purge is scoped to this run's writes; it does not guess filenames.

        A legacy stale file the failing adapter never touched is NOT deleted -- the
        non-zero exit is what stops it being read as fresh, and the next successful
        parse overwrites it. Deleting an un-attributable file would re-introduce the
        load-side ``{kind}_{slug}.parquet`` glob as a hand-kept parallel list.
        """
        staging = tmp_path / "staging"
        stale = _write_parquet(staging / "hadiths_sanadset.parquet")
        _patch_registry(monkeypatch, [_StubAdapter("sanadset", _producer_defect_parse)])
        with pytest.raises(ParseProducerError):
            run_all(tmp_path / "raw", staging)
        assert stale.exists()

    def test_clean_sources_keep_their_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A source that parsed before the defective one keeps its written parquet."""
        staging = tmp_path / "staging"
        _patch_registry(
            monkeypatch,
            [
                _StubAdapter("good", _good_parse("hadiths_good.parquet")),
                _StubAdapter("sanadset", _producer_defect_parse),
            ],
        )
        with pytest.raises(ParseProducerError):
            run_all(tmp_path / "raw", staging)
        assert (staging / "hadiths_good.parquet").exists()


class TestNarrowingPreserved:
    def test_transient_failure_is_still_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A non-producer exception is logged and skipped, NOT made fatal.

        The da#386 narrowing is limited to the producer gate; run_all must still
        continue-on-failure for a genuinely transient error.
        """
        _patch_registry(
            monkeypatch,
            [
                _StubAdapter("good", _good_parse("hadiths_good.parquet")),
                _StubAdapter("flaky", _transient_failure_parse),
            ],
        )
        results = run_all(tmp_path / "raw", tmp_path / "staging")  # must NOT raise
        assert results["flaky"] == []
        assert results["good"]  # the good source still produced output

    def test_transient_failure_alongside_producer_defect(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A transient failure does not suppress the producer-defect raise."""
        _patch_registry(
            monkeypatch,
            [
                _StubAdapter("flaky", _transient_failure_parse),
                _StubAdapter("sanadset", _producer_defect_parse),
            ],
        )
        with pytest.raises(ParseProducerError) as err:
            run_all(tmp_path / "raw", tmp_path / "staging")
        assert set(err.value.defects) == {"sanadset"}  # flaky is skipped, not a defect


class TestCmdParseExitCode:
    def test_cmd_parse_exits_with_producer_defect_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``isnad-ingest parse`` exits PARSE_PRODUCER_DEFECT, not 0 (da#386)."""
        _patch_registry(monkeypatch, [_StubAdapter("sanadset", _producer_defect_parse)])
        monkeypatch.setattr(
            config_mod,
            "get_settings",
            lambda: SimpleNamespace(
                data_raw_dir=str(tmp_path / "raw"), data_staging_dir=str(tmp_path / "staging")
            ),
        )
        with pytest.raises(SystemExit) as exc:
            cli._cmd_parse()
        assert exc.value.code == ExitCode.PARSE_PRODUCER_DEFECT

    def test_cmd_parse_succeeds_when_no_defect(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No defect -> no SystemExit; the happy path is unchanged."""
        _patch_registry(monkeypatch, [_StubAdapter("good", _good_parse("hadiths_good.parquet"))])
        monkeypatch.setattr(
            config_mod,
            "get_settings",
            lambda: SimpleNamespace(
                data_raw_dir=str(tmp_path / "raw"), data_staging_dir=str(tmp_path / "staging")
            ),
        )
        cli._cmd_parse()  # must not raise SystemExit
