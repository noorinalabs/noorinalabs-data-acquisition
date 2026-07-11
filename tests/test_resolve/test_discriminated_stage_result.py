"""Discriminated stage result: da#360 (exit non-zero on a raised stage) + da#378
(detector provenance makes a degraded artifact distinguishable from a true negative).

Both halves are proved RED-FIRST — each test names the behaviour on ``origin/main``
that it would fail against:

da#360
    ``run_all`` wrapped every stage in ``except Exception`` and logged-and-continued,
    reached ``resolve_pipeline_complete`` and returned a ``dict`` — the CLI printed
    "Resolution complete." and exited ``0``. A 7.5-hour run could report success
    having skipped a stage. The tests below assert ``run_all`` RAISES
    ``ResolveStageError`` (mapped to a non-zero exit) when a stage raises; on origin
    it returned normally (DID NOT RAISE).

da#378
    The composed ``parallel_links.parquet`` carried no detector provenance, so a
    ``DEDUP_REQUIRE_ML=false`` degraded artifact was **byte-identical** to a true
    negative (both a plain ``PARALLEL_LINKS_SCHEMA.empty_table()``). The tests below
    first pin that byte-identity for the un-stamped write, then assert the stamped
    writes SEPARATE the two classes — the discriminator is verified on BOTH, not
    merely observed to return a value on one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import (
    ResolveStageError,
    StageErrored,
    StageRan,
    bio_promote,
    date_reconcile,
    dedup,
    disambiguate,
    fuzzy_cluster,
    muhaddithat_links,
    narrator_split,
    ner,
    parallels,
    run_all,
    tabaqa_dates,
)
from src.resolve._provenance import (
    DetectorProvenance,
    DetectorStatus,
    read_provenance,
    write_parallel_links,
)
from src.resolve.dedup import _write_empty_output, run_dedup
from src.resolve.schemas import PARALLEL_LINKS_SCHEMA


def _staging_with_parquet(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    pq.write_table(pa.table({"x": [1]}), staging / "hadiths_test.parquet")
    return staging, output


def _install_benign_spies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every stage with a no-op so a test can make exactly one of them raise."""
    monkeypatch.setattr(ner, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(disambiguate, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(bio_promote, "promote_bios_to_canonical", lambda *_a, **_k: None)
    metrics = type("M", (), {"merged_records": 0, "multi_member_clusters": 0})()
    monkeypatch.setattr(fuzzy_cluster, "cluster_canonical_narrators", lambda *_a, **_k: metrics)
    monkeypatch.setattr(narrator_split, "split_generic_narrators", lambda *_a, **_k: None)
    monkeypatch.setattr(date_reconcile, "reconcile_canonical_dates", lambda *_a, **_k: None)
    monkeypatch.setattr(tabaqa_dates, "apply_tabaqa_fallback", lambda *_a, **_k: None)
    monkeypatch.setattr(
        muhaddithat_links, "build_muhaddithat_mention_links", lambda *_a, **_k: None
    )
    monkeypatch.setattr(dedup, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(parallels, "run", lambda *_a, **_k: [])


# ---------------------------------------------------------------------------
# da#360 — a raised stage makes run_all exit non-zero (never 0).
# ---------------------------------------------------------------------------
class TestRaisedStageExitsNonZero:
    def test_run_all_raises_when_a_stage_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED on origin: run_all swallowed the raise and returned a dict (exit 0)."""
        staging, output = _staging_with_parquet(tmp_path)
        _install_benign_spies(monkeypatch)

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise ValueError("ner blew up mid-run")

        monkeypatch.setattr(ner, "run", _boom)

        with pytest.raises(ResolveStageError) as excinfo:
            run_all(tmp_path / "raw", staging, output)

        # The aggregate names the failed stage and its exception type.
        assert [e.step for e in excinfo.value.errored] == ["ner"]
        assert excinfo.value.errored[0].exc_type == "ValueError"

    def test_sweep_continues_but_still_exits_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-pipeline failure does not abort the dependency-aware sweep, but the
        run still refuses to report success and aggregates EVERY failed stage."""
        staging, output = _staging_with_parquet(tmp_path)
        _install_benign_spies(monkeypatch)

        ran_after: list[str] = []

        def _raise(name: str):  # type: ignore[no-untyped-def]
            def _fn(*_a: Any, **_k: Any) -> Any:
                raise RuntimeError(f"{name} failed")

            return _fn

        # NER precedes bio_promote; disambiguate is skipped when NER fails, but
        # bio_promote runs regardless (da#99/da#117) — proving the sweep continues.
        monkeypatch.setattr(ner, "run", _raise("ner"))
        monkeypatch.setattr(disambiguate, "run", _raise("disambiguate"))
        monkeypatch.setattr(
            bio_promote,
            "promote_bios_to_canonical",
            lambda *_a, **_k: ran_after.append("bio_promote"),
        )

        with pytest.raises(ResolveStageError) as excinfo:
            run_all(tmp_path / "raw", staging, output)

        # bio_promote still ran after the NER failure ...
        assert "bio_promote" in ran_after
        # ... yet the run exits non-zero, and only the stage that actually RAISED is
        # errored (disambiguate was skipped for want of mentions, not a failure).
        assert [e.step for e in excinfo.value.errored] == ["ner"]

    def test_parallels_raise_survives_compose_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raised ``parallels`` must not be relabelled StageRan by the compose step.

        RED on the pre-guard head: on the COMMON path — ``dedup`` produced links,
        ``parallels.run`` raises — the compose block sees a non-None ``composed``
        (the semantic side alone composes) and unconditionally did
        ``outcomes["parallels"] = StageRan(...)``, clobbering the StageErrored the
        parallels step recorded. The ``errored`` scan then found nothing and
        ``run_all`` returned normally (exit 0) — the exact da#360 swallow, one
        block down. The guard keeps the StageErrored, so ``run_all`` still raises.
        (Nikolaos Papadopoulos, PR#404 review.)
        """
        staging, output = _staging_with_parquet(tmp_path)
        _install_benign_spies(monkeypatch)

        # dedup succeeds AND writes a real parallel_links.parquet, so semantic_links
        # is non-None and _compose_parallel_links returns a path (the trigger).
        def _dedup_writes_links(_staging: Path, _out: Path, **_k: Any) -> list[Path]:
            return [
                write_parallel_links(
                    PARALLEL_LINKS_SCHEMA.empty_table(),
                    _staging / "parallel_links.parquet",
                    DetectorProvenance(DetectorStatus.RAN, 0, DetectorStatus.NOT_RUN, 0),
                )
            ]

        def _parallels_boom(*_a: Any, **_k: Any) -> Any:
            raise ValueError("parallels blew up after dedup produced links")

        monkeypatch.setattr(dedup, "run", _dedup_writes_links)
        monkeypatch.setattr(parallels, "run", _parallels_boom)

        with pytest.raises(ResolveStageError) as excinfo:
            run_all(tmp_path / "raw", staging, output)

        assert [e.step for e in excinfo.value.errored] == ["parallels"]

    def test_clean_run_returns_discriminated_outcomes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stage raised → a ``{step: StageOutcome}`` map, no ``ResolveStageError``.

        A stage that produced no file is a ``StageRan`` with empty ``files`` — the
        honest "ran, no output", NOT the ``[]`` that pre-da#360 also left for a
        raised stage. There is no fourth ambiguous state.
        """
        staging, output = _staging_with_parquet(tmp_path)
        _install_benign_spies(monkeypatch)

        outcomes = run_all(tmp_path / "raw", staging, output)

        assert isinstance(outcomes["ner"], StageRan)
        assert outcomes["ner"].files == []
        assert not any(isinstance(o, StageErrored) for o in outcomes.values())

    def test_cli_resolve_exits_stage_failed_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CLI maps ``ResolveStageError`` to a non-zero ``EXIT_STAGE_FAILED``."""
        from src import cli
        from src.exit_codes import ExitCode

        def _raise_stage(*_a: Any, **_k: Any) -> Any:
            raise ResolveStageError([StageErrored("dedup", "RuntimeError", "tb")])

        monkeypatch.setattr("src.resolve.run_all", _raise_stage)

        with pytest.raises(SystemExit) as excinfo:
            cli._cmd_resolve()

        assert excinfo.value.code == ExitCode.RESOLVE_STAGE_FAILED
        assert int(ExitCode.RESOLVE_STAGE_FAILED) != 0


# ---------------------------------------------------------------------------
# da#378 — a degraded artifact is distinguishable from a true negative.
# ---------------------------------------------------------------------------
class TestDetectorProvenanceSeparatesDegradedFromTrueNegative:
    def test_unstamped_empty_writes_are_byte_identical(self, tmp_path: Path) -> None:
        """Pin the pre-fix conflation: two plain empty writes ARE byte-identical.

        This is what ``origin`` produced for BOTH the degraded and the true-negative
        cases — the same ``PARALLEL_LINKS_SCHEMA.empty_table()`` written the same way.
        """
        a = tmp_path / "a.parquet"
        b = tmp_path / "b.parquet"
        pq.write_table(PARALLEL_LINKS_SCHEMA.empty_table(), a)
        pq.write_table(PARALLEL_LINKS_SCHEMA.empty_table(), b)
        assert a.read_bytes() == b.read_bytes()
        # And an un-stamped artifact reads back NO provenance to discriminate on.
        assert read_provenance(a) is None

    def test_degraded_write_path_stamps_degraded_no_ml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive ``run_dedup``'s REAL degraded branch (deps missing + require_ml=False)."""
        staging = tmp_path / "staging"
        staging.mkdir()
        # Report the ml deps as absent so the degraded branch (not the model path) runs.
        monkeypatch.setattr("src.resolve.dedup.missing_dependencies", lambda _d: ["faiss"])

        path = run_dedup(staging, require_ml=False)

        prov = read_provenance(path)
        assert prov is not None
        assert prov.semantic is DetectorStatus.DEGRADED_NO_ML
        assert prov.semantic_rows == 0
        # The rows are still an empty, schema-valid table.
        assert pq.read_table(path).num_rows == 0

    def test_discriminator_separates_the_two_classes(self, tmp_path: Path) -> None:
        """Instrument verified on BOTH classes: degraded→DEGRADED_NO_ML, true-neg→RAN.

        A discriminator that returned the same value on both — or that only labelled
        one — would not separate the classes. This asserts the separation directly,
        and that the two artifacts are no longer byte-identical.
        """
        degraded_dir = tmp_path / "degraded"
        degraded_dir.mkdir()
        degraded = _write_empty_output(degraded_dir, DetectorStatus.DEGRADED_NO_ML)

        # A true negative: the semantic detector RAN and found zero pairs — exactly
        # what run_dedup's success path writes at 0 rows (semantic=RAN, 0 rows).
        tn_dir = tmp_path / "true_negative"
        tn_dir.mkdir()
        true_negative = write_parallel_links(
            PARALLEL_LINKS_SCHEMA.empty_table(),
            tn_dir / "parallel_links.parquet",
            DetectorProvenance(
                semantic=DetectorStatus.RAN,
                semantic_rows=0,
                deterministic=DetectorStatus.NOT_RUN,
                deterministic_rows=0,
            ),
        )

        degraded_prov = read_provenance(degraded)
        tn_prov = read_provenance(true_negative)
        assert degraded_prov is not None
        assert tn_prov is not None

        # Both empty (0 rows) — the case that was byte-identical on origin ...
        assert pq.read_table(degraded).num_rows == 0
        assert pq.read_table(true_negative).num_rows == 0
        # ... but the discriminator SEPARATES them, and the bytes now differ.
        assert degraded_prov.semantic is DetectorStatus.DEGRADED_NO_ML
        assert tn_prov.semantic is DetectorStatus.RAN
        assert degraded_prov.semantic != tn_prov.semantic
        assert degraded.read_bytes() != true_negative.read_bytes()

    def test_run_all_composed_artifact_carries_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the composed artifact run_all writes records semantic status.

        Degraded scenario — dedup writes the DEGRADED_NO_ML empty artifact, parallels
        runs — the composed parquet the graph loader reads reports semantic=DEGRADED_NO_ML,
        deterministic=RAN, so an operator can tell dedup did NOT run its algorithm.
        """
        staging, output = _staging_with_parquet(tmp_path)
        _install_benign_spies(monkeypatch)

        def _dedup_degraded(_staging: Path, _out: Path, **_k: Any) -> list[Path]:
            return [_write_empty_output(_staging, DetectorStatus.DEGRADED_NO_ML)]

        def _parallels_ran(_staging: Path, _out: Path, **_k: Any) -> list[Path]:
            return [
                write_parallel_links(
                    PARALLEL_LINKS_SCHEMA.empty_table(),
                    _staging / "parallel_links.parquet",
                    DetectorProvenance(DetectorStatus.NOT_RUN, 0, DetectorStatus.RAN, 0),
                )
            ]

        monkeypatch.setattr(dedup, "run", _dedup_degraded)
        monkeypatch.setattr(parallels, "run", _parallels_ran)

        run_all(tmp_path / "raw", staging, output)

        prov = read_provenance(staging / "parallel_links.parquet")
        assert prov is not None
        assert prov.semantic is DetectorStatus.DEGRADED_NO_ML
        assert prov.deterministic is DetectorStatus.RAN
