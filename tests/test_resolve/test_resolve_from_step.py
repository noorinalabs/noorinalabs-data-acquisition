"""Step-skip CLI plumbing for ``src.resolve.run_all(from_step=...)`` (da#268).

Asserts the gating that lets a crashed pipeline resume without redoing completed
stages: steps BEFORE ``from_step`` are skipped, ``from_step`` and later run, and a
skipped NER is treated as complete iff its mention output already exists (so
disambiguate can consume + resume against it).

The per-step callables are replaced with lightweight spies so the test exercises
the ordering/gating logic — not the heavy real steps — deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import (
    RESOLVE_STEP_ORDER,
    _resolve_start_index,
    bio_promote,
    date_reconcile,
    dedup,
    disambiguate,
    fuzzy_cluster,
    muhaddithat_links,
    ner,
    parallels,
    run_all,
    tabaqa_dates,
)


def test_resolve_start_index() -> None:
    assert _resolve_start_index(None) == 0
    assert _resolve_start_index("ner") == 0
    assert _resolve_start_index("disambiguate") == 1
    assert _resolve_start_index("dedup") == RESOLVE_STEP_ORDER.index("dedup")


def test_unknown_from_step_raises() -> None:
    with pytest.raises(ValueError, match="unknown resolve step"):
        run_all(Path("/x"), Path("/y"), Path("/z"), from_step="nope")


def _install_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every resolve step with a call-recording spy; return the call log."""
    called: list[str] = []

    def spy(name: str, ret: object):  # type: ignore[no-untyped-def]
        def _fn(*_a: object, **_k: object) -> object:
            called.append(name)
            return ret

        return _fn

    monkeypatch.setattr(ner, "run", spy("ner", []))
    monkeypatch.setattr(disambiguate, "run", spy("disambiguate", []))
    monkeypatch.setattr(bio_promote, "promote_bios_to_canonical", spy("bio_promote", None))
    # fuzzy_cluster returns a metrics object; give it the attributes run_all reads.
    cluster_metrics = type("M", (), {"merged_records": 0, "multi_member_clusters": 0})()
    monkeypatch.setattr(
        fuzzy_cluster, "cluster_canonical_narrators", spy("cluster", cluster_metrics)
    )
    monkeypatch.setattr(date_reconcile, "reconcile_canonical_dates", spy("reconcile", None))
    monkeypatch.setattr(tabaqa_dates, "apply_tabaqa_fallback", spy("tabaqa_dates", None))
    monkeypatch.setattr(
        muhaddithat_links, "build_muhaddithat_mention_links", spy("muhaddithat_links", None)
    )
    monkeypatch.setattr(dedup, "run", spy("dedup", []))
    monkeypatch.setattr(parallels, "run", spy("parallels", []))
    return called


def _staging_with_parquet(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    # run_all's pre-flight requires at least one staging parquet.
    pq.write_table(pa.table({"x": [1]}), staging / "hadiths_test.parquet")
    return staging, output


def test_from_step_disambiguate_skips_ner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = _install_spies(monkeypatch)
    staging, output = _staging_with_parquet(tmp_path)
    # Precomputed NER output present so the skipped-ner gate treats it as complete.
    (output / "narrator_mentions_resolved.parquet").write_bytes(b"placeholder")

    run_all(tmp_path / "raw", staging, output, from_step="disambiguate")

    assert "ner" not in called, "ner must be skipped by --from-step disambiguate"
    assert "disambiguate" in called, "disambiguate must run"
    assert "bio_promote" in called and "dedup" in called, "later steps must run"


def test_from_step_bio_promote_skips_ner_and_disambiguate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = _install_spies(monkeypatch)
    staging, output = _staging_with_parquet(tmp_path)

    run_all(tmp_path / "raw", staging, output, from_step="bio_promote")

    assert "ner" not in called
    assert "disambiguate" not in called
    assert "bio_promote" in called
    assert "cluster" in called


def test_no_from_step_runs_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = _install_spies(monkeypatch)
    staging, output = _staging_with_parquet(tmp_path)

    run_all(tmp_path / "raw", staging, output)

    for step in ("ner", "disambiguate", "bio_promote", "cluster", "dedup", "parallels"):
        assert step in called, f"{step} should run in a full pipeline"


def test_resume_flag_threads_to_resumable_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_all(resume=...)`` (CLI ``--no-resume``, da#272) must reach every stage
    that keeps an intra-stage checkpoint: disambiguate, dedup, parallels."""
    _install_spies(monkeypatch)
    staging, output = _staging_with_parquet(tmp_path)
    (output / "narrator_mentions_resolved.parquet").write_bytes(b"placeholder")

    seen: dict[str, bool] = {}

    def _spy(name: str, ret: object):  # type: ignore[no-untyped-def]
        def _fn(*_a: object, resume: bool = True, **_k: object) -> object:
            seen[name] = resume
            return ret

        return _fn

    monkeypatch.setattr(disambiguate, "run", _spy("disambiguate", []))
    monkeypatch.setattr(dedup, "run", _spy("dedup", []))
    monkeypatch.setattr(parallels, "run", _spy("parallels", []))

    run_all(tmp_path / "raw", staging, output, resume=False)
    assert seen == {"disambiguate": False, "dedup": False, "parallels": False}
