"""Missing-required-input fail-loud guards for resolve stages (da#361).

Before da#361 several ``src/resolve`` stages returned a success-shaped empty
result — ``[]`` / ``None`` / a zero-row parquet — when a **required** input
artifact was absent (an upstream defect: parse never ran, or ran against the
wrong dir). That empty was indistinguishable to every caller, and to the process
exit status, from an input that ran and legitimately produced nothing. This is
the #309 defect class (a missing *dependency*) one layer up (a missing *input*),
and it is landable now that #360 makes ``run_all`` surface a raise instead of
swallowing it.

Each test below is RED on origin: the stage returned an empty success. The
paired "present-but-empty" test pins the *other* side of the distinction — a
genuine empty must still succeed honestly — so the fix cannot be a blanket
"empty == error".

The dedup / bio_promote / narrator_split sites are covered in their own suites
(``test_dedup.py``, ``test_bio_promote.py``, ``test_narrator_split.py``); this
file covers the ner + disambiguate sites, the ``MissingInputError`` contract, and
the ``run_all`` → exit-code integration.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve import (
    MissingDependencyError,
    MissingInputError,
    ResolveStageError,
    bio_promote,
    date_reconcile,
    dedup,
    disambiguate,
    fuzzy_cluster,
    muhaddithat_links,
    narrator_split,
    parallels,
    run_all,
    tabaqa_dates,
)
from src.resolve import ner as ner_mod
from src.resolve._inputs import require_input
from src.resolve.disambiguate import _load_mentions
from src.resolve.disambiguate import run as disambiguate_run
from src.resolve.ner import _extract_from_hadiths, _load_phase1_mentions
from src.resolve.ner import run as ner_run
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA
from tests.factories import build_hadith_table


# --------------------------------------------------------------------------- #
# Fixtures / writers
# --------------------------------------------------------------------------- #
def _write_bios(staging: Path, rows: list[dict[str, object]]) -> Path:
    """Write a schema-valid narrators_bio_test.parquet from partial rows."""
    full = [{**{f.name: None for f in NARRATOR_BIO_SCHEMA}, **r} for r in rows]
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATOR_BIO_SCHEMA}
    path = staging / "narrators_bio_test.parquet"
    pq.write_table(pa.table(arrays, schema=NARRATOR_BIO_SCHEMA), path)
    return path


def _write_mentions(output: Path, rows: list[dict[str, object]]) -> Path:
    """Write a schema-valid narrator_mentions_resolved.parquet from partial rows."""
    full = [{**{f.name: None for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}, **r} for r in rows]
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    path = output / "narrator_mentions_resolved.parquet"
    pq.write_table(pa.table(arrays, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA), path)
    return path


def _staging_output(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    return staging, output


# --------------------------------------------------------------------------- #
# MissingInputError contract
# --------------------------------------------------------------------------- #
class TestMissingInputErrorContract:
    def test_is_an_exception_not_a_baseexception(self) -> None:
        """Unlike ``MissingDependencyError`` (which sails PAST ``run_all``'s
        ``except Exception`` to its own exit code), ``MissingInputError`` is a
        plain ``Exception`` ON PURPOSE, so ``run_all`` CATCHES it as a
        ``StageErrored`` and reuses the da#360 fail-loud machinery."""
        assert issubclass(MissingInputError, Exception)
        assert not issubclass(MissingDependencyError, Exception)  # BaseException-only

    def test_require_input_raises_on_absent_and_passes_on_present(self) -> None:
        # present=True → no raise.
        require_input(stage="s", present=True, input_desc="x", produced_by="p", remediation="r")
        # present=False → raise, message naming stage + input + remediation.
        with pytest.raises(MissingInputError) as excinfo:
            require_input(
                stage="ner",
                present=False,
                input_desc="foo.parquet",
                produced_by="parse",
                remediation="re-run parse",
            )
        msg = str(excinfo.value)
        assert "ner" in msg and "foo.parquet" in msg and "re-run parse" in msg


# --------------------------------------------------------------------------- #
# NER — a STAGED corpus missing its expected input is a defect; an UNSTAGED
# configured source simply absent from a partial staging subset is legitimate.
# --------------------------------------------------------------------------- #
class TestNerMissingInput:
    def test_staged_phase1_corpus_missing_mentions_raises(self, tmp_path: Path) -> None:
        """RED on origin: sanadset's hadiths are staged (so it is a real corpus in
        this run) but its Phase-1 mentions file is absent — NER returned ``[]`` and
        the pipeline marched on. Now it fails loud."""
        staging, output = _staging_output(tmp_path)
        # sanadset staged (via a hadiths shard carrying source_corpus=sanadset) but
        # its required narrator_mentions_sanadset.parquet is NOT written.
        table = build_hadith_table([{"source_id": "s-1", "source_corpus": "sanadset"}])
        pq.write_table(table, staging / "hadiths_sanadset.parquet")

        with pytest.raises(MissingInputError) as excinfo:
            ner_run(staging, output)
        assert excinfo.value.stage == "ner"
        assert "narrator_mentions_sanadset.parquet" in str(excinfo.value)

    def test_empty_staging_does_not_raise(self, tmp_path: Path) -> None:
        """Distinction: an EMPTY staging tree stages no corpus, so every configured
        source is unstaged (required=False) — a legitimate skip, not a defect. This
        is why partial-staging / standalone runs keep working."""
        staging, output = _staging_output(tmp_path)
        assert ner_run(staging, output) == []

    def test_unstaged_configured_source_absent_is_legitimate(self, tmp_path: Path) -> None:
        """Only ONE corpus staged (thaqalayn, Arabic-routed); sanadset/lk/sunnah are
        configured but unstaged, so their absent files do NOT raise."""
        staging, output = _staging_output(tmp_path)
        # thaqalayn present with a real isnad so it extracts; no other corpus staged.
        table = build_hadith_table(
            [{"source_id": "t-1", "source_corpus": "thaqalayn", "isnad_raw_ar": "حدثنا محمد"}]
        )
        pq.write_table(table, staging / "hadiths_thaqalayn.parquet")
        # Must not raise (sanadset/lk phase1 + sunnah english are unstaged).
        result = ner_run(staging, output)
        assert isinstance(result, list)

    def test_extract_helper_required_absent_raises(self, tmp_path: Path) -> None:
        """Helper-level: ``required=True`` + no hadith files → raise."""
        staging, _ = _staging_output(tmp_path)
        with pytest.raises(MissingInputError):
            _extract_from_hadiths(staging, "thaqalayn", language="ar", required=True)

    def test_extract_helper_not_required_absent_returns_empty(self, tmp_path: Path) -> None:
        """Helper-level: ``required=False`` (default) + no hadith files → honest []."""
        staging, _ = _staging_output(tmp_path)
        assert _extract_from_hadiths(staging, "thaqalayn", language="ar") == []

    def test_phase1_helper_required_absent_raises(self, tmp_path: Path) -> None:
        staging, _ = _staging_output(tmp_path)
        with pytest.raises(MissingInputError):
            _load_phase1_mentions(staging, "sanadset", "missing.parquet", required=True)

    def test_phase1_helper_not_required_absent_returns_empty(self, tmp_path: Path) -> None:
        staging, _ = _staging_output(tmp_path)
        assert _load_phase1_mentions(staging, "sanadset", "missing.parquet") == []


# --------------------------------------------------------------------------- #
# Disambiguate — candidate bios (staging) + NER mention output (output_dir).
# --------------------------------------------------------------------------- #
class TestDisambiguateMissingInput:
    def test_no_bio_files_raises(self, tmp_path: Path) -> None:
        """RED on origin: no candidate bio shards → ``[]`` (silent). Now raises."""
        staging, output = _staging_output(tmp_path)
        with pytest.raises(MissingInputError) as excinfo:
            disambiguate_run(staging, output)
        assert excinfo.value.stage == "disambiguate"
        assert "narrators_bio_*.parquet" in str(excinfo.value)

    def test_present_but_empty_bio_shard_is_honest_empty(self, tmp_path: Path) -> None:
        """Distinction: a bio shard that EXISTS but has zero rows is a genuine empty
        (no candidates) — an honest ``[]``, NOT a raise."""
        staging, output = _staging_output(tmp_path)
        _write_bios(staging, [])  # present, zero candidate rows
        assert disambiguate_run(staging, output) == []

    def test_bios_present_mentions_absent_is_honest_empty(self, tmp_path: Path) -> None:
        """da#361 carve-out: candidates present but the NER mention output ABSENT is
        NOT a missing input — NER legitimately produces zero mentions for a bio-only
        corpus (e.g. muhaddithat), so disambiguate no-ops honestly (``[]``), never
        raises. (The bio candidates ARE required and DO raise when absent.)"""
        staging, output = _staging_output(tmp_path)
        _write_bios(
            staging,
            [{"bio_id": "b1", "source": "itqan", "name_ar": "محمد", "name_ar_normalized": "محمد"}],
        )
        # No narrator_mentions_resolved.parquet in output_dir — legitimate empty.
        assert disambiguate_run(staging, output) == []

    def test_bios_present_mentions_present_but_empty_is_honest_empty(self, tmp_path: Path) -> None:
        """Same carve-out with the mentions file present but zero rows — honest []."""
        staging, output = _staging_output(tmp_path)
        _write_bios(
            staging,
            [{"bio_id": "b1", "source": "itqan", "name_ar": "محمد", "name_ar_normalized": "محمد"}],
        )
        _write_mentions(output, [])  # present, zero mention rows
        assert disambiguate_run(staging, output) == []

    def test_dead_code_load_mentions_helper_raises_on_absent(self, tmp_path: Path) -> None:
        """``_load_mentions`` is currently uncalled (``run`` uses ``_count_mentions``)
        but is kept honest so a future re-wiring inherits the fail-loud contract."""
        staging, _ = _staging_output(tmp_path)
        with pytest.raises(MissingInputError):
            _load_mentions(staging)


# --------------------------------------------------------------------------- #
# Optional inputs — date_reconcile + tabaqa_dates are carved OUT of fail-loud.
# --------------------------------------------------------------------------- #
class TestOptionalInputsDoNotRaise:
    def test_date_reconcile_no_canonical_is_noop(self, tmp_path: Path) -> None:
        from src.resolve.date_reconcile import reconcile_canonical_dates

        staging, output = _staging_output(tmp_path)
        assert reconcile_canonical_dates(staging, output) is None

    def test_tabaqa_dates_no_canonical_is_noop(self, tmp_path: Path) -> None:
        from src.resolve.tabaqa_dates import apply_tabaqa_fallback

        _, output = _staging_output(tmp_path)
        assert apply_tabaqa_fallback(output) is None


# --------------------------------------------------------------------------- #
# run_all integration — a MissingInputError from a real stage is caught as a
# StageErrored (da#360) and surfaces as ResolveStageError, never a silent exit 0.
# --------------------------------------------------------------------------- #
def test_run_all_surfaces_missing_input_as_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that raises ``MissingInputError`` inside ``run_all`` must be recorded
    as a ``StageErrored`` and re-raised as ``ResolveStageError`` — i.e. da#360's
    machinery catches it (proving the ``Exception``-not-``BaseException`` choice)."""
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    # Pre-flight needs a staging parquet.
    pq.write_table(pa.table({"x": [1]}), staging / "hadiths_test.parquet")

    # Neutralise every other stage (dedup would raise MissingDependencyError without
    # ml; narrator_split would raise its own MissingInputError) so exactly
    # bio_promote raises MissingInputError.
    monkeypatch.setattr(ner_mod, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(disambiguate, "run", lambda *_a, **_k: [])
    _metrics = type("M", (), {"merged_records": 0, "multi_member_clusters": 0})()
    monkeypatch.setattr(fuzzy_cluster, "cluster_canonical_narrators", lambda *_a, **_k: _metrics)
    monkeypatch.setattr(narrator_split, "split_generic_narrators", lambda *_a, **_k: None)
    monkeypatch.setattr(date_reconcile, "reconcile_canonical_dates", lambda *_a, **_k: None)
    monkeypatch.setattr(tabaqa_dates, "apply_tabaqa_fallback", lambda *_a, **_k: None)
    monkeypatch.setattr(
        muhaddithat_links, "build_muhaddithat_mention_links", lambda *_a, **_k: None
    )
    monkeypatch.setattr(dedup, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(parallels, "run", lambda *_a, **_k: [])

    def _boom(*_a: object, **_k: object) -> object:
        raise MissingInputError(
            stage="bio_promote",
            input_desc="narrators_bio_*.parquet",
            produced_by="parse",
            remediation="re-run parse",
        )

    monkeypatch.setattr(bio_promote, "promote_bios_to_canonical", _boom)

    with pytest.raises(ResolveStageError) as excinfo:
        run_all(tmp_path / "raw", staging, output)
    assert "bio_promote" in [e.step for e in excinfo.value.errored]
