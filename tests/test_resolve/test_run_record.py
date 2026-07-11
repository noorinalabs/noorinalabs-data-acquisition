"""The resolve run record — the producer half of the prune completeness binding (da#428).

Every test here defends one property: **the declared count is a tally of the rows the
writer held in memory, and NOT a read-back of the file it wrote.** A read-back agrees
perfectly with a truncated parquet, and da#426 measured that shape deleting 83.9% of the
production graph with ``missing == 0`` and every derived gate green.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.graph.prune import read_canonical_ids
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve import _run_record
from src.resolve._run_record import (
    RESOLVE_RUN_FILENAME,
    RunStatus,
    canonical_id_tally,
    current_run_id,
    file_md5,
    finalize_run,
    read_run_record,
    write_canonical,
)
from src.resolve.bio_promote import promote_bios_to_canonical
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic

CANONICAL = "narrators_canonical.parquet"


def _canonical_table(canonical_ids: list[str | None], *, death_year: int | None = None) -> pa.Table:
    """A canonical table carrying exactly ``canonical_ids`` (dups/empties allowed).

    ``death_year`` models what the date stages actually do: ``reconcile`` and
    ``tabaqa_dates`` rewrite the file with the SAME row set but different date columns.
    Without it a fixture for those two stages would be byte-identical to the previous
    write, and any md5-changed assertion over them would be inert.
    """
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate(canonical_ids):
        base: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
        base["canonical_id"] = cid
        base["name_ar"] = f"اسم {i}"
        base["name_ar_normalized"] = f"اسم {i}"
        base["mention_count"] = 0
        base["death_year_ah"] = death_year
        rows.append(base)
    arrays = {f.name: [r[f.name] for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    return pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)


def _write_bio(staging: Path, bio_id: str, name: str) -> None:
    base: dict[str, Any] = {f.name: None for f in NARRATOR_BIO_SCHEMA}
    base.update(
        {
            "bio_id": bio_id,
            "source": "itqan",
            "name_ar": name,
            "name_ar_normalized": normalize_arabic(name),
        }
    )
    arrays = {f.name: [base[f.name]] for f in NARRATOR_BIO_SCHEMA}
    pq.write_table(
        pa.table(arrays, schema=NARRATOR_BIO_SCHEMA),
        staging / f"narrators_bio_{bio_id.replace(':', '_')}.parquet",
    )


# --- The definition: distinct, non-empty — NOT a row count -------------------
def test_tally_is_distinct_non_empty_not_a_row_count() -> None:
    """A duplicate and a falsy id must make ``canonical_ids < rows``, red-first.

    If the producer declared ``len(df)`` the consumer's equality would false-FAIL on
    any dup/empty — and a false-FAIL on a destructive op sends an operator to "restore
    from backup" after a prune that did exactly the right thing.

    The ``None`` here is defensive-only and cannot reach a written parquet:
    ``canonical_id`` is NON-NULLABLE in ``NARRATORS_CANONICAL_SCHEMA``, so the cast in
    ``write_parquet`` rejects it. The consumer filters falsy anyway, and this tally
    matches that behaviour exactly, so the two agree on inputs neither can receive.
    The *representable* hazards — a duplicate and an empty string — are exercised on
    the real write path in ``test_producer_tally_equals_consumer_read_canonical_ids``.
    """
    table = _canonical_table(["nar:a", "nar:b", "nar:b", None, ""])

    assert table.num_rows == 5
    assert canonical_id_tally(table) == 2  # {nar:a, nar:b}
    assert canonical_id_tally(table) < table.num_rows


def test_producer_tally_equals_consumer_read_canonical_ids(tmp_path: Path) -> None:
    """Producer and consumer provably count the SAME thing on the same fixture.

    Same definition, different derivation: the tally comes from the in-memory rows,
    ``read_canonical_ids`` from the written file. This is the equality the whole prune
    gate hangs on, so it is pinned against the real consumer, not a restatement of it.

    A duplicate and an empty-string id are both present and both survive the write, so
    ``canonical_ids < rows`` here on a real artifact — the two are NOT interchangeable.
    """
    table = _canonical_table(["nar:a", "nar:b", "nar:b", "", "nar:c"])
    path = tmp_path / CANONICAL

    write_canonical(table, path, stage="disambiguate")

    record = read_run_record(tmp_path)
    assert record is not None
    assert record.canonical_ids == len(read_canonical_ids(path)) == 3
    assert record.canonical_ids < pq.read_table(path).num_rows == 5


# --- THE FALSIFIER ----------------------------------------------------------
def test_tally_is_the_in_memory_count_not_a_read_back_of_a_short_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer is handed 100 rows; the disk receives 20. The record must say **100**.

    This is the acceptance criterion of da#428 — *if this test cannot be written, the
    implementation is a read-back and the issue is not done* — and getting it right took
    two attempts, which is the whole lesson.

    **A truncation applied AFTER the write does not falsify anything.** A read-back taken
    at write time has already read the complete file by then, so it reports 100 too, and
    the test passes under BOTH implementations while appearing to prove the point. That
    version of this test was written first and a read-back mutant sailed straight through
    it (``feedback_passing_repro_masks_bug``: a green repro proves nothing if it exercises
    a different shape than the real failure).

    The shape that actually separates them is the **partial write** — the mode the issue's
    own table lists as "NOT caught" by a read-back manifest. So ``write_parquet`` is made
    lossy here: ``write_canonical`` intends 100 rows and the file ends up with 20. An
    in-memory tally declares what the writer INTENDED (100) and therefore disagrees with
    the short file — and that disagreement is the entire completeness signal. A read-back
    declares 20 and agrees with it perfectly, which is precisely how a truncated parquet
    sails through every gate and deletes 83.9% of the graph (da#426).
    """
    full = _canonical_table([f"nar:{i}" for i in range(100)])
    path = tmp_path / CANONICAL

    real_write_parquet = _run_record.write_parquet

    def _partial_write(table: pa.Table, p: Path, schema: pa.Schema | None = None) -> Path:
        """The disk gets a fifth of what the writer handed it. Valid parquet, short."""
        return real_write_parquet(table.slice(0, 20), p, schema=schema)

    monkeypatch.setattr(_run_record, "write_parquet", _partial_write)

    write_canonical(full, path, stage="cluster")

    # The mutation is EFFECTIVE: the file really is short, and really is still a valid,
    # readable parquet — the dangerous shape, not a corrupt one any reader would reject.
    assert pq.read_table(path).num_rows == 20
    assert len(read_canonical_ids(path)) == 20

    # KNOWN-POSITIVE CONTROL: a read-back implementation would declare 20 — it agrees
    # with the short file by construction. If the record below also said 20, this test
    # could not tell the two implementations apart and would be worthless.
    assert canonical_id_tally(pq.read_table(path)) == 20

    # THE PROPERTY: the record declares what the writer INTENDED to write.
    record = read_run_record(tmp_path)
    assert record is not None
    assert record.canonical_ids == 100, "the tally came from the FILE, not from memory"
    assert record.canonical_ids != len(read_canonical_ids(path))


def test_record_is_not_re_derived_when_the_parquet_changes_underneath_it(
    tmp_path: Path,
) -> None:
    """Truncating the file after the fact must not retroactively change the declaration,
    and must break the md5 binding so the publish refuses the bytes.

    This is NOT the falsifier (see above — a read-back survives it). It pins the weaker
    but still necessary property that the record is a stored declaration rather than
    something recomputed on read, and that the tally is bound to the bytes it was taken
    over.
    """
    full = _canonical_table([f"nar:{i}" for i in range(100)])
    path = tmp_path / CANONICAL

    write_canonical(full, path, stage="cluster")
    pq.write_table(full.slice(0, 20), path)  # valid, short

    assert len(read_canonical_ids(path)) == 20

    record = read_run_record(tmp_path)
    assert record is not None
    assert record.canonical_ids == 100
    assert record.parquet_md5 != file_md5(path), "the md5 binding did not notice the rewrite"


# --- The four-(really six-)writer composition -------------------------------
def test_bio_promote_after_disambiguate_declares_the_post_bio_promote_count(
    tmp_path: Path,
) -> None:
    """The declared count is the LAST writer's, not the first writer's.

    ``narrators_canonical.parquet`` has six writers. ``disambiguate`` OVERWRITES it;
    ``bio_promote`` MERGEs into it — and that is not marginal: 43,109 of 129,234
    canonical ids are bio-only ``mention_count=0`` entries (da#352 ruled these real
    narrators). A tally taken at ``disambiguate`` would declare ~86k while the file
    holds ~129k, and the consumer's equality would false-FAIL on every healthy run.

    This drives the REAL ``promote_bios_to_canonical`` stage over a canonical file the
    disambiguate writer already produced — a test exercising only ``disambiguate`` would
    pass while being wrong ([[feedback_fixture_makes_guard_assertion_inert]]).
    """
    staging = tmp_path / "staging"
    curated = tmp_path / "curated"
    staging.mkdir()
    curated.mkdir()
    path = curated / CANONICAL

    # Writer 1 — disambiguate OVERWRITES with 2 mention-derived ids.
    write_canonical(_canonical_table(["nar:a", "nar:b"]), path, stage="disambiguate")
    assert read_run_record(curated).canonical_ids == 2  # type: ignore[union-attr]

    # Writer 2 — bio_promote MERGEs 2 bio-only narrators in, for real.
    _write_bio(staging, "itqan:1", "سعيد بن سماك")
    _write_bio(staging, "itqan:2", "فاطمة بنت محمد")
    promote_bios_to_canonical(staging, curated)

    # The mutation is effective: bio_promote genuinely ADDED rows.
    after = read_canonical_ids(path)
    assert len(after) == 4, "fixture inert — bio_promote added no ids, so it proves nothing"

    record = read_run_record(curated)
    assert record is not None
    assert record.last_writer == "bio_promote"
    assert record.canonical_ids == 4, "declared the disambiguate count, not the merged one"
    assert record.canonical_ids == len(read_canonical_ids(path))


def test_every_writer_re_mints_the_tally_so_the_last_one_wins(tmp_path: Path) -> None:
    """Each of the six writers re-mints; whichever runs LAST is the one that stands.

    ``reconcile`` and ``tabaqa_dates`` run AFTER ``narrator_split`` and preserve the row
    set while rewriting the BYTES — so a tally pinned to a named "last" stage would keep
    a stale md5 even when the count happened to be right. Binding the tally to the write
    rather than to a stage name is what makes that impossible.
    """
    path = tmp_path / CANONICAL
    seen: list[tuple[str, int, str]] = []

    # death_year models the date stages genuinely rewriting the file with the same rows.
    for stage, ids, death_year in [
        ("disambiguate", ["nar:a", "nar:b", "nar:c"], None),
        ("bio_promote", ["nar:a", "nar:b", "nar:c", "nar:d"], None),
        ("cluster", ["nar:a", "nar:b", "nar:d"], None),  # fuzzy merge DROPS an id
        ("narrator_split", ["nar:a", "nar:b", "nar:d", "nar:e"], None),  # peeled row APPENDED
        ("reconcile", ["nar:a", "nar:b", "nar:d", "nar:e"], 110),  # rows same, BYTES change
        ("tabaqa_dates", ["nar:a", "nar:b", "nar:d", "nar:e"], 120),  # rows same, BYTES change
    ]:
        write_canonical(_canonical_table(ids, death_year=death_year), path, stage=stage)
        record = read_run_record(tmp_path)
        assert record is not None
        assert record.last_writer == stage
        assert record.canonical_ids == len(set(ids))
        assert record.parquet_md5 == file_md5(path), f"{stage} left a stale md5"
        seen.append((stage, record.canonical_ids, record.parquet_md5))

    # The count really does move around across writers (drops AND appends), so the
    # "last writer wins" assertion above is not vacuously true of a constant.
    assert [c for _, c, _ in seen] == [3, 4, 3, 4, 4, 4]

    # And the last two writers — which PRESERVE the row set — still changed the BYTES.
    # This is why the tally cannot be pinned to a stage chosen by name: `narrator_split`
    # is the last writer to change the COUNT, but `tabaqa_dates` is the last to change
    # the FILE, and a record naming the former would carry a stale md5 for the latter.
    split_md5 = next(m for s, _, m in seen if s == "narrator_split")
    reconcile_md5 = next(m for s, _, m in seen if s == "reconcile")
    tabaqa_md5 = next(m for s, _, m in seen if s == "tabaqa_dates")
    assert split_md5 != reconcile_md5 != tabaqa_md5
    assert tabaqa_md5 == file_md5(path)


# --- Fail-closed run status --------------------------------------------------
def test_write_stamps_incomplete_and_only_finalize_upgrades(tmp_path: Path) -> None:
    """A write can never declare the run finished; only the end of ``run_all`` can.

    So a killed process, a raised stage or a ``--stop-after`` probe all leave
    ``incomplete`` on disk. Nothing has to remember to mark failure.
    """
    path = tmp_path / CANONICAL
    write_canonical(_canonical_table(["nar:a"]), path, stage="disambiguate")

    assert read_run_record(tmp_path).run_status is RunStatus.INCOMPLETE  # type: ignore[union-attr]

    finalize_run(tmp_path, ("step=ner outcome=ran files=1",), complete=True)
    record = read_run_record(tmp_path)
    assert record is not None
    assert record.run_status is RunStatus.COMPLETE
    assert record.canonical_ids == 1  # the tally is preserved, not recomputed
    assert record.stages == ("step=ner outcome=ran files=1",)


def test_finalize_with_errored_stage_leaves_incomplete(tmp_path: Path) -> None:
    path = tmp_path / CANONICAL
    write_canonical(_canonical_table(["nar:a"]), path, stage="disambiguate")

    finalize_run(tmp_path, (), complete=False)

    assert read_run_record(tmp_path).run_status is RunStatus.INCOMPLETE  # type: ignore[union-attr]


def test_finalize_must_not_upgrade_a_record_it_did_not_mint(tmp_path: Path) -> None:
    """A resumed run that touched no canonical writer must NOT promote a partial run's
    tally to ``complete``.

    ``--from-step dedup`` (da#268/da#272) starts past every canonical writer. If
    ``finalize_run`` stamped ``complete`` on whatever record it found, an earlier run
    that died half-way would have its short tally blessed by a later run that merely
    finished — a false PASS reached by an entirely legitimate operation, on a
    destructive op, with every other gate green.
    """
    path = tmp_path / CANONICAL
    write_canonical(_canonical_table(["nar:a", "nar:b"]), path, stage="disambiguate")

    # Forge the record so it looks like it was minted by a DIFFERENT (earlier) process.
    record_path = tmp_path / RESOLVE_RUN_FILENAME
    foreign = record_path.read_text().replace(f"run_id={current_run_id()}", "run_id=someone-else")
    record_path.write_text(foreign)

    # The mutation is effective: the record no longer claims this run.
    assert read_run_record(tmp_path).run_id != current_run_id()  # type: ignore[union-attr]

    finalize_run(tmp_path, (), complete=True)

    record = read_run_record(tmp_path)
    assert record is not None
    assert record.run_status is RunStatus.INCOMPLETE, "a foreign partial run was blessed complete"


def test_absent_record_reads_as_none_never_as_zero(tmp_path: Path) -> None:
    """A missing instrument reading is a stop, never a silent zero."""
    assert read_run_record(tmp_path) is None
    (tmp_path / RESOLVE_RUN_FILENAME).write_text("GARBAGE not-a-record\n")
    assert read_run_record(tmp_path) is None


# --- run_all actually CALLS finalize_run (da#429 review) ---------------------
def _run_all_with_spies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, boom: bool = False
) -> Path:
    """Drive the real ``run_all`` with no-op stages, except a ``disambiguate`` that
    genuinely mints a tally through the sanctioned writer.

    ``boom=True`` makes a later stage raise, so ``run_all`` finishes its sweep and then
    raises ``ResolveStageError`` (da#360) — the errored path.
    """
    from src.resolve import (
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

    staging = tmp_path / "staging"
    output = tmp_path / "curated"
    staging.mkdir()
    output.mkdir()
    pq.write_table(pa.table({"x": [1]}), staging / "hadiths_test.parquet")

    def _mint(*_a: object, **_k: object) -> list[Path]:
        write_canonical(
            _canonical_table(["nar:a", "nar:b"]), output / CANONICAL, stage="disambiguate"
        )
        return []

    monkeypatch.setattr(ner, "run", lambda *_a, **_k: [])
    monkeypatch.setattr(disambiguate, "run", _mint)
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

    def _boom(*_a: object, **_k: object) -> list[Path]:
        raise ValueError("parallels blew up mid-run")

    monkeypatch.setattr(parallels, "run", _boom if boom else (lambda *_a, **_k: []))

    if boom:
        from src.resolve import ResolveStageError

        with pytest.raises(ResolveStageError):
            run_all(tmp_path / "raw", staging, output)
    else:
        run_all(tmp_path / "raw", staging, output)
    return output


def test_run_all_finalizes_a_successful_run_to_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_all`` must actually CALL ``finalize_run`` — the seam, not the function.

    ``write_canonical`` stamps ``incomplete`` on every write, so ``complete`` can ONLY
    appear if ``run_all`` reached its finalize call. Replace that call with ``pass`` and
    the entire 2,229-test suite stays green (Oyunbileg Batbayar): nothing asserted that a
    successful run yields ``complete``. The failure is fail-CLOSED — publish would refuse
    every healthy run — but it would land as a false-FAIL at the end of a 7.5-hour re-run,
    on the launch path, which is precisely how an operator gets trained to weaken a gate.
    """
    output = _run_all_with_spies(tmp_path, monkeypatch)

    record = read_run_record(output)
    assert record is not None
    assert record.run_status is RunStatus.COMPLETE, "run_all never finalized the record"
    assert record.canonical_ids == 2, "the tally was recomputed, not preserved"
    assert record.last_writer == "disambiguate"
    # The per-stage forensics landed too — a hand-authored record cannot fake these.
    assert any(s.startswith("step=ner ") for s in record.stages)


def test_run_all_leaves_an_errored_run_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction. A one-sided assertion is half a control.

    Without this, ``finalize_run(..., complete=True)`` unconditionally would satisfy the
    test above while blessing a run that raised — the exact 'resolve stopped early' mode
    that no count equality can see.
    """
    output = _run_all_with_spies(tmp_path, monkeypatch, boom=True)

    record = read_run_record(output)
    assert record is not None
    assert record.run_status is RunStatus.INCOMPLETE, "a run with a raised stage was blessed"


# --- The branch that calls it: no seventh writer may bypass write_canonical ---
# Every function name that can put bytes into a parquet file, in any call form. The
# detector matches the bare name (`write_parquet(...)`) AND the dotted attribute form
# (`pq.write_table(...)`, `base.write_parquet(...)`) — a gate must match every syntactic
# form of the thing it forbids, or it merely relocates the bypass to the form it cannot
# see. `pq.write_table` is exactly the form the scrub scripts use and the most natural
# way a future writer would reach for (Jean-Claude Habimana, da#429 review).
_PARQUET_WRITE_FUNCS = frozenset({"write_parquet", "write_table"})

# The two names that identify the CANONICAL master as the target. `canonical_path` is the
# load-bearing one: it survives a call form that passes no schema at all.
_CANONICAL_MARKERS = frozenset({"canonical_path", "NARRATORS_CANONICAL_SCHEMA"})


def _bypassing_writers(source: str) -> list[int]:
    """Line numbers of parquet writes that target the canonical master directly.

    Such a call writes ``narrators_canonical.parquet`` WITHOUT minting a tally, leaving
    the record describing bytes that no longer exist. A guard is pinned by the branch that
    calls it, not by the existence of the function — so this walks the AST of every
    resolve module rather than trusting a convention.

    Residual limit, stated rather than hidden: a schema reached through an alias variable
    and a path built inline (``pq.write_table(t, out / "narrators_canonical.parquet")``)
    would slip past. That is a narrower early-warning signal, NOT a safety hole — the md5
    binding in :func:`write_canonical` is form-agnostic (it hashes bytes and does not care
    how they arrived), so any bypass still fails closed at publish.
    """
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name not in _PARQUET_WRITE_FUNCS:
            continue
        refs = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if refs & _CANONICAL_MARKERS:
            found.append(node.lineno)
    return found


@pytest.mark.parametrize(
    ("form", "is_bypass"),
    [
        # Known-POSITIVES — every call form that really does write the canonical master.
        ("write_parquet(t, canonical_path, schema=NARRATORS_CANONICAL_SCHEMA)", True),
        ("pq.write_table(t, canonical_path)", True),  # the form the scrub scripts use
        ("base.write_parquet(t, canonical_path)", True),  # dotted
        ("pyarrow.parquet.write_table(t, canonical_path, compression='snappy')", True),
        # Known-NEGATIVES — must NOT fire.
        ("write_canonical(t, canonical_path, stage='cluster')", False),
        ("write_parquet(t, audit_path, schema=NARRATOR_SPLITS_SCHEMA)", False),
        ("writer.write_table(table)", False),  # streaming ParquetWriter, other artifact
        ("pq.read_table(canonical_path)", False),  # a READ of the canonical master
    ],
)
def test_the_bypass_detector_separates_its_two_classes(form: str, is_bypass: bool) -> None:
    """Verify the instrument on BOTH classes before reading it anywhere else.

    The first version of this detector matched only the bare-name ``write_parquet`` form
    and was blind to ``pq.write_table(t, canonical_path)`` — a gate that cannot see the
    most natural way to commit the very bypass it forbids. A detector that cannot separate
    the two classes it exists to separate is not producing a measurement.
    """
    assert bool(_bypassing_writers(form)) is is_bypass


# The ONE sanctioned site: write_canonical's own write_parquet call, which mints the
# tally around it. Exempting it by name (rather than by "the detector happens not to see
# it") keeps the detector able to fire on every other module — including any future one.
_SANCTIONED_WRITER_MODULE = "_run_record.py"


def test_the_sanctioned_writer_is_itself_the_only_exempted_site() -> None:
    """The exemption is a real, load-bearing hole, so it is asserted rather than assumed.

    If ``write_canonical`` ever stopped writing the canonical master, this fails — and
    the test below would then be passing while guarding nothing.
    """
    source = Path(__file__).resolve().parents[2] / "src" / "resolve" / _SANCTIONED_WRITER_MODULE
    assert _bypassing_writers(source.read_text(encoding="utf-8")), (
        "write_canonical no longer writes narrators_canonical.parquet — the exemption "
        "below is now covering nothing, and the bypass gate has quietly gone inert."
    )


def test_no_resolve_module_writes_the_canonical_master_directly() -> None:
    """Every canonical write goes through :func:`write_canonical`. A seventh writer that
    bypasses it fails HERE, rather than silently publishing a stale tally.

    A guard is pinned by the BRANCH THAT CALLS IT, not by the existence of the function:
    ``write_canonical`` could be perfect and the binding still be worthless if one stage
    quietly kept calling ``write_parquet`` on the canonical path.
    """
    resolve_dir = Path(__file__).resolve().parents[2] / "src" / "resolve"
    offenders = {
        py.name: lines
        for py in sorted(resolve_dir.glob("*.py"))
        if py.name != _SANCTIONED_WRITER_MODULE
        if (lines := _bypassing_writers(py.read_text(encoding="utf-8")))
    }
    assert offenders == {}, (
        f"these write narrators_canonical.parquet without minting a tally: {offenders}. "
        "Use write_canonical(table, path, stage=...) — the tally must be re-minted by "
        "every writer, because whichever runs LAST is the one the consumer trusts."
    )
