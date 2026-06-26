"""Tests for multi-source narrator date reconciliation (da#165).

Covers: agreeing sources, conflicting sources, the circa+multi-alternative case,
per-source noise robustness + outlier down-weighting, sanadset scalar folding,
open-bound (after/before) reconciliation, the always-concrete-precision invariant,
inclusive-window CE conversion, and the end-to-end canonical-table stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve.date_reconcile import (
    DateObservation,
    ReconciledDate,
    ce_bounds,
    observation_from_row,
    parse_source_notation,
    reconcile_canonical_dates,
    reconcile_event,
    reconcile_narrator,
)
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.hijri import ah_year_to_ce_range


def _obs(
    point: int | None = None,
    earliest: int | None = None,
    latest: int | None = None,
    precision: DatePrecision = DatePrecision.EXACT,
) -> DateObservation:
    return DateObservation(point=point, earliest=earliest, latest=latest, precision=precision)


def _exact(year: int) -> DateObservation:
    return _obs(point=year, earliest=year, latest=year, precision=DatePrecision.EXACT)


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def test_agreeing_sources_collapse_to_exact() -> None:
    rec = reconcile_event([_exact(256), _exact(256)])
    assert rec.precision is DatePrecision.EXACT
    assert (rec.point, rec.earliest, rec.latest) == (256, 256, 256)
    assert rec.conflict is False


def test_within_tolerance_is_agreement_not_conflict() -> None:
    # ±1-2 AH years is routine; the band is tight but flagged as non-conflict.
    rec = reconcile_event([_exact(256), _exact(257)])
    assert rec.precision is DatePrecision.RANGE
    assert (rec.earliest, rec.latest) == (256, 257)
    assert rec.conflict is False  # within AGREEMENT_TOLERANCE


# --------------------------------------------------------------------------- #
# Conflict
# --------------------------------------------------------------------------- #
def test_conflicting_sources_produce_honest_tight_range() -> None:
    # A: died 256, B: died 255 — keep the envelope, do not invent a winner.
    # ±1 year is the routine classical wafāt disagreement: a tight RANGE, and
    # within AGREEMENT_TOLERANCE so it is not flagged as a conflict.
    rec = reconcile_event([_exact(256), _exact(255)])
    assert rec.precision is DatePrecision.RANGE
    assert rec.earliest == 255
    assert rec.latest == 256
    assert rec.conflict is False


def test_wide_disagreement_flags_conflict() -> None:
    # A real disagreement (beyond AGREEMENT_TOLERANCE) still keeps the honest
    # envelope but is surfaced as a conflict for QA.
    rec = reconcile_event([_exact(256), _exact(250)])
    assert rec.precision is DatePrecision.RANGE
    assert (rec.earliest, rec.latest) == (250, 256)
    assert rec.conflict is True


# --------------------------------------------------------------------------- #
# circa + multi-alternative (Nikolaos's carry-forward)
# --------------------------------------------------------------------------- #
def test_circa_with_alternatives_keeps_envelope() -> None:
    # da#164's parser collapses this to a single CIRCA point; we keep the band.
    parsed = parse_source_notation("١٨٠ هـ تقريبا أو ١٨٥")
    assert parsed is not None
    assert parsed.precision is DatePrecision.CIRCA
    assert (parsed.earliest, parsed.latest) == (180, 185)
    assert parsed.point == 180


def test_circa_single_year_stays_point() -> None:
    parsed = parse_source_notation("circa 150")
    assert parsed is not None
    assert parsed.precision is DatePrecision.CIRCA
    assert (parsed.earliest, parsed.latest) == (150, 150)


def test_circa_only_sources_reconcile_as_circa_band() -> None:
    rec = reconcile_event(
        [
            _obs(point=180, earliest=180, latest=185, precision=DatePrecision.CIRCA),
            _obs(point=182, earliest=182, latest=182, precision=DatePrecision.CIRCA),
        ]
    )
    assert rec.precision is DatePrecision.CIRCA
    assert rec.earliest == 180
    assert rec.latest == 185


# --------------------------------------------------------------------------- #
# Noise robustness
# --------------------------------------------------------------------------- #
def test_lone_wide_bound_does_not_widen_envelope() -> None:
    # One source's death RANGE has a stray plausible integer inflating `latest`.
    noisy = _obs(point=256, earliest=256, latest=290, precision=DatePrecision.RANGE)
    rec = reconcile_event([_exact(256), _exact(257), noisy])
    # Anchored on the 256/257 agreement; the 290 noise (>NOISE_GAP beyond) is rejected.
    assert rec.earliest == 256
    assert rec.latest == 257


def test_lone_outlier_point_is_downweighted() -> None:
    rec = reconcile_event([_exact(255), _exact(256), _exact(300)])
    # 300 is > OUTLIER_GAP from the 255/256 median → dropped from the consensus.
    assert rec.earliest == 255
    assert rec.latest == 256
    assert rec.conflict is True  # raw spread still recorded as a conflict


# --------------------------------------------------------------------------- #
# sanadset scalar point-years
# --------------------------------------------------------------------------- #
def test_sanadset_scalar_unknown_precision_folds_as_point() -> None:
    # Bare year, precision "unknown", null bounds — the value IS known.
    scalar = _obs(point=256, precision=DatePrecision.UNKNOWN)
    rec = reconcile_event([scalar])
    assert rec.precision is DatePrecision.EXACT
    assert (rec.point, rec.earliest, rec.latest) == (256, 256, 256)


def test_sanadset_scalar_agrees_with_attested_source() -> None:
    rec = reconcile_event([_obs(point=256, precision=DatePrecision.UNKNOWN), _exact(256)])
    assert rec.precision is DatePrecision.EXACT
    assert rec.point == 256


# --------------------------------------------------------------------------- #
# Open bounds (after / before)
# --------------------------------------------------------------------------- #
def test_after_only_reconciles_as_after() -> None:
    rec = reconcile_event([_obs(point=130, earliest=130, precision=DatePrecision.AFTER)])
    assert rec.precision is DatePrecision.AFTER
    assert (rec.earliest, rec.latest) == (130, None)


def test_after_and_before_bracket_a_range() -> None:
    rec = reconcile_event(
        [
            _obs(point=130, earliest=130, precision=DatePrecision.AFTER),
            _obs(point=170, latest=170, precision=DatePrecision.BEFORE),
        ]
    )
    assert rec.precision is DatePrecision.RANGE
    assert (rec.earliest, rec.latest) == (130, 170)


def test_contradictory_open_bounds_flag_conflict() -> None:
    rec = reconcile_event(
        [
            _obs(point=170, earliest=170, precision=DatePrecision.AFTER),
            _obs(point=130, latest=130, precision=DatePrecision.BEFORE),
        ]
    )
    assert rec.conflict is True
    assert rec.precision is DatePrecision.AFTER


# --------------------------------------------------------------------------- #
# Always-concrete-precision invariant
# --------------------------------------------------------------------------- #
def test_no_observations_is_unknown_never_none() -> None:
    rec = reconcile_event([])
    assert rec.precision is DatePrecision.UNKNOWN
    assert rec.precision is not None
    assert (rec.point, rec.earliest, rec.latest) == (None, None, None)


def test_all_silent_sources_is_unknown() -> None:
    rec = reconcile_event(
        [_obs(precision=DatePrecision.UNKNOWN), _obs(precision=DatePrecision.UNKNOWN)]
    )
    assert rec.precision is DatePrecision.UNKNOWN


def test_null_precision_string_reads_as_unknown() -> None:
    row: dict[str, Any] = {
        "death_year_ah": None,
        "death_year_ah_earliest": None,
        "death_year_ah_latest": None,
        "death_date_precision": None,
    }
    obs = observation_from_row(row, "death")
    assert obs.precision is DatePrecision.UNKNOWN


def test_reconcile_narrator_emits_concrete_precision_strings() -> None:
    cols = reconcile_narrator(birth=[], death=[_exact(256)])
    assert cols["birth_date_precision"] == DatePrecision.UNKNOWN.value
    assert cols["death_date_precision"] == DatePrecision.EXACT.value
    assert cols["death_year_ah"] == 256


# --------------------------------------------------------------------------- #
# CE conversion contract — inclusive window, never the point converter
# --------------------------------------------------------------------------- #
def test_ce_bounds_use_inclusive_window() -> None:
    rec = ReconciledDate(256, 256, 256, DatePrecision.EXACT)
    assert ce_bounds(rec) == (ah_year_to_ce_range(256)[0], ah_year_to_ce_range(256)[1])


def test_ce_bounds_open_side_is_none() -> None:
    rec = ReconciledDate(130, 130, None, DatePrecision.AFTER)
    earliest, latest = ce_bounds(rec)
    assert earliest == ah_year_to_ce_range(130)[0]
    assert latest is None


# --------------------------------------------------------------------------- #
# End-to-end canonical-table stage
# --------------------------------------------------------------------------- #
def _write_bios(staging: Path, suffix: str, rows: list[dict[str, Any]]) -> Path:
    full = []
    for r in rows:
        base = {f.name: None for f in NARRATOR_BIO_SCHEMA}
        base.update(r)
        full.append(base)
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATOR_BIO_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    path = staging / f"narrators_bio_{suffix}.parquet"
    pq.write_table(table, path)
    return path


def _write_canonical(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    full = []
    for r in rows:
        base = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
        base.update(r)
        full.append(base)
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATORS_CANONICAL_SCHEMA}
    table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    path = out_dir / "narrators_canonical.parquet"
    pq.write_table(table, path)
    return path


def _bio_date_row(**kw: Any) -> dict[str, Any]:
    return kw


def test_stage_reconciles_conflict_and_enforces_concrete_precision(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    name = "زائدة بن قدامة"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)

    # Two sources disagree on the death year (160 vs 161).
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:1",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=161,
                death_year_ah_earliest=161,
                death_year_ah_latest=161,
                death_date_precision="exact",
            )
        ],
    )
    _write_bios(
        staging,
        "fawaz",
        [
            _bio_date_row(
                bio_id="fawaz:1",
                source="fawaz",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=160,
                death_year_ah_earliest=160,
                death_year_ah_latest=160,
                death_date_precision="exact",
            )
        ],
    )

    # Canonical table: the bio-backed narrator + a mention-only narrator whose
    # precision columns are null (must come out concrete as "unknown").
    mention_cid = make_canonical_id(normalize_arabic("شعبة"))
    _write_canonical(
        curated,
        [
            {
                "canonical_id": cid,
                "name_ar": name,
                "name_ar_normalized": norm,
                "death_year_ah": 161,  # bio_promote's arbitrary back-fill
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            },
            {
                "canonical_id": mention_cid,
                "name_ar": "شعبة",
                "name_ar_normalized": normalize_arabic("شعبة"),
                "source_corpus": "sunnah",
                "source_corpora": ["sunnah"],
                "mention_count": 5,
            },
        ],
    )

    out = reconcile_canonical_dates(staging, curated)
    assert out is not None
    by_id = {r["canonical_id"]: r for r in pq.read_table(out).to_pylist()}

    bio_rec = by_id[cid]
    assert bio_rec["death_year_ah_earliest"] == 160
    assert bio_rec["death_year_ah_latest"] == 161
    assert bio_rec["death_date_precision"] == "range"
    assert bio_rec["birth_date_precision"] == "unknown"  # no birth stated → concrete

    mention_rec = by_id[mention_cid]
    assert mention_rec["death_date_precision"] == "unknown"
    assert mention_rec["birth_date_precision"] == "unknown"

    # Invariant: every emitted row carries concrete precision strings.
    for row in by_id.values():
        assert row["birth_date_precision"] is not None
        assert row["death_date_precision"] is not None


def test_stage_is_idempotent(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    name = "سفيان الثوري"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:9",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=161,
                death_year_ah_earliest=161,
                death_year_ah_latest=161,
                death_date_precision="exact",
            )
        ],
    )
    _write_canonical(
        curated,
        [
            {
                "canonical_id": cid,
                "name_ar": name,
                "name_ar_normalized": norm,
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            }
        ],
    )

    reconcile_canonical_dates(staging, curated)
    first = pq.read_table(curated / "narrators_canonical.parquet").to_pylist()
    reconcile_canonical_dates(staging, curated)
    second = pq.read_table(curated / "narrators_canonical.parquet").to_pylist()
    assert first == second


def test_stage_no_canonical_is_noop(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()
    assert reconcile_canonical_dates(staging, curated) is None
