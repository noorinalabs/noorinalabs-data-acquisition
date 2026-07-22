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


def test_circa_with_ah_ce_slash_does_not_spuriously_widen() -> None:
    # A slash is an AH/CE separator, not an alternative marker (Kavitha): the CE
    # mirror (795م) must not be mistaken for a 2nd candidate AH year.
    parsed = parse_source_notation("circa 180هـ/795م")
    assert parsed is not None
    assert parsed.precision is DatePrecision.CIRCA
    assert (parsed.earliest, parsed.latest) == (180, 180)


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


def test_widening_is_order_independent() -> None:
    # MUST-FIX 1 (Jean-Claude): with a competing point cluster, the NOISE_GAP gate
    # is anchored on the FIXED consensus band, so a chain of in-band bounds cannot
    # walk the envelope past consensus and the result is independent of iteration
    # order. Consensus points {100, 101}; lower bounds 92 (within NOISE_GAP of
    # core_lo=100) and 83 (beyond it) must reconcile identically either way.
    obs = [
        _exact(100),
        _exact(101),
        _obs(point=100, earliest=92, latest=100, precision=DatePrecision.RANGE),
        _obs(point=101, earliest=83, latest=101, precision=DatePrecision.RANGE),
    ]
    forward = reconcile_event(obs)
    backward = reconcile_event(list(reversed(obs)))
    assert forward == backward
    # 92 is supported (gap 8 ≤ NOISE_GAP); 83 is noise (gap 17 > NOISE_GAP).
    assert forward.earliest == 92


def test_single_source_wide_range_is_preserved_not_collapsed() -> None:
    # MUST-FIX 2 (Kavitha): da#164 emits a closed RANGE for "between 200 and 230
    # AH" (point=200, earliest=200, latest=230). With no competing source, the
    # source's own span must survive in full as RANGE [200, 230] — never clipped
    # against its own point into a spurious EXACT 200.
    rec = reconcile_event(
        [_obs(point=200, earliest=200, latest=230, precision=DatePrecision.RANGE)]
    )
    assert rec.precision is DatePrecision.RANGE
    assert (rec.earliest, rec.latest) == (200, 230)
    assert rec.point == 200


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
    # A bracketed open range is legitimate, not a disagreement (Jean-Claude).
    assert rec.conflict is False


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


# --------------------------------------------------------------------------- #
# Implausible attested death scrub (da#454) — the input-side companion to
# narrator_split's da#446 output-side gate: a single UNCONTESTED late-collector
# attested death (no competing source, so reconciliation sees no "conflict") must
# still not survive reconciliation when it is impossible for the narrator's own
# generation.
# --------------------------------------------------------------------------- #
def test_stage_scrubs_implausible_attested_death_for_generation(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    # A Companion (sahabi, plausible window ~[11, 110] AH) whose single bio source
    # attests a d. 720 AH death — the late-collector pollution class da#446/#454
    # documents: a commentator's death mislabelled onto an early isnad node.
    name = "صحابي وهمي"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:454a",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=720,
                death_year_ah_earliest=720,
                death_year_ah_latest=720,
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
                "generation": "sahabi",
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            }
        ],
    )

    out = reconcile_canonical_dates(staging, curated)
    assert out is not None
    rec = next(r for r in pq.read_table(out).to_pylist() if r["canonical_id"] == cid)

    # The whole death dating is scrubbed back to undated — not just the point —
    # so the ṭabaqa fallback (da#166) sees a genuinely undated narrator next.
    assert rec["death_year_ah"] is None
    assert rec["death_year_ah_earliest"] is None
    assert rec["death_year_ah_latest"] is None
    assert rec["death_date_precision"] == "unknown"


def test_stage_scrubs_implausible_death_over_preexisting_canonical_point(tmp_path: Path) -> None:
    """The scrub must pierce the back-fill-protection guard (da#454, Jean-Claude must-fix).

    Reproduces the idempotent-re-run / already-dated-canonical hole: the canonical
    record ALREADY carries a legacy pre-fix corrupt death point (d. 720 AH welded
    onto a Companion), exactly the state a persisted pre-fix
    ``narrators_canonical.parquet`` carries into a ``resolve --from-step reconcile``
    re-run. The scrub nulls the reconciled death dating, but the back-fill guard
    (``value is None and rec.get(key) is not None -> continue``) would otherwise
    keep the surviving 720 point while its bounds + precision are nulled — a
    self-inconsistent half-scrub that ``_death_is_undated`` then reads as "dated",
    skipping the ṭabaqa fallback. Assert the WHOLE death dating is cleared: point
    AND both bounds AND precision, with no surviving point behind a nulled envelope.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    name = "صحابي ملوث"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:454d",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=720,
                death_year_ah_earliest=720,
                death_year_ah_latest=720,
                death_date_precision="exact",
            )
        ],
    )
    # Canonical record already dated with the pre-fix corrupt point — the guard
    # branch (rec.get("death_year_ah") is not None) MUST fire for this to exercise
    # the hole rather than the clean from-scratch path.
    _write_canonical(
        curated,
        [
            {
                "canonical_id": cid,
                "name_ar": name,
                "name_ar_normalized": norm,
                "generation": "sahabi",
                "death_year_ah": 720,
                "death_year_ah_earliest": 720,
                "death_year_ah_latest": 720,
                "death_date_precision": "exact",
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            }
        ],
    )

    out = reconcile_canonical_dates(staging, curated)
    assert out is not None
    rec = next(r for r in pq.read_table(out).to_pylist() if r["canonical_id"] == cid)

    # No half-scrubbed survivor: point, both bounds, and precision all cleared.
    assert rec["death_year_ah"] is None
    assert rec["death_year_ah_earliest"] is None
    assert rec["death_year_ah_latest"] is None
    assert rec["death_date_precision"] == "unknown"


def test_stage_scrubs_implausible_attested_death_absolute_envelope(tmp_path: Path) -> None:
    """No/unknown generation still bounds against the absolute plausibility envelope."""
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    name = "مجهول الطبقة"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:454b",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=805,
                death_year_ah_earliest=805,
                death_year_ah_latest=805,
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
                # generation left unset — the absolute envelope still applies.
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            }
        ],
    )

    out = reconcile_canonical_dates(staging, curated)
    assert out is not None
    rec = next(r for r in pq.read_table(out).to_pylist() if r["canonical_id"] == cid)
    assert rec["death_year_ah"] is None
    assert rec["death_date_precision"] == "unknown"


def test_stage_keeps_plausible_attested_death_for_generation(tmp_path: Path) -> None:
    """Regression: a plausible generation-consistent death is reconciled as before."""
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    name = "تابعي حقيقي"
    norm = normalize_arabic(name)
    cid = make_canonical_id(norm)
    _write_bios(
        staging,
        "itqan",
        [
            _bio_date_row(
                bio_id="itqan:454c",
                source="itqan",
                name_ar=name,
                name_ar_normalized=norm,
                death_year_ah=110,
                death_year_ah_earliest=110,
                death_year_ah_latest=110,
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
                "generation": "tabii",
                "source_corpus": "itqan",
                "source_corpora": ["itqan"],
                "mention_count": 0,
            }
        ],
    )

    out = reconcile_canonical_dates(staging, curated)
    assert out is not None
    rec = next(r for r in pq.read_table(out).to_pylist() if r["canonical_id"] == cid)
    assert rec["death_year_ah"] == 110
    assert rec["death_date_precision"] == "exact"
