"""Tests for the ṭabaqa → estimated-window date fallback (da#166).

Covers: the generation → death-window table values, the always-concrete
``tabaqa_estimate`` precision, AH→CE via the inclusive-window converter, the
"known ṭabaqa only" gate, and the end-to-end canonical-table stage — a
known-ṭabaqa undated narrator gets an estimate, a reconciled/parsed date is
never overwritten, a null-ṭabaqa narrator stays undated + UNKNOWN, and the
stage is idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision, NarratorGeneration
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.resolve.tabaqa_dates import (
    _GENERATION_DEATH_WINDOW_AH,
    TabaqaEstimate,
    apply_tabaqa_fallback,
    estimate_death_window,
    generation_from_value,
)
from src.utils.hijri import ah_year_to_ce_range


# --------------------------------------------------------------------------- #
# generation_from_value — normalisation
# --------------------------------------------------------------------------- #
def test_generation_from_value_known() -> None:
    assert generation_from_value("sahabi") is NarratorGeneration.SAHABI
    assert generation_from_value("taba_tabii") is NarratorGeneration.TABA_TABII


def test_generation_from_value_null_or_garbled_is_unknown() -> None:
    assert generation_from_value(None) is NarratorGeneration.UNKNOWN
    assert generation_from_value("") is NarratorGeneration.UNKNOWN
    assert generation_from_value("not-a-generation") is NarratorGeneration.UNKNOWN
    assert generation_from_value("unknown") is NarratorGeneration.UNKNOWN


# --------------------------------------------------------------------------- #
# estimate_death_window — the principled ṭabaqa → window table
# --------------------------------------------------------------------------- #
def test_window_table_values() -> None:
    # The documented generational death-year (AH) spans (Ibn Ḥajar's Taqrīb
    # periodization). Pinned so a silent edit to the mapping fails the test.
    assert _GENERATION_DEATH_WINDOW_AH == {
        NarratorGeneration.SAHABI: (11, 110),
        NarratorGeneration.TABII: (90, 180),
        NarratorGeneration.TABA_TABII: (150, 250),
        NarratorGeneration.ATBA_TABA_TABIIN: (220, 300),
        NarratorGeneration.LATER: (280, 400),
    }


def test_estimate_is_midpoint_with_concrete_tabaqa_precision() -> None:
    est = estimate_death_window(NarratorGeneration.TABA_TABII)
    assert est is not None
    assert est.death_year_ah_earliest == 150
    assert est.death_year_ah_latest == 250
    assert est.death_year_ah == 200  # window midpoint
    assert est.precision is DatePrecision.TABAQA_ESTIMATE


def test_estimate_ce_uses_inclusive_window_converter() -> None:
    est = estimate_death_window(NarratorGeneration.SAHABI)
    assert est is not None
    # CE bounds run from the START of the earliest AH year to the END of the
    # latest — exactly ah_year_to_ce_range on each end, never the point ah_to_ce.
    assert est.death_ce_earliest == ah_year_to_ce_range(11)[0]
    assert est.death_ce_latest == ah_year_to_ce_range(110)[1]


def test_estimate_every_known_generation_is_ordered_and_concrete() -> None:
    for generation in _GENERATION_DEATH_WINDOW_AH:
        est = estimate_death_window(generation)
        assert est is not None
        assert est.death_year_ah_earliest <= est.death_year_ah <= est.death_year_ah_latest
        assert est.precision is DatePrecision.TABAQA_ESTIMATE


def test_estimate_unknown_generation_yields_none() -> None:
    # No known ṭabaqa class → no fabricated window (contract #2).
    assert estimate_death_window(NarratorGeneration.UNKNOWN) is None


def test_estimate_dataclass_default_precision() -> None:
    est = TabaqaEstimate(150, 100, 200, 700, 800)
    assert est.precision is DatePrecision.TABAQA_ESTIMATE


# --------------------------------------------------------------------------- #
# Stage — apply_tabaqa_fallback over narrators_canonical.parquet
# --------------------------------------------------------------------------- #
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


def test_stage_fills_undated_known_tabaqa_narrator(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [
            {
                "canonical_id": "nar:undated-tabii",
                "name_ar": "راو مجهول التاريخ",
                "generation": "tabii",
                "death_date_precision": "unknown",
            }
        ],
    )

    path = apply_tabaqa_fallback(out)
    assert path is not None
    rec = pq.read_table(path).to_pylist()[0]
    assert rec["death_year_ah_earliest"] == 90
    assert rec["death_year_ah_latest"] == 180
    assert rec["death_year_ah"] == 135
    assert rec["death_date_precision"] == "tabaqa_estimate"


def test_stage_does_not_overwrite_reconciled_date(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [
            {
                "canonical_id": "nar:dated-thawri",
                "name_ar": "سفيان الثوري",
                "generation": "taba_tabii",
                "death_year_ah": 161,
                "death_year_ah_earliest": 161,
                "death_year_ah_latest": 161,
                "death_date_precision": "exact",
            }
        ],
    )

    path = apply_tabaqa_fallback(out)
    assert path is not None
    rec = pq.read_table(path).to_pylist()[0]
    # The attested EXACT dating is preserved untouched — no ṭabaqa overwrite.
    assert rec["death_year_ah"] == 161
    assert rec["death_year_ah_earliest"] == 161
    assert rec["death_year_ah_latest"] == 161
    assert rec["death_date_precision"] == "exact"


def test_stage_does_not_overwrite_open_after_bound(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [
            {
                "canonical_id": "nar:after-130",
                "generation": "tabii",
                # "died after 130 AH" → earliest set, latest None, precision AFTER.
                "death_year_ah_earliest": 130,
                "death_date_precision": "after",
            }
        ],
    )

    path = apply_tabaqa_fallback(out)
    assert path is not None
    rec = pq.read_table(path).to_pylist()[0]
    # A reconciled open bound is a real dating — the fallback leaves it alone.
    assert rec["death_year_ah_earliest"] == 130
    assert rec["death_year_ah_latest"] is None
    assert rec["death_date_precision"] == "after"


def test_stage_leaves_null_tabaqa_undated(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [
            {
                "canonical_id": "nar:no-tabaqa",
                "generation": None,
                "death_date_precision": "unknown",
            },
            {
                "canonical_id": "nar:unknown-tabaqa",
                "generation": "unknown",
                "death_date_precision": "unknown",
            },
        ],
    )

    path = apply_tabaqa_fallback(out)
    assert path is not None
    by_id = {r["canonical_id"]: r for r in pq.read_table(path).to_pylist()}
    for cid in ("nar:no-tabaqa", "nar:unknown-tabaqa"):
        rec = by_id[cid]
        assert rec["death_year_ah"] is None
        assert rec["death_year_ah_earliest"] is None
        assert rec["death_year_ah_latest"] is None
        assert rec["death_date_precision"] == "unknown"


def test_stage_does_not_touch_birth(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [{"canonical_id": "nar:b", "generation": "sahabi", "death_date_precision": "unknown"}],
    )
    rec = pq.read_table(apply_tabaqa_fallback(out)).to_pylist()[0]
    # Death-anchored fallback only: birth is left as the reconcile stage left it.
    assert rec["birth_year_ah"] is None
    assert rec["birth_year_ah_earliest"] is None
    assert rec["birth_year_ah_latest"] is None


def test_stage_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    _write_canonical(
        out,
        [{"canonical_id": "nar:idem", "generation": "later", "death_date_precision": "unknown"}],
    )

    apply_tabaqa_fallback(out)
    first = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    apply_tabaqa_fallback(out)
    second = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    assert first == second
    assert first[0]["death_date_precision"] == "tabaqa_estimate"


def test_stage_no_canonical_is_noop(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir()
    assert apply_tabaqa_fallback(out) is None
