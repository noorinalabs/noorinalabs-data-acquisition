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
    DEATH_PLAUSIBILITY_MARGIN_AH,
    NARRATOR_DEATH_MAX_AH,
    NARRATOR_DEATH_MIN_AH,
    TabaqaEstimate,
    apply_tabaqa_fallback,
    clamp_death_ah,
    estimate_death_window,
    generation_from_value,
    is_plausible_narrator_death_ah,
    plausible_attested_death_year,
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
# da#446 — isnad-narrator death-year plausibility envelope + clamp
# --------------------------------------------------------------------------- #
def test_plausibility_envelope_derives_from_generation_windows() -> None:
    # The envelope is DERIVED from _GENERATION_DEATH_WINDOW_AH so the two cannot drift:
    # floor = the Prophet's death (min generation low, no downward margin); ceiling =
    # the last window's high + margin.
    assert NARRATOR_DEATH_MIN_AH == min(lo for lo, _ in _GENERATION_DEATH_WINDOW_AH.values())
    assert NARRATOR_DEATH_MIN_AH == 11
    assert NARRATOR_DEATH_MAX_AH == (
        max(hi for _, hi in _GENERATION_DEATH_WINDOW_AH.values()) + DEATH_PLAUSIBILITY_MARGIN_AH
    )
    assert NARRATOR_DEATH_MAX_AH == 450


def test_plausible_absolute_envelope_no_generation() -> None:
    # In-envelope attested deaths pass; the boundaries are inclusive.
    assert is_plausible_narrator_death_ah(11) is True
    assert is_plausible_narrator_death_ah(256) is True
    assert is_plausible_narrator_death_ah(450) is True
    # The da#446 pollution — late collectors/commentators d. ~700–780 AH, and the
    # observed prod max of 859 AH — is impossible for an isnad narrator.
    for polluted in (451, 754, 780, 805, 859):
        assert is_plausible_narrator_death_ah(polluted) is False
    # Sub-11 (before the Prophet's death) and negative axis artifacts are impossible.
    for early in (-44, 0, 10):
        assert is_plausible_narrator_death_ah(early) is False


def test_plausible_generation_aware_tightening() -> None:
    # A KNOWN generation tightens to its own window ± margin: a Companion (11–110)
    # dated 300 AH sits inside the loose [11, 450] envelope but is impossible for a
    # sahabi, so the generation-aware check rejects it.
    assert is_plausible_narrator_death_ah(300, NarratorGeneration.SAHABI) is False
    # …while a death just past the rounded window edge, within the margin, is kept
    # (110 + 50 = 160) — the margin protects genuine boundary narrators.
    assert is_plausible_narrator_death_ah(160, NarratorGeneration.SAHABI) is True
    assert is_plausible_narrator_death_ah(161, NarratorGeneration.SAHABI) is False
    # A genuine LATER narrator (280–400) at 390 stays plausible; an UNKNOWN generation
    # falls back to the absolute envelope only.
    assert is_plausible_narrator_death_ah(390, NarratorGeneration.LATER) is True
    assert is_plausible_narrator_death_ah(300, NarratorGeneration.UNKNOWN) is True


def test_clamp_death_ah_bounds_axis_estimates() -> None:
    # The −44 / 805 AH the issue reports clamp back into the envelope; an in-range
    # value is returned unchanged.
    assert clamp_death_ah(-44) == NARRATOR_DEATH_MIN_AH == 11
    assert clamp_death_ah(-36) == 11  # student-of-Companion boundary underflow
    assert clamp_death_ah(805) == NARRATOR_DEATH_MAX_AH == 450
    assert clamp_death_ah(497) == 450  # student-of-latest-plausible boundary overflow
    assert clamp_death_ah(256) == 256


# --------------------------------------------------------------------------- #
# da#454 — plausible_attested_death_year, the stamp-time companion to the
# da#446 output-side scrub: bio_promote/disambiguate call this so an implausible
# attested year never enters narrators_canonical or the candidate pool.
# --------------------------------------------------------------------------- #
def test_plausible_attested_death_year_passes_through_valid_year() -> None:
    assert plausible_attested_death_year(256) == 256
    assert plausible_attested_death_year(160, "sahabi") == 160


def test_plausible_attested_death_year_none_is_none() -> None:
    # A bio with no attested death stays None — never fabricated.
    assert plausible_attested_death_year(None) is None
    assert plausible_attested_death_year(None, "sahabi") is None


def test_plausible_attested_death_year_scrubs_absolute_envelope_violation() -> None:
    assert plausible_attested_death_year(805) is None
    assert plausible_attested_death_year(754, "unknown") is None


def test_plausible_attested_death_year_scrubs_generation_mismatch() -> None:
    # 300 AH sits inside the loose absolute envelope but is impossible for a
    # Companion (sahabi) — matches is_plausible_narrator_death_ah's own gate.
    assert plausible_attested_death_year(300, "sahabi") is None
    assert plausible_attested_death_year(300, "later") == 300


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
