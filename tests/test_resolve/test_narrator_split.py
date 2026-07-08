"""Tests for src.resolve.narrator_split — da#337 same-name (generic-name) split.

The load-bearing property is the **over-split guard** (Ivana's PR#338 constraint):
the recall-first ``is_generic_name`` screen admits genuinely-single people
(Sufyān al-Thawrī, al-Zuhrī), and the split MUST gate on independent death-band
evidence — ≥2 well-separated, well-supported bands — never on the screen alone. A
single-band name abstains and its node is left whole. Coverage:

* pure ``plan_split`` band logic (abstain vs peel, every gate);
* the ``سفيان الثوري`` case Ivana flagged — screens IN, death-band gate ABSTAINS;
* end-to-end stage over parquet fixtures — peel-not-partition, undatable remainder
  stays, registered-mononym deferral, remap correctness, idempotence, and a
  protected-hub regression (dominant node keeps ≥90% of its mentions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from src.resolve.generic_name import is_generic_name
from src.resolve.narrator_split import (
    MID_GAP,
    SPLIT_MAX_CLUSTERS,
    SPLIT_MIN_SUPPORT,
    DatableMention,
    plan_split,
    split_generic_narrators,
)
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic

_ABU_ABDALLAH = normalize_arabic("أبو عبد الله")
_ZUHRI = normalize_arabic("الزهري")
_THAWRI = normalize_arabic("سفيان الثوري")
_SUFYAN = normalize_arabic("سفيان")


# --------------------------------------------------------------------------- #
# plan_split — pure band logic
# --------------------------------------------------------------------------- #
def _dm(estimate: int, n: int, anchor: str = "anchor") -> list[DatableMention]:
    """``n`` datable mentions at one death estimate (distinct mention ids)."""
    return [
        DatableMention(
            mention_id=f"{anchor}-{estimate}-{i}", estimate=estimate, anchor_names=(anchor,)
        )
        for i in range(n)
    ]


def test_single_band_abstains() -> None:
    # All mentions in one death band → one cluster → no split (the al-Zuhrī case).
    plan = plan_split(_ZUHRI, _dm(124, 40))
    assert not plan.is_split
    assert plan.peeled == ()


def test_thawri_screens_in_but_death_band_gate_abstains() -> None:
    # The exact case Ivana flagged: سفيان الثوري IS admitted by the recall-first
    # screen, yet clusters to a single death band (d.161) so the split ABSTAINS.
    assert is_generic_name(_THAWRI, 5000) is True
    plan = plan_split(_THAWRI, _dm(161, 200))
    assert not plan.is_split


def test_two_separated_supported_bands_split() -> None:
    # Two ~100y-apart, well-supported bands → primary keeps the larger; the other peels.
    datable = _dm(150, 20, "early") + _dm(250, 12, "late")
    plan = plan_split(_ABU_ABDALLAH, datable)
    assert plan.is_split
    assert len(plan.peeled) == 1
    band = plan.peeled[0]
    # The larger (150) band is retained under the bare id; the 250 band peels.
    assert band.midpoint_ah == 250
    assert band.mention_count == 12
    assert band.new_id == make_discriminated_canonical_id(_ABU_ABDALLAH, band.discriminator)
    assert band.new_id != make_canonical_id(_ABU_ABDALLAH)
    # Only the peeled band's mentions are remapped; the retained band's are not.
    remap = plan.remap()
    assert len(remap) == 12
    assert all(v == band.new_id for v in remap.values())


def test_below_support_band_not_peeled() -> None:
    # A big band + a below-SPLIT_MIN_SUPPORT band → only one qualifying → abstain.
    datable = _dm(150, 30) + _dm(250, SPLIT_MIN_SUPPORT - 1, "tiny")
    plan = plan_split(_ABU_ABDALLAH, datable)
    assert not plan.is_split


def test_too_many_bands_is_noise_abstain() -> None:
    # More than SPLIT_MAX_CLUSTERS well-supported bands = generic bucket → abstain.
    datable: list[DatableMention] = []
    for k in range(SPLIT_MAX_CLUSTERS + 1):
        datable += _dm(120 + k * 100, SPLIT_MIN_SUPPORT + 2, f"b{k}")
    plan = plan_split(_ABU_ABDALLAH, datable)
    assert not plan.is_split


def test_within_band_gap_estimates_stay_one_band() -> None:
    # Estimates 40y apart are <= SPLIT_BAND_GAP (80) so they do NOT cut a new band —
    # ordinary within-person estimate scatter must not fragment a node.
    datable = _dm(200, 15) + _dm(240, 15, "b2")
    assert not plan_split(_ABU_ABDALLAH, datable).is_split


def test_same_band_label_collision_tiebreak_by_anchor() -> None:
    # Two peeled bands both inside the TABA_TABII window (single-match label d:150-250)
    # get distinct ids via the anchor-name tie-break. Use 3 bands so two share a label.
    datable = _dm(20, 15, "aaa") + _dm(160, 15, "zzz") + _dm(245, 15, "mmm")
    plan = plan_split(_ABU_ABDALLAH, datable)
    assert plan.is_split
    ids = {b.new_id for b in plan.peeled}
    assert len(ids) == len(plan.peeled)  # every peeled id distinct


# --------------------------------------------------------------------------- #
# Stage — split_generic_narrators over parquet fixtures
# --------------------------------------------------------------------------- #
def _canonical_row(cid: str, name_norm: str, mention_count: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    base["canonical_id"] = cid
    base["name_ar"] = name_norm
    base["name_ar_normalized"] = name_norm
    base["mention_count"] = mention_count
    base["death_date_precision"] = "unknown"
    base["birth_date_precision"] = "unknown"
    base.update(over)
    return base


def _anchor_row(cid: str, death: int) -> dict[str, Any]:
    # A fully-specified (3-token) attested anchor — never a split candidate itself.
    return _canonical_row(
        cid,
        normalize_arabic(f"راو مؤرخ رقم {death}"),
        5,
        death_year_ah=death,
        death_date_precision="exact",
    )


def _mention_row(mention_id: str, hadith_id: str, position: int, cid: str) -> dict[str, Any]:
    base: dict[str, Any] = {f.name: None for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    base["mention_id"] = mention_id
    base["hadith_id"] = hadith_id
    base["source_corpus"] = "bukhari"
    base["position_in_chain"] = position
    base["canonical_narrator_id"] = cid
    return base


def _write(out: Path, canonical: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    c_arrays = {f.name: [r[f.name] for r in canonical] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(
        pa.table(c_arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        out / "narrators_canonical.parquet",
    )
    m_arrays = {f.name: [r[f.name] for r in mentions] for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    pq.write_table(
        pa.table(m_arrays, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA),
        out / "narrator_mentions_resolved.parquet",
    )


def _hadiths_against_anchor(
    candidate_id: str,
    anchor_id: str,
    count: int,
    start: int,
    prefix: str,
) -> list[dict[str, Any]]:
    """``count`` two-narrator chains: anchor at pos 0 (teacher), candidate at pos 1."""
    rows: list[dict[str, Any]] = []
    for i in range(count):
        hid = f"{prefix}:h{start + i}"
        rows.append(_mention_row(f"{prefix}-a{start + i}", hid, 0, anchor_id))
        rows.append(_mention_row(f"{prefix}-c{start + i}", hid, 1, candidate_id))
    return rows


def _read_canonical(out: Path) -> dict[str, dict[str, Any]]:
    return {
        r["canonical_id"]: r for r in pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    }


def test_stage_no_canonical_is_noop(tmp_path: Path) -> None:
    assert split_generic_narrators(tmp_path) is None


def test_stage_splits_two_bands_and_peels(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    cand = make_canonical_id(_ABU_ABDALLAH)
    early_death = 150 - MID_GAP  # candidate estimate ≈ 150 (teacher + MID_GAP)
    late_death = 250 - MID_GAP  # candidate estimate ≈ 250
    canonical = [
        _canonical_row(cand, _ABU_ABDALLAH, 54),
        _anchor_row("nar:early", early_death),
        _anchor_row("nar:late", late_death),
    ]
    mentions = _hadiths_against_anchor(cand, "nar:early", 40, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 14, 0, "l")
    _write(out, canonical, mentions)

    result = split_generic_narrators(out)
    assert result is not None

    by_id = _read_canonical(out)
    # Primary retained the larger (early) band + kept its bare id; count reduced by 14.
    assert by_id[cand]["mention_count"] == 40
    # Exactly one peeled node, carrying the same normalized name, ~250 window.
    peeled = [
        r
        for r in by_id.values()
        if r["name_ar_normalized"] == _ABU_ABDALLAH and r["canonical_id"] != cand
    ]
    assert len(peeled) == 1
    assert peeled[0]["death_year_ah"] == 250
    assert peeled[0]["death_date_precision"] == "tabaqa_estimate"
    assert peeled[0]["mention_count"] == 14

    # Mentions of the late band are remapped to the peeled id; early band unchanged.
    m = {
        r["mention_id"]: r
        for r in pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    }
    assert m["l-c0"]["canonical_narrator_id"] == peeled[0]["canonical_id"]
    assert m["e-c0"]["canonical_narrator_id"] == cand

    # Audit report has one row per peel.
    audit = pq.read_table(out / "narrator_splits.parquet").to_pylist()
    assert len(audit) == 1
    assert audit[0]["primary_id"] == cand
    assert audit[0]["new_id"] == peeled[0]["canonical_id"]


def test_stage_undatable_remainder_stays_on_primary(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    cand = make_canonical_id(_ABU_ABDALLAH)
    canonical = [
        _canonical_row(cand, _ABU_ABDALLAH, 54),
        _anchor_row("nar:early", 150 - MID_GAP),
        _anchor_row("nar:late", 250 - MID_GAP),
        # An undated anchor (precision unknown, no year) → its neighbours are undatable.
        _canonical_row("nar:undated", normalize_arabic("راو غير مؤرخ اطلاقا"), 5),
    ]
    mentions = _hadiths_against_anchor(cand, "nar:early", 20, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 20, 0, "l")
    mentions += _hadiths_against_anchor(cand, "nar:undated", 10, 0, "u")
    _write(out, canonical, mentions)

    assert split_generic_narrators(out) is not None
    m = {
        r["mention_id"]: r
        for r in pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    }
    # The undatable mentions keep the primary id (peel-not-partition).
    assert m["u-c0"]["canonical_narrator_id"] == cand
    assert m["u-c9"]["canonical_narrator_id"] == cand


def test_stage_registered_mononym_is_deferred(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    # Bare سفيان is owned by mononym_split → this stage never splits it, even with
    # two clean death bands present.
    cand = make_canonical_id(_SUFYAN)
    canonical = [
        _canonical_row(cand, _SUFYAN, 44),
        _anchor_row("nar:early", 150 - MID_GAP),
        _anchor_row("nar:late", 250 - MID_GAP),
    ]
    mentions = _hadiths_against_anchor(cand, "nar:early", 22, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 22, 0, "l")
    _write(out, canonical, mentions)

    # No candidates (سفيان is registered) → no-op.
    assert split_generic_narrators(out) is None


def test_stage_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    cand = make_canonical_id(_ABU_ABDALLAH)
    canonical = [
        _canonical_row(cand, _ABU_ABDALLAH, 54),
        _anchor_row("nar:early", 150 - MID_GAP),
        _anchor_row("nar:late", 250 - MID_GAP),
    ]
    mentions = _hadiths_against_anchor(cand, "nar:early", 40, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 14, 0, "l")
    _write(out, canonical, mentions)

    assert split_generic_narrators(out) is not None
    first_c = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    first_m = pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()

    # Second run is a strict no-op (every node now a single band → abstain).
    assert split_generic_narrators(out) is None
    second_c = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    second_m = pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    assert first_c == second_c
    assert first_m == second_m


def test_stage_protected_hub_keeps_dominant_node(tmp_path: Path) -> None:
    # A dominant single-band hub (al-Zuhrī) coexisting with a real split must end as
    # ONE node retaining ≥90% of its mentions (here 100% — single band abstains).
    out = tmp_path / "curated"
    zuhri = make_canonical_id(_ZUHRI)
    cand = make_canonical_id(_ABU_ABDALLAH)
    canonical = [
        _canonical_row(zuhri, _ZUHRI, 100),
        _canonical_row(cand, _ABU_ABDALLAH, 54),
        _anchor_row("nar:early", 150 - MID_GAP),
        _anchor_row("nar:late", 250 - MID_GAP),
        _anchor_row("nar:zuhri_anchor", 124 - MID_GAP),
    ]
    mentions = _hadiths_against_anchor(zuhri, "nar:zuhri_anchor", 100, 0, "z")
    mentions += _hadiths_against_anchor(cand, "nar:early", 40, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 14, 0, "l")
    _write(out, canonical, mentions)

    assert split_generic_narrators(out) is not None
    by_id = _read_canonical(out)
    # al-Zuhrī stayed whole: still one node, mention_count intact, no same-name peel.
    assert by_id[zuhri]["mention_count"] == 100
    zuhri_peels = [
        r
        for r in by_id.values()
        if r["name_ar_normalized"] == _ZUHRI and r["canonical_id"] != zuhri
    ]
    assert zuhri_peels == []
