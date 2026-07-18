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
import pytest

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from src.resolve import MissingInputError
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
_SHUBA = normalize_arabic("شعبة")
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
    # PR-3a: a fixture that *provably triggers* the same-label anchor tie-break (the old
    # fixture at 20/160/245 did NOT — 160→d:150 and 245→d:225 land in DISTINCT 25-year
    # buckets, so it passed trivially without exercising the branch). Two peeled bands
    # inside the wide LATER generation window (280-400) — d.310 and d.395 — BOTH resolve
    # to the single generation label ``d:280-400`` (its uniquely-matching sub-range
    # 301-400 is 99 AH wide, > SPLIT_BAND_GAP=80, so two distinct bands CAN share it).
    # A third, larger, earlier band (d.120) is the retained primary. Only the anchor-name
    # tie-break keeps the two colliding peeled ids distinct.
    datable = _dm(120, 30, "ret") + _dm(310, 15, "aaa") + _dm(395, 15, "bbb")
    plan = plan_split(_ABU_ABDALLAH, datable)
    assert plan.is_split
    assert len(plan.peeled) == 2
    # Both peeled bands collapsed to the SAME base generation label — the collision the
    # tie-break exists for — so each discriminator is label|anchor (contains "|").
    assert all(b.discriminator.startswith("d:280-400") for b in plan.peeled)
    assert all("|" in b.discriminator for b in plan.peeled), (
        "the same-label tie-break must fire (discriminator = label|anchor)"
    )
    # The tie-break appends the band's smallest anchor name, so the two ids are distinct.
    assert {b.discriminator for b in plan.peeled} == {"d:280-400|aaa", "d:280-400|bbb"}
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


def _mention_row(
    mention_id: str,
    hadith_id: str,
    position: int,
    cid: str,
    chain_index: int | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {f.name: None for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    base["mention_id"] = mention_id
    base["hadith_id"] = hadith_id
    base["source_corpus"] = "bukhari"
    base["position_in_chain"] = position
    base["canonical_narrator_id"] = cid
    # None coalesces to chain 0 (single-isnad hadith / legacy file), so callers that
    # do not care about multi-isnad grouping are unaffected.
    base["chain_index"] = chain_index
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


def _hadiths_candidate_teaches(
    candidate_id: str,
    anchor_id: str,
    count: int,
    start: int,
    prefix: str,
) -> list[dict[str, Any]]:
    """``count`` two-narrator chains: candidate at pos 0, anchor at pos 1 (student).

    The mirror of :func:`_hadiths_against_anchor` — here the attested anchor is the
    candidate's *student* (position +1), which dies a transmission gap *later*, so the
    candidate's per-mention estimate is ``anchor_death - MID_GAP``. Pairing both helpers
    for one hub gives it teacher-generation AND student-generation neighbours (a real
    multi-generation spread) that the ±MID_GAP correction must fold into one band.
    """
    rows: list[dict[str, Any]] = []
    for i in range(count):
        hid = f"{prefix}:h{start + i}"
        rows.append(_mention_row(f"{prefix}-c{start + i}", hid, 0, candidate_id))
        rows.append(_mention_row(f"{prefix}-a{start + i}", hid, 1, anchor_id))
    return rows


def _read_canonical(out: Path) -> dict[str, dict[str, Any]]:
    return {
        r["canonical_id"]: r for r in pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    }


def test_stage_no_canonical_raises_missing_input(tmp_path: Path) -> None:
    """da#361: an absent canonical table = an upstream defect (disambiguate/bio_promote
    did not run this pass), not a no-op. RED before da#361: returned ``None``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(MissingInputError) as excinfo:
        split_generic_narrators(tmp_path)
    assert excinfo.value.stage == "narrator_split"
    assert "narrators_canonical.parquet" in str(excinfo.value)


def test_stage_canonical_present_no_mentions_is_honest_noop(tmp_path: Path) -> None:
    """da#361 carve-out: canonical present but the NER mention output ABSENT is NOT a
    missing input — a bio-only corpus has a canonical table (from bio_promote) and no
    chain mentions to split, so the stage no-ops honestly (``None``), never raises."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    # Write only the canonical table; leave narrator_mentions_resolved.parquet absent.
    c = [_canonical_row("nar:x", normalize_arabic("محمد"), 3)]
    c_arrays = {f.name: [r[f.name] for r in c] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(
        pa.table(c_arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        out / "narrators_canonical.parquet",
    )
    assert split_generic_narrators(out) is None


def test_stage_empty_canonical_is_honest_noop(tmp_path: Path) -> None:
    """da#361 distinction: both inputs PRESENT but the canonical table is genuinely
    empty (zero records) — an honest no-op ``None``, NOT a raise."""
    out = tmp_path / "out"
    _write(out, canonical=[], mentions=[])
    assert split_generic_narrators(out) is None


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
    # da#340: peeled rows carry the first-class ISNAD_ESTIMATE precision, not the old
    # TABAQA_ESTIMATE overload — the window is an isnad-adjacency estimate.
    assert peeled[0]["death_date_precision"] == "isnad_estimate"
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


def test_stage_peel_re_derives_attestation(tmp_path: Path) -> None:
    """da#370: narrator_split mutates mention_count AFTER the last derivation site
    (fuzzy_cluster), so it must re-derive attestation on BOTH the reduced primary and
    each peeled band. Without the two re-derivations the primary keeps a stale tag and
    every peeled band loads as the loader's "unknown" default.

    The peeled band carries real isnad mentions -> isnad_attested. The primary's
    recorded count is set equal to the peeled total, so the ``max(0, ...)`` reduction
    lands it at 0 -> biographical_only — the branch a primary hits when peeling consumes
    its whole recorded count. The assertion is that each tag follows the FINAL count.

    Two equal bands of GENERIC_MIN_MENTIONS: the recorded count (50) both screens the
    name in as generic AND equals the single peeled band, so the residual clamps to 0."""
    out = tmp_path / "curated"
    cand = make_canonical_id(_ABU_ABDALLAH)
    early_death = 150 - MID_GAP
    late_death = 250 - MID_GAP
    canonical = [
        # Recorded count == the peeled (late) band, so the reduction clamps to 0.
        _canonical_row(cand, _ABU_ABDALLAH, 50),
        _anchor_row("nar:early", early_death),
        _anchor_row("nar:late", late_death),
    ]
    mentions = _hadiths_against_anchor(cand, "nar:early", 50, 0, "e")
    mentions += _hadiths_against_anchor(cand, "nar:late", 50, 0, "l")
    _write(out, canonical, mentions)

    assert split_generic_narrators(out) is not None

    by_id = _read_canonical(out)
    peeled = [
        r
        for r in by_id.values()
        if r["name_ar_normalized"] == _ABU_ABDALLAH and r["canonical_id"] != cand
    ]
    assert len(peeled) == 1
    # Site 1 — _peeled_record: a band with real isnad mentions is isnad_attested.
    assert peeled[0]["mention_count"] == 50
    assert peeled[0]["attestation"] == "isnad_attested"
    # Site 2 — apply loop: the primary reduced to 0 re-tags to biographical_only
    # (a stale carried-over "isnad_attested" would be wrong).
    assert by_id[cand]["mention_count"] == 0
    assert by_id[cand]["attestation"] == "biographical_only"


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

    # da#340: the peeled row carries ISNAD_ESTIMATE — and it is precisely because the
    # evidence gate excludes that precision (via _ESTIMATED_PRECISIONS) that the re-run
    # below is a strict no-op. If a peeled estimate re-seeded the attested pool, the
    # second run could re-split; idempotence and the precision change are one property.
    peeled_first = [
        r for r in first_c if r["name_ar_normalized"] == _ABU_ABDALLAH and r["canonical_id"] != cand
    ]
    assert len(peeled_first) == 1
    assert peeled_first[0]["death_date_precision"] == DatePrecision.ISNAD_ESTIMATE.value

    # Second run is a strict no-op (every node now a single band → abstain).
    assert split_generic_narrators(out) is None
    second_c = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    second_m = pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    assert first_c == second_c
    assert first_m == second_m


def _hub_multigen_mentions(
    cand: str,
    canonical: list[dict[str, Any]],
    teacher_deaths: list[int],
    student_deaths: list[int],
    per_anchor: int,
    prefix: str,
) -> list[dict[str, Any]]:
    """A hub with teacher-generation AND student-generation attested neighbours.

    ``teacher_deaths`` are earlier-dying neighbours (candidate is the student, +MID_GAP);
    ``student_deaths`` later-dying (candidate is the teacher, -MID_GAP). Appends the anchor
    rows to ``canonical`` and returns the chain mentions. The point is a genuine spread of
    attested neighbour death-years that the ±MID_GAP correction must fold back into ONE
    band — a delta-function of identical neighbours (the old fixture) never exercises it.
    """
    mentions: list[dict[str, Any]] = []
    for k, d in enumerate(teacher_deaths):
        aid = f"nar:{prefix}_t{k}"
        canonical.append(_anchor_row(aid, d))
        mentions += _hadiths_against_anchor(cand, aid, per_anchor, k * 1000, f"{prefix}t{k}")
    for k, d in enumerate(student_deaths):
        aid = f"nar:{prefix}_s{k}"
        canonical.append(_anchor_row(aid, d))
        mentions += _hadiths_candidate_teaches(cand, aid, per_anchor, k * 1000, f"{prefix}s{k}")
    return mentions


def test_stage_protected_hubs_multigen_spread_abstain(tmp_path: Path) -> None:
    # LOAD-BEARING (da#337 PR-3a): two real hubs — al-Zuhrī (d.124) and Shuʿba (d.160) —
    # each mentioned across chains whose ±1 neighbours span a *range* of attested death
    # years (a teacher generation dying decades earlier + a student generation dying
    # decades later). Raw neighbour deaths span ~110 AH, so the ±MID_GAP correction is
    # genuinely exercised: it must pull that spread back to ONE band per hub so each hub
    # ABSTAINS (stays whole). This is the property the old delta-function fixture (100
    # identical mentions at one death year) never tested — an abstain there was free.
    out = tmp_path / "curated"
    zuhri = make_canonical_id(_ZUHRI)
    shuba = make_canonical_id(_SHUBA)
    canonical: list[dict[str, Any]] = [
        _canonical_row(zuhri, _ZUHRI, 60),
        _canonical_row(shuba, _SHUBA, 60),
    ]
    # al-Zuhrī d.124: teachers d.70/85 → est 117/132; students d.165/180 → est 118/133.
    mentions = _hub_multigen_mentions(zuhri, canonical, [70, 85], [165, 180], 15, "z")
    # Shuʿba d.160: teachers d.105/120 → est 152/167; students d.200/215 → est 153/168.
    mentions += _hub_multigen_mentions(shuba, canonical, [105, 120], [200, 215], 15, "s")
    _write(out, canonical, mentions)

    # Both hubs screen IN as candidates (so the death-band gate really runs)…
    assert is_generic_name(_ZUHRI, 60) is True
    assert is_generic_name(_SHUBA, 60) is True
    # …and both abstain, so the stage is a whole-corpus no-op (no node split at all).
    assert split_generic_narrators(out) is None
    by_id = _read_canonical(out)
    for hub in (zuhri, shuba):
        assert by_id[hub]["mention_count"] == 60  # intact
        assert [
            r
            for r in by_id.values()
            if r["name_ar_normalized"] == by_id[hub]["name_ar_normalized"]
            and r["canonical_id"] != hub
        ] == []  # no same-name peel


def test_stage_hub_peels_small_spurious_band_keeps_ge_90pct(tmp_path: Path) -> None:
    # The case that actually proves the ≥90%-retention constraint the stage exists for:
    # a dominant band (92% of mentions) coexisting with a small, well-separated, still-
    # qualifying spurious band (8%) that peels off. The primary MUST retain ≥90% AND the
    # small band MUST split to its own node — an abstain (which keeps 100%) would NOT
    # exercise this, and a partition that shed >10% would violate the guard.
    out = tmp_path / "curated"
    cand = make_canonical_id(_ABU_ABDALLAH)
    total = 200
    small = 16  # 8% — ≥ SPLIT_MIN_SUPPORT (10) so it genuinely qualifies and peels
    dominant = total - small  # 184 = 92%
    canonical = [
        _canonical_row(cand, _ABU_ABDALLAH, total),
        _anchor_row("nar:dominant", 150 - MID_GAP),  # candidate est ≈ 150
        _anchor_row("nar:spurious", 260 - MID_GAP),  # candidate est ≈ 260 (>80 away)
    ]
    mentions = _hadiths_against_anchor(cand, "nar:dominant", dominant, 0, "dom")
    mentions += _hadiths_against_anchor(cand, "nar:spurious", small, 0, "spur")
    _write(out, canonical, mentions)

    assert split_generic_narrators(out) is not None
    by_id = _read_canonical(out)
    # Primary kept its bare id and ≥90% of its mentions (184/200 = 92%).
    retained = by_id[cand]["mention_count"]
    assert retained == dominant
    assert retained >= 0.90 * total
    # The 8% spurious band peeled to its own same-name node with exactly `small` mentions.
    peeled = [
        r
        for r in by_id.values()
        if r["name_ar_normalized"] == _ABU_ABDALLAH and r["canonical_id"] != cand
    ]
    assert len(peeled) == 1
    assert peeled[0]["mention_count"] == small
    assert peeled[0]["death_year_ah"] == 260
    assert peeled[0]["death_date_precision"] == DatePrecision.ISNAD_ESTIMATE.value


# --------------------------------------------------------------------------- #
# da#411 — multi-isnad chain keying (no cross-chain ±1 neighbours)
# --------------------------------------------------------------------------- #
def test_multi_isnad_no_cross_chain_neighbour(tmp_path: Path) -> None:
    """da#411: a multi-isnad hadith must keep its isnads separate for the ±1 lookup.

    One ``hadith_id`` carries two isnads (``chain_index`` 0 and 1), each numbered from
    position 0 with distinct narrators; the split candidate sits at position 1 in BOTH.
    Keying the splitter's chain index on ``hadith_id`` alone (the pre-da#411 bug)
    flattens the two isnads into one position-sorted list, so the candidate's pos±1
    neighbour lookup picks up the *other* isnad's narrators — teacher/student
    adjacencies that exist in no real chain and that corrupt the death-band evidence.
    Grouping on the composite ``(hadith_id, chain_index)`` key the graph loaders use
    (``src.graph.load_edges._chain_index``, da#282) confines each mention's neighbours
    to its own chain.

    RED on the old hadith_id-only code (each mention's anchors leak across chains);
    GREEN after the composite-key fix. Exercises ``_build_chain_index`` (the keying)
    and ``_collect_datable`` (the neighbour window) directly.
    """
    from src.resolve.narrator_split import _build_chain_index, _collect_datable

    cand = make_canonical_id(_ABU_ABDALLAH)
    # Distinct attested anchors per isnad of the SAME hadith. Chain 0: teacher d.100
    # (pos 0) + student d.170 (pos 2). Chain 1: teacher d.250 (pos 0) + student d.320
    # (pos 2). Their death years are far apart so a cross-chain leak is unmistakable.
    ta0 = _anchor_row("nar:t_a0", 100)
    sa0 = _anchor_row("nar:s_a0", 170)
    tb1 = _anchor_row("nar:t_b1", 250)
    sb1 = _anchor_row("nar:s_b1", 320)
    canonical = [_canonical_row(cand, _ABU_ABDALLAH, 2), ta0, sa0, tb1, sb1]
    mentions = [
        # isnad 0
        _mention_row("m-t-a0", "h1", 0, "nar:t_a0", chain_index=0),
        _mention_row("m-c-0", "h1", 1, cand, chain_index=0),
        _mention_row("m-s-a0", "h1", 2, "nar:s_a0", chain_index=0),
        # isnad 1 — same hadith_id, different chain_index
        _mention_row("m-t-b1", "h1", 0, "nar:t_b1", chain_index=1),
        _mention_row("m-c-1", "h1", 1, cand, chain_index=1),
        _mention_row("m-s-b1", "h1", 2, "nar:s_b1", chain_index=1),
    ]
    out = tmp_path / "curated"
    _write(out, canonical, mentions)

    attested_by_id = {r["canonical_id"]: r["death_year_ah"] for r in (ta0, sa0, tb1, sb1)}
    name_by_id = {r["canonical_id"]: r["name_ar_normalized"] for r in canonical}

    chains, candidate_mentions = _build_chain_index(
        out / "narrator_mentions_resolved.parquet", {cand}
    )
    datable = _collect_datable(candidate_mentions[cand], chains, attested_by_id, name_by_id)

    by_mention = {d.mention_id: d for d in datable}
    chain0_anchors = {name_by_id["nar:t_a0"], name_by_id["nar:s_a0"]}
    chain1_anchors = {name_by_id["nar:t_b1"], name_by_id["nar:s_b1"]}

    # Each mention draws its ±1 anchors ONLY from its own isnad…
    assert set(by_mention["m-c-0"].anchor_names) == chain0_anchors
    assert set(by_mention["m-c-1"].anchor_names) == chain1_anchors
    # …with no cross-chain leak in either direction (the load-bearing assertion).
    assert not (set(by_mention["m-c-0"].anchor_names) & chain1_anchors)
    assert not (set(by_mention["m-c-1"].anchor_names) & chain0_anchors)


def test_gappy_chain_datable_across_dropped_position(tmp_path: Path) -> None:
    """da#439: a position gap (from a dropped unresolved mention) must NOT hide a neighbour.

    Chain positions 0..4, but positions 1 and 3 are **unresolved** (null
    ``canonical_narrator_id``) and so are dropped from ``chains`` in
    ``_build_chain_index``. The candidate sits at position 2; its only resolved
    neighbours are the teacher at position 0 and the student at position 4 — each two
    positions away.

    The loader (``load_edges._build_chain_pairs``) pairs *consecutive resolved*
    mentions, so it emits ``teacher -> candidate`` and ``candidate -> student`` across
    those gaps. The old exact-``position ± 1`` window looked only at positions 1 and 3
    (both dropped), found nothing, and left the candidate **undatable** — a systematic
    under-count relative to the loaded graph (RED). Reading the immediate resolved
    neighbours in the position-sorted chain recovers both adjacencies (GREEN).
    """
    from src.resolve.narrator_split import _build_chain_index, _collect_datable

    cand = make_canonical_id(_ABU_ABDALLAH)
    teacher = _anchor_row("nar:teacher", 100)  # pos 0 → dies earlier → sign +1
    student = _anchor_row("nar:student", 180)  # pos 4 → dies later → sign −1
    canonical = [_canonical_row(cand, _ABU_ABDALLAH, 1), teacher, student]
    mentions = [
        _mention_row("m-teacher", "h1", 0, "nar:teacher", chain_index=0),
        _mention_row("m-gap1", "h1", 1, None, chain_index=0),  # unresolved → dropped
        _mention_row("m-cand", "h1", 2, cand, chain_index=0),
        _mention_row("m-gap3", "h1", 3, None, chain_index=0),  # unresolved → dropped
        _mention_row("m-student", "h1", 4, "nar:student", chain_index=0),
    ]
    out = tmp_path / "curated"
    _write(out, canonical, mentions)

    attested_by_id = {r["canonical_id"]: r["death_year_ah"] for r in (teacher, student)}
    name_by_id = {r["canonical_id"]: r["name_ar_normalized"] for r in canonical}

    chains, candidate_mentions = _build_chain_index(
        out / "narrator_mentions_resolved.parquet", {cand}
    )
    datable = _collect_datable(candidate_mentions[cand], chains, attested_by_id, name_by_id)

    # The candidate is datable ACROSS the gaps — the exact-±1 window would have left
    # `datable` empty (the divergence da#439 fixes).
    assert len(datable) == 1
    dm = datable[0]
    assert set(dm.anchor_names) == {
        name_by_id["nar:teacher"],
        name_by_id["nar:student"],
    }
    # Estimate folds both neighbours with the loader-consistent sign convention:
    # teacher (earlier position) + MID_GAP, student (later position) − MID_GAP.
    assert dm.estimate == round(((100 + MID_GAP) + (180 - MID_GAP)) / 2)


def test_split_adjacency_matches_loader_transmitted_to(tmp_path: Path) -> None:
    """da#439: the split's neighbour set == the loader's TRANSMITTED_TO edges (same chain).

    The reconciliation this issue asks for is *agreement* between two definitions of
    "adjacent". This test runs BOTH on the same gappy fixture and asserts the candidate's
    splitter neighbours are exactly the narrators the loader connects it to — the
    cross-function invariant, not just a splitter-internal fixture.
    """
    from src.graph.load_edges import _build_chain_pairs
    from src.resolve.narrator_split import _build_chain_index, _collect_datable

    cand = make_canonical_id(_ABU_ABDALLAH)
    teacher = _anchor_row("nar:teacher", 100)
    student = _anchor_row("nar:student", 180)
    canonical = [_canonical_row(cand, _ABU_ABDALLAH, 1), teacher, student]
    # Same single (hadith_id, chain_index) chain with two interior gaps.
    mention_rows = [
        _mention_row("m-teacher", "hdt:bukhari:1", 0, "nar:teacher", chain_index=0),
        _mention_row("m-gap1", "hdt:bukhari:1", 1, None, chain_index=0),
        _mention_row("m-cand", "hdt:bukhari:1", 2, cand, chain_index=0),
        _mention_row("m-gap3", "hdt:bukhari:1", 3, None, chain_index=0),
        _mention_row("m-student", "hdt:bukhari:1", 4, "nar:student", chain_index=0),
    ]
    out = tmp_path / "curated"
    _write(out, canonical, mention_rows)

    # Loader side: TRANSMITTED_TO edges incident to the candidate → the other endpoints.
    loader_pairs = _build_chain_pairs(mention_rows)
    loader_neighbours = {p["to_id"] for p in loader_pairs if p["from_id"] == cand} | {
        p["from_id"] for p in loader_pairs if p["to_id"] == cand
    }
    assert loader_neighbours == {"nar:teacher", "nar:student"}

    # Splitter side: the anchors it derived for the candidate, mapped back to ids.
    attested_by_id = {r["canonical_id"]: r["death_year_ah"] for r in (teacher, student)}
    name_by_id = {r["canonical_id"]: r["name_ar_normalized"] for r in canonical}
    id_by_name = {v: k for k, v in name_by_id.items()}
    chains, candidate_mentions = _build_chain_index(
        out / "narrator_mentions_resolved.parquet", {cand}
    )
    datable = _collect_datable(candidate_mentions[cand], chains, attested_by_id, name_by_id)
    split_neighbours = {id_by_name[n] for d in datable for n in d.anchor_names}

    # The two notions of adjacency AGREE (the da#439 invariant).
    assert split_neighbours == loader_neighbours
