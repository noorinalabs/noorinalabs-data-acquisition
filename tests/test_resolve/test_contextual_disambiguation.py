"""Tests for src.resolve.contextual_disambiguation — da#346 isnad-neighbour peel.

The load-bearing property (feedback_silent_zero_is_not_a_measurement /
feedback_drop_gate_bidirectional_ab): the peel is driven by a hand-verified signature
seed, and the instrument MUST SEPARATE the classes — a signature fires on the genuine
multi-referent mentions it names (bare ʿAbd Allāh ⟵Prophet ⟶Nāfiʿ ⇒ Ibn ʿUmar) and does
NOT fire on the residual (a different ʿAbd Allāh) or on a partial-match (only one role).
An unknown or ambiguous (≥2-signature) neighbour pair is never peeled; it stays on the
bare primary, which over_merged_flag keeps flagged. Coverage:

* signature loading + validation (an unconstrained signature is a hard error);
* ``ContextualSignature.matches`` / ``match_signature`` role logic + ambiguity guard;
* ``neighbour_pair_coverage`` blast-radius tally;
* end-to-end stage over parquet fixtures — instrument separation (peel the matched class,
  leave the control class AND the partial-match), residual stays, primary count reduced,
  coverage report emitted, empty-seed / absent-primary no-op, and idempotence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from src.resolve.contextual_disambiguation import (
    ContextualSignature,
    apply_contextual_disambiguation,
    load_contextual_signatures,
    match_signature,
    neighbour_pair_coverage,
)
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic

_ABDALLAH = normalize_arabic("عبد الله")
_PROPHET = normalize_arabic("رسول الله")
_NAFI = normalize_arabic("نافع")
_ZUHRI = normalize_arabic("الزهري")
_MALIK = normalize_arabic("مالك")

_IBN_UMAR_SIG = ContextualSignature(
    name_ar="عبد الله",
    teacher_ar="رسول الله",
    student_ar="نافع",
    discriminator="ctx:ibn-umar",
    name_en="Abd Allah ibn Umar",
    note="Prophet -> Ibn Umar -> Nafi textbook isnad; hand-verified.",
)


# --------------------------------------------------------------------------- #
# Pure signature / match / coverage logic
# --------------------------------------------------------------------------- #
def test_signature_matches_requires_every_constrained_role() -> None:
    both = _IBN_UMAR_SIG
    assert both.matches(frozenset({_PROPHET}), frozenset({_NAFI})) is True
    # Partial: teacher matches but student does not -> no match (both roles required).
    assert both.matches(frozenset({_PROPHET}), frozenset({_MALIK})) is False
    assert both.matches(frozenset({_ZUHRI}), frozenset({_NAFI})) is False
    # Empty neighbour context never matches a constrained signature.
    assert both.matches(frozenset(), frozenset()) is False


def test_signature_single_role_matches_on_that_role_only() -> None:
    teacher_only = ContextualSignature(
        name_ar="عبد الله",
        teacher_ar="رسول الله",
        student_ar=None,
        discriminator="ctx:t",
        name_en=None,
        note="n",
    )
    assert teacher_only.matches(frozenset({_PROPHET}), frozenset()) is True
    assert teacher_only.matches(frozenset({_ZUHRI}), frozenset({_PROPHET})) is False


def test_match_signature_zero_and_ambiguous_return_none() -> None:
    teacher_only = ContextualSignature(
        name_ar="عبد الله",
        teacher_ar="رسول الله",
        student_ar=None,
        discriminator="ctx:t",
        name_en=None,
        note="n",
    )
    student_only = ContextualSignature(
        name_ar="عبد الله",
        teacher_ar=None,
        student_ar="نافع",
        discriminator="ctx:s",
        name_en=None,
        note="n",
    )
    sigs = [teacher_only, student_only]
    # Exactly one matches -> that one.
    assert match_signature(frozenset({_PROPHET}), frozenset({_MALIK}), sigs) is teacher_only
    # Zero match -> None.
    assert match_signature(frozenset({_ZUHRI}), frozenset({_MALIK}), sigs) is None
    # BOTH match (ambiguous) -> None (never guess between curated referents).
    assert match_signature(frozenset({_PROPHET}), frozenset({_NAFI}), sigs) is None


def test_neighbour_pair_coverage_tally() -> None:
    neighbours = {
        "m1": (frozenset({_PROPHET}), frozenset({_NAFI})),  # matches Ibn Umar
        "m2": (frozenset({_PROPHET}), frozenset({_MALIK})),  # partial, no match
        "m3": (frozenset(), frozenset()),  # no neighbour at all
        "m4": (frozenset({_ZUHRI}), frozenset()),  # teacher only, no match
    }
    cov = neighbour_pair_coverage(
        _ABDALLAH, make_canonical_id(_ABDALLAH), neighbours, [_IBN_UMAR_SIG]
    )
    assert cov.total_mentions == 4
    assert cov.with_any_neighbour == 3
    assert cov.with_teacher == 3
    assert cov.with_student == 2
    assert cov.signature_matched == 1
    assert cov.ambiguous_multimatch == 0


def test_load_signatures_validation(tmp_path: Path) -> None:
    assert load_contextual_signatures(tmp_path / "absent.yaml") == ()
    good = tmp_path / "good.yaml"
    good.write_text(
        "signatures:\n"
        '  - name_ar: "عبد الله"\n'
        '    teacher_ar: "رسول الله"\n'
        '    student_ar: "نافع"\n'
        '    discriminator: "ctx:ibn-umar"\n'
        '    note: "hand-verified"\n',
        encoding="utf-8",
    )
    sigs = load_contextual_signatures(good)
    assert len(sigs) == 1 and sigs[0].discriminator == "ctx:ibn-umar"

    unconstrained = tmp_path / "bad.yaml"
    unconstrained.write_text(
        'signatures:\n  - name_ar: "عبد الله"\n    discriminator: "d"\n    note: "n"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="teacher_ar and/or student_ar"):
        load_contextual_signatures(unconstrained)


def test_ships_seed_is_loadable_and_bidirectionally_correct() -> None:
    """The shipped seed loads, and its Ibn ʿUmar signature fires on the golden chain
    while NOT firing on a genuine other-ʿAbd-Allāh pair (instrument separation)."""
    sigs = load_contextual_signatures()
    assert len(sigs) >= 1
    ibn_umar = next(s for s in sigs if s.discriminator == "ctx:ibn-umar")
    # Fires on Prophet -> Nafi (the multi-referent class it names)…
    assert ibn_umar.matches(frozenset({_PROPHET}), frozenset({_NAFI})) is True
    # …and does NOT fire on a different pair (the control class stays whole).
    assert ibn_umar.matches(frozenset({_ZUHRI}), frozenset({_MALIK})) is False


# --------------------------------------------------------------------------- #
# Stage — apply_contextual_disambiguation over parquet fixtures
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


def _mention_row(
    mention_id: str, hadith_id: str, position: int, cid: str, chain_index: int = 0
) -> dict[str, Any]:
    base: dict[str, Any] = {f.name: None for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    base["mention_id"] = mention_id
    base["hadith_id"] = hadith_id
    base["source_corpus"] = "bukhari"
    base["position_in_chain"] = position
    base["canonical_narrator_id"] = cid
    base["chain_index"] = chain_index
    return base


def _chain(
    hid: str, teacher_id: str, cand_id: str, student_id: str, tag: str
) -> list[dict[str, Any]]:
    """A 3-narrator isnad: teacher (pos0) -> candidate (pos1) -> student (pos2)."""
    return [
        _mention_row(f"{tag}-t", hid, 0, teacher_id),
        _mention_row(f"{tag}-c", hid, 1, cand_id),
        _mention_row(f"{tag}-s", hid, 2, student_id),
    ]


def _write(out: Path, canonical: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    c_arrays = {f.name: [r[f.name] for r in canonical] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(
        pa.table(c_arrays, schema=NARRATORS_CANONICAL_SCHEMA), out / "narrators_canonical.parquet"
    )
    m_arrays = {f.name: [r[f.name] for r in mentions] for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    pq.write_table(
        pa.table(m_arrays, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA),
        out / "narrator_mentions_resolved.parquet",
    )


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "sig.yaml"
    p.write_text(
        "signatures:\n"
        '  - name_ar: "عبد الله"\n'
        '    teacher_ar: "رسول الله"\n'
        '    student_ar: "نافع"\n'
        '    name_en: "Abd Allah ibn Umar"\n'
        '    discriminator: "ctx:ibn-umar"\n'
        '    note: "hand-verified golden chain"\n',
        encoding="utf-8",
    )
    return p


def _read_canonical(out: Path) -> dict[str, dict[str, Any]]:
    return {
        r["canonical_id"]: r for r in pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    }


def test_stage_peels_matched_class_leaves_control_and_partial(tmp_path: Path) -> None:
    """Instrument separation: only Prophet->Nafi mentions peel to Ibn ʿUmar; a control
    (Malik->Zuhri) and a partial-match (Prophet->Malik) both STAY on the bare primary."""
    out = tmp_path / "curated"
    bare = make_canonical_id(_ABDALLAH)
    canonical = [
        _canonical_row(bare, _ABDALLAH, 5),
        _canonical_row(
            make_canonical_id(_PROPHET), _PROPHET, 9, death_year_ah=11, death_date_precision="exact"
        ),
        _canonical_row(make_canonical_id(_NAFI), _NAFI, 9),
        _canonical_row(make_canonical_id(_MALIK), _MALIK, 9),
        _canonical_row(make_canonical_id(_ZUHRI), _ZUHRI, 9),
    ]
    mentions: list[dict[str, Any]] = []
    # 3 golden-chain mentions (Prophet -> AbdAllah -> Nafi) => Ibn ʿUmar.
    mentions += _chain("h1", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g1")
    mentions += _chain("h2", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g2")
    mentions += _chain("h3", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g3")
    # 1 partial (Prophet -> AbdAllah -> Malik): teacher matches, student does not.
    mentions += _chain("h4", make_canonical_id(_PROPHET), bare, make_canonical_id(_MALIK), "p1")
    # 1 control (Malik -> AbdAllah -> Zuhri): a different ʿAbd Allāh, no role matches.
    mentions += _chain("h5", make_canonical_id(_MALIK), bare, make_canonical_id(_ZUHRI), "c1")
    _write(out, canonical, mentions)

    result = apply_contextual_disambiguation(out, seed_path=_seed(tmp_path))
    assert result is not None

    by_id = _read_canonical(out)
    target = make_discriminated_canonical_id(_ABDALLAH, "ctx:ibn-umar")
    # The Ibn ʿUmar node was minted with exactly the 3 golden-chain mentions.
    assert target in by_id
    assert by_id[target]["mention_count"] == 3
    assert by_id[target]["name_en"] == "Abd Allah ibn Umar"
    assert by_id[target]["over_merged"] is False
    # Primary reduced by 3 (5 recorded - 3 peeled = 2); partial + control stay.
    assert by_id[bare]["mention_count"] == 2

    m = {
        r["mention_id"]: r
        for r in pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    }
    assert m["g1-c"]["canonical_narrator_id"] == target
    assert m["g2-c"]["canonical_narrator_id"] == target
    assert m["p1-c"]["canonical_narrator_id"] == bare  # partial NOT peeled
    assert m["c1-c"]["canonical_narrator_id"] == bare  # control NOT peeled

    # Coverage (blast-radius) report emitted with the right tally.
    cov = {
        r["primary_id"]: r for r in pq.read_table(out / "contextual_coverage.parquet").to_pylist()
    }
    assert cov[bare]["total_mentions"] == 5
    assert cov[bare]["signature_matched"] == 3
    assert cov[bare]["with_any_neighbour"] == 5

    # Audit report: one row per peeled target.
    audit = pq.read_table(out / "contextual_splits.parquet").to_pylist()
    assert len(audit) == 1 and audit[0]["new_id"] == target and audit[0]["mention_count"] == 3


def test_stage_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    bare = make_canonical_id(_ABDALLAH)
    canonical = [
        _canonical_row(bare, _ABDALLAH, 3),
        _canonical_row(make_canonical_id(_PROPHET), _PROPHET, 5),
        _canonical_row(make_canonical_id(_NAFI), _NAFI, 5),
    ]
    mentions = _chain("h1", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g1")
    mentions += _chain("h2", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g2")
    _write(out, canonical, mentions)
    seed = _seed(tmp_path)

    assert apply_contextual_disambiguation(out, seed_path=seed) is not None
    first_c = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
    first_m = pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist()
    # Second run: peeled mentions now point at the Ibn ʿUmar id, residual matches nothing.
    assert apply_contextual_disambiguation(out, seed_path=seed) is None
    assert pq.read_table(out / "narrators_canonical.parquet").to_pylist() == first_c
    assert pq.read_table(out / "narrator_mentions_resolved.parquet").to_pylist() == first_m


def test_stage_empty_seed_is_noop(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    bare = make_canonical_id(_ABDALLAH)
    _write(
        out,
        [_canonical_row(bare, _ABDALLAH, 3)],
        _chain("h1", make_canonical_id(_PROPHET), bare, make_canonical_id(_NAFI), "g1"),
    )
    empty = tmp_path / "empty.yaml"
    empty.write_text("signatures: []\n", encoding="utf-8")
    assert apply_contextual_disambiguation(out, seed_path=empty) is None
    assert not (out / "contextual_coverage.parquet").exists()


def test_stage_absent_primary_is_noop(tmp_path: Path) -> None:
    """A seed whose bare name is not in the canonical table peels nothing and mints
    nothing (the over_merged_flag no-fabricate discipline)."""
    out = tmp_path / "curated"
    other = make_canonical_id(_ZUHRI)
    _write(out, [_canonical_row(other, _ZUHRI, 3)], [_mention_row("z", "h1", 0, other)])
    assert apply_contextual_disambiguation(out, seed_path=_seed(tmp_path)) is None


def test_stage_no_canonical_is_noop(tmp_path: Path) -> None:
    out = tmp_path / "curated"
    out.mkdir(parents=True, exist_ok=True)
    assert apply_contextual_disambiguation(out, seed_path=_seed(tmp_path)) is None
