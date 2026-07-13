"""Tests for src.resolve.over_merged_flag — the da#445 honest-leaderboard flag.

The load-bearing test is the **bidirectional acceptance gate**
(:class:`TestBidirectionalAcceptance`): the curated seed MUST flag every known
over-merged chimera and MUST NOT flag any genuine hub. The exclude direction is
sacred — flagging a genuinely-central transmitter (al-Zuhrī, Abū Hurayra, …) is the
silent-zero error this whole flag-not-split design exists to avoid (#337 §0 / da#423),
so a seed that ever named a hub fails CI here, exactly as a split that shattered him
would have. There is deliberately NO corpus-internal threshold in scope: the seed is a
hand-verified curated list and this fixture is its gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.identity import make_canonical_id
from src.resolve import over_merged_flag as omf
from src.resolve._run_record import read_run_record
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic

# The two control sets are the spec's locked lists (#337 §2). MUST_INCLUDE names are the
# provably-definitional bare generics seeded this wave; MUST_EXCLUDE are genuinely-single
# hubs that must never be flagged. Both are keyed by their raw Arabic name and resolved
# to a canonical id the same way the node is minted.
_MUST_INCLUDE = (
    "عبد الله",  # ʿAbd Allāh — betweenness #1, spike-proven mega-blob
    "سفيان",  # Sufyān — betweenness #2, al-Thawrī + ibn ʿUyayna fusion
    "أبو عبد الله",  # Abū ʿAbd Allāh — most common bare kunya
    "أبو جعفر",  # Abū Jaʿfar
    "محمد",  # Muḥammad
    "أحمد",  # Aḥmad
    "عبد الرحمن",  # ʿAbd al-Raḥmān
    "أبو بكر",  # Abū Bakr — Companion-adjacent bare kunya (rijāl-reviewed)
)
_MUST_EXCLUDE = (
    "الزهري",  # al-Zuhrī
    "أبو هريرة",  # Abū Hurayra
    "مالك",  # Mālik
    "ابن عباس",  # Ibn ʿAbbās
    "شعبة",  # Shuʿba
    "قتادة",  # Qatāda
    "الأعمش",  # al-Aʿmash
    "نافع",  # Nāfiʿ
    "معمر",  # Maʿmar
)


def _cid(name: str) -> str:
    return make_canonical_id(normalize_arabic(name))


class TestBidirectionalAcceptance:
    """CI-binding: the curated seed flags every chimera and no genuine hub."""

    def test_every_seeded_chimera_is_present(self) -> None:
        flagged = set(omf.over_merged_flags(omf.load_over_merged_seed()))
        missing = [n for n in _MUST_INCLUDE if _cid(n) not in flagged]
        assert not missing, f"seed is missing known over-merged chimeras: {missing}"

    def test_no_genuine_hub_is_flagged(self) -> None:
        # The SACRED direction: a flag list that names a genuine hub fails here, exactly
        # as a split that shattered al-Zuhrī would have (#337 §0 / da#423).
        flagged = set(omf.over_merged_flags(omf.load_over_merged_seed()))
        offenders = [n for n in _MUST_EXCLUDE if _cid(n) in flagged]
        assert not offenders, f"seed FLAGS a genuine hub — silent-zero violation: {offenders}"


class TestSeedIntegrity:
    def test_every_entry_carries_a_note(self) -> None:
        for entry in omf.load_over_merged_seed():
            assert entry.note.strip(), (
                f"seed entry {entry.name_ar or entry.canonical_id} has no note"
            )

    def test_resolved_ids_are_distinct(self) -> None:
        ids = [e.resolved_id() for e in omf.load_over_merged_seed()]
        assert len(ids) == len(set(ids)), "duplicate canonical id in the seed"

    def test_malformed_entry_raises(self, tmp_path: Path) -> None:
        # A seed entry with neither a canonical_id nor a name_ar is a defect, not a
        # silently-dropped row (a dropped chimera is the very silent zero we guard).
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "over_merged:\n  - name_en: nameless\n    over_merge_note: x\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="canonical_id or a name_ar"):
            omf.load_over_merged_seed(bad)

    def test_entry_without_note_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text('over_merged:\n  - name_ar: "محمد"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="over_merge_note"):
            omf.load_over_merged_seed(bad)


# --------------------------------------------------------------------------- #
# Stage — apply_over_merged_flags over a synthetic canonical table
# --------------------------------------------------------------------------- #
def _canonical_row(cid: str, name_norm: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    base["canonical_id"] = cid
    base["name_ar_normalized"] = name_norm
    base["mention_count"] = 100
    base.update(over)
    return base


def _write_canonical(curated: Path, rows: list[dict[str, Any]]) -> None:
    curated.mkdir(parents=True, exist_ok=True)
    arrays = {f.name: [r.get(f.name) for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(
        pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        curated / "narrators_canonical.parquet",
    )


def _read_by_id(curated: Path) -> dict[str, dict[str, Any]]:
    return {
        r["canonical_id"]: r
        for r in pq.read_table(curated / "narrators_canonical.parquet").to_pylist()
    }


def test_stage_flags_curated_rows_and_leaves_others(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    abd_allah = _cid("عبد الله")
    sufyan = _cid("سفيان")
    zuhri = _cid("الزهري")
    rows = [
        _canonical_row(abd_allah, normalize_arabic("عبد الله")),
        _canonical_row(sufyan, normalize_arabic("سفيان")),
        _canonical_row(zuhri, normalize_arabic("الزهري")),
        _canonical_row("nar:ordinary", normalize_arabic("محمد بن اسماعيل البخاري")),
    ]
    _write_canonical(curated, rows)

    assert omf.apply_over_merged_flags(curated) is not None
    by_id = _read_by_id(curated)
    # Chimeras flagged True and carry a note.
    for cid in (abd_allah, sufyan):
        assert by_id[cid]["over_merged"] is True
        assert by_id[cid]["over_merge_note"]
    # Genuine hub + ordinary node explicitly False with no note.
    for cid in (zuhri, "nar:ordinary"):
        assert by_id[cid]["over_merged"] is False
        assert by_id[cid]["over_merge_note"] is None
    # No node minted / dropped — pure annotation.
    assert len(by_id) == 4


def test_stage_matches_real_production_node_ids() -> None:
    # The seed resolves the two top contaminants to the exact stg leaderboard node ids
    # (project memory): bare ʿAbd Allāh nar:9b79… and bare Sufyān nar:afc3…. Pins that
    # the name-keyed resolution mints the SAME id the corpus does, so the seed will flag
    # the real nodes on the re-run, not phantoms.
    flags = omf.over_merged_flags(omf.load_over_merged_seed())
    assert _cid("عبد الله").startswith("nar:9b793066")
    assert _cid("سفيان").startswith("nar:afc30d02")
    assert _cid("عبد الله") in flags
    assert _cid("سفيان") in flags


def test_stage_no_canonical_table_is_noop(tmp_path: Path) -> None:
    assert omf.apply_over_merged_flags(tmp_path / "curated") is None


def test_stage_empty_canonical_is_noop(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    _write_canonical(curated, [])
    assert omf.apply_over_merged_flags(curated) is None


def test_stage_seed_matching_no_row_is_noop_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A canonical table with none of the seeded ids: the stage flags nothing (no-op None)
    # and NEVER fabricates a node for an absent seed id — it warns instead.
    curated = tmp_path / "curated"
    _write_canonical(curated, [_canonical_row("nar:someone-else", normalize_arabic("خالد"))])
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(omf.logger, "warning", lambda e, **kw: events.append((e, kw)))
    monkeypatch.setattr(omf.logger, "info", lambda e, **kw: events.append((e, kw)))

    assert omf.apply_over_merged_flags(curated) is None
    # The one canonical row is untouched; no seed id was invented as a new row.
    assert len(_read_by_id(curated)) == 1
    assert any(e == "over_merged_flag_seed_absent" for e, _ in events)


def test_stage_is_idempotent(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    _write_canonical(
        curated,
        [
            _canonical_row(_cid("عبد الله"), normalize_arabic("عبد الله")),
            _canonical_row("nar:ordinary", normalize_arabic("محمد بن اسماعيل البخاري")),
        ],
    )
    assert omf.apply_over_merged_flags(curated) is not None
    first = pq.read_table(curated / "narrators_canonical.parquet").to_pylist()
    assert omf.apply_over_merged_flags(curated) is not None
    second = pq.read_table(curated / "narrators_canonical.parquet").to_pylist()
    assert first == second


def test_stage_goes_through_write_canonical(tmp_path: Path) -> None:
    # da#428: the canonical rewrite must re-mint the completeness tally through
    # write_canonical (the AST bypass gate + this record are how a 7th writer is caught).
    curated = tmp_path / "curated"
    _write_canonical(
        curated,
        [
            _canonical_row(_cid("عبد الله"), normalize_arabic("عبد الله")),
            _canonical_row("nar:ordinary", normalize_arabic("محمد بن اسماعيل البخاري")),
        ],
    )
    assert omf.apply_over_merged_flags(curated) is not None
    record = read_run_record(curated)
    assert record is not None
    assert record.last_writer == "over_merged_flag"
    assert record.canonical_ids == 2
