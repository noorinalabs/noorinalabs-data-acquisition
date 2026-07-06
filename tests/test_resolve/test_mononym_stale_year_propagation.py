"""Integration test for da#266 — a mononym split must write the REFINED person's
death year into the chain-context index, not the pre-split ambiguous mononym
bio's.

The da#248 split re-resolves a bare over-merged mononym (``سفيان``) to a specific
person using the death years of its chain neighbours (the incrementally-populated
``death_year_index``). The bug this test guards: on a split, ``disambiguate.run``
used to write the *ambiguous pre-split bio's* death year (``c.death_year_ah``)
into that index rather than the refined person's (``person.death_year_ah``). A
downstream **chain-adjacent registered mononym** then reads a stale year and can
mis-select among genuinely-distinct persons.

The fixture is a single three-narrator chain, processed in position order:

    pos 0  الحسن البصري   (dated bio anchor, d.110 — NOT a registered mononym)
    pos 1  سفيان          (registered mononym → refines to al-Thawrī, d.161)
    pos 2  سفيان          (registered mononym → must refine to ibn ʿUyayna, d.198)

The ``سفيان`` bio is deliberately **undated** (``death_year_ah=None``) — realistic
for a merged mononym, and it neutralizes the temporal filter (an undated candidate
always passes) so the ONLY variable between the buggy and fixed code is which year
the split writes to the index:

* pos 1 reads pos 0's year [110] → uniquely al-Thawrī (d.161); the merged bio is
  undated. Fixed: writes 161. Buggy: writes None.
* pos 2 reads pos 1's index slot.
  - Fixed (161): ``refine(سفيان, [161])`` → al-Thawrī implausible (gap 0), ibn
    ʿUyayna plausible (gap 37) → uniquely **ibn ʿUyayna**.
  - Buggy (None): ``_adjacent_death_years`` skips None → no evidence → the split
    abstains → pos 2 stays the bare merged ``سفيان`` node.

So asserting pos 2 resolves to ``سفيان بن عيينة`` fails on the buggy write and
passes only once the refined year propagates. Revert the da#266 one-liner in
``disambiguate.run`` and ``test_second_mononym_resolves_off_refined_year`` fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.identity import make_canonical_id
from src.resolve import disambiguate, ner
from src.utils.arabic import normalize_arabic

_HADITH_ID = "hdt:test:mononym-chain"

# Chain, in position order. pos 0 is a dated, non-registered anchor whose death
# year uniquely selects al-Thawrī for the pos-1 سفيان; pos 1 and pos 2 are both
# the bare registered mononym سفيان.
_ANCHOR = "الحسن البصري"  # al-Ḥasan al-Baṣrī — two-token nasab, never a registry key
_SUFYAN = "سفيان"

# Canonical ids of the three possible سفيان nodes (same id contract the split uses).
_ID_BARE_SUFYAN = make_canonical_id(normalize_arabic("سفيان"))
_ID_THAWRI = make_canonical_id(normalize_arabic("سفيان الثوري"))
_ID_IBN_UYAYNA = make_canonical_id(normalize_arabic("سفيان بن عيينة"))


def _write_phase1_mentions(staging_dir: Path) -> None:
    names = [_ANCHOR, _SUFYAN, _SUFYAN]
    table = pa.table(
        {
            "name_ar": pa.array(names, type=pa.string()),
            "name_en": pa.array([None, None, None], type=pa.string()),
            "name_ar_normalized": pa.array([normalize_arabic(n) for n in names], type=pa.string()),
            "source_hadith_id": pa.array([_HADITH_ID] * 3, type=pa.string()),
            "position_in_chain": pa.array([0, 1, 2], type=pa.int32()),
            "transmission_method": pa.array(["حدثنا", "عن", "عن"], type=pa.string()),
        }
    )
    pq.write_table(table, staging_dir / "narrator_mentions_sanadset.parquet")


def _write_bio_candidates(staging_dir: Path) -> None:
    """Two bios: the dated anchor, and the UNDATED merged سفيان mononym.

    The سفيان bio carries ``death_year_ah=None`` on purpose — a merged mononym has
    no single authoritative death year, and an undated candidate always passes the
    temporal filter, isolating the index-write as the only variable under test.
    """

    def _col(values: list[str | int | None], typ: pa.DataType) -> pa.Array:
        return pa.array(values, type=typ)

    table = pa.table(
        {
            "bio_id": _col(["bio:anchor", "bio:sufyan"], pa.string()),
            "name_ar": _col([_ANCHOR, _SUFYAN], pa.string()),
            "name_en": _col([None, None], pa.string()),
            "name_ar_normalized": _col(
                [normalize_arabic(_ANCHOR), normalize_arabic(_SUFYAN)], pa.string()
            ),
            "kunya": _col([None, None], pa.string()),
            "nisba": _col([None, None], pa.string()),
            "birth_year_ah": _col([None, None], pa.int32()),
            # Anchor d.110 uniquely selects al-Thawrī (d.161, gap 51) over ibn
            # ʿUyayna (d.198, gap 88 > 80). The merged سفيان bio is undated.
            "death_year_ah": _col([110, None], pa.int32()),
            "birth_location": _col([None, None], pa.string()),
            "death_location": _col([None, None], pa.string()),
            "generation": _col([None, None], pa.string()),
            "gender": _col(["male", "male"], pa.string()),
            "trustworthiness": _col([None, None], pa.string()),
            "external_id": _col([None, None], pa.string()),
            "source": _col(["test", "test"], pa.string()),
        }
    )
    pq.write_table(table, staging_dir / "narrators_bio_test.parquet")


def _read_mentions(output_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(output_dir / "narrator_mentions_resolved.parquet")
    return cast(list[dict[str, object]], table.to_pylist())


def _canonical_id_at(mentions: list[dict[str, object]], position: int) -> str | None:
    for m in mentions:
        if int(m["position_in_chain"]) == position:  # type: ignore[call-overload]
            cid = m.get("canonical_narrator_id")
            return None if cid is None else str(cid)
    raise AssertionError(f"no mention at position {position}")


def test_second_mononym_resolves_off_refined_year(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _write_phase1_mentions(staging)
    _write_bio_candidates(staging)

    ner.run(staging, output)
    disambiguate.run(staging, output)
    mentions = _read_mentions(output)
    assert len(mentions) == 3, "all three chain mentions must survive to resolution"

    # pos 1 splits off the anchor's real year — al-Thawrī. This holds under both the
    # buggy and fixed code (the fix changes only what pos 1 *writes*, not how it
    # resolves), so it is the stable anchor for the pos-2 discriminator below.
    assert _canonical_id_at(mentions, 1) == _ID_THAWRI

    # pos 2 is the discriminator. It resolves to ibn ʿUyayna ONLY if pos 1 wrote its
    # REFINED year (161) into the index. Under the bug pos 1 wrote the merged bio's
    # (None) year, so pos 2 sees no evidence and abstains to the bare سفيان node.
    assert _canonical_id_at(mentions, 2) == _ID_IBN_UYAYNA

    # The two Sufyāns landed on distinct, specific nodes — neither is the bare
    # over-merged mononym, and they are not each other.
    ids = {_canonical_id_at(mentions, 1), _canonical_id_at(mentions, 2)}
    assert _ID_BARE_SUFYAN not in ids
    assert len(ids) == 2
