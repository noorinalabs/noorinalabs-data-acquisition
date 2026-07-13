"""Flag over-merged bare-generic narrator nodes for the honest leaderboard (da#445).

Background (#337 / #346 flag-now)
---------------------------------
A bare-generic narrator name — a bare ism (``محمد``, ``سفيان``) or a bare kunya
(``أبو عبد الله``) with no distinguishing nasab/nisba — collapses many
historically-distinct people onto one ``nar:`` node (``make_canonical_id`` is a pure
function of the normalized name). That fused node then reports a spuriously inflated
betweenness centrality and pollutes the top of the choke-point leaderboard.

**We do NOT split and do NOT mint nodes.** The #337 spike proved algorithmic
context-splitting of these nodes both infeasible (the chimera does not separate — bare
``عبد الله``'s largest community is an 850-year mega-blob welded by the collectors'
shared late teachers) and unsafe (it false-splits genuine hubs — al-Zuhrī peeled into
five fabricated nodes, one dated 594 AH for a man who died 124 AH). So the owner chose
to *annotate* the leaderboard now and defer a real split to external rijāl evidence
(da#443). This stage is that annotation: one boolean column, zero fabricated nodes.

Why a curated list and not a threshold
--------------------------------------
There is **no corpus-internal threshold** that separates over-merged chimeras from
genuine hubs (the #337 §0 negative result): the genuine hubs Shuʿba/Qatāda/Mālik/Ibn
ʿAbbās span an attested-neighbour-death range as wide as the chimeras, and bare Sufyān
is as neighbour-concentrated as a hub. A pure-threshold auto-flag would re-commit the
silent-zero error at the detector — false-flagging a genuinely central transmitter of
the Prophet's hadith. So the flagged set is a **hand-verified curated list**
(:data:`_SEED_PATH`) under a **bidirectional acceptance fixture**
(``tests/test_resolve/test_over_merged_flag.py``): every genuine-hub control name MUST
be absent from the list and every seeded chimera MUST be present — the da#423
silent-zero discipline, the exclude direction sacred.

What this stage does
--------------------
Reads ``over_merged_narrators.yaml``, resolves each entry to a canonical id (by
``canonical_id`` when given, else ``make_canonical_id(normalize_arabic(name_ar))`` — the
same rule that minted the node), and sets ``over_merged = True`` + ``over_merge_note`` on
the matching canonical rows (every other row is set ``over_merged = False``, note None).
It fabricates nothing and remaps nothing. Placed after the canonical table is fully
built (after ``tabaqa_dates``); it is the last writer of ``narrators_canonical.parquet``
and, like every canonical writer, goes through :func:`write_canonical` so the da#428
completeness tally is re-minted. Idempotent: a re-run reads the already-flagged table
and re-applies the same booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.parse.identity import make_canonical_id
from src.resolve._run_record import write_canonical
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

# The curated seed ships beside this module (imported wherever the package installs).
_SEED_PATH = Path(__file__).with_name("over_merged_narrators.yaml")

__all__ = [
    "OverMergedEntry",
    "load_over_merged_seed",
    "over_merged_flags",
    "apply_over_merged_flags",
]


@dataclass(frozen=True)
class OverMergedEntry:
    """One curated over-merged bare-generic narrator.

    Keyed either by an explicit ``canonical_id`` or by ``name_ar`` (resolved to the id
    ``make_canonical_id`` mints — the SAME rule that created the node, so the flag lands
    on the very node the over-merge produced). ``note`` is the confirming spike/rijāl
    evidence, required so a flagged node always carries its justification.
    """

    canonical_id: str | None
    name_ar: str | None
    name_en: str | None
    note: str

    def resolved_id(self) -> str:
        """The canonical id this entry flags — explicit id, else minted from the name."""
        if self.canonical_id:
            return self.canonical_id
        return make_canonical_id(normalize_arabic(self.name_ar or ""))


def load_over_merged_seed(path: Path = _SEED_PATH) -> tuple[OverMergedEntry, ...]:
    """Parse the curated ``over_merged_narrators.yaml`` seed.

    Returns the entries in file order (empty when the file is absent). Raises
    ``ValueError`` on a malformed entry — a seed with a missing key is a defect, not a
    silently-dropped row (an unflagged chimera is exactly the silent zero this list
    exists to prevent).
    """
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("over_merged", []) if isinstance(raw, dict) else raw
    entries: list[OverMergedEntry] = []
    for row in rows:
        cid = row.get("canonical_id")
        name_ar = row.get("name_ar")
        note = row.get("over_merge_note")
        if not cid and not name_ar:
            raise ValueError("over_merged seed entry needs a canonical_id or a name_ar")
        if not note:
            raise ValueError(
                f"over_merged seed entry {cid or name_ar!r} is missing its over_merge_note"
            )
        entries.append(
            OverMergedEntry(
                canonical_id=cid,
                name_ar=name_ar,
                name_en=row.get("name_en"),
                note=note,
            )
        )
    return tuple(entries)


def over_merged_flags(entries: Iterable[OverMergedEntry]) -> dict[str, str]:
    """``canonical_id -> over_merge_note`` for every seeded entry."""
    return {entry.resolved_id(): entry.note for entry in entries}


def apply_over_merged_flags(output_dir: Path, *, seed_path: Path = _SEED_PATH) -> Path | None:
    """Set ``over_merged`` (+ note) on curated canonical rows; pure annotation (da#445).

    Reads ``narrators_canonical.parquet`` from ``output_dir`` (the curated dir), flags the
    rows whose canonical id is in the curated seed, and rewrites the table via
    :func:`write_canonical`. Returns the canonical path when at least one row was flagged,
    or ``None`` on a no-op (no canonical table, empty table, or the seed matched no row).
    Never fabricates a node — a seed id that matches no canonical row is logged (loudly),
    never invented.
    """
    canonical_path = output_dir / "narrators_canonical.parquet"
    if not canonical_path.exists():
        logger.warning("over_merged_flag_no_canonical", path=str(canonical_path))
        return None

    flags = over_merged_flags(load_over_merged_seed(seed_path))
    records: list[dict[str, Any]] = pq.read_table(canonical_path).to_pylist()
    if not records:
        return None

    matched = 0
    for rec in records:
        cid = rec.get("canonical_id")
        note = flags.get(cid) if cid else None
        if note is not None:
            rec["over_merged"] = True
            rec["over_merge_note"] = note
            matched += 1
        else:
            rec["over_merged"] = False
            rec["over_merge_note"] = None

    # A seed id matching no canonical row is a stale/typo'd entry, NOT a licence to mint a
    # node — surface it so an unapplied flag is observable, never silent (da#428 spirit;
    # feedback_silent_zero_is_not_a_measurement).
    present = {rec.get("canonical_id") for rec in records}
    unmatched = sorted(cid for cid in flags if cid not in present)
    if unmatched:
        logger.warning("over_merged_flag_seed_absent", unmatched=unmatched, count=len(unmatched))

    if matched == 0:
        logger.info("over_merged_flag_no_matches", seed=len(flags), canonical_total=len(records))
        return None

    arrays = {f.name: [rec.get(f.name) for rec in records] for f in NARRATORS_CANONICAL_SCHEMA}
    table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_canonical(table, canonical_path, stage="over_merged_flag")

    logger.info(
        "over_merged_flag_complete",
        flagged=matched,
        seed=len(flags),
        unmatched_seed=len(unmatched),
        canonical_total=len(records),
    )
    return canonical_path
