"""Merge curated UNDER-merged narrator surface forms onto one canonical (da#431/da#347).

Background
----------
The canonical narrator id is a pure function of the normalized name
(``make_canonical_id``), and ``fuzzy_cluster`` only *merges* two canonicals when they
clear a precision guard: at least :data:`~src.resolve.fuzzy_cluster._MIN_SHARED_TOKENS`
shared significant name tokens, no death-year conflict, no gender conflict (da#138). That
guard is correct — loosening it re-commits the da#423 over-merge that deleted real
figures — but it structurally *cannot* bridge two references to the SAME person that share
too few tokens:

* **kunya ↔ ism** (da#431): ``أبو وائل`` (kunya) and ``شقيق بن سلمة`` (ism+nasab) share
  ZERO significant tokens, so the ~3,117 mentions on the kunya node never join the ism
  node, and the rijāl biography (promoted to its own node by ``bio_promote``) sits on a
  zero-mention node while the mentions sit on a bio-less one.
* **bare ism ↔ qualified** (da#347): bare ``أنس`` and ``أنس بن مالك`` share ONE token
  (``< _MIN_SHARED_TOKENS``), so Anas b. Mālik is split across two nodes, halving his
  centrality.

This stage does NOT loosen the guard. It supplies **curated, corroborated external
evidence** for a hand-verified set of confirmed single identities
(:data:`_SEED_PATH`, ``narrator_unify.yaml``) and merges ONLY those, reusing the exact
merge ``fuzzy_cluster`` performs (:func:`~src.resolve.fuzzy_cluster._merge_cluster` +
:func:`~src.resolve.fuzzy_cluster._remap_mention_canonical_ids`). It is the mirror of
``over_merged_flag`` — a curated list under a **bidirectional acceptance fixture**
(``tests/test_resolve/test_narrator_unify.py``): every golden multi-node-of-one-person
group MUST merge, and every distinct-narrator control MUST NOT.

The gate is the da#423 safety net, not the decision
----------------------------------------------------
The curated list *is* the decision (hand-verified, like ``over_merged_narrators.yaml``).
:func:`_group_admissible` REFUSES a group — loudly, never merging it — when any of these
holds, so a curation error cannot silently sweep in a distinct person:

1. **over-merge exclusion** — a member is a curated over-merged bare generic
   (``over_merged_narrators.yaml``); those fuse many people and can never be a unify member.
2. **corroborated death conflict** — two members carry ``death_year_provenance ==
   "corroborated"`` death years disagreeing beyond
   :data:`~src.resolve.fuzzy_cluster._DEATH_YEAR_TOLERANCE`.
3. **gross death spread** — two members' death years disagree by more than
   :data:`_UNIFY_MAX_DEATH_SPREAD` regardless of provenance (a loose sanity band that
   catches a distinct namesake even when neither year is corroborated — e.g. the d.200
   taba-tabiʿī bare ``شقيق`` against the d.82 Companion-era Abū Wāʾil).
4. **gender conflict** — two members carry explicit, differing gender.
5. **uncorroborated evidence** — the group's declared evidence is not attested in-corpus
   (for ``bio_kunya_ism_bridge``: no resolved member spells out BOTH the kunya and the ism).

Placement & idempotency
-----------------------
Runs after ``tabaqa_dates`` (dates + mention_count final) and before ``over_merged_flag``
(which stays the last writer of ``narrators_canonical.parquet``). Rewrites the canonical
table and remaps the absorbed mentions. Idempotent: after a merge the absorbed surfaces
survive only as aliases, so a re-run resolves each group to a single survivor node and
skips it (a group resolving to < 2 distinct canonicals is a no-op).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.resolve._run_record import write_canonical
from src.resolve.fuzzy_cluster import (
    _DEATH_YEAR_TOLERANCE,
    _genders_conflict,
    _merge_cluster,
    _remap_mention_canonical_ids,
)
from src.resolve.over_merged_flag import load_over_merged_seed
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

_SEED_PATH = Path(__file__).with_name("narrator_unify.yaml")

# Loose sanity band (in AH years) on the death-year spread within a unify group,
# applied regardless of provenance. Tighter than "any conflict" would be — the curated
# members legitimately carry noisy/default death years (e.g. the generic sahabi d.60) —
# but wide gaps (> this) mark a distinct namesake, not one person. Chosen so a genuine
# generation-default noise gap (Anas: bare d.60 vs qualified d.91 = 31) passes while a
# cross-generation namesake (bare Shaqīq d.200 vs Abū Wāʾil d.82 = 118) is refused.
_UNIFY_MAX_DEATH_SPREAD = 50

__all__ = [
    "UnifyGroup",
    "load_unify_seed",
    "apply_narrator_unification",
]


@dataclass(frozen=True)
class UnifyGroup:
    """One curated group of surface forms that are a single narrator.

    ``members`` are ``name_ar_normalized`` surfaces (the stored column) — NOT recomputed
    ids, because these nodes carry narrator_split-discriminated ids. ``member_ids`` are
    optional explicit canonical ids that take precedence when resolving. ``evidence`` names
    the corroboration the gate must confirm; ``bio_kunya_ism_bridge`` also reads
    ``kunya_norm``/``ism_norm``.
    """

    primary_label: str
    issue: str
    evidence: str
    note: str
    members: tuple[str, ...]
    member_ids: tuple[str, ...] = ()
    kunya_norm: str | None = None
    ism_norm: str | None = None


def load_unify_seed(path: Path = _SEED_PATH) -> tuple[UnifyGroup, ...]:
    """Parse the curated ``narrator_unify.yaml`` seed (empty when the file is absent).

    Raises ``ValueError`` on a malformed group — a dropped member is exactly the silent
    under-merge this list exists to fix, so a missing key fails loudly, never silently.
    """
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("unify", []) if isinstance(raw, dict) else raw
    groups: list[UnifyGroup] = []
    for row in rows:
        members = tuple(normalize_arabic(m) for m in (row.get("members") or []))
        member_ids = tuple(row.get("member_ids") or ())
        if len(members) + len(member_ids) < 2:
            raise ValueError(
                f"unify group {row.get('primary_label')!r} needs >= 2 members to merge"
            )
        evidence = row.get("evidence")
        note = row.get("note")
        if not evidence or not note:
            raise ValueError(
                f"unify group {row.get('primary_label')!r} needs both `evidence` and `note`"
            )
        kunya = row.get("kunya_norm")
        ism = row.get("ism_norm")
        if evidence == "bio_kunya_ism_bridge" and not (kunya and ism):
            raise ValueError(
                f"unify group {row.get('primary_label')!r} evidence bio_kunya_ism_bridge "
                "requires kunya_norm and ism_norm"
            )
        groups.append(
            UnifyGroup(
                primary_label=row.get("primary_label") or (members[0] if members else "?"),
                issue=row.get("issue") or "",
                evidence=evidence,
                note=note,
                members=members,
                member_ids=member_ids,
                kunya_norm=normalize_arabic(kunya) if kunya else None,
                ism_norm=normalize_arabic(ism) if ism else None,
            )
        )
    return tuple(groups)


def _over_merged_norms(seed_path: Path | None = None) -> frozenset[str]:
    """Normalized names of every curated over-merged bare generic (the exclusion set)."""
    entries = load_over_merged_seed(seed_path) if seed_path else load_over_merged_seed()
    return frozenset(normalize_arabic(e.name_ar) for e in entries if e.name_ar)


def _resolve_members(
    group: UnifyGroup, by_norm: dict[str, list[dict[str, Any]]], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Canonical records for a group's members (by explicit id, else by normalized name).

    A surface matching more than one row (a narrator_split-discriminated twin sharing a
    normalized name) resolves to ALL matching rows; the gate then vets the set. Deduped on
    canonical_id, preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def _add(rec: dict[str, Any]) -> None:
        cid = rec.get("canonical_id")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(rec)

    for mid in group.member_ids:
        rec = by_id.get(mid)
        if rec is not None:
            _add(rec)
    for surface in group.members:
        for rec in by_norm.get(surface, []):
            _add(rec)
    return out


def _corroborated_death_conflict(members: list[dict[str, Any]]) -> tuple[int, int] | None:
    """First pair of members whose CORROBORATED death years disagree beyond tolerance."""
    dated = [(m.get("death_year_ah"), m.get("death_year_provenance")) for m in members]
    for i in range(len(members)):
        yi, pi = dated[i]
        if not (isinstance(yi, int) and pi == "corroborated"):
            continue
        for j in range(i + 1, len(members)):
            yj, pj = dated[j]
            if (
                isinstance(yj, int)
                and pj == "corroborated"
                and abs(yi - yj) > _DEATH_YEAR_TOLERANCE
            ):
                return (yi, yj)
    return None


def _gross_death_spread(members: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Widest pair of member death years when it exceeds the unify sanity band."""
    years: list[int] = [y for m in members if isinstance((y := m.get("death_year_ah")), int)]
    if len(years) < 2:
        return None
    lo, hi = min(years), max(years)
    return (lo, hi) if hi - lo > _UNIFY_MAX_DEATH_SPREAD else None


def _gender_conflict(members: list[dict[str, Any]]) -> bool:
    """True when any two members carry explicit, differing gender."""
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if _genders_conflict(members[i], members[j]):
                return True
    return False


def _evidence_corroborated(group: UnifyGroup, members: list[dict[str, Any]]) -> bool:
    """Confirm the group's declared evidence is attested in the resolved members.

    ``bio_kunya_ism_bridge``: a resolved member's normalized name must contain BOTH the
    declared kunya and the declared ism — the in-corpus full-name node that spells out the
    kunya↔ism identity, so the bridge is attested, not asserted. Unknown evidence kinds are
    treated as NOT corroborated (fail closed).
    """
    if group.evidence == "bio_kunya_ism_bridge":
        kunya, ism = group.kunya_norm, group.ism_norm
        if not (kunya and ism):
            return False
        return any(
            kunya in (m.get("name_ar_normalized") or "")
            and ism in (m.get("name_ar_normalized") or "")
            for m in members
        )
    return False


def _group_admissible(
    group: UnifyGroup, members: list[dict[str, Any]], over_merged: frozenset[str]
) -> tuple[bool, str]:
    """Gate a resolved group. Returns (ok, reason) — reason is empty when admissible."""
    distinct = {m.get("canonical_id") for m in members}
    if len(distinct) < 2:
        return False, "fewer than 2 distinct canonical nodes (already unified / not present)"

    hit = sorted(
        normalize_arabic(m.get("name_ar_normalized") or "")
        for m in members
        if normalize_arabic(m.get("name_ar_normalized") or "") in over_merged
    )
    if hit:
        return False, f"member is a curated over-merged bare generic: {hit}"

    conflict = _corroborated_death_conflict(members)
    if conflict is not None:
        return False, f"corroborated death-year conflict {conflict[0]} vs {conflict[1]} AH"

    spread = _gross_death_spread(members)
    if spread is not None:
        return (
            False,
            f"gross death-year spread {spread[0]}-{spread[1]} AH (> {_UNIFY_MAX_DEATH_SPREAD})",
        )

    if _gender_conflict(members):
        return False, "explicit gender conflict between members"

    if not _evidence_corroborated(group, members):
        return False, f"declared evidence {group.evidence!r} not corroborated in-corpus"

    return True, ""


def _survivor_row(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the merged survivor row, preserving every canonical column.

    :func:`~src.resolve.fuzzy_cluster._merge_cluster` sums mentions, unions aliases,
    back-fills the scalar bio, and re-derives attestation. It does not touch the extra
    canonical columns (date bounds/precision, ``death_year_provenance``, ``external_id``,
    sect, over-merge flags), so those are back-filled here from the members
    (representative first) so the survivor keeps the full biography rather than dropping it.
    """
    merged = _merge_cluster(members)
    survivor_id = merged["canonical_id"]
    rep = next((m for m in members if m.get("canonical_id") == survivor_id), members[0])
    ordered = [rep, *[m for m in members if m is not rep]]
    row: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    for name, value in merged.items():
        row[name] = value
    for field in NARRATORS_CANONICAL_SCHEMA:
        if row.get(field.name) in (None, ""):
            for m in ordered:
                if m.get(field.name) not in (None, ""):
                    row[field.name] = m.get(field.name)
                    break
    return row


def apply_narrator_unification(output_dir: Path, *, seed_path: Path = _SEED_PATH) -> Path | None:
    """Merge each admissible curated unify group; remap absorbed mentions (da#431/da#347).

    Reads ``narrators_canonical.parquet`` from ``output_dir``, resolves + gates each seed
    group, merges the admissible ones onto a single survivor, rewrites the canonical table
    via :func:`write_canonical`, and remaps the absorbed mentions on
    ``narrator_mentions_resolved.parquet``. Returns the canonical path when at least one
    group merged, else ``None`` (no table, empty seed, or every group a no-op/refused).
    Never fabricates a node; a refused group is logged loudly, never merged.
    """
    canonical_path = output_dir / "narrators_canonical.parquet"
    if not canonical_path.exists():
        logger.warning("narrator_unify_no_canonical", path=str(canonical_path))
        return None

    groups = load_unify_seed(seed_path)
    if not groups:
        return None

    records: list[dict[str, Any]] = pq.read_table(canonical_path).to_pylist()
    if not records:
        return None

    over_merged = _over_merged_norms()
    by_id = {r["canonical_id"]: r for r in records if r.get("canonical_id")}
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_norm.setdefault(normalize_arabic(r.get("name_ar_normalized") or ""), []).append(r)

    remap: dict[str, str] = {}
    absorbed_ids: set[str] = set()
    survivors: dict[str, dict[str, Any]] = {}
    merged_groups = 0

    for group in groups:
        members = _resolve_members(group, by_norm, by_id)
        ok, reason = _group_admissible(group, members, over_merged)
        if not ok:
            logger.warning(
                "narrator_unify_group_refused",
                group=group.primary_label,
                issue=group.issue,
                resolved=len(members),
                reason=reason,
            )
            continue
        survivor = _survivor_row(members)
        survivor_id = survivor["canonical_id"]
        survivors[survivor_id] = survivor
        for m in members:
            cid = m.get("canonical_id")
            if cid and cid != survivor_id:
                absorbed_ids.add(cid)
            if cid:
                remap[cid] = survivor_id
        merged_groups += 1
        logger.info(
            "narrator_unify_group_merged",
            group=group.primary_label,
            issue=group.issue,
            members=len(members),
            survivor=survivor_id,
            mentions=survivor.get("mention_count"),
        )

    if merged_groups == 0:
        logger.info("narrator_unify_no_op", groups=len(groups))
        return None

    # Rebuild: drop absorbed rows, replace/insert survivors (dedupe on canonical_id so a
    # survivor whose id already existed as a member is not duplicated).
    rebuilt: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for r in records:
        cid = r.get("canonical_id")
        if cid in absorbed_ids:
            continue
        if cid in survivors:
            if cid in emitted:
                continue
            rebuilt.append(survivors[cid])
            emitted.add(cid)
        else:
            rebuilt.append(r)
    for sid, srow in survivors.items():
        if sid not in emitted:
            rebuilt.append(srow)
            emitted.add(sid)

    arrays = {f.name: [r.get(f.name) for r in rebuilt] for f in NARRATORS_CANONICAL_SCHEMA}
    table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_canonical(table, canonical_path, stage="narrator_unify")

    remapped = _remap_mention_canonical_ids(
        output_dir / "narrator_mentions_resolved.parquet", remap
    )

    logger.info(
        "narrator_unify_complete",
        groups=len(groups),
        merged=merged_groups,
        absorbed=len(absorbed_ids),
        canonical_before=len(records),
        canonical_after=len(rebuilt),
        mentions_remapped=remapped,
    )
    return canonical_path


def unify_summary(groups: Iterable[UnifyGroup]) -> str:
    """One-line human summary of the seed (for the run record / logs)."""
    gs = tuple(groups)
    return f"{len(gs)} curated unify group(s): " + "; ".join(g.primary_label for g in gs)
