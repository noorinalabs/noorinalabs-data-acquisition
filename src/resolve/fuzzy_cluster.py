"""Fuzzy cross-source narrator clustering — recall increment on the exact-name pass.

The exact-name cross-source collapse (da#99, in ``disambiguate``/``bio_promote``)
keys every canonical narrator on ``make_canonical_id(name_ar_normalized)``: two
records merge **only** when their normalized names are byte-identical. That is
high precision, but it leaves *recall* on the table — the same real person whose
spelling survives Arabic normalization differently across sources (kunya-only
forms, nasab/nisba expansion or truncation, an optional ``ابن``/``بن``,
transliteration drift, honorifics) lands as two separate ``nar:`` nodes.

This module is that recall increment. It runs **on top of** — never replacing —
the exact-name pass: it reads the already-collapsed ``narrators_canonical.parquet``
and clusters records that are high-confidence name variants of one person, then
merges each cluster into a single canonical record. It is the bio/canonical-level
analogue of the rapidfuzz mention stage in ``disambiguate`` (which matches
*mentions* to *bios*); here both sides are canonical narrators.

Signals (issue da#118)
----------------------
* **rapidfuzz ``token_set_ratio``** over each record's match keys — its
  ``name_ar_normalized`` **plus** every ``alias`` (the da#94 Itqan name-variant
  feed): token-set is order- and extra-token-insensitive, so a nasab expansion
  (``محمد بن اسماعيل`` ↔ ``محمد بن اسماعيل البخاري``) scores high while a genuinely
  different name does not.
* **Token blocking** — only records sharing a significant (non-connector) name
  token are ever compared, keeping the pass near-linear instead of O(n²).

Precision guards (against over-merging distinct same-named narrators)
--------------------------------------------------------------------
* A **conservative default threshold** (:data:`_CLUSTER_RATIO_THRESHOLD`),
  tunable per the precision/recall tradeoff in the acceptance criteria.
* A **death-year guard**: when both records carry ``death_year_ah`` and they
  disagree by more than :data:`_DEATH_YEAR_TOLERANCE`, the merge is blocked even
  on a perfect name match — two scholars a century apart are not one person.
* A **gender guard**: two records with explicit, differing ``gender`` never merge.

Identity & ordering invariants
-------------------------------
* The merged record's id is ``make_canonical_id(representative.name_ar_normalized)``
  (``src.parse.identity``) — never a parallel id scheme (da#110). The
  representative is the existing canonical record the cluster collapses onto, so
  its id is itself a prior ``make_canonical_id`` output: clustering re-keys nothing.
* It operates **downstream of** ``disambiguate → bio_promote`` (da#117): those
  produce the canonical set, this clusters within it. It does not touch their
  ordering, and is idempotent — re-running on an already-clustered table is a
  no-op (the variant records are already gone).
* When a mentions file is present, the absorbed-id → representative-id remap is
  applied to ``narrator_mentions_resolved.parquet`` so the graph's NARRATED edges
  follow the merge instead of dangling on a node that no longer exists (#109).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src.parse.base import safe_str, write_parquet
from src.parse.identity import make_canonical_id
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.resolve.sect_affiliation import derive_sect_affiliation, normalize_corpus, primary_corpus
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ClusterMetrics", "cluster_records", "cluster_assignment", "cluster_canonical_narrators"]

# ---------------------------------------------------------------------------
# Thresholds (tunable — the da#118 precision/recall tradeoff)
# ---------------------------------------------------------------------------
# rapidfuzz token_set_ratio at/above which two records are a merge candidate.
# Conservative by default: token_set_ratio already discounts word order and
# extra tokens, so a high bar still catches nasab/kunya variants while keeping
# distinct same-prefix names apart.
_CLUSTER_RATIO_THRESHOLD = 90.0

# When both records carry a death year, a disagreement beyond this many AH years
# blocks the merge regardless of name score (two narrators a generation+ apart
# are not the same person). One source rounding a death year by a year is fine.
_DEATH_YEAR_TOLERANCE = 2

# Name tokens that carry no disambiguating signal (Arabic genealogical
# connectors / honorific particles). Excluded as *blocking* keys so a block does
# not balloon to "every name containing بن"; still scored as part of the name.
_CONNECTOR_TOKENS = frozenset(
    {
        "بن",
        "ابن",
        "ابو",
        "ابي",
        "ام",
        "ال",
        "بنت",
        "عن",
    }
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterMetrics:
    """Outcome of a fuzzy clustering pass over a canonical narrator table."""

    input_records: int
    output_records: int
    merged_records: int
    multi_member_clusters: int
    cross_source_clusters: int
    mentions_remapped: int = 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"fuzzy-cluster: {self.input_records} → {self.output_records} canonical "
            f"({self.merged_records} merged into {self.multi_member_clusters} clusters, "
            f"{self.cross_source_clusters} cross-source); "
            f"{self.mentions_remapped} mentions remapped"
        )


# ---------------------------------------------------------------------------
# Match keys & scoring
# ---------------------------------------------------------------------------
def _match_keys(record: dict[str, Any]) -> list[str]:
    """Every normalized name string a record can match on: its name + aliases (da#94)."""
    keys: list[str] = []
    name = safe_str(record.get("name_ar_normalized"))
    if name:
        keys.append(name)
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        for a in aliases:
            av = safe_str(a)
            if av and av not in keys:
                keys.append(av)
    return keys


def _significant_tokens(keys: list[str]) -> set[str]:
    """Blocking tokens for a record: name tokens minus genealogical connectors."""
    tokens: set[str] = set()
    for key in keys:
        for tok in key.split():
            if len(tok) >= 2 and tok not in _CONNECTOR_TOKENS:
                tokens.add(tok)
    return tokens


def _name_similarity(keys_a: list[str], keys_b: list[str]) -> float:
    """Best token_set_ratio across the cross product of two records' match keys.

    Taking the max over names + aliases means a record matches if *any* of its
    spellings (including a da#94 variant) is a strong token-set match for any of
    the other's — exactly the cross-source-variant recall this pass targets.
    """
    best = 0.0
    for a in keys_a:
        for b in keys_b:
            score = fuzz.token_set_ratio(a, b)
            if score > best:
                best = score
                if best >= 100.0:
                    return best
    return best


def _death_years_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when both carry a death year that disagree beyond the tolerance."""
    da, db = a.get("death_year_ah"), b.get("death_year_ah")
    if isinstance(da, int) and isinstance(db, int):
        return abs(da - db) > _DEATH_YEAR_TOLERANCE
    return False


def _genders_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when both carry an explicit, differing gender."""
    ga, gb = safe_str(a.get("gender")), safe_str(b.get("gender"))
    return bool(ga and gb and ga != gb)


def _can_merge(a: dict[str, Any], b: dict[str, Any], *, threshold: float) -> bool:
    """Decide whether two canonical records are the same person (precision-guarded)."""
    if _death_years_conflict(a, b) or _genders_conflict(a, b):
        return False
    return _name_similarity(_match_keys(a), _match_keys(b)) >= threshold


# ---------------------------------------------------------------------------
# Union-find clustering
# ---------------------------------------------------------------------------
class _UnionFind:
    """Minimal disjoint-set over record indices."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def groups(self) -> list[list[int]]:
        clusters: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            clusters.setdefault(self.find(i), []).append(i)
        return list(clusters.values())


def cluster_records(
    records: list[dict[str, Any]],
    *,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
) -> list[list[int]]:
    """Cluster canonical narrator records into same-person groups (as index lists).

    Token-blocks the records, scores each candidate pair within a block with
    rapidfuzz ``token_set_ratio`` over name + aliases, applies the death-year and
    gender precision guards, and unions the survivors. Returns one index list per
    cluster (singletons included), so the i-th record's cluster is recoverable.
    Pure and deterministic — the IO wrapper and the quality test both drive it.
    """
    n = len(records)
    uf = _UnionFind(n)

    # Inverted index: significant token -> record indices carrying it (blocking).
    token_index: dict[str, list[int]] = {}
    record_keys: list[list[str]] = []
    for i, rec in enumerate(records):
        keys = _match_keys(rec)
        record_keys.append(keys)
        for tok in _significant_tokens(keys):
            token_index.setdefault(tok, []).append(i)

    # Candidate pairs = records sharing ≥1 significant token. Dedup pairs so each
    # is scored once even when two records share several tokens.
    seen: set[tuple[int, int]] = set()
    for members in token_index.values():
        for pos, i in enumerate(members):
            for j in members[pos + 1 :]:
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                if _can_merge(records[i], records[j], threshold=threshold):
                    uf.union(i, j)

    return uf.groups()


def cluster_assignment(
    records: list[dict[str, Any]],
    *,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
) -> dict[str, str]:
    """Map each record's ``canonical_id`` to its cluster's representative id.

    The label is the representative's id (see :func:`_choose_representative`), so
    feeding this straight into ``quality.pairwise_quality`` against a gold map
    measures the recall increment over the exact-name baseline (where every
    record is its own cluster).
    """
    assignment: dict[str, str] = {}
    for cluster in cluster_records(records, threshold=threshold):
        rep = _choose_representative([records[i] for i in cluster])
        rep_label = str(rep.get("canonical_id"))
        for i in cluster:
            cid = safe_str(records[i].get("canonical_id"))
            if cid:
                assignment[cid] = rep_label
    return assignment


# ---------------------------------------------------------------------------
# Cluster merge
# ---------------------------------------------------------------------------
def _completeness(record: dict[str, Any]) -> int:
    """Count of populated biographical scalar fields — a tie-break for representative."""
    fields = (
        "name_ar",
        "name_en",
        "birth_year_ah",
        "death_year_ah",
        "generation",
        "gender",
        "trustworthiness",
        "external_id",
    )
    return sum(1 for f in fields if record.get(f) not in (None, ""))


def _choose_representative(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the canonical record a cluster collapses onto.

    Deterministic ranking: most mentions, then most complete bio, then longest
    normalized name, then smallest canonical_id. The winner's id (already a
    ``make_canonical_id`` output) becomes the cluster's id, so the merge routes
    through the identity contract and re-keys nothing.
    """

    def _rank(rec: dict[str, Any]) -> tuple[int, int, int]:
        mc = rec.get("mention_count")
        return (
            mc if isinstance(mc, int) else 0,
            _completeness(rec),
            len(safe_str(rec.get("name_ar_normalized")) or ""),
        )

    # Rank on (mentions, completeness, name length); break a remaining tie on the
    # smallest canonical_id so the choice is fully deterministic.
    best_rank = max(_rank(m) for m in members)
    top = [m for m in members if _rank(m) == best_rank]
    return min(top, key=lambda m: safe_str(m.get("canonical_id")) or "")


def _merge_cluster(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge a cluster of canonical records into one, keyed on the representative.

    Unions multi-valued provenance (source_ids, source_corpora, aliases), sums
    mention_count, and back-fills each scalar bio field from the representative
    first then any member that has it. The non-representative members' normalized
    names (and all members' existing aliases) become aliases of the survivor, so
    no source spelling is lost. source_corpus + sect_affiliation are recomputed
    from the unioned corpora (da#103).
    """
    rep = _choose_representative(members)
    rep_norm = safe_str(rep.get("name_ar_normalized")) or ""
    canonical_id = make_canonical_id(rep_norm) if rep_norm else str(rep.get("canonical_id"))

    merged: dict[str, Any] = {
        "canonical_id": canonical_id,
        "name_ar": safe_str(rep.get("name_ar")),
        "name_en": safe_str(rep.get("name_en")),
        "name_ar_normalized": rep_norm or None,
        "external_id": safe_str(rep.get("external_id")),
    }

    source_ids: list[str] = []
    corpora: list[str] = []
    aliases: list[str] = []
    mention_count = 0
    scalar_fields = (
        "name_ar",
        "name_en",
        "birth_year_ah",
        "death_year_ah",
        "generation",
        "gender",
        "trustworthiness",
        "external_id",
    )

    # Representative first so its values win the back-fill; others fill the gaps.
    for rec in [rep, *[m for m in members if m is not rep]]:
        mc = rec.get("mention_count")
        mention_count += mc if isinstance(mc, int) else 0

        for sid in rec.get("source_ids") or []:
            s = safe_str(sid)
            if s and s not in source_ids:
                source_ids.append(s)

        for corp in rec.get("source_corpora") or []:
            nc = normalize_corpus(safe_str(corp))
            if nc and nc not in corpora:
                corpora.append(nc)

        for field_name in scalar_fields:
            if merged.get(field_name) in (None, "") and rec.get(field_name) not in (None, ""):
                merged[field_name] = rec.get(field_name)

        # A non-representative spelling becomes an alias; carry existing aliases.
        rec_norm = safe_str(rec.get("name_ar_normalized"))
        if rec_norm and rec_norm != rep_norm and rec_norm not in aliases:
            aliases.append(rec_norm)
        for a in rec.get("aliases") or []:
            av = safe_str(a)
            if av and av != rep_norm and av not in aliases:
                aliases.append(av)

    merged["aliases"] = aliases
    merged["source_ids"] = source_ids
    merged["mention_count"] = mention_count
    merged["source_corpora"] = sorted(set(corpora))
    merged["source_corpus"] = primary_corpus(corpora)
    merged["sect_affiliation"] = derive_sect_affiliation(corpora)
    return merged


# ---------------------------------------------------------------------------
# Parquet IO
# ---------------------------------------------------------------------------
def _build_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Project merged rows onto NARRATORS_CANONICAL_SCHEMA."""
    arrays = {f.name: [r.get(f.name) for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    return pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)


def _remap_mention_canonical_ids(mentions_path: Path, remap: dict[str, str]) -> int:
    """Rewrite absorbed canonical ids on the mention rows to the cluster survivor.

    A merge dissolves the absorbed records' ``nar:`` nodes; mentions backfilled
    with one of those ids (da#99/#109) would otherwise key a NARRATED edge on a
    node ``load_nodes`` no longer creates. Streams row-group by row-group and
    rewrites in place (mirrors ``disambiguate._backfill_mention_canonical_ids``).
    Idempotent and bounded-memory. Returns the count of rows remapped.
    """
    if not remap or not mentions_path.exists():
        return 0

    pf = pq.ParquetFile(mentions_path)
    id_idx = NARRATOR_MENTIONS_RESOLVED_SCHEMA.get_field_index("canonical_narrator_id")
    tmp_path = mentions_path.with_name(mentions_path.name + ".cluster.tmp")
    writer = pq.ParquetWriter(tmp_path, NARRATOR_MENTIONS_RESOLVED_SCHEMA, compression="snappy")
    remapped = 0
    try:
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx).cast(NARRATOR_MENTIONS_RESOLVED_SCHEMA)
            ids = table.column("canonical_narrator_id").to_pylist()
            new_ids: list[str | None] = []
            for cid in ids:
                target = remap.get(cid) if cid is not None else None
                if target is not None and target != cid:
                    new_ids.append(target)
                    remapped += 1
                else:
                    new_ids.append(cid)
            table = table.set_column(
                id_idx, "canonical_narrator_id", pa.array(new_ids, type=pa.string())
            )
            writer.write_table(table)
    finally:
        writer.close()

    tmp_path.replace(mentions_path)
    logger.info("cluster_mentions_remapped", path=str(mentions_path), remapped=remapped)
    return remapped


def cluster_canonical_narrators(
    canonical_path: Path,
    *,
    mentions_path: Path | None = None,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
) -> ClusterMetrics:
    """Fuzzy-cluster ``narrators_canonical.parquet`` in place; return metrics.

    Reads the canonical table the exact-name pass produced, merges high-confidence
    cross-source name variants (see module docstring), and rewrites the file. When
    ``mentions_path`` is given, the absorbed-id → survivor-id remap is applied to
    the mentions so the graph edges follow the merge. A no-op (file untouched)
    when the table is missing/empty or nothing clusters.
    """
    if not canonical_path.exists():
        logger.warning("cluster_canonical_missing", path=str(canonical_path))
        return ClusterMetrics(0, 0, 0, 0, 0)

    records = pq.read_table(canonical_path).to_pylist()
    if not records:
        return ClusterMetrics(0, 0, 0, 0, 0)

    clusters = cluster_records(records, threshold=threshold)

    merged_rows: list[dict[str, Any]] = []
    remap: dict[str, str] = {}
    multi_member = 0
    cross_source = 0
    for cluster in clusters:
        members = [records[i] for i in cluster]
        merged = _merge_cluster(members)
        merged_rows.append(merged)
        if len(members) > 1:
            multi_member += 1
            survivor_id = str(merged["canonical_id"])
            for m in members:
                old_id = safe_str(m.get("canonical_id"))
                if old_id and old_id != survivor_id:
                    remap[old_id] = survivor_id
            if len(merged.get("source_corpora") or []) > 1:
                cross_source += 1

    write_parquet(_build_table(merged_rows), canonical_path, schema=NARRATORS_CANONICAL_SCHEMA)

    remapped = (
        _remap_mention_canonical_ids(mentions_path, remap) if mentions_path is not None else 0
    )

    metrics = ClusterMetrics(
        input_records=len(records),
        output_records=len(merged_rows),
        merged_records=len(records) - len(merged_rows),
        multi_member_clusters=multi_member,
        cross_source_clusters=cross_source,
        mentions_remapped=remapped,
    )
    logger.info("cluster_canonical_complete", summary=metrics.summary())
    return metrics
