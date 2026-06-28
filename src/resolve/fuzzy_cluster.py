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
* A **token-order guard** (da#138): ``token_set_ratio`` is order-*insensitive*, so
  a pure nasab reversal of two **different** people — ``محمد بن عبد الله``
  (Muḥammad son of ʿAbdullāh) ↔ ``عبد الله بن محمد`` (ʿAbdullāh son of Muḥammad) —
  scores a perfect 100 and, with no death-year/gender to corroborate, would
  falsely merge. The guard requires the shared significant tokens to appear in the
  **same relative order** in the two matched spellings. A genuine variant only
  *adds or drops* tokens (kunya/nisba/nasab expansion), which preserves the order
  of the shared tokens, so recall is untouched; only a reordering is rejected.
  This precision defect — and the validation harness that quantified its
  false-merge rate — is the da#138 tech-debt work; raising the numeric threshold
  cannot fix a perfect-100 reversal score and would only cost recall.

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

import itertools
import time
from collections import defaultdict
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

# Minimum number of *shared* significant (non-connector) name tokens required to
# merge. token_set_ratio scores a bare given name as a perfect (100) subset of
# every fuller name carrying it (``محمد`` ⊂ ``محمد بن اسماعيل البخاري``), and a
# sparse bare-name record has no death-year/gender to trip the other guards — so
# name score alone would over-merge distinct people who merely share one common
# given name. Requiring ≥2 shared significant tokens means a pure single-token
# subset never clusters, while a genuine variant (which shares the nasab/nisba
# stem) still does.
_MIN_SHARED_TOKENS = 2

# Emit a `cluster_progress` log line every this many candidate pairs scored, so a
# long clustering pass over a large canonical set is not silent for hours (the
# pre-optimization pass ran >5h with zero progress output and an unbounded `seen`
# set that grew until OOM — see the composite-key blocking note in
# :func:`cluster_records`).
_CLUSTER_PROGRESS_INTERVAL = 500_000

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


def _significant_token_sequence(key: str) -> list[str]:
    """Significant (non-connector) tokens of a single name string, in name order.

    The ordered counterpart of :func:`_significant_tokens` — the token-order guard
    compares relative order, so it needs the sequence, not the set.
    """
    return [tok for tok in key.split() if len(tok) >= 2 and tok not in _CONNECTOR_TOKENS]


def _token_order_consistent(a: str, b: str) -> bool:
    """True when the tokens shared by names ``a`` and ``b`` keep the same order.

    A genuine variant only adds/removes tokens, so the shared tokens stay in the
    same relative order; a nasab reversal of two different people permutes them.
    Fewer than two shared significant tokens is left to :data:`_MIN_SHARED_TOKENS`
    — this guard only adjudicates the high-overlap case a reversal can exploit.
    """
    seq_a = _significant_token_sequence(a)
    seq_b = _significant_token_sequence(b)
    shared = set(seq_a) & set(seq_b)
    if len(shared) < 2:
        return True
    return [t for t in seq_a if t in shared] == [t for t in seq_b if t in shared]


def _name_match(keys_a: list[str], keys_b: list[str], *, threshold: float) -> bool:
    """True when some key pair is a strong AND token-order-consistent match.

    A pair counts only if its ``token_set_ratio`` clears ``threshold`` *and* it
    survives the da#138 token-order guard. Taking the best over the cross product
    of each record's match keys (its name + da#94 aliases) keeps the alias-driven
    recall — a record matches if *any* of its spellings is both a strong and
    order-consistent match for any of the other's.
    """
    for a in keys_a:
        for b in keys_b:
            if fuzz.token_set_ratio(a, b) >= threshold and _token_order_consistent(a, b):
                return True
    return False


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


def _shared_significant_token_count(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Number of significant (non-connector) name tokens the two records share."""
    return len(_significant_tokens(_match_keys(a)) & _significant_tokens(_match_keys(b)))


def _can_merge(a: dict[str, Any], b: dict[str, Any], *, threshold: float) -> bool:
    """Decide whether two canonical records are the same person (precision-guarded).

    A merge requires ALL of: no death-year conflict, no gender conflict, at least
    :data:`_MIN_SHARED_TOKENS` shared significant name tokens (so a bare
    single-token subset never clusters), and a matched key pair that clears
    ``threshold`` *and* is token-order-consistent (so a nasab reversal of two
    different people does not merge — da#138). Pairwise and symmetric — the cluster
    post-validation relies on that to refuse any cluster harbouring a
    guard-conflicting pair.
    """
    if _death_years_conflict(a, b) or _genders_conflict(a, b):
        return False
    if _shared_significant_token_count(a, b) < _MIN_SHARED_TOKENS:
        return False
    return _name_match(_match_keys(a), _match_keys(b), threshold=threshold)


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


def _safe_partition(
    group: list[int],
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[list[int]]:
    """Split a union-find group into sub-clusters with NO guard-conflicting pair.

    Union-find is transitive but :func:`_can_merge` is pairwise, so a bridge
    record can chain two endpoints that fail the guards against each other
    (A–B ✓, B–C ✓, but A–C death-conflict): the conflicting A–C pair would
    survive in one cluster and merge two real people. This greedily places each
    member into the first existing sub-cluster it is ``_can_merge``-compatible
    with *every* member of, opening a new sub-cluster otherwise. By induction
    each sub-cluster is a clique under ``_can_merge``, so no conflicting pair can
    co-occur. Deterministic: members are processed in ``canonical_id`` order.
    """
    if len(group) <= 1:
        return [list(group)]

    ordered = sorted(group, key=lambda i: safe_str(records[i].get("canonical_id")) or "")
    subclusters: list[list[int]] = []
    for i in ordered:
        for sub in subclusters:
            if all(_can_merge(records[i], records[j], threshold=threshold) for j in sub):
                sub.append(i)
                break
        else:
            subclusters.append([i])
    return subclusters


def cluster_records(
    records: list[dict[str, Any]],
    *,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
) -> list[list[int]]:
    """Cluster canonical narrator records into same-person groups (as index lists).

    Token-blocks the records, scores each candidate pair within a block with
    rapidfuzz ``token_set_ratio`` over name + aliases, applies the precision
    guards (death-year, gender, ≥2 shared significant tokens), and unions the
    survivors. Each transitive union-find group is then re-partitioned into
    cliques (:func:`_safe_partition`) so a bridge record can never chain a
    guard-conflicting pair into one cluster. Returns one index list per cluster
    (singletons included), so the i-th record's cluster is recoverable. Pure and
    deterministic — the IO wrapper and the quality test both drive it.

    Blocking on **unordered token pairs** (not single tokens)
    --------------------------------------------------------
    ``_can_merge`` already requires :data:`_MIN_SHARED_TOKENS` (=2) shared
    significant tokens, so any pair that can possibly merge shares ≥2 tokens —
    i.e. it co-occurs in the posting list of at least one *2-token* composite
    key. We therefore block on ``(tok_a, tok_b)`` pairs rather than single
    tokens. This is **behaviour-identical** to single-token blocking + the
    ≥2-shared-token guard (the set of pairs that reach ``_can_merge`` and pass it
    is unchanged, and union-find components are invariant to edge order), but it
    eliminates the pathology that made the old pass blow up: a single
    ultra-frequent name token (``محمد``/``احمد``…) put tens of thousands of
    records in one block, so the inner ``O(|block|²)`` enumeration scored
    billions of pairs that the ≥2-token guard then rejected — pegging CPU and
    growing the ``seen`` set until it exhausted memory. Requiring two *specific*
    shared tokens makes the blocks small, so both time and peak memory drop by
    orders of magnitude on a real (~300k-record) canonical set.
    """
    n = len(records)
    uf = _UnionFind(n)

    # Inverted index: unordered significant-token PAIR -> record indices whose
    # name/alias tokens include both members of the pair (composite blocking).
    # Each record contributes C(t, 2) keys for its t significant tokens (t is
    # small — a name has a handful of tokens), so the index stays compact.
    pair_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        toks = sorted(_significant_tokens(_match_keys(rec)))
        for key in itertools.combinations(toks, 2):
            pair_index[key].append(i)

    candidate_pairs_est = sum(len(m) * (len(m) - 1) // 2 for m in pair_index.values())
    max_block = max((len(m) for m in pair_index.values()), default=0)
    logger.info(
        "cluster_blocking_built",
        records=n,
        blocks=len(pair_index),
        candidate_pairs_est=candidate_pairs_est,
        max_block=max_block,
    )

    # Within each composite block every pair already shares the two key tokens,
    # so it clears the ≥2-shared-token guard by construction. Dedup pairs so a
    # pair sharing >2 tokens (which lands in several composite blocks) is scored
    # once; with composite keys `seen` is bounded by the *viable* pair count, not
    # the old all-pairs-sharing-any-token explosion.
    seen: set[tuple[int, int]] = set()
    scored = 0
    merged = 0
    t0 = time.monotonic()
    for members in pair_index.values():
        for pos, i in enumerate(members):
            for j in members[pos + 1 :]:
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                scored += 1
                if _can_merge(records[pair[0]], records[pair[1]], threshold=threshold):
                    uf.union(pair[0], pair[1])
                    merged += 1
                if scored % _CLUSTER_PROGRESS_INTERVAL == 0:
                    logger.info(
                        "cluster_progress",
                        scored=scored,
                        candidate_pairs_est=candidate_pairs_est,
                        pct=(
                            round(scored / candidate_pairs_est * 100, 1)
                            if candidate_pairs_est
                            else 100.0
                        ),
                        merged=merged,
                        unique_pairs=len(seen),
                        elapsed_seconds=round(time.monotonic() - t0, 1),
                    )

    logger.info(
        "cluster_pairs_scored",
        scored=scored,
        merged=merged,
        unique_pairs=len(seen),
        elapsed_seconds=round(time.monotonic() - t0, 1),
    )

    # Re-partition each transitive group into guard-safe cliques (closes the
    # bridge-bypass: A–B ✓, B–C ✓, A–C conflict must NOT yield one {A,B,C}).
    clusters: list[list[int]] = []
    for group in uf.groups():
        clusters.extend(_safe_partition(group, records, threshold=threshold))
    return clusters


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
