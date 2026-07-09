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
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from src.models.enums import DatePrecision
from src.parse.base import safe_str, write_parquet
from src.parse.identity import make_canonical_id
from src.resolve._checkpoint import (
    CheckpointController,
    checkpoint_dir,
    clear_checkpoint,
    hash_parquet_column_groups,
    hash_strings,
    load_checkpoint,
    log_resume,
    resolve_cadence,
    save_checkpoint,
)
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

# Default cap on composite-block size. A 2-token block with more than this many
# members means those two significant tokens co-occur in >K canonical records —
# i.e. they are common *given* names (عبد+الله, محمد+علي), not a discriminating
# signal. Such a block contributes O(K²) candidate pairs (a single 28k-member
# block on real data ≈ 400M pairs) that are almost all distinct people, so we
# skip blocks larger than this. A pair whose records ALSO share a rarer token
# pair is still scored via that smaller block, so the only merges lost are those
# justified *solely* by a common-name pair — the least reliable ones. Tunable
# (and disable-able with ``None``) so it can be swept against the da#118
# precision/recall harness; ``None`` restores the exact uncapped behaviour.
#
# Set to 250 (down from 1000) for the main#723 reload: cdist cost is O(K²) per
# block, and on the real ~245k-canonical set the 250–1000-member blocks dominated
# the candidate volume — ~550M pairs, ~36h even fully parallel across 14 cores.
# Capping at 250 makes the pass tractable; the dropped merges are exactly the
# lowest-reliability common-name-pair-only ones (cluster is itself a recall
# *increment* on the exact-name pass, so this trades a little extra recall for a
# runnable pass). Raising it back is a deliberate precision/recall + runtime call.
_DEFAULT_MAX_BLOCK_SIZE = 250

# Block-scoring strategy. Tiny blocks (≤ this many members) are scored by a plain
# scalar Python loop — they have too few pairs to amortize building a numpy matrix
# or dispatching a future. Every larger block is scored by a single-threaded C
# ``cdist`` (``workers=1``) dispatched to ONE shared thread pool across all blocks
# (see ``_block_merge_pairs`` / ``cluster_records``).
#
# This is the fix for the per-block thread-pool pathology: the previous code
# called ``cdist(..., workers=-1)`` *per block*, spawning a fresh all-cores pool
# for every medium block (64 threads each doing ~3 rows of a ~200-key matrix →
# ~99% thread spawn/join overhead, ~1.2 effective cores, ~40h projected on the
# real ~245k-canonical set). rapidfuzz releases the GIL during the C matrix
# computation, so many *single-threaded* cdist calls running on one persistent
# pool achieve true multi-core throughput with ZERO per-block spawn cost.
_SCALAR_BLOCK_MAX = 8

# Worker count for the shared block-scoring pool. Capped so an oversubscribed
# host (many cores) does not thrash; cdist itself is single-threaded per call, so
# parallelism is purely across blocks.
_CLUSTER_MAX_WORKERS = min(16, (os.cpu_count() or 2))

# How many block-scoring futures may be in flight at once (× workers). Bounds peak
# memory (each future holds one block's flattened keys + score matrix) while
# keeping the pool saturated.
_CLUSTER_INFLIGHT_FACTOR = 4

# Emit a `cluster_progress` log line every this many candidate pairs scored, so a
# long clustering pass over a large canonical set is not silent for hours (the
# pre-optimization pass ran >5h with zero progress output and an unbounded `seen`
# set that grew until OOM — see the composite-key blocking note in
# :func:`cluster_records`).
_CLUSTER_PROGRESS_INTERVAL = 500_000

# Safe-partition progress telemetry (da#306). A union-find group larger than this
# threshold gets a periodic progress line from :func:`_safe_partition` — run 4's
# union-find produced a single 168,773-member mega-group whose partition phase ran
# SILENTLY for 17h+ (the phase had no logging at all). Sized so only genuinely
# large groups emit; a normal cluster (≤ tens of members) never logs.
_SAFE_PARTITION_PROGRESS_THRESHOLD = 5_000
_SAFE_PARTITION_PROGRESS_INTERVAL = 10_000

# Top-k group sizes logged in the post-union `cluster_group_sizes` line (da#306).
_GROUP_SIZE_TOP_K = 10

# Per-record match-key cap (da#270). Block scoring is a ``process.cdist`` over the
# block's flattened match keys (name + aliases of every member); its cost is
# O(K²) in that flattened key count, and profiling the real ~210k-canonical set
# showed it is ~99% of clustering wall-time. A pathological tail of records — a
# handful of prolific narrators that accreted hundreds of alias spellings across
# sources, plus pollution nodes whose "name" is a captured isnad fragment — carry
# up to ~740 match keys (p99 is 19), so a single such record inside a block blows
# that block's K (and thus its cdist cost) super-linearly, and because those
# records also carry hundreds of significant tokens they land in a huge number of
# blocks. Capping each record to its first ``_MAX_MATCH_KEYS_PER_RECORD`` keys
# (name always first, then aliases in their existing deterministic order) bounds
# every record's per-block contribution. At 64 this touches only ~0.1% of records
# (>64 keys) yet cut cdist time ~3.5× on the worst real blocks; the only lost
# merges are those justified *solely* by a record's 65th-plus alias spelling — the
# least-reliable tail. ``None`` disables the cap (exact pre-da#270 behaviour, used
# by the equivalence/precision-recall harness). Applied in :func:`_match_keys` so
# every consumer — blocking, cdist, and the scalar path — sees one bounded key set.
_MAX_MATCH_KEYS_PER_RECORD: int | None = 64

# Per-record blocking-token cap (da#270). Composite blocking enumerates
# ``C(t, 2)`` token pairs per record for its ``t`` significant tokens; a pollution
# record with a 1,000-token "name" contributes ~500k posting entries and joins
# that many blocks, which is what made the blocking build itself balloon to
# multi-GB. Only the first ``_MAX_BLOCKING_TOKENS_PER_RECORD`` significant tokens
# (deterministic sorted order) generate blocking pairs; the record is still fully
# scored inside whatever blocks it does join, and the precision guards still read
# its full token set. At 64 this affects <1% of records (the same pathological
# tail) and never a normal name (p99 significant-token count is 34). ``None``
# disables the cap.
_MAX_BLOCKING_TOKENS_PER_RECORD: int | None = 64

# Crash-resume for the multi-day block-scoring pass (da#272, PR2). The
# checkpointed state is the union-find parent array + the set of applied block
# indices + the scored/merged counters. Correctness leans on two properties of
# this stage that make it MORE robust than the streaming stages: the final
# clusters are connected components (invariant to union order) and ``uf.union`` is
# idempotent — so a resume that applies only the not-yet-applied blocks yields the
# same components as an uninterrupted run regardless of order or partial re-work.
#
# Cadence is keyed to SCORED-PAIR progress, not block count: block cost varies by
# orders of magnitude (an 8-member scalar block vs a 250-member cdist block), so
# "every N blocks" would checkpoint wildly unevenly. Checkpointing every
# ``_CLUSTER_CHECKPOINT_SCORED_INTERVAL`` scored pairs bounds crash-loss to that
# many pairs of re-work while keeping the O(n) parent-array serialization
# infrequent (a full ~1.7B-pair run ⇒ a few dozen writes).
_CLUSTER_CHECKPOINT_SCORED_INTERVAL = 50_000_000
_CLUSTER_CHECKPOINT_SCHEMA_VERSION = 1
# Canonical columns whose content determines the clustering output (name/alias
# match keys + the death-year/gender guards + record identity & row order, which
# fix the block indices). A change to any of them discards the checkpoint.
_FINGERPRINT_CANONICAL_COLS = (
    "canonical_id",
    "name_ar",
    "name_en",
    "name_ar_normalized",
    "aliases",
    "death_year_ah",
    "gender",
)

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
    # da#376: records that produced no blocking key and were therefore never offered to
    # `_can_merge`. Reported so "did not merge" can be told apart from "was never scored".
    unblockable_records: int = 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        unblockable = (
            f"; {self.unblockable_records} never scored (no blocking key)"
            if self.unblockable_records
            else ""
        )
        return (
            f"fuzzy-cluster: {self.input_records} → {self.output_records} canonical "
            f"({self.merged_records} merged into {self.multi_member_clusters} clusters, "
            f"{self.cross_source_clusters} cross-source); "
            f"{self.mentions_remapped} mentions remapped{unblockable}"
        )


# ---------------------------------------------------------------------------
# Match keys & scoring
# ---------------------------------------------------------------------------
def _match_keys(record: dict[str, Any]) -> list[str]:
    """Every normalized name string a record can match on: its name + aliases (da#94).

    Truncated to :data:`_MAX_MATCH_KEYS_PER_RECORD` (da#270): the name is always
    kept first, then aliases in their existing deterministic order, so a
    pathological record with hundreds of alias spellings cannot blow up the O(K²)
    per-block ``cdist``. ``None`` keeps every key (exact pre-da#270 behaviour).
    """
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
    if _MAX_MATCH_KEYS_PER_RECORD is not None:
        return keys[:_MAX_MATCH_KEYS_PER_RECORD]
    return keys


def _significant_tokens(keys: list[str]) -> set[str]:
    """Blocking tokens for a record: name tokens minus genealogical connectors."""
    tokens: set[str] = set()
    for key in keys:
        for tok in key.split():
            if len(tok) >= 2 and tok not in _CONNECTOR_TOKENS:
                tokens.add(tok)
    return tokens


def count_unblockable(records: list[dict[str, Any]]) -> int:
    """Records that produce no blocking key and are therefore never scored (da#376).

    Blocking keys are ``combinations(significant_tokens, 2)``, so a record with fewer
    than :data:`_MIN_SHARED_TOKENS` significant tokens joins no block and is never
    offered to :func:`_can_merge` — at any threshold. Exposed so callers can report the
    count instead of inferring "no merge" from silence.
    """
    return sum(1 for r in records if len(_significant_tokens(_match_keys(r))) < _MIN_SHARED_TOKENS)


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


def _can_merge_cached(
    i: int,
    j: int,
    records: list[dict[str, Any]],
    keys_cache: dict[int, list[str]],
    tok_cache: dict[int, set[str]],
    *,
    threshold: float,
) -> bool:
    """Cached-key equivalent of :func:`_can_merge` for two record indices (da#306).

    Returns the **byte-identical** decision to
    ``_can_merge(records[i], records[j], threshold=threshold)``. The ONLY
    difference is that the two per-record derived values — the match-key list
    (:func:`_match_keys`) and the significant-token set (:func:`_significant_tokens`)
    — are read from ``keys_cache``/``tok_cache`` instead of being recomputed on
    every call. It reuses the exact same guard predicates (:func:`_death_years_conflict`,
    :func:`_genders_conflict`, :func:`_name_match`) so the merge *rule* has a single
    source of truth; only the key/token derivation is memoized. The equivalence is:

    * ``_shared_significant_token_count(a, b)`` is
      ``len(_significant_tokens(_match_keys(a)) & _significant_tokens(_match_keys(b)))``,
      and ``tok_cache[k] == _significant_tokens(_match_keys(records[k]))`` — so
      ``len(tok_cache[i] & tok_cache[j])`` is the same integer.
    * ``keys_cache[k] == _match_keys(records[k])`` — so the ``_name_match`` call
      sees the identical key lists.
    """
    a, b = records[i], records[j]
    if _death_years_conflict(a, b) or _genders_conflict(a, b):
        return False
    if len(tok_cache[i] & tok_cache[j]) < _MIN_SHARED_TOKENS:
        return False
    return _name_match(keys_cache[i], keys_cache[j], threshold=threshold)


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

    Token-indexed candidate lookup (da#306) — output-identical, not just faster
    -----------------------------------------------------------------------------
    The naive version scans **every** existing sub-cluster for each member, so
    run 4's single 168,773-member mega-group (a transitive chain, not a clique)
    was worst-case ~n²/2 ≈ 14.2B :func:`_can_merge` calls and ran the phase
    silently for 17h+. The optimization rests on one exact fact about the merge
    rule: :func:`_can_merge` requires ``≥ _MIN_SHARED_TOKENS`` (≥1) shared
    significant tokens with **every** member of a sub-cluster, so a sub-cluster in
    which *no* member shares even a single significant token with the incoming
    member can NEVER accept it — the naive scan tests it and rejects on its first
    member anyway. We therefore maintain a ``token -> sub-cluster indices`` index
    and, for each member, consider only the sub-clusters that share ≥1 token with
    it (a **superset** of the true acceptors, so nothing is dropped), tested in
    **ascending sub-cluster index = original creation order**, first-accepting
    wins. Each considered sub-cluster is still verified with the full
    ``all(_can_merge(...))`` predicate. Formally: the first accepting sub-cluster
    in the naive full scan is necessarily a candidate here (it accepted, so the
    member shares ≥1 token with each of its members), every sub-cluster before it
    that the naive scan rejected is either also tested-and-rejected here or a
    non-candidate the naive scan would reject on its first member — so the chosen
    sub-cluster (or the decision to open a new one) is identical, and the emitted
    partition is byte-identical.

    Per-record caching (da#306): each member's match keys (:func:`_match_keys`)
    and significant-token set (:func:`_significant_tokens`) are computed once up
    front and read from the caches by :func:`_can_merge_cached` — the naive path
    re-derived both inside every ``_can_merge`` call, which at billions of calls
    dominated. The decision is unchanged (see :func:`_can_merge_cached`).
    """
    if len(group) <= 1:
        return [list(group)]

    ordered = sorted(group, key=lambda i: safe_str(records[i].get("canonical_id")) or "")

    # Per-record derived values, computed ONCE per member (da#306). Keyed by record
    # index; every index tested below (both the incoming member and every existing
    # sub-cluster member) is in ``ordered``, so both operands of every
    # ``_can_merge_cached`` call are present.
    keys_cache: dict[int, list[str]] = {i: _match_keys(records[i]) for i in ordered}
    tok_cache: dict[int, set[str]] = {i: _significant_tokens(keys_cache[i]) for i in ordered}

    subclusters: list[list[int]] = []
    # significant token -> indices of sub-clusters containing ≥1 member with it.
    token_index: dict[str, set[int]] = defaultdict(set)

    log_progress = len(group) > _SAFE_PARTITION_PROGRESS_THRESHOLD
    t0 = time.monotonic()

    for placed, i in enumerate(ordered):
        # Candidate sub-clusters: those sharing ≥1 significant token with i. A
        # sub-cluster sharing zero tokens can never accept i (a zero-overlap pair
        # fails _can_merge's ≥_MIN_SHARED_TOKENS guard), so the naive scan would
        # reject it on its first member — skipping it is free of output effect.
        candidates: set[int] = set()
        for tok in tok_cache[i]:
            candidates.update(token_index[tok])

        for s in sorted(candidates):  # ascending index == original creation order
            sub = subclusters[s]
            if all(
                _can_merge_cached(i, j, records, keys_cache, tok_cache, threshold=threshold)
                for j in sub
            ):
                sub.append(i)
                for tok in tok_cache[i]:
                    token_index[tok].add(s)
                break
        else:
            s = len(subclusters)
            subclusters.append([i])
            for tok in tok_cache[i]:
                token_index[tok].add(s)

        if log_progress and (placed + 1) % _SAFE_PARTITION_PROGRESS_INTERVAL == 0:
            logger.info(
                "safe_partition_progress",
                members_placed=placed + 1,
                group_size=len(group),
                subclusters=len(subclusters),
                elapsed_seconds=round(time.monotonic() - t0, 1),
            )

    return subclusters


def _log_group_size_distribution(groups: list[list[int]]) -> None:
    """Log the union-find group-size distribution before partitioning (da#306).

    Surfaces the mega-group pathology that made the safe-partition phase silent:
    group count, how many are multi-member, the largest, the top-k sizes, and how
    many records sit inside multi-member groups (the partition workload).
    """
    sizes = sorted((len(g) for g in groups), reverse=True)
    multi = sum(1 for s in sizes if s > 1)
    logger.info(
        "cluster_group_sizes",
        groups=len(groups),
        multi_member_groups=multi,
        largest=sizes[0] if sizes else 0,
        top_k_sizes=sizes[:_GROUP_SIZE_TOP_K],
        members_in_multi_member_groups=sum(s for s in sizes if s > 1),
    )


def _stateless_merge_ok(
    records: list[dict[str, Any]],
    members: list[int],
    owner: list[int],
    flat_keys: list[str],
    p: int,
    q: int,
) -> tuple[int, int] | None:
    """Apply the STATELESS non-name guards to an over-threshold key pair.

    ``p``/``q`` index into the block's flattened key list. Applies the same guards
    as :func:`_can_merge` for an already-name-matched key pair, EXCEPT the
    union-find ``already-merged`` check (that one is stateful and stays on the main
    thread): different owning records, no death-year / gender conflict, and the
    matched key pair token-order-consistent (da#138). Returns the canonical
    ``(i, j)`` record-index pair (``i < j``) on pass, else ``None``.

    This is pure (reads ``records``/``flat_keys`` read-only, mutates nothing), so
    it runs INSIDE the parallel block workers — moving the per-pair guard cost
    (the real bottleneck on the ~550M candidate pairs) off the serial main thread,
    which is then left only the cheap ``uf.find``/``uf.union`` on survivors.
    """
    a_pos, b_pos = owner[p], owner[q]
    if a_pos == b_pos:
        return None  # two keys (aliases) of the same record
    i, j = members[a_pos], members[b_pos]
    if _death_years_conflict(records[i], records[j]) or _genders_conflict(records[i], records[j]):
        return None
    if not _token_order_consistent(flat_keys[p], flat_keys[q]):
        return None
    return (i, j) if i < j else (j, i)


def _block_merge_pairs(
    records: list[dict[str, Any]],
    members: list[int],
    owner: list[int],
    flat_keys: list[str],
    threshold: float,
) -> list[tuple[int, int]]:
    """Pure, thread-safe: cdist a block + apply the stateless guards → merge pairs.

    Computes the full key×key ``token_set_ratio`` matrix in C with ``workers=1``
    (single-threaded), then runs every over-threshold key pair through
    :func:`_stateless_merge_ok` and returns the DEDUPED surviving ``(i, j)``
    record-index pairs. Holds no shared state, so it is dispatched concurrently to
    a shared :class:`ThreadPoolExecutor`: rapidfuzz releases the GIL during the C
    matrix computation, and the per-pair guard work (death/gender/token-order — the
    bottleneck across ~550M candidate pairs) runs HERE in the worker rather than on
    the serial main thread, which is left only the cheap ``uf.find``/``uf.union``.

    A record pair survives if ANY of its key pairs passes the guards (the union-find
    semantics): the per-block dedup keeps the returned list — and the main thread's
    work — to the distinct record pairs, not the far larger key-pair count.
    """
    if len(flat_keys) < 2:
        return []
    mat = process.cdist(
        flat_keys,
        flat_keys,
        scorer=fuzz.token_set_ratio,
        dtype=np.float64,
        workers=1,
    )
    rows, cols = np.where(mat >= threshold)
    out: set[tuple[int, int]] = set()
    for p, q in zip(rows.tolist(), cols.tolist()):
        if p >= q:
            continue
        pair = _stateless_merge_ok(records, members, owner, flat_keys, p, q)
        if pair is not None:
            out.add(pair)
    return list(out)


def cluster_records(
    records: list[dict[str, Any]],
    *,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
    max_block_size: int | None = _DEFAULT_MAX_BLOCK_SIZE,
    ckpt_dir: Path | None = None,
    fingerprint: str = "",
    resume: bool = True,
    stop_after: int | None = None,
    checkpoint_every_n_intervals: int | None = None,
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
    tokens, eliminating the single-ultra-frequent-token blocks (``محمد``…) that
    made the old single-token pass enumerate billions of guaranteed-reject pairs
    and grow ``seen`` until it OOM-crashed. Per-record token sets and match keys
    are computed **once** and reused across every pair (the old ``_can_merge``
    recomputed them per pair — the dominant cost at this scale).

    ``max_block_size`` (the common-name cap)
    ----------------------------------------
    On real data even *composite* keys stay dense: ``عبد``+``الله`` co-occur in
    ~28k records, so that one block alone is ~400M pairs of mostly-distinct
    people. A block with more than ``max_block_size`` members is therefore
    **skipped** — its two tokens are common given names, not a discriminating
    signal. A pair whose records also share a *rarer* token pair is still scored
    via that smaller block, so the only merges lost are those justified solely by
    a common-name pair (the least reliable). Pass ``None`` to disable the cap and
    score every ≥2-shared-token pair (exact pre-cap behaviour; the
    behaviour-identical-to-single-token-blocking mode used by the equivalence
    test and the precision/recall harness).

    Crash-resume (da#272, PR2)
    --------------------------
    When ``ckpt_dir`` is given (by :func:`cluster_canonical_narrators`; the pure
    callers pass ``None`` and are unaffected), the block-scoring pass checkpoints
    the union-find parent array + the set of applied block indices + the
    scored/merged counters every ``_CLUSTER_CHECKPOINT_SCORED_INTERVAL`` scored
    pairs. On resume it restores that state and re-scores only the blocks not in
    the applied set. This is output-identical to an uninterrupted run because the
    final clusters are connected components (invariant to union order) and
    ``uf.union`` is idempotent — order and any incidental re-work cannot change the
    components. ``resume=False`` cold-starts; ``stop_after`` stops after N
    checkpoint writes (``--stop-after``), leaving the checkpoint and writing no
    output.
    """
    n = len(records)
    uf = _UnionFind(n)

    # Compute each record's match keys + significant-token set ONCE (not per
    # pair). At billions of candidate pairs the repeated _match_keys/
    # _significant_tokens calls dominated; caching turns them into O(n) prep.
    keys_cache: list[list[str]] = [_match_keys(rec) for rec in records]
    tok_cache: list[set[str]] = [_significant_tokens(keys_cache[i]) for i in range(n)]

    # da#376 — a record that is never SCORED must not look like a record that was
    # scored and refused. Blocking below keys on `combinations(tokens, 2)`, so a record
    # with fewer than two significant tokens contributes no key, joins no block, and is
    # never offered to `_can_merge` — at any threshold. Today that outcome (no merge, no
    # log, no counter) is observationally identical to a guard rejection: the same
    # success-shaped silence as da#309. Count it and say so.
    _unblockable = [i for i in range(n) if len(tok_cache[i]) < _MIN_SHARED_TOKENS]
    if _unblockable:
        _unblockable_mentions = sum(int(records[i].get("mention_count") or 0) for i in _unblockable)
        logger.warning(
            "cluster_records_unblockable",
            records=len(_unblockable),
            pct_of_records=round(len(_unblockable) / n * 100, 1),
            mentions_held=_unblockable_mentions,
            min_shared_tokens=_MIN_SHARED_TOKENS,
            examples=[safe_str(records[i].get("name_ar_normalized")) for i in _unblockable[:5]],
            detail=(
                "fewer than _MIN_SHARED_TOKENS significant tokens -> no blocking key -> "
                "never scored by _can_merge at any threshold (structural, not statistical)"
            ),
        )

    # Inverted index: unordered significant-token PAIR -> record indices whose
    # name/alias tokens include both members of the pair (composite blocking).
    # Each record contributes C(t, 2) keys for its t significant tokens (t is
    # small — a name has a handful of tokens), so the index stays compact.
    #
    # ``_MAX_BLOCKING_TOKENS_PER_RECORD`` (da#270) caps t per record before the
    # C(t, 2) fan-out: a pollution record whose "name" is a captured isnad fragment
    # can carry >1,000 significant tokens, and its ~500k posting entries were what
    # ballooned this build to multi-GB. Taking the first N sorted tokens is
    # deterministic and bounds the fan-out; the record is still fully scored inside
    # whatever blocks it joins, and the precision guards below read its full token
    # set (``tok_cache``), so only the *reach* of the pathological tail is trimmed.
    max_block_tokens = _MAX_BLOCKING_TOKENS_PER_RECORD
    pair_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(n):
        block_tokens = sorted(tok_cache[i])
        if max_block_tokens is not None:
            block_tokens = block_tokens[:max_block_tokens]
        for key in itertools.combinations(block_tokens, 2):
            pair_index[key].append(i)

    # Apply the common-name cap: drop over-dense blocks up front so neither the
    # candidate estimate nor the scoring loop pays for them.
    if max_block_size is not None:
        blocks = [m for m in pair_index.values() if len(m) <= max_block_size]
        skipped = [m for m in pair_index.values() if len(m) > max_block_size]
    else:
        blocks = list(pair_index.values())
        skipped = []

    candidate_pairs_est = sum(len(m) * (len(m) - 1) // 2 for m in blocks)
    skipped_pairs_est = sum(len(m) * (len(m) - 1) // 2 for m in skipped)
    logger.info(
        "cluster_blocking_built",
        records=n,
        blocks=len(blocks),
        candidate_pairs_est=candidate_pairs_est,
        max_block=max((len(m) for m in pair_index.values()), default=0),
        max_block_size=max_block_size,
        skipped_blocks=len(skipped),
        skipped_pairs_est=skipped_pairs_est,
    )

    # Score each block's pairs by vectorizing the rapidfuzz call instead of a
    # Python loop of per-pair ``token_set_ratio`` calls — at ~1.7B candidate pairs
    # the per-pair Python+call overhead was the bottleneck (~27k pairs/s → ~18h).
    # ``process.cdist`` computes a block's full key×key score matrix in C. Within a
    # composite block every pair already shares the two key tokens, so the
    # ≥2-shared-token guard holds by construction; only the name score + death/
    # gender + token-order guards remain. We keep NO ``seen`` set (it would hold
    # ~1B tuples → the original OOM relocated): union-find is idempotent and the
    # ``uf.find`` check skips already-merged pairs, so memory stays bounded to the
    # blocking index + the union-find. float64 matches the scalar
    # ``token_set_ratio`` exactly, so the ≥threshold cut is identical across the
    # scalar and cdist paths (no float32 borderline flips).
    #
    # Parallelism is ACROSS blocks on one shared pool, not within a block: each
    # cdist runs single-threaded (``workers=1``) but releases the GIL, so the pool
    # gets true multi-core throughput without the per-block thread-pool spawn that
    # made ``workers=-1`` project to ~40h (see ``_block_merge_pairs`` /
    # ``_SCALAR_BLOCK_MAX``). Tiny blocks are scored inline (scalar) — too few
    # pairs to amortize a future. Union-find is mutated only here in the main
    # thread (it is not thread-safe); workers return pure index-pair lists.
    scored = 0
    merged = 0
    # Set of `blocks` indices already unioned into `uf` — the resume skip-set.
    applied: set[int] = set()

    # --- Crash-resume (da#272): restore union-find + applied-block set. See the
    # module "Crash-resume" note: correctness holds regardless of union order.
    controller: CheckpointController | None = None
    if ckpt_dir is not None:
        cadence = resolve_cadence(
            checkpoint_every_n_intervals,
            "CLUSTER_CHECKPOINT_EVERY_N_INTERVALS",
            1,
        )
        if not resume:
            clear_checkpoint(ckpt_dir)
        else:
            ckpt = load_checkpoint(ckpt_dir)
            if ckpt is not None:
                layout_ok = ckpt.get("schema_version") == _CLUSTER_CHECKPOINT_SCHEMA_VERSION
                fp_ok = ckpt.get("fingerprint") == fingerprint
                parent_ok = len(ckpt.get("parent", [])) == n
                if layout_ok and fp_ok and parent_ok:
                    uf._parent = [int(x) for x in ckpt["parent"]]
                    applied = {int(b) for b in ckpt["applied_blocks"]}
                    scored = int(ckpt["scored"])
                    merged = int(ckpt["merged"])
                    log_resume("cluster", skipped=len(applied), total=len(blocks), merged=merged)
                else:
                    clear_checkpoint(ckpt_dir)
        controller = CheckpointController(cadence, stop_after=stop_after)

    next_log = scored + _CLUSTER_PROGRESS_INTERVAL
    next_checkpoint_at = scored + _CLUSTER_CHECKPOINT_SCORED_INTERVAL
    t0 = time.monotonic()

    def _flatten(members: list[int]) -> tuple[list[str], list[int]]:
        """Flatten a block's members' match keys + a parallel owner→member-pos map."""
        fk: list[str] = []
        ow: list[int] = []
        for pos in range(len(members)):
            for name_key in keys_cache[members[pos]]:
                fk.append(name_key)
                ow.append(pos)
        return fk, ow

    def _snapshot() -> dict[str, object]:
        """Serialize the resumable state: union-find + applied blocks + counters."""
        return {
            "schema_version": _CLUSTER_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "parent": uf._parent,
            "applied_blocks": sorted(applied),
            "scored": scored,
            "merged": merged,
        }

    # future -> (block_idx, m) for in-flight cdist blocks. Each future RESOLVES to
    # the block's already-guard-filtered, deduped (i, j) merge pairs; the block_idx
    # lets ``_apply`` record it in the resume skip-set. Drained in COMPLETION order
    # (not submission order) so one slow block never stalls the pool.
    pending: dict[Future[list[tuple[int, int]]], tuple[int, int]] = {}
    inflight_cap = _CLUSTER_MAX_WORKERS * _CLUSTER_INFLIGHT_FACTOR

    def _apply(block_idx: int, m: int, pairs: list[tuple[int, int]]) -> None:
        """Union a block's pre-guarded merge pairs (the ONLY stateful, serial step).

        ``pairs`` are ``(i, j)`` record indices that already cleared every stateless
        guard in the worker; all that remains is the union-find ``already-merged``
        check + union, both O(α(n)) ≈ O(1). Runs only on the main thread, so the
        checkpoint it may write (reading ``uf._parent``) sees a consistent state.
        """
        nonlocal scored, merged, next_log, next_checkpoint_at
        for i, j in pairs:
            if uf.find(i) != uf.find(j):
                uf.union(i, j)
                merged += 1
        scored += m * (m - 1) // 2
        applied.add(block_idx)
        if scored >= next_log:
            next_log = scored + _CLUSTER_PROGRESS_INTERVAL
            logger.info(
                "cluster_progress",
                scored=scored,
                candidate_pairs_est=candidate_pairs_est,
                pct=(
                    round(scored / candidate_pairs_est * 100, 1) if candidate_pairs_est else 100.0
                ),
                merged=merged,
                elapsed_seconds=round(time.monotonic() - t0, 1),
            )
        # Checkpoint on scored-pair progress (block cost varies by orders of
        # magnitude, so block-count is a poor cadence). da#272 / da#276.
        if controller is not None and ckpt_dir is not None and scored >= next_checkpoint_at:
            next_checkpoint_at = scored + _CLUSTER_CHECKPOINT_SCORED_INTERVAL
            if controller.batch_complete():
                save_checkpoint(ckpt_dir, _snapshot())
                logger.info(
                    "cluster_checkpoint_saved",
                    applied=len(applied),
                    total=len(blocks),
                    scored=scored,
                    merged=merged,
                )
                if controller.checkpoint_written():
                    controller.stop(
                        "cluster", processed=len(applied), total=len(blocks), merged=merged
                    )

    def _drain_completed(block_until_one: bool) -> None:
        """Apply every finished block's pairs; optionally block until ≥1 finishes.

        Union-find is mutated only here on the main thread (it is not thread-safe).
        Final clusters are connected components, which are invariant to union order,
        so draining in completion order is deterministic (the equivalence test holds).
        """
        if not pending:
            return
        done: set[Future[list[tuple[int, int]]]] = (
            wait(pending.keys(), return_when=FIRST_COMPLETED).done
            if block_until_one
            else {f for f in pending if f.done()}
        )
        for f in done:
            block_idx, m = pending.pop(f)
            _apply(block_idx, m, f.result())

    with ThreadPoolExecutor(max_workers=_CLUSTER_MAX_WORKERS) as pool:
        for block_idx, members in enumerate(blocks):
            if block_idx in applied:
                continue  # already unioned before the resume point — skip
            m = len(members)
            if m < 2:
                continue
            flat_keys, owner = _flatten(members)

            if m <= _SCALAR_BLOCK_MAX:
                # Tiny block: plain scalar loop + the stateless guards inline — fewer
                # pairs than the cost of a future round-trip, so score on this thread.
                seen: set[tuple[int, int]] = set()
                for p in range(len(flat_keys)):
                    for q in range(p + 1, len(flat_keys)):
                        if fuzz.token_set_ratio(flat_keys[p], flat_keys[q]) >= threshold:
                            pair = _stateless_merge_ok(records, members, owner, flat_keys, p, q)
                            if pair is not None:
                                seen.add(pair)
                _apply(block_idx, m, list(seen))
                continue

            # Larger block: dispatch cdist + guard-filtering to the shared pool.
            pending[
                pool.submit(_block_merge_pairs, records, members, owner, flat_keys, threshold)
            ] = (block_idx, m)
            # Opportunistically reap anything already done (free slots, no blocking);
            # only block once the in-flight set hits the cap.
            _drain_completed(block_until_one=False)
            if len(pending) >= inflight_cap:
                _drain_completed(block_until_one=True)

        # Drain the remaining in-flight blocks.
        while pending:
            _drain_completed(block_until_one=True)

    logger.info(
        "cluster_pairs_scored",
        scored=scored,
        merged=merged,
        elapsed_seconds=round(time.monotonic() - t0, 1),
    )

    # Re-partition each transitive group into guard-safe cliques (closes the
    # bridge-bypass: A–B ✓, B–C ✓, A–C conflict must NOT yield one {A,B,C}).
    groups = uf.groups()
    _log_group_size_distribution(groups)
    clusters: list[list[int]] = []
    for group in groups:
        clusters.extend(_safe_partition(group, records, threshold=threshold))
    return clusters


def cluster_assignment(
    records: list[dict[str, Any]],
    *,
    threshold: float = _CLUSTER_RATIO_THRESHOLD,
    max_block_size: int | None = _DEFAULT_MAX_BLOCK_SIZE,
) -> dict[str, str]:
    """Map each record's ``canonical_id`` to its cluster's representative id.

    The label is the representative's id (see :func:`_choose_representative`), so
    feeding this straight into ``quality.pairwise_quality`` against a gold map
    measures the recall increment over the exact-name baseline (where every
    record is its own cluster). Pass ``max_block_size=None`` to measure recall
    without the common-name cap.
    """
    assignment: dict[str, str] = {}
    for cluster in cluster_records(records, threshold=threshold, max_block_size=max_block_size):
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
    # Precision columns default to UNKNOWN (never null), matching disambiguate and
    # the da#161 Narrator non-Optional invariant — merged rows may carry no precision
    # key, so the generic projection above would leave them null (da#239).
    for col in ("birth_date_precision", "death_date_precision"):
        arrays[col] = [r.get(col) or DatePrecision.UNKNOWN.value for r in rows]
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
    max_block_size: int | None = _DEFAULT_MAX_BLOCK_SIZE,
    staging_dir: Path | None = None,
    resume: bool = True,
    stop_after: int | None = None,
) -> ClusterMetrics:
    """Fuzzy-cluster ``narrators_canonical.parquet`` in place; return metrics.

    Reads the canonical table the exact-name pass produced, merges high-confidence
    cross-source name variants (see module docstring), and rewrites the file. When
    ``mentions_path`` is given, the absorbed-id → survivor-id remap is applied to
    the mentions so the graph edges follow the merge. A no-op (file untouched)
    when the table is missing/empty or nothing clusters.

    Crash-resume (da#272, PR2): when ``staging_dir`` is given the block-scoring
    pass is checkpointed under ``<staging_dir>/.cluster_checkpoint`` and resumes
    after a crash; ``resume=False`` cold-starts and ``stop_after`` bounds the run
    to N checkpoint writes (``--stop-after``, raising ``StopAfterReached`` before
    the canonical table is rewritten). The checkpoint is cleared on success.
    """
    if not canonical_path.exists():
        logger.warning("cluster_canonical_missing", path=str(canonical_path))
        return ClusterMetrics(0, 0, 0, 0, 0)

    records = pq.read_table(canonical_path).to_pylist()
    if not records:
        return ClusterMetrics(0, 0, 0, 0, 0)

    # Fingerprint the clustering-driver columns of the input so a checkpoint taken
    # against a different canonical set / threshold / cap is discarded (da#272).
    ckpt_dir: Path | None = None
    fingerprint = ""
    if staging_dir is not None:
        ckpt_dir = checkpoint_dir(staging_dir, "cluster")
        digests, _rows = hash_parquet_column_groups(
            canonical_path, {"content": _FINGERPRINT_CANONICAL_COLS}
        )
        # The da#270 caps (_MAX_MATCH_KEYS_PER_RECORD / _MAX_BLOCKING_TOKENS_PER_RECORD)
        # are module constants, not threaded params, but they shape the block
        # UNIVERSE — the match-key cap feeds keys_cache/tok_cache and the
        # blocking-token cap feeds pair_index → the `blocks` list membership,
        # order, and count. Because the resume skip-set is POSITIONAL block indices,
        # a resume across a cap change would otherwise match the fingerprint yet
        # restore indices pointing at different blocks. Hashing the caps discards
        # the checkpoint on any cap change (da#303).
        fingerprint = hash_strings(
            digests["content"],
            round(threshold, 6),
            max_block_size,
            len(records),
            _MAX_MATCH_KEYS_PER_RECORD,
            _MAX_BLOCKING_TOKENS_PER_RECORD,
        )

    clusters = cluster_records(
        records,
        threshold=threshold,
        max_block_size=max_block_size,
        ckpt_dir=ckpt_dir,
        fingerprint=fingerprint,
        resume=resume,
        stop_after=stop_after,
    )

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

    # Clustering completed and the canonical table is rewritten — drop the
    # checkpoint so the next cold run doesn't spuriously resume (da#272).
    if ckpt_dir is not None:
        clear_checkpoint(ckpt_dir)

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
        unblockable_records=count_unblockable(records),
    )
    logger.info("cluster_canonical_complete", summary=metrics.summary())
    return metrics
