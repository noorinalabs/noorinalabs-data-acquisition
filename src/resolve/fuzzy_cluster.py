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
    """
    n = len(records)
    uf = _UnionFind(n)

    # Compute each record's match keys + significant-token set ONCE (not per
    # pair). At billions of candidate pairs the repeated _match_keys/
    # _significant_tokens calls dominated; caching turns them into O(n) prep.
    keys_cache: list[list[str]] = [_match_keys(rec) for rec in records]
    tok_cache: list[set[str]] = [_significant_tokens(keys_cache[i]) for i in range(n)]

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
    next_log = _CLUSTER_PROGRESS_INTERVAL
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

    # future -> m (member count) for in-flight cdist blocks. Each future RESOLVES to
    # the block's already-guard-filtered, deduped (i, j) merge pairs, so the main
    # thread needs nothing but the count for the progress estimate. Drained in
    # COMPLETION order (not submission order) so one slow block never stalls the
    # pool: a submission-order drain (waiting on the oldest future) idles every
    # other worker behind a slow block and collapses throughput to ~1 core.
    pending: dict[Future[list[tuple[int, int]]], int] = {}
    inflight_cap = _CLUSTER_MAX_WORKERS * _CLUSTER_INFLIGHT_FACTOR

    def _apply(m: int, pairs: list[tuple[int, int]]) -> None:
        """Union a block's pre-guarded merge pairs (the ONLY stateful, serial step).

        ``pairs`` are ``(i, j)`` record indices that already cleared every stateless
        guard in the worker; all that remains is the union-find ``already-merged``
        check + union, both O(α(n)) ≈ O(1), so the main thread is no longer the
        bottleneck even across hundreds of millions of candidate pairs.
        """
        nonlocal scored, merged, next_log
        for i, j in pairs:
            if uf.find(i) != uf.find(j):
                uf.union(i, j)
                merged += 1
        scored += m * (m - 1) // 2
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
            _apply(pending.pop(f), f.result())

    with ThreadPoolExecutor(max_workers=_CLUSTER_MAX_WORKERS) as pool:
        for members in blocks:
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
                _apply(m, list(seen))
                continue

            # Larger block: dispatch cdist + guard-filtering to the shared pool.
            pending[
                pool.submit(_block_merge_pairs, records, members, owner, flat_keys, threshold)
            ] = m
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
    clusters: list[list[int]] = []
    for group in uf.groups():
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

    clusters = cluster_records(records, threshold=threshold, max_block_size=max_block_size)

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
