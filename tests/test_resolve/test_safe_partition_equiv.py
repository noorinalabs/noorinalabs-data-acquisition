"""Equivalence guard for the da#306 token-indexed ``_safe_partition`` rewrite.

The optimization (token-indexed candidate lookup + per-record key/token caching)
must be **output-identical** to the previous naive greedy scan — the recovery
plan cherry-picks it onto run 4's checkpoint and expects byte-identical
partitions. This module pins that: it keeps a verbatim copy of the OLD
implementation as :func:`_safe_partition_reference` (calling the real, uncached
:func:`fuzzy_cluster._can_merge`, exactly as the pre-da#306 code did) and asserts
the current :func:`fuzzy_cluster._safe_partition` returns the identical
sub-cluster lists across a large space of randomized synthetic groups — including
the guard-conflict *bridge* topology (A–B ✓, B–C ✓, A–C conflict) that
``_safe_partition`` exists to split.

A microbenchmark on a synthetic mega-group is run out-of-tree (timing assertions
are flaky in CI, so it is not committed as a pytest test); its before/after
numbers are quoted in the PR body.
"""

from __future__ import annotations

import random
from typing import Any

from src.parse.base import safe_str
from src.parse.identity import make_canonical_id
from src.resolve import fuzzy_cluster as fc


# ---------------------------------------------------------------------------
# Verbatim copy of the PRE-da#306 implementation (the reference oracle).
# ---------------------------------------------------------------------------
def _safe_partition_reference(
    group: list[int],
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[list[int]]:
    """The naive greedy scan as it stood before da#306 — the equivalence oracle.

    Kept byte-for-byte (modulo the rename) so this test is self-contained: it does
    NOT depend on the git history of ``fuzzy_cluster.py``. Uses the real,
    uncached ``fc._can_merge`` — the exact predicate the old code called.
    """
    if len(group) <= 1:
        return [list(group)]

    ordered = sorted(group, key=lambda i: safe_str(records[i].get("canonical_id")) or "")
    subclusters: list[list[int]] = []
    for i in ordered:
        for sub in subclusters:
            if all(fc._can_merge(records[i], records[j], threshold=threshold) for j in sub):
                sub.append(i)
                break
        else:
            subclusters.append([i])
    return subclusters


# ---------------------------------------------------------------------------
# Synthetic-record generation
# ---------------------------------------------------------------------------
# A small pool of significant tokens so records share tokens frequently (this is
# what exercises the token index AND creates the near-miss / bridge topologies).
_TOKENS = [
    "محمد",
    "احمد",
    "علي",
    "حسن",
    "حسين",
    "عبدالله",
    "اسماعيل",
    "ابراهيم",
    "يوسف",
    "عمر",
    "البخاري",
    "الكوفي",
    "البصري",
    "المدني",
    "الشيباني",
]


def _mk_record(idx: int, rng: random.Random) -> dict[str, Any]:
    """A synthetic canonical record with random name tokens + optional guards.

    Names are built from a small shared token pool so token overlap — and thus
    high ``token_set_ratio`` scores, shared-token counts, and guard-conflict
    bridges — arises densely, stressing the token-index candidate lookup and the
    clique-splitting path rather than the trivially-disjoint case.
    """
    n_tokens = rng.randint(1, 3)
    tokens = rng.sample(_TOKENS, n_tokens)
    name = " بن ".join(tokens)
    # Aliases: sometimes a re-ordered / extended spelling (da#94 feed analogue).
    aliases: list[str] = []
    if rng.random() < 0.35:
        extra = rng.choice(_TOKENS)
        aliases.append(f"{name} {extra}")
    record: dict[str, Any] = {
        # Unique id per record so canonical_id ordering is total and stable.
        "canonical_id": f"{make_canonical_id(name)}::{idx:04d}",
        "name_ar_normalized": name,
        "aliases": aliases,
    }
    # Death year present ~60% of the time, drawn from a spread wide enough that
    # two present values frequently conflict (> _DEATH_YEAR_TOLERANCE) → this is
    # what manufactures the A–B ✓, B–C ✓, A–C death-conflict bridge topology.
    if rng.random() < 0.6:
        record["death_year_ah"] = rng.choice([120, 121, 200, 201, 256, 300, 400])
    # Explicit gender ~25% of the time (the other guard).
    if rng.random() < 0.25:
        record["gender"] = rng.choice(["male", "female"])
    return record


def _random_group(rng: random.Random, size: int) -> list[dict[str, Any]]:
    return [_mk_record(i, rng) for i in range(size)]


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------
def test_safe_partition_matches_reference_across_random_groups() -> None:
    """Byte-identical partitions vs the pre-da#306 oracle over many random groups.

    Sweeps a range of group sizes and seeds; the dense shared-token pool ensures
    the runs hit real merges, real rejections, and guard-conflict bridges — not
    just trivially-disjoint singletons.
    """
    threshold = fc._CLUSTER_RATIO_THRESHOLD
    for seed in range(60):
        rng = random.Random(seed)
        size = rng.choice([2, 3, 5, 8, 13, 21, 40])
        records = _random_group(rng, size)
        group = list(range(len(records)))

        expected = _safe_partition_reference(group, records, threshold=threshold)
        actual = fc._safe_partition(group, records, threshold=threshold)

        assert actual == expected, (
            f"partition mismatch at seed={seed} size={size}\n"
            f"  expected={expected}\n  actual  ={actual}"
        )


def test_safe_partition_splits_death_year_bridge() -> None:
    """The canonical bridge: A–B ✓, B–C ✓, A–C death-conflict must NOT co-cluster.

    B is a name-compatible hub for both A and C, but A and C carry death years a
    century apart. The greedy clique split must place A and C in different
    sub-clusters; the reference and the optimized impl must agree exactly.
    """
    # All three share the same two significant tokens (≥ _MIN_SHARED_TOKENS) and
    # score a perfect token_set_ratio, so only the death-year guard separates them.
    a = {
        "canonical_id": "nar:a",
        "name_ar_normalized": "محمد بن اسماعيل",
        "aliases": [],
        "death_year_ah": 200,
    }
    b = {
        "canonical_id": "nar:b",
        "name_ar_normalized": "محمد بن اسماعيل",
        "aliases": [],
        # No death year → conflicts with neither A nor C: the bridge.
    }
    c = {
        "canonical_id": "nar:c",
        "name_ar_normalized": "محمد بن اسماعيل",
        "aliases": [],
        "death_year_ah": 300,
    }
    records = [a, b, c]
    group = [0, 1, 2]
    threshold = fc._CLUSTER_RATIO_THRESHOLD

    expected = _safe_partition_reference(group, records, threshold=threshold)
    actual = fc._safe_partition(group, records, threshold=threshold)

    assert actual == expected
    # And concretely: A and C (death-conflict) never share a sub-cluster.
    for sub in actual:
        assert not (0 in sub and 2 in sub)
    # B greedily joins A's sub-cluster (created first in canonical_id order);
    # C opens its own. Two sub-clusters total.
    assert sorted(len(s) for s in actual) == [1, 2]


def test_safe_partition_singleton_and_empty() -> None:
    """Degenerate inputs behave identically (guards the ``len(group) <= 1`` path)."""
    records = [{"canonical_id": "nar:x", "name_ar_normalized": "محمد", "aliases": []}]
    threshold = fc._CLUSTER_RATIO_THRESHOLD
    assert fc._safe_partition([], records, threshold=threshold) == [[]]
    assert fc._safe_partition([0], records, threshold=threshold) == [[0]]
    assert fc._safe_partition([0], records, threshold=threshold) == _safe_partition_reference(
        [0], records, threshold=threshold
    )
