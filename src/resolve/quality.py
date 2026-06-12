"""Cluster-quality metrics for narrator-identity resolution (da#99).

Pairwise precision / recall / F1 over predicted vs. gold narrator clusters — the
standard entity-resolution dedup-quality measure. Each mention is assigned a
cluster label (its canonical narrator id for the prediction; the true narrator
identity for the gold). We compare the set of *same-cluster mention pairs*:

* **precision** — of all mention pairs the resolver placed together, the
  fraction that truly are the same narrator (over-merging lowers this).
* **recall** — of all truly-same mention pairs, the fraction the resolver caught
  (under-merging / missed variants lowers this).

Pure Python, no ML dependencies, so it runs in unit tests and CI. Only mentions
present in BOTH the predicted and gold maps are scored, so a labelled *sample*
can be evaluated against a full prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

__all__ = ["ClusterQuality", "pairwise_quality"]


@dataclass(frozen=True)
class ClusterQuality:
    """Pairwise precision/recall/F1 of a clustering against a gold standard."""

    precision: float
    recall: float
    f1: float
    true_positive_pairs: int
    predicted_pairs: int
    gold_pairs: int

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"precision={self.precision:.3f} recall={self.recall:.3f} f1={self.f1:.3f} "
            f"(tp={self.true_positive_pairs}, pred={self.predicted_pairs}, "
            f"gold={self.gold_pairs})"
        )


def _same_cluster_pairs(assignment: Mapping[str, str]) -> set[tuple[str, str]]:
    """Set of unordered (a, b) mention-id pairs sharing a cluster label."""
    groups: dict[str, list[str]] = {}
    for key, label in assignment.items():
        groups.setdefault(label, []).append(key)
    pairs: set[tuple[str, str]] = set()
    for members in groups.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return pairs


def pairwise_quality(
    predicted: Mapping[str, str],
    gold: Mapping[str, str],
) -> ClusterQuality:
    """Compute pairwise precision/recall/F1 of *predicted* against *gold*.

    *predicted* and *gold* both map ``mention_id -> cluster_label`` (any hashable
    label). Only mention ids present in both are scored. With no positive pairs on
    a side the corresponding rate is defined as 1.0 (vacuously perfect).
    """
    keys = set(predicted) & set(gold)
    pred_pairs = _same_cluster_pairs({k: predicted[k] for k in keys})
    gold_pairs = _same_cluster_pairs({k: gold[k] for k in keys})

    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 1.0
    recall = tp / len(gold_pairs) if gold_pairs else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ClusterQuality(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive_pairs=tp,
        predicted_pairs=len(pred_pairs),
        gold_pairs=len(gold_pairs),
    )
