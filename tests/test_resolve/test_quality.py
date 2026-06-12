"""Tests for cross-source resolution quality metrics (da#99)."""

from __future__ import annotations

from src.resolve.quality import pairwise_quality


class TestPairwiseQuality:
    def test_perfect_clustering(self) -> None:
        """Identical predicted and gold clusters → precision = recall = 1."""
        gold = {"m1": "A", "m2": "A", "m3": "B"}
        pred = {"m1": "x", "m2": "x", "m3": "y"}  # labels differ, partition matches
        q = pairwise_quality(pred, gold)
        assert q.precision == 1.0
        assert q.recall == 1.0
        assert q.f1 == 1.0
        assert q.true_positive_pairs == 1  # (m1, m2)

    def test_over_merge_lowers_precision(self) -> None:
        """Merging two distinct narrators → precision < 1, recall = 1."""
        gold = {"m1": "A", "m2": "A", "m3": "B"}
        pred = {"m1": "x", "m2": "x", "m3": "x"}  # all one cluster (over-merge)
        q = pairwise_quality(pred, gold)
        assert q.recall == 1.0  # the true (m1,m2) pair is still together
        assert q.precision < 1.0  # but (m1,m3)/(m2,m3) are wrong
        assert q.true_positive_pairs == 1
        assert q.predicted_pairs == 3

    def test_under_merge_lowers_recall(self) -> None:
        """Splitting one narrator across clusters → recall < 1, precision = 1."""
        gold = {"m1": "A", "m2": "A", "m3": "A"}
        pred = {"m1": "x", "m2": "x", "m3": "y"}  # m3 split off (under-merge)
        q = pairwise_quality(pred, gold)
        assert q.precision == 1.0
        assert q.recall < 1.0
        assert q.true_positive_pairs == 1
        assert q.gold_pairs == 3

    def test_all_singletons(self) -> None:
        """No same-cluster pairs on either side → vacuously perfect."""
        gold = {"m1": "A", "m2": "B"}
        pred = {"m1": "x", "m2": "y"}
        q = pairwise_quality(pred, gold)
        assert q.precision == 1.0
        assert q.recall == 1.0
        assert q.predicted_pairs == 0
        assert q.gold_pairs == 0

    def test_only_overlapping_keys_scored(self) -> None:
        """Mentions absent from the gold sample are ignored, not penalized."""
        gold = {"m1": "A", "m2": "A"}  # labelled sample of 2
        pred = {"m1": "x", "m2": "x", "m3": "x", "m4": "z"}  # full prediction
        q = pairwise_quality(pred, gold)
        assert q.precision == 1.0
        assert q.recall == 1.0
        assert q.gold_pairs == 1
        assert q.predicted_pairs == 1  # only (m1, m2) among scored keys
