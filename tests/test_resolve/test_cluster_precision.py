"""Precision-validation harness for fuzzy cross-source narrator clustering (da#138).

The fuzzy pass (``src.resolve.fuzzy_cluster``) trades a little precision for recall:
it merges high-confidence name variants of one narrator that the exact-name pass
leaves split. The risk is *over-merging* — collapsing two genuinely different
people into one canonical narrator. ``test_fuzzy_cluster`` already proves a handful
of individual guards; this module is the aggregate **precision guard**: a labeled
gold set of narrator records run end-to-end through ``cluster_assignment`` and
scored with ``quality.pairwise_quality``, asserting precision stays at its
documented baseline so a future change that loosens clustering is caught here.

Findings that motivated the harness (measured on ``cluster_precision_gold.json``):

* Before da#138 the clustering **falsely merged a nasab reversal** — ``محمد بن عبد
  الله`` (Muḥammad b. ʿAbdullāh) and ``عبد الله بن محمد`` (ʿAbdullāh b. Muḥammad),
  two different people — because ``rapidfuzz.token_set_ratio`` is order-insensitive
  and scores that pair a perfect ``100``. With no death-year/gender metadata to
  corroborate, the merge went through: precision ``0.667`` / **false-merge rate
  ``0.333``** on this set. Raising the numeric threshold cannot fix a 100 score and
  would only cost recall, so the fix is the token-order guard, not a higher bar.
* With the da#138 token-order guard in place the same set scores precision
  ``1.000`` (**false-merge rate ``0.000``**) at recall ``1.000`` — the legitimate
  cross-source variants still merge. The sensitivity test below re-introduces the
  regression to prove this harness would catch it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.parse.identity import make_canonical_id
from src.resolve import fuzzy_cluster
from src.resolve.fuzzy_cluster import cluster_assignment
from src.resolve.quality import pairwise_quality
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic

# The documented precision floor for the labeled gold set. The fuzzy pass must not
# over-merge any distinct narrator in the set — every predicted same-narrator pair
# must be a true one (false-merge rate 0.0). A drop here means a change loosened
# clustering precision; investigate before lowering this number.
PRECISION_BASELINE = 1.0

# Recall floor on the same set: a precision guard is worthless if it is satisfied
# by refusing every merge, so we also pin recall — the legitimate cross-source
# variants in the fixture must still cluster.
RECALL_FLOOR = 1.0

_FIXTURE = Path(__file__).parent / "fixtures" / "cluster_precision_gold.json"


def _load_gold() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build canonical records + the gold ``canonical_id -> person`` map from JSON.

    Each fixture entry becomes a full ``NARRATORS_CANONICAL_SCHEMA`` record keyed,
    like the real producers, on ``make_canonical_id(normalize_arabic(name_ar))`` so
    the harness exercises the genuine identity contract. Distinct people never share
    a normalized name (asserted), so canonical ids are unique and the gold map is
    lossless.
    """
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    gold: dict[str, str] = {}
    for entry in raw["records"]:
        norm = normalize_arabic(entry["name_ar"])
        canonical_id = make_canonical_id(norm)
        record: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
        record.update(
            {
                "canonical_id": canonical_id,
                "name_ar": entry["name_ar"],
                "name_ar_normalized": norm,
                "aliases": [normalize_arabic(a) for a in entry.get("aliases", [])],
                "source_ids": [],
                "source_corpora": entry.get("source_corpora", []),
                "mention_count": entry.get("mention_count", 1),
            }
        )
        for field in ("death_year_ah", "gender", "external_id"):
            if field in entry:
                record[field] = entry[field]
        records.append(record)
        gold[canonical_id] = entry["gold_person"]
    return records, gold


def test_gold_fixture_is_well_formed() -> None:
    """The fixture has no distinct-person name collisions and ≥1 true merge pair."""
    records, gold = _load_gold()
    ids = [r["canonical_id"] for r in records]
    assert len(ids) == len(set(ids)), "distinct records collide on canonical_id"
    # At least one gold cluster has >1 member, else recall is vacuous.
    person_counts: dict[str, int] = {}
    for person in gold.values():
        person_counts[person] = person_counts.get(person, 0) + 1
    assert any(n > 1 for n in person_counts.values())


def test_clustering_precision_meets_baseline() -> None:
    """End-to-end: clustering the gold set over-merges nothing (precision floor)."""
    records, gold = _load_gold()
    predicted = cluster_assignment(records)
    quality = pairwise_quality(predicted, gold)

    assert quality.precision >= PRECISION_BASELINE, (
        f"clustering precision regressed: {quality.summary()} "
        f"(false-merge rate {quality.false_merge_rate:.3f})"
    )
    # Precision floor of 1.0 means zero false merges — keep the two phrasings in sync.
    assert quality.false_merge_rate <= 1.0 - PRECISION_BASELINE
    # And the guard is not met by under-merging: the real variants still cluster.
    assert quality.recall >= RECALL_FLOOR, f"recall regressed: {quality.summary()}"


def test_nasab_reversal_does_not_false_merge() -> None:
    """The da#138 trap pair (reversed nasab, different people) stays in two clusters."""
    records, _ = _load_gold()
    forward = make_canonical_id(normalize_arabic("محمد بن عبد الله"))
    reversed_ = make_canonical_id(normalize_arabic("عبد الله بن محمد"))
    predicted = cluster_assignment(records)
    assert predicted[forward] != predicted[reversed_]


def test_harness_detects_a_precision_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling the da#138 token-order guard must drop the harness's precision.

    This is the harness's own meta-test: if a future refactor silently removed the
    order guard, ``test_clustering_precision_meets_baseline`` must start failing.
    We simulate that removal by neutralizing ``_token_order_consistent`` and confirm
    precision falls below the baseline and the false-merge rate rises — i.e. the
    reversed-nasab pair gets wrongly merged again.
    """
    records, gold = _load_gold()
    monkeypatch.setattr(fuzzy_cluster, "_token_order_consistent", lambda a, b: True)

    regressed = pairwise_quality(cluster_assignment(records), gold)
    assert regressed.precision < PRECISION_BASELINE
    assert regressed.false_merge_rate > 0.0
