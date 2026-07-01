"""Tests for fuzzy cross-source narrator clustering (da#118).

The exact-name pass (#99) collapses narrators only on byte-identical normalized
names. These tests prove the fuzzy increment:

* **recall gain** — variant-spelling records of one person across ≥2 sources that
  the exact pass leaves split now cluster, measured via ``quality.pairwise_quality``
  against a gold standard, at a precision floor; and
* **precision guard** — distinct narrators with similar names but conflicting
  death-year / gender do NOT wrongly merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.resolve.fuzzy_cluster import (
    cluster_assignment,
    cluster_canonical_narrators,
    cluster_records,
)
from src.resolve.quality import pairwise_quality
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA, NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic


def _rec(name: str, **over: Any) -> dict[str, Any]:
    """A canonical record keyed (like the real producers) on its normalized name."""
    norm = normalize_arabic(name)
    base: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    base.update(
        {
            "canonical_id": make_canonical_id(norm),
            "name_ar": name,
            "name_ar_normalized": norm,
            "aliases": [],
            "source_ids": [],
            "source_corpora": [],
            "mention_count": 1,
        }
    )
    base.update(over)
    return base


def _write_canonical(path: Path, records: list[dict[str, Any]]) -> None:
    arrays = {f.name: [r.get(f.name) for r in records] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA), path)


# ---------------------------------------------------------------------------
# Recall increment
# ---------------------------------------------------------------------------
def test_variant_spelling_across_sources_clusters() -> None:
    """A nasab-expanded variant of one person across two sources clusters as one."""
    # Same scholar; Itqan carries the fuller nisba, Sanadset the shorter form.
    a = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"], death_year_ah=256)
    b = _rec("محمد بن اسماعيل", source_corpora=["sanadset"], death_year_ah=256)
    # An unrelated narrator that must stay on its own.
    c = _rec("فاطمة بنت قيس", source_corpora=["muhaddithat"])

    clusters = cluster_records([a, b, c])
    sizes = sorted(len(grp) for grp in clusters)
    assert sizes == [1, 2]  # {a,b} merged, c alone


def test_recall_gain_over_exact_name_baseline() -> None:
    """pairwise_quality recall rises over the exact-name pass at precision = 1.0.

    Gold: a, b are the same person (one cluster); c is distinct. The exact-name
    baseline keeps every record its own cluster (each id maps to itself) → it
    misses the (a,b) gold pair, so recall < 1. The fuzzy pass recovers it.
    """
    a = _rec("احمد بن حنبل الشيباني", source_corpora=["itqan"], death_year_ah=241)
    b = _rec("احمد بن حنبل", source_corpora=["sanadset"], death_year_ah=241)
    c = _rec("مالك بن انس", source_corpora=["itqan"], death_year_ah=179)
    records = [a, b, c]

    gold = {
        a["canonical_id"]: "person-1",
        b["canonical_id"]: "person-1",
        c["canonical_id"]: "person-2",
    }
    # Exact-name baseline: each canonical id is its own cluster.
    exact = {r["canonical_id"]: r["canonical_id"] for r in records}
    fuzzy = cluster_assignment(records)

    exact_q = pairwise_quality(exact, gold)
    fuzzy_q = pairwise_quality(fuzzy, gold)

    assert exact_q.recall < 1.0  # the exact pass under-merges the variant pair
    assert fuzzy_q.recall > exact_q.recall  # the increment
    assert fuzzy_q.recall == 1.0
    assert fuzzy_q.precision == 1.0  # precision floor held — nothing over-merged


def test_alias_drives_a_merge() -> None:
    """A da#94 alias key, not the primary name, carries the match."""
    # Primary names diverge, but a's Itqan alias equals b's primary spelling.
    a = _rec(
        "ابو عبد الله محمد البخاري",
        aliases=[normalize_arabic("محمد بن اسماعيل البخاري")],
        source_corpora=["itqan"],
    )
    b = _rec("محمد بن اسماعيل البخاري", source_corpora=["sanadset"])

    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [2]


# ---------------------------------------------------------------------------
# Precision guards
# ---------------------------------------------------------------------------
def test_conflicting_death_years_block_merge() -> None:
    """Two same-named narrators a generation apart must NOT merge."""
    a = _rec("محمد بن عبد الله", source_corpora=["itqan"], death_year_ah=150)
    b = _rec("محمد بن عبد الله", source_corpora=["sanadset"], death_year_ah=320)
    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [1, 1]  # stayed apart


def test_conflicting_gender_blocks_merge() -> None:
    """Similar names with explicit differing gender must NOT merge."""
    a = _rec("عبد الله بن محمد", source_corpora=["itqan"], gender="male")
    b = _rec("عبد الله بن محمد", source_corpora=["muhaddithat"], gender="female")
    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [1, 1]


def test_distinct_names_do_not_cluster() -> None:
    """Genuinely different narrators sharing a common token stay separate."""
    a = _rec("سفيان الثوري", source_corpora=["itqan"], death_year_ah=161)
    b = _rec("سفيان بن عيينة", source_corpora=["sanadset"], death_year_ah=198)
    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [1, 1]


def test_bare_single_token_subset_does_not_merge() -> None:
    """Over-merge vector 1: a bare given name is a 100-score subset but must NOT merge.

    ``token_set_ratio("محمد", "محمد بن اسماعيل البخاري") == 100`` and the sparse
    bare-name record carries no death-year/gender to trip the other guards — so
    only the ≥2-shared-significant-token rule keeps them apart.
    """
    bare = _rec("محمد", source_corpora=["sanadset"])  # one significant token
    full = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"])
    assert fuzz.token_set_ratio(bare["name_ar_normalized"], full["name_ar_normalized"]) == 100.0
    clusters = cluster_records([bare, full])
    assert sorted(len(g) for g in clusters) == [1, 1]  # stayed split


def test_transitive_bridge_does_not_merge_conflicting_endpoints() -> None:
    """Over-merge vector 2: a bridge must not chain a guard-conflicting pair.

    A (Bukhari, d256) –bridge– B (bare ``محمد بن اسماعيل``, no death year) –bridge–
    C (Kufi, d320): A–B and B–C each pass, but A–C conflicts on death year. The
    transitive union must be re-partitioned so A and C never share a cluster.
    """
    a = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"], death_year_ah=256)
    b = _rec("محمد بن اسماعيل", source_corpora=["sanadset"])  # bridge, no death year
    c = _rec("محمد بن اسماعيل الكوفي", source_corpora=["thaqalayn"], death_year_ah=320)

    clusters = cluster_records([a, b, c])
    cluster_of = {idx: ci for ci, grp in enumerate(clusters) for idx in grp}
    # records passed as [a, b, c] → indices 0, 1, 2.
    assert cluster_of[0] != cluster_of[2]  # A and C never co-cluster
    # No cluster harbours the conflicting A–C pair; B attaches to exactly one side.
    assert sorted(len(g) for g in clusters) == [1, 2]


# ---------------------------------------------------------------------------
# Merge mechanics & identity contract
# ---------------------------------------------------------------------------
def test_merged_record_routes_through_make_canonical_id(tmp_path: Path) -> None:
    """The survivor's id is make_canonical_id(rep name) — never a parallel scheme."""
    rep = _rec(
        "محمد بن اسماعيل البخاري",
        source_corpora=["itqan"],
        mention_count=10,
        death_year_ah=256,
        external_id="320",
    )
    var = _rec(
        "محمد بن اسماعيل",
        source_corpora=["sanadset"],
        mention_count=2,
        death_year_ah=256,
    )
    canonical = tmp_path / "narrators_canonical.parquet"
    _write_canonical(canonical, [rep, var])

    metrics = cluster_canonical_narrators(canonical)
    assert metrics.input_records == 2
    assert metrics.output_records == 1
    assert metrics.merged_records == 1
    assert metrics.cross_source_clusters == 1

    rows = pq.read_table(canonical).to_pylist()
    assert len(rows) == 1
    merged = rows[0]
    # Representative (more mentions) keeps the fuller name; id routes through identity.
    assert merged["canonical_id"] == make_canonical_id(normalize_arabic("محمد بن اسماعيل البخاري"))
    # Provenance unioned; absorbed spelling preserved as an alias; mentions summed.
    assert sorted(merged["source_corpora"]) == ["itqan", "sanadset"]
    assert normalize_arabic("محمد بن اسماعيل") in merged["aliases"]
    assert merged["mention_count"] == 12
    assert merged["external_id"] == "320"


def test_clustering_is_idempotent(tmp_path: Path) -> None:
    """Re-running on an already-clustered table changes nothing."""
    a = _rec("احمد بن حنبل الشيباني", source_corpora=["itqan"], death_year_ah=241, mention_count=5)
    b = _rec("احمد بن حنبل", source_corpora=["sanadset"], death_year_ah=241, mention_count=1)
    canonical = tmp_path / "narrators_canonical.parquet"
    _write_canonical(canonical, [a, b])

    first = cluster_canonical_narrators(canonical)
    assert first.merged_records == 1
    second = cluster_canonical_narrators(canonical)
    assert second.merged_records == 0
    assert second.input_records == 1
    assert second.output_records == 1


def test_mentions_remapped_to_survivor(tmp_path: Path) -> None:
    """Mentions backfilled with an absorbed id are rewritten to the survivor (#109)."""
    rep = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"], mention_count=10)
    var = _rec("محمد بن اسماعيل", source_corpora=["sanadset"], mention_count=2)
    survivor_id = make_canonical_id(normalize_arabic("محمد بن اسماعيل البخاري"))
    absorbed_id = var["canonical_id"]

    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    canonical = out_dir / "narrators_canonical.parquet"
    _write_canonical(canonical, [rep, var])

    mentions = out_dir / "narrator_mentions_resolved.parquet"
    mrows = [
        {f.name: None for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA} | row
        for row in (
            {
                "mention_id": "m1",
                "hadith_id": "h1",
                "source_corpus": "sanadset",
                "position_in_chain": 0,
                "canonical_narrator_id": absorbed_id,
            },
            {
                "mention_id": "m2",
                "hadith_id": "h2",
                "source_corpus": "itqan",
                "position_in_chain": 0,
                "canonical_narrator_id": survivor_id,
            },
        )
    ]
    arrays = {f.name: [r[f.name] for r in mrows] for f in NARRATOR_MENTIONS_RESOLVED_SCHEMA}
    pq.write_table(pa.table(arrays, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA), mentions)

    metrics = cluster_canonical_narrators(canonical, mentions_path=mentions)
    assert metrics.mentions_remapped == 1

    got = {r["mention_id"]: r["canonical_narrator_id"] for r in pq.read_table(mentions).to_pylist()}
    assert got["m1"] == survivor_id  # absorbed → survivor
    assert got["m2"] == survivor_id  # already on survivor — untouched


def test_empty_and_missing_inputs_are_noops(tmp_path: Path) -> None:
    """Missing or empty canonical table → a clean no-op, not a crash."""
    missing = tmp_path / "absent.parquet"
    m = cluster_canonical_narrators(missing)
    assert m.input_records == 0 and m.output_records == 0

    empty = tmp_path / "narrators_canonical.parquet"
    _write_canonical(empty, [])
    m2 = cluster_canonical_narrators(empty)
    assert m2.input_records == 0 and m2.merged_records == 0


def test_unset_precision_defaults_to_unknown(tmp_path: Path) -> None:
    """da#239: records carrying no precision key must round-trip through the
    builder with both precision columns defaulted to UNKNOWN — never null."""
    # Two distinct narrators (no merge) — each _rec leaves precision unset (None).
    a = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"])
    b = _rec("فاطمة بنت قيس", source_corpora=["muhaddithat"])
    canonical = tmp_path / "narrators_canonical.parquet"
    _write_canonical(canonical, [a, b])

    cluster_canonical_narrators(canonical)

    rows = pq.read_table(canonical).to_pylist()
    assert len(rows) == 2
    for rec in rows:
        assert rec["birth_date_precision"] == DatePrecision.UNKNOWN.value
        assert rec["death_date_precision"] == DatePrecision.UNKNOWN.value
        assert rec["birth_date_precision"] is not None
        assert rec["death_date_precision"] is not None
