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
import pytest
from rapidfuzz import fuzz

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.resolve import fuzzy_cluster as fc
from src.resolve.disambiguate import _MERGE_LOG_SCHEMA
from src.resolve.fuzzy_cluster import (
    _match_keys,
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


# ---------------------------------------------------------------------------
# da#380 — the death-year veto is weighted by `death_year_provenance`. An
# `uncorroborated` year (a weak Stage-2 fuzzy bio year, persisted but excluded
# from steering identity — disambiguate.py da#356) must NOT veto a real merge at
# the tight `_DEATH_YEAR_TOLERANCE`; only a gross spread still blocks it. These
# guards go RED if the provenance tag is ignored (the pre-da#380 behaviour).
# ---------------------------------------------------------------------------
def test_uncorroborated_death_year_does_not_veto_small_spread() -> None:
    """A small disagreement with one `uncorroborated` year still merges.

    Spread = 5 AH — beyond `_DEATH_YEAR_TOLERANCE` (2) but far under
    `_UNCORROBORATED_DEATH_SPREAD` (50). Ignoring the provenance tag (the old
    year-blind veto) would split this real pair — this assertion is the guard.
    """
    a = _rec("محمد بن عبد الله", source_corpora=["itqan"], death_year_ah=150)
    b = _rec(
        "محمد بن عبد الله",
        source_corpora=["sanadset"],
        death_year_ah=155,
        death_year_provenance="uncorroborated",
    )
    # Predicate-level guard: the discount is what clears the veto.
    assert fc._death_years_conflict(a, b) is False
    assert fc._death_years_conflict(b, a) is False  # symmetric
    # End-to-end: the pair merges into one cluster.
    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [2]


def test_uncorroborated_gross_death_spread_still_blocks() -> None:
    """A generations-apart namesake still blocks even when a year is uncorroborated."""
    a = _rec("محمد بن عبد الله", source_corpora=["itqan"], death_year_ah=150)
    b = _rec(
        "محمد بن عبد الله",
        source_corpora=["sanadset"],
        death_year_ah=320,  # spread 170 > _UNCORROBORATED_DEATH_SPREAD (50)
        death_year_provenance="uncorroborated",
    )
    assert fc._death_years_conflict(a, b) is True
    clusters = cluster_records([a, b])
    assert sorted(len(g) for g in clusters) == [1, 1]  # stayed apart


def test_corroborated_death_years_keep_tight_tolerance() -> None:
    """Two `corroborated` years 5 AH apart still conflict — the tight band is preserved.

    Same spread (5) as the merging uncorroborated case above; here BOTH years are
    corroborated, so the discount must NOT apply and the veto must still fire.
    """
    a = _rec(
        "محمد بن عبد الله",
        source_corpora=["itqan"],
        death_year_ah=150,
        death_year_provenance="corroborated",
    )
    b = _rec(
        "محمد بن عبد الله",
        source_corpora=["sanadset"],
        death_year_ah=155,
        death_year_provenance="corroborated",
    )
    assert fc._death_years_conflict(a, b) is True
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


def test_merge_retags_bio_only_survivor_as_attested(tmp_path: Path) -> None:
    """da#370: folding a bio-only (mention_count 0) variant into an attested narrator
    lifts the summed mention_count above 0, so the survivor must RE-DERIVE to
    isnad_attested — a carried-over biographical_only would be stale."""
    attested = _rec(
        "محمد بن اسماعيل البخاري",
        mention_count=5,
        attestation="isnad_attested",
        death_year_ah=256,
    )
    bio_only = _rec(
        "محمد بن اسماعيل",
        mention_count=0,
        attestation="biographical_only",
        death_year_ah=256,
    )
    canonical = tmp_path / "narrators_canonical.parquet"
    _write_canonical(canonical, [attested, bio_only])

    metrics = cluster_canonical_narrators(canonical)
    assert metrics.merged_records == 1

    rows = pq.read_table(canonical).to_pylist()
    assert len(rows) == 1
    merged = rows[0]
    assert merged["mention_count"] == 5
    assert merged["attestation"] == "isnad_attested"


def test_merge_cluster_drops_bare_relational_alias() -> None:
    """da#391: the cluster alias-union never carries a bare relational-pronoun.

    A member's own spelling and its existing aliases both become aliases of the
    survivor — the ``list<string>`` blind spot ``clean_narrator_name`` never reaches.
    A deixis token ("خالته") riding in a member's alias list, or as a member's whole
    name, must not propagate into the merged aliases; a real variant spelling still
    does. RED on pre-da#391 code: ``خالته`` is appended unconditionally.
    """
    rep = _rec("محمد بن اسماعيل البخاري", mention_count=10)
    var = _rec(
        "محمد بن اسماعيل",
        mention_count=2,
        aliases=[normalize_arabic("خالته"), normalize_arabic("محمد البخاري")],
    )

    merged = fc._merge_cluster([rep, var])

    assert normalize_arabic("محمد بن اسماعيل") in merged["aliases"]  # real variant kept
    assert normalize_arabic("محمد البخاري") in merged["aliases"]  # real alias kept
    assert normalize_arabic("خالته") not in merged["aliases"]  # deixis dropped


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


def _build_cluster_fixture(out_dir: Path) -> tuple[Path, Path, str, str]:
    """A two-record canonical set (``var`` absorbed into ``rep``) plus a merge_log
    with one row on the absorbed id and one on the survivor. Returns the canonical
    path, merge_log path, absorbed id and survivor id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = _rec("محمد بن اسماعيل البخاري", source_corpora=["itqan"], mention_count=10)
    var = _rec("محمد بن اسماعيل", source_corpora=["sanadset"], mention_count=2)
    survivor_id = make_canonical_id(normalize_arabic("محمد بن اسماعيل البخاري"))
    absorbed_id = var["canonical_id"]

    canonical = out_dir / "narrators_canonical.parquet"
    _write_canonical(canonical, [rep, var])

    merge_log = out_dir / "merge_log.parquet"
    log_rows = {
        "canonical_id": [absorbed_id, survivor_id],
        "mention_id": ["m1", "m2"],
        "mention_text": ["محمد بن اسماعيل", "محمد بن اسماعيل البخاري"],
        "merge_stage": ["fuzzy", "exact"],
        "score": [0.95, 1.0],
    }
    pq.write_table(pa.table(log_rows, schema=_MERGE_LOG_SCHEMA), merge_log)
    return canonical, merge_log, absorbed_id, survivor_id


def _merge_log_orphans(canonical: Path, merge_log: Path) -> list[str]:
    live = set(pq.read_table(canonical).column("canonical_id").to_pylist())
    return [c for c in pq.read_table(merge_log).column("canonical_id").to_pylist() if c not in live]


def test_merge_log_orphaned_without_remap(tmp_path: Path) -> None:
    """Control (da#313 red witness): clustering that rewrites the canonical set but
    is NOT told about the merge_log leaves the absorbed id dangling in the log."""
    canonical, merge_log, absorbed_id, _survivor = _build_cluster_fixture(tmp_path / "curated")

    cluster_canonical_narrators(canonical)  # merge_log_path omitted

    orphans = _merge_log_orphans(canonical, merge_log)
    assert orphans == [absorbed_id]  # the absorbed id no longer exists as a canonical node


def test_merge_log_remapped_to_survivor(tmp_path: Path) -> None:
    """da#313: passing ``merge_log_path`` routes every absorbed id to its cluster
    survivor, so no merge_log row references a dissolved node — and no row is
    dropped (identity routing, not a silent filter)."""
    canonical, merge_log, absorbed_id, survivor_id = _build_cluster_fixture(tmp_path / "curated")

    metrics = cluster_canonical_narrators(canonical, merge_log_path=merge_log)
    assert metrics.merge_log_remapped == 1

    rows = pq.read_table(merge_log).to_pylist()
    assert len(rows) == 2  # both rows preserved — nothing filtered
    by_mention = {r["mention_id"]: r["canonical_id"] for r in rows}
    assert by_mention["m1"] == survivor_id  # absorbed → survivor
    assert by_mention["m2"] == survivor_id  # already on survivor — untouched
    assert _merge_log_orphans(canonical, merge_log) == []
    assert absorbed_id not in by_mention.values()


def test_merge_log_remap_is_idempotent(tmp_path: Path) -> None:
    """A second clustering pass over the already-collapsed set remaps nothing."""
    canonical, merge_log, _absorbed, _survivor = _build_cluster_fixture(tmp_path / "curated")

    cluster_canonical_narrators(canonical, merge_log_path=merge_log)
    second = cluster_canonical_narrators(canonical, merge_log_path=merge_log)
    assert second.merge_log_remapped == 0
    assert _merge_log_orphans(canonical, merge_log) == []


def test_empty_and_missing_inputs_are_noops(tmp_path: Path) -> None:
    """Missing or empty canonical table → a clean no-op, not a crash."""
    missing = tmp_path / "absent.parquet"
    m = cluster_canonical_narrators(missing)
    assert m.input_records == 0 and m.output_records == 0

    empty = tmp_path / "narrators_canonical.parquet"
    _write_canonical(empty, [])
    m2 = cluster_canonical_narrators(empty)
    assert m2.input_records == 0 and m2.merged_records == 0


# ---------------------------------------------------------------------------
# da#270: per-record match-key + blocking-token caps (throughput guard)
# ---------------------------------------------------------------------------
def test_match_keys_caps_alias_explosion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record with hundreds of aliases contributes only the capped key count.

    The O(K²) block cdist is the clustering bottleneck (da#270); capping keys per
    record bounds a pathological record's contribution. The name is always kept
    first so exact-name/single-token matching is never lost.
    """
    monkeypatch.setattr(fc, "_MAX_MATCH_KEYS_PER_RECORD", 8)
    rec = _rec("محمد بن اسماعيل البخاري", aliases=[f"اسم رقم {i}" for i in range(200)])
    keys = _match_keys(rec)
    assert len(keys) == 8
    assert keys[0] == rec["name_ar_normalized"], "name must survive the cap, first"


def test_match_keys_cap_none_keeps_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is disable-able (exact pre-da#270 behaviour for the harness)."""
    monkeypatch.setattr(fc, "_MAX_MATCH_KEYS_PER_RECORD", None)
    rec = _rec("محمد البخاري", aliases=[f"variant {i}" for i in range(50)])
    assert len(_match_keys(rec)) == 51  # name + 50 aliases


def test_alias_rich_record_still_clusters_on_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capping keys must not lose a merge justified by the name itself.

    Even truncated to a tiny key budget, an alias-bloated record still carries its
    name first, so a genuine nasab-expansion variant still clusters with it.
    """
    monkeypatch.setattr(fc, "_MAX_MATCH_KEYS_PER_RECORD", 4)
    bloated = _rec(
        "محمد بن اسماعيل البخاري",
        source_corpora=["itqan"],
        death_year_ah=256,
        aliases=[f"لقب {i}" for i in range(300)],
    )
    variant = _rec("محمد بن اسماعيل", source_corpora=["sanadset"], death_year_ah=256)
    clusters = cluster_records([bloated, variant])
    assert sorted(len(g) for g in clusters) == [2]  # merged despite the cap


def test_blocking_token_cap_bounds_reach(monkeypatch: pytest.MonkeyPatch) -> None:
    """The blocking-token cap trims a pathological record's block reach.

    With the cap at 2, a record is blocked only on its two lowest-sorted
    significant tokens; a genuine variant that shares *only* higher-sorted tokens
    is no longer co-blocked, so it does not merge. Uncapped it does. This is the
    lever that stops a 1,000-token pollution "name" from joining ~500k blocks.
    """
    # Shared tokens ثيثا/ثيثب sort AFTER the padding tokens اا/اب, so a cap of 2
    # (which keeps only اا/اب on ``poll``) drops the pair that would block these two
    # together. ``variant`` is a pure token-subset so its token_set_ratio is 100.
    poll = _rec("اا اب ثيثا ثيثب", source_corpora=["x"])
    variant = _rec("ثيثا ثيثب", source_corpora=["y"])

    monkeypatch.setattr(fc, "_MAX_BLOCKING_TOKENS_PER_RECORD", None)
    uncapped = cluster_records([poll, variant])

    monkeypatch.setattr(fc, "_MAX_BLOCKING_TOKENS_PER_RECORD", 2)
    capped = cluster_records([poll, variant])

    assert sorted(len(g) for g in uncapped) == [2], "shared tokens merge when uncapped"
    assert sorted(len(g) for g in capped) == [1, 1], "cap drops the late-token block"


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
