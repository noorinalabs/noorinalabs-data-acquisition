"""Tests for the DuckDB Parquet-exploration helper (da#273).

Covers the pure dataset-planning logic (view naming + schema-guarded shard
grouping) against the real on-disk dataset names, and an end-to-end registration
over tiny synthetic Parquet files (read-only, tmp paths — never the real corpus).
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.tools import duck

# The real staging/curated file stems this tool must group correctly (from the
# wave-23 data dirs). Schema keys are stand-ins: shards of one dataset share a key.
_STAGING_SCHEMAS: dict[str, str] = {
    "collections_bihar": "collections",
    "collections_fawaz": "collections",
    "collections_sunnah_scraped": "collections",
    "hadiths_bihar": "hadiths",
    "hadiths_sanadset": "hadiths",
    "hadiths_sunnah_scraped": "hadiths",
    "narrator_aliases_itqan": "aliases",
    "narrator_mentions_lk": "mentions",
    "narrator_mentions_sanadset": "mentions",
    "narrators_bio_itqan": "bio",
    "narrators_bio_kaggle": "bio",
    "narrators_bio_muhaddithat": "bio",
    "network_edges_itqan": "edges",
    "network_edges_mis": "edges",
    "parallel_links": "parallel",
}


def _plan(schemas: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Run ``plan_datasets`` over stems with a dict-driven schema key."""
    files = [Path(f"/d/{stem}.parquet") for stem in schemas]

    def schema_key(p: Path) -> Hashable:
        return schemas[p.stem]

    return {
        ds.name: tuple(f.stem for f in ds.files) for ds in duck.plan_datasets(files, schema_key)
    }


def test_every_file_gets_a_per_file_view() -> None:
    plan = _plan(_STAGING_SCHEMAS)
    for stem in _STAGING_SCHEMAS:
        assert stem in plan, f"{stem} must have its own per-file view"
        assert plan[stem] == (stem,)


def test_shards_form_combined_views() -> None:
    plan = _plan(_STAGING_SCHEMAS)
    assert set(plan["collections"]) == {
        "collections_bihar",
        "collections_fawaz",
        "collections_sunnah_scraped",
    }
    assert set(plan["hadiths"]) == {"hadiths_bihar", "hadiths_sanadset", "hadiths_sunnah_scraped"}
    assert set(plan["narrator_mentions"]) == {"narrator_mentions_lk", "narrator_mentions_sanadset"}
    assert set(plan["network_edges"]) == {"network_edges_itqan", "network_edges_mis"}


def test_longest_prefix_wins_for_same_fileset() -> None:
    """narrators_bio_* names the view ``narrators_bio``, never the shorter ``narrators``."""
    plan = _plan(_STAGING_SCHEMAS)
    assert "narrators_bio" in plan
    assert set(plan["narrators_bio"]) == {
        "narrators_bio_itqan",
        "narrators_bio_kaggle",
        "narrators_bio_muhaddithat",
    }
    assert "narrators" not in plan


def test_schema_mismatch_blocks_bogus_grouping() -> None:
    """aliases_* and mentions_* share the ``narrator`` prefix but differ in schema → no merge."""
    plan = _plan(_STAGING_SCHEMAS)
    assert "narrator" not in plan


def test_single_file_dataset_has_no_combined_view() -> None:
    plan = _plan(_STAGING_SCHEMAS)
    # parallel_links is the only parallel_* file → per-file view only.
    assert plan["parallel_links"] == ("parallel_links",)
    assert "parallel" not in plan


def _write_parquet(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    cols = {f.name: pa.array([r[f.name] for r in rows], type=f.type) for f in schema}
    pq.write_table(pa.table(cols, schema=schema), path)


def test_build_registry_end_to_end(tmp_path: Path) -> None:
    """A real DuckDB session registers the expected views and queries them read-only."""
    staging = tmp_path / "staging"
    curated = tmp_path / "curated"
    staging.mkdir()
    curated.mkdir()

    h_schema = pa.schema([("hadith_id", pa.string()), ("matn", pa.string())])
    _write_parquet(staging / "hadiths_a.parquet", [{"hadith_id": "h1", "matn": "x"}], h_schema)
    _write_parquet(staging / "hadiths_b.parquet", [{"hadith_id": "h2", "matn": "y"}], h_schema)

    canon_schema = pa.schema([("canonical_id", pa.string()), ("mention_count", pa.int32())])
    _write_parquet(
        curated / "narrators_canonical.parquet",
        [{"canonical_id": "nar:1", "mention_count": 7}],
        canon_schema,
    )

    registry = duck.build_registry(staging, curated)
    names = {v.name for v in registry.views}
    # per-file + combined
    assert {"staging_hadiths_a", "staging_hadiths_b", "staging_hadiths"} <= names
    assert "curated_narrators_canonical" in names

    con = registry.connection
    # Combined view unions both shards.
    assert con.sql("SELECT count(*) FROM staging_hadiths").fetchone()[0] == 2
    # Curated single-file view reads through.
    assert con.sql("SELECT mention_count FROM curated_narrators_canonical").fetchone()[0] == 7


def test_missing_dirs_are_tolerated(tmp_path: Path) -> None:
    """Absent data dirs yield an empty (but usable) registry, not a crash."""
    registry = duck.build_registry(tmp_path / "nope", tmp_path / "also_nope")
    assert registry.views == []


def test_query_helper_csv_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (tmp_path / "curated").mkdir()
    schema = pa.schema([("n", pa.int64())])
    _write_parquet(staging / "nums.parquet", [{"n": 1}, {"n": 2}], schema)

    rc = duck.main(["--staging", str(staging), "--curated", str(tmp_path / "curated"), "--list"])
    assert rc == 0
    assert "staging_nums" in capsys.readouterr().out

    rc = duck.main(
        [
            "--staging",
            str(staging),
            "--curated",
            str(tmp_path / "curated"),
            "-c",
            "SELECT sum(n) AS total FROM staging_nums",
            "--csv",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "total" in out
    assert "3" in out
