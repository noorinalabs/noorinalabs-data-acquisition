"""Tests for the bio-direct canonical promoter (da#92a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.identity import make_canonical_id
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve.bio_promote import promote_bios_to_canonical
from src.utils.arabic import normalize_arabic


def _write_bios(staging: Path, suffix: str, rows: list[dict[str, Any]]) -> Path:
    """Write a narrators_bio_<suffix>.parquet with schema defaults filled in."""
    full = []
    for r in rows:
        base = {f.name: None for f in NARRATOR_BIO_SCHEMA}
        base.update(r)
        full.append(base)
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATOR_BIO_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    path = staging / f"narrators_bio_{suffix}.parquet"
    pq.write_table(table, path)
    return path


def test_promotes_bios_to_canonical_with_nar_ids(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "سعيد بن سماك"
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": normalize_arabic(name),
                "trustworthiness": "matruk",
                "external_id": "320",
            }
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    table = pq.read_table(path)
    rows = table.to_pylist()
    assert len(rows) == 1
    rec = rows[0]
    assert rec["canonical_id"].startswith("nar:")
    assert rec["canonical_id"] == make_canonical_id(normalize_arabic(name))
    assert rec["trustworthiness"] == "matruk"
    assert rec["external_id"] == "320"
    assert rec["source_ids"] == ["itqan:320"]
    assert rec["mention_count"] == 0


def test_same_name_dedups_to_one_canonical(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "عمر بن حفص"
    norm = normalize_arabic(name)
    # The second bio carries an int death year; back-fill must keep it an int
    # (regression: it was previously stringified, breaking the int32 column).
    _write_bios(
        staging,
        "itqan",
        [
            {"bio_id": "itqan:1", "source": "itqan", "name_ar": name, "name_ar_normalized": norm},
            {
                "bio_id": "itqan:2",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": norm,
                "death_year_ah": 231,
            },
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 1
    assert sorted(rows[0]["source_ids"]) == ["itqan:1", "itqan:2"]
    assert rows[0]["death_year_ah"] == 231


def test_source_filter(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:1", "source": "itqan", "name_ar": "أ", "name_ar_normalized": "ا"}],
    )
    _write_bios(
        staging,
        "muhaddithat",
        [{"bio_id": "m:1", "source": "muhaddithat", "name_ar": "ب", "name_ar_normalized": "ب"}],
    )
    path = promote_bios_to_canonical(staging, out_dir, sources={"itqan"})
    assert path is not None
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["source_ids"] == ["itqan:1"]


def test_merges_into_existing_canonical(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    # Pre-existing canonical (as if from the disambiguator) with a different id.
    from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA

    existing = {f.name: [None] for f in NARRATORS_CANONICAL_SCHEMA}
    existing["canonical_id"] = ["nar:preexisting"]
    existing["source_ids"] = [["sanadset:9"]]
    existing["aliases"] = [[]]
    existing["mention_count"] = [5]
    pq.write_table(
        pa.table(existing, schema=NARRATORS_CANONICAL_SCHEMA),
        out_dir / "narrators_canonical.parquet",
    )

    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:1", "source": "itqan", "name_ar": "ج", "name_ar_normalized": "ج"}],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    ids = {r["canonical_id"] for r in pq.read_table(path).to_pylist()}
    assert "nar:preexisting" in ids  # not clobbered
    assert len(ids) == 2  # plus the new itqan-derived narrator


def test_no_bio_files_returns_none(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    assert promote_bios_to_canonical(staging, out_dir) is None
