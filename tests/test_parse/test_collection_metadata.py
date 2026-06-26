"""Tests for sourced Collection metadata enrichment (da#230 / da#153 item #5).

Covers the curated, *sourced* fill of ``name_ar`` + ``expected_count`` for
collections that arrive blank from upstream — riyadussalihin is the named
example — and verifies the enrichment flows through the Sunnah.com parser into
the emitted ``collections_*`` Parquet, and that the Collection model carries the
new field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from src.models.collection import Collection
from src.parse.collection_metadata import (
    COLLECTION_METADATA,
    apply_collection_metadata,
    lookup,
)
from src.parse.schemas import COLLECTION_SCHEMA
from src.parse.sunnah_api import run


def _blank_sunnah_row(slug: str) -> dict[str, object]:
    """A collection row as a source parser would emit it before enrichment."""
    return {
        "collection_id": f"sunnah:{slug}",
        "name_ar": None,
        "name_en": slug,
        "compiler_name": None,
        "compilation_year_ah": None,
        "sect": "sunni",
        "total_hadiths": None,
        "source_corpus": "sunnah",
    }


class TestCuratedTable:
    def test_riyadussalihin_is_sourced(self) -> None:
        meta = lookup("riyadussalihin")
        assert meta is not None
        assert meta.name_ar == "رياض الصالحين"
        assert meta.expected_count == 1896
        # "Sourced" — the value must carry where it came from.
        assert meta.source
        assert meta.source.startswith("http")

    def test_every_entry_carries_provenance(self) -> None:
        # No unsourced guesses: every curated value cites a source.
        for slug, meta in COLLECTION_METADATA.items():
            assert meta.source, f"{slug} is missing a provenance citation"
            assert meta.expected_count is None or meta.expected_count > 0

    def test_lookup_is_case_insensitive(self) -> None:
        assert lookup("RiyadusSalihin") is lookup("riyadussalihin")


class TestApplyCollectionMetadata:
    def test_fills_blank_name_ar_and_expected_count(self) -> None:
        enriched = apply_collection_metadata(_blank_sunnah_row("riyadussalihin"))
        assert enriched["name_ar"] == "رياض الصالحين"
        assert enriched["expected_count"] == 1896

    def test_does_not_override_source_supplied_name_ar(self) -> None:
        row = _blank_sunnah_row("riyadussalihin")
        row["name_ar"] = "اسم من المصدر"
        enriched = apply_collection_metadata(row)
        assert enriched["name_ar"] == "اسم من المصدر"
        # expected_count is still filled (the source did not supply one).
        assert enriched["expected_count"] == 1896

    def test_unknown_slug_gets_present_but_null_expected_count(self) -> None:
        enriched = apply_collection_metadata(_blank_sunnah_row("not-a-real-collection"))
        # Key must exist so the row satisfies COLLECTION_SCHEMA ...
        assert "expected_count" in enriched
        # ... but absence of a curated entry is never a guess.
        assert enriched["expected_count"] is None
        assert enriched["name_ar"] is None

    def test_does_not_mutate_input(self) -> None:
        row = _blank_sunnah_row("riyadussalihin")
        apply_collection_metadata(row)
        assert row["name_ar"] is None
        assert "expected_count" not in row


class TestModelCarriesField:
    def test_collection_accepts_expected_count(self) -> None:
        c = Collection(
            id="col:riyadussalihin",
            name_ar="رياض الصالحين",
            name_en="Riyad as-Salihin",
            sect="sunni",  # type: ignore[arg-type]
            expected_count=1896,
        )
        assert c.expected_count == 1896

    def test_expected_count_defaults_none(self) -> None:
        c = Collection(
            id="col:x",
            name_ar="ا",
            name_en="X",
            sect="sunni",  # type: ignore[arg-type]
        )
        assert c.expected_count is None


class TestSunnahParserEmitsExpectedCount:
    def test_riyadussalihin_enriched_in_parquet(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        sunnah_dir = raw_dir / "sunnah"
        sunnah_dir.mkdir(parents=True)

        # Source provides no Arabic title and no count for riyadussalihin.
        collections = [{"name": "riyadussalihin"}]
        (sunnah_dir / "collections.json").write_text(json.dumps(collections), encoding="utf-8")

        run(raw_dir, staging_dir)

        coll_path = staging_dir / "collections_sunnah.parquet"
        assert coll_path.exists()
        table = pq.read_table(coll_path)
        assert table.schema.equals(COLLECTION_SCHEMA)
        rows = table.to_pylist()
        riyad = next(r for r in rows if r["collection_id"] == "sunnah:riyadussalihin")
        assert riyad["name_ar"] == "رياض الصالحين"
        assert riyad["expected_count"] == 1896
