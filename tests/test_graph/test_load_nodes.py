"""Tests for src.graph.load_nodes — Neo4j node loading with mock client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.graph.load_nodes import LoadResult, load_all_nodes
from tests.test_graph.conftest import (
    MockNeo4jClient,
    write_collections,
    write_hadiths,
    write_historical_events_yaml,
    write_locations_yaml,
    write_narrator_mentions,
    write_narrators_canonical,
)


class TestLoadResult:
    def test_frozen(self) -> None:
        r = LoadResult("Narrator", 5, 2, 1, ["err"])
        with pytest.raises(AttributeError):
            r.node_type = "other"  # type: ignore[misc]

    def test_default_errors_empty(self) -> None:
        r = LoadResult("Hadith", 0, 0, 0)
        assert r.validation_errors == []


class TestLoadNarrators:
    def test_valid_narrators(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:abu-hurayra",
                    "name_ar": "أبو هريرة",
                    "name_en": "Abu Hurayra",
                },
                {"canonical_id": "nar:anas", "name_ar": "أنس", "name_en": "Anas"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        narrator_result = results[0]
        assert narrator_result.node_type == "Narrator"
        assert narrator_result.created + narrator_result.merged == 2
        assert narrator_result.skipped == 0

    def test_narrator_sets_sect_and_corpus(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#103: Narrator nodes carry sect_affiliation + source_corpus/_corpora."""
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:cross-sect",
                    "name_en": "Cross Sect",
                    "source_corpus": "sunnah",
                    "source_corpora": ["sunnah", "thaqalayn"],
                    "sect_affiliation": "neutral",
                },
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        query, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        # The MERGE explicitly SETs each property (no blanket ``SET n += row``).
        assert "n.source_corpus" in query
        assert "n.source_corpora" in query
        assert "n.sect_affiliation" in query
        assert isinstance(batch, list)
        row = batch[0]
        assert row["source_corpus"] == "sunnah"
        assert row["source_corpora"] == ["sunnah", "thaqalayn"]
        assert row["sect_affiliation"] == "neutral"

    def test_narrator_sect_corpus_defaults_when_absent(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A legacy canonical row with null sect/corpus loads with safe defaults."""
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:legacy", "name_en": "Legacy"}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        _, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        assert isinstance(batch, list)
        row = batch[0]
        assert row["source_corpus"] == ""
        assert row["source_corpora"] == []
        assert row["sect_affiliation"] == "unknown"

    def test_invalid_canonical_id_skipped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [
                {"canonical_id": "nar:valid", "name_en": "Valid"},
                {"canonical_id": "INVALID", "name_en": "Bad"},  # no nar: prefix
                {"canonical_id": "", "name_en": "Empty"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        narrator_result = results[0]
        assert narrator_result.skipped == 2
        assert len(narrator_result.validation_errors) == 2

    def test_strict_missing_file_raises(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Missing required file"):
            load_all_nodes(mock_client, staging_dir, curated_dir, strict=True)

    def test_lenient_missing_file_returns_zeros(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        narrator_result = results[0]
        assert narrator_result.created == 0
        assert narrator_result.merged == 0

    def test_canonical_read_path_matches_resolve_write_path(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#112 contract: the loader reads narrators_canonical.parquet from the
        SAME dir the resolve stage writes it to (the curated/resolve-output dir),
        and NOT from staging.

        ``write_narrators_canonical`` writes into ``curated_dir`` (mirroring
        ``disambiguate.run`` / ``bio_promote`` whose ``output_dir`` the CLI maps
        to ``DATA_CURATED_DIR``). A stray copy in ``staging_dir`` must be ignored,
        proving writer-path == reader-path.
        """
        # Resolve-output location (curated): loaded.
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:from-curated", "name_en": "From Curated"}],
        )
        # A stale staging copy that the OLD loader would have read: must be ignored.
        write_narrators_canonical(
            staging_dir,
            [
                {"canonical_id": "nar:stale-staging-a", "name_en": "Stale A"},
                {"canonical_id": "nar:stale-staging-b", "name_en": "Stale B"},
            ],
        )

        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        narrator_result = results[0]
        assert narrator_result.node_type == "Narrator"
        # Exactly the single curated row — never the two staging rows.
        assert narrator_result.created + narrator_result.merged == 1
        assert narrator_result.skipped == 0

        loaded_ids = {
            row["id"]
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            for row in batch
            if isinstance(row, dict) and str(row.get("id", "")).startswith("nar:")
        }
        assert loaded_ids == {"nar:from-curated"}


class TestNarratorNameEnFallback:
    """da#159: every loaded Narrator gets a non-empty English display name.

    Almost no canonical records carry a sourced ``name_en`` (113 / 47,199), so
    the loader synthesizes a deterministic transliteration of ``name_ar`` when
    one is absent — fixing hollow English search/display on ``/graph`` and
    narrator pages.
    """

    @staticmethod
    def _narrator_batch(client: MockNeo4jClient) -> list[dict[str, object]]:
        _query, batch = next(
            (q, b) for q, b in client.calls if isinstance(b, list) and "MERGE (n:Narrator" in q
        )
        assert isinstance(batch, list)
        return batch

    def test_missing_name_en_filled_from_transliteration(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:m", "name_ar": "محمد", "name_en": None}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        row = self._narrator_batch(mock_client)[0]
        assert row["name_en"] == "Muhammad"

    def test_empty_string_name_en_filled(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:ah", "name_ar": "أبو هريرة", "name_en": ""}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        row = self._narrator_batch(mock_client)[0]
        assert row["name_en"] == "Abu Hurayra"

    def test_sourced_name_en_preserved(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A real sourced English name is used verbatim, never overwritten."""
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:prophet",
                    "name_ar": "محمد",
                    "name_en": "Prophet Muhammad",
                }
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        row = self._narrator_batch(mock_client)[0]
        assert row["name_en"] == "Prophet Muhammad"

    def test_falls_back_to_normalized_when_name_ar_missing(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:nn",
                    "name_ar": None,
                    "name_ar_normalized": "علي",
                    "name_en": None,
                }
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        row = self._narrator_batch(mock_client)[0]
        assert row["name_en"] == "Ali"

    def test_no_arabic_anywhere_stays_empty(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:bare", "name_ar": None, "name_en": None}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        row = self._narrator_batch(mock_client)[0]
        assert row["name_en"] == ""

    def test_coverage_rises_to_full_on_production_shaped_batch(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Production-shaped: many name_ar-only records, one sourced English name.

        Mirrors the staging reality (coverage 113/47,199) at small scale: only a
        single record arrives with a sourced ``name_en``; the rest are Arabic-only
        and must come out of the loader with a non-empty, ASCII English name.
        """
        arabic_only = [
            "محمد",
            "علي",
            "عبد الله",
            "أبو هريرة",
            "عائشة",
            "محمد بن إسماعيل",
            "عمر بن الخطاب",
            "الزهري",
            "سفيان الثوري",
            "أنس بن مالك",
        ]
        rows = [
            {"canonical_id": f"nar:{i}", "name_ar": ar, "name_en": None}
            for i, ar in enumerate(arabic_only)
        ]
        rows.append(
            {"canonical_id": "nar:sourced", "name_ar": "محمد", "name_en": "Prophet Muhammad"}
        )
        write_narrators_canonical(curated_dir, rows)

        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        batch = self._narrator_batch(mock_client)
        assert len(batch) == len(rows)

        non_empty = [r for r in batch if str(r["name_en"]).strip()]
        # Coverage is total (was ~0% pre-fix for the Arabic-only majority).
        assert len(non_empty) == len(rows)
        # Names are sane: ASCII and capitalized (the definite article ``al-``
        # legitimately leads lowercase, e.g. "al-Zuhri").
        for r in batch:
            name_en = str(r["name_en"])
            assert name_en.isascii()
            assert name_en[0].isupper() or name_en.startswith("al-")
        # Sourced name preserved; transliterations applied to the rest.
        by_id = {r["id"]: r["name_en"] for r in batch}
        assert by_id["nar:sourced"] == "Prophet Muhammad"
        assert by_id["nar:2"] == "Abd Allah"
        assert by_id["nar:3"] == "Abu Hurayra"


class TestLoadHadiths:
    def test_valid_hadiths(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1", "matn_ar": "text1"},
                {"source_id": "h-2", "matn_ar": "text2"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        hadith_result = results[1]
        assert hadith_result.node_type == "Hadith"
        assert hadith_result.created + hadith_result.merged == 2

    def _hadith_batch_rows(self, mock_client: MockNeo4jClient) -> list[dict[str, Any]]:
        for _query, batch in mock_client.calls:
            if (
                isinstance(batch, list)
                and batch
                and isinstance(batch[0], dict)
                and str(batch[0].get("id", "")).startswith("hdt:")
            ):
                return batch
        return []

    def test_matn_ar_falls_back_to_full_text_ar(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Sources that only populate full_text_ar (halimbahae/open_hadith/bihar)
        must not land textless: matn_ar falls back to full_text_ar (da#190)."""
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-empty", "matn_ar": "", "full_text_ar": "النص الكامل"},
                {"source_id": "h-none", "matn_ar": None, "full_text_ar": "نص آخر"},
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        rows = {r["id"]: r["matn_ar"] for r in self._hadith_batch_rows(mock_client)}
        assert rows["hdt:h-empty"] == "النص الكامل"
        assert rows["hdt:h-none"] == "نص آخر"

    def test_matn_ar_preserved_when_present(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A real matn_ar is never overwritten by full_text_ar."""
        write_hadiths(
            staging_dir,
            [{"source_id": "h-1", "matn_ar": "المتن", "full_text_ar": "الإسناد والمتن"}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        rows = {r["id"]: r["matn_ar"] for r in self._hadith_batch_rows(mock_client)}
        assert rows["hdt:h-1"] == "المتن"

    def test_composition_skips_non_canonical_collections(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Non-canonical (source, collection) Hadith are skipped at load (da#191):
        halimbahae keeps only its unique books; mis loads no Hadith nodes."""
        write_hadiths(
            staging_dir,
            [
                # halimbahae unique book -> kept
                {
                    "source_id": "hb-kept",
                    "source_corpus": "halimbahae",
                    "collection_name": "musnad_ahmad_ibn-hanbal",
                    "matn_ar": "متن",
                },
                # halimbahae six-book duplicate of lk -> dropped
                {
                    "source_id": "hb-drop",
                    "source_corpus": "halimbahae",
                    "collection_name": "sahih_al-bukhari",
                    "matn_ar": "متن",
                },
                # mis -> no Hadith nodes at all
                {
                    "source_id": "mis-drop",
                    "source_corpus": "mis",
                    "collection_name": "Sahih Muslim",
                    "matn_ar": "متن",
                },
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = {r["id"] for r in self._hadith_batch_rows(mock_client)}
        assert ids == {"hdt:hb-kept"}

    def test_adds_hdt_prefix(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(staging_dir, [{"source_id": "bukhari-1"}])
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        # Find the hadith write batch call
        hadith_batches = [
            batch
            for query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0]["id"].startswith("hdt:")
        ]
        assert len(hadith_batches) >= 1
        assert hadith_batches[0][0]["id"] == "hdt:bukhari-1"

    def test_node_omits_collection_position_props(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """book/chapter/hadith_number live on the APPEARS_IN edge, not the node (#35)."""
        write_hadiths(
            staging_dir,
            [{"source_id": "bukhari-1", "book_number": 1, "chapter_number": 1, "hadith_number": 1}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        hadith_batches = [
            batch
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0]["id"].startswith("hdt:")
        ]
        assert len(hadith_batches) >= 1
        row = hadith_batches[0][0]
        assert "book_number" not in row
        assert "chapter_number" not in row
        assert "hadith_number" not in row

    def test_invalid_source_id_skipped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1"},
                {"source_id": ""},  # empty
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        hadith_result = results[1]
        assert hadith_result.skipped == 1

    def test_strict_missing_hadiths_raises(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Provide narrators but no hadiths
        write_narrators_canonical(
            curated_dir,
            [
                {"canonical_id": "nar:test", "name_en": "Test"},
            ],
        )
        with pytest.raises(FileNotFoundError, match="hadiths_"):
            load_all_nodes(mock_client, staging_dir, curated_dir, strict=True)


class TestLoadCollections:
    def test_valid_collections(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_collections(
            staging_dir,
            [
                {"collection_id": "bukhari", "name_en": "Sahih al-Bukhari"},
                {"collection_id": "col:muslim", "name_en": "Sahih Muslim"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        coll_result = results[2]
        assert coll_result.node_type == "Collection"
        assert coll_result.created + coll_result.merged == 2

    def test_adds_col_prefix(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_collections(staging_dir, [{"collection_id": "bukhari", "name_en": "Bukhari"}])
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        coll_batches = [
            batch
            for query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0].get("id", "").startswith("col:")
        ]
        assert len(coll_batches) >= 1
        assert coll_batches[0][0]["id"] == "col:bukhari"


class TestLoadChains:
    def test_chains_from_mentions(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrator_mentions(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "source_hadith_id": "h-1",
                    "position_in_chain": 0,
                    "name_ar": "n1",
                },
                {
                    "mention_id": "m2",
                    "source_hadith_id": "h-1",
                    "position_in_chain": 1,
                    "name_ar": "n2",
                },
                {
                    "mention_id": "m3",
                    "source_hadith_id": "h-2",
                    "position_in_chain": 0,
                    "name_ar": "n3",
                },
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        chain_result = results[3]
        assert chain_result.node_type == "Chain"
        assert chain_result.created + chain_result.merged == 2  # 2 distinct hadiths


class TestLoadGradings:
    def test_gradings_from_hadiths(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1", "grade": "sahih"},
                {"source_id": "h-2", "grade": "hasan"},
                {"source_id": "h-3"},  # no grade
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        grading_result = results[4]
        assert grading_result.node_type == "Grading"
        assert grading_result.created + grading_result.merged == 2

    def test_grading_carries_normalized_display_grade(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Raw Arabic grade is preserved and a normalized display grade added (da#148)."""
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1", "grade": "صحيح"},  # raw Arabic
                {"source_id": "h-2", "grade": "Da'if"},
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        grading_batches = [
            batch
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0]["id"].startswith("grd:")
        ]
        assert len(grading_batches) >= 1
        by_hadith = {r["hadith_id"]: r for r in grading_batches[0]}
        assert by_hadith["hdt:h-1"]["grade"] == "صحيح"  # raw preserved
        assert by_hadith["hdt:h-1"]["grade_normalized"] == "sahih"
        assert by_hadith["hdt:h-2"]["grade_normalized"] == "daif"


class TestLoadHistoricalEvents:
    def test_valid_events(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_historical_events_yaml(
            curated_dir,
            [
                {"id": "evt:ridda", "name_en": "Ridda Wars", "year_start_ah": 11},
                {"id": "evt:badr", "name_en": "Battle of Badr", "year_start_ah": 2},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        event_result = results[5]
        assert event_result.node_type == "HistoricalEvent"
        assert event_result.created + event_result.merged == 2

    def test_missing_name_skipped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_historical_events_yaml(
            curated_dir,
            [
                {"id": "evt:1", "name_en": "Valid"},
                {"id": "evt:2"},  # missing name_en
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        event_result = results[5]
        assert event_result.skipped == 1

    def test_invalid_id_skipped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_historical_events_yaml(
            curated_dir,
            [
                {"id": "", "name_en": "Bad ID"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        event_result = results[5]
        assert event_result.skipped == 1


class TestLoadLocations:
    def test_valid_locations(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_locations_yaml(
            curated_dir,
            [
                {"id": "loc:medina", "name_en": "Medina", "region": "Hejaz"},
                {"id": "mecca", "name_en": "Mecca"},
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        loc_result = results[6]
        assert loc_result.node_type == "Location"
        assert loc_result.created + loc_result.merged == 2

    def test_missing_file_returns_zeros(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Locations are optional even in strict mode
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        loc_result = results[6]
        assert loc_result.created == 0


class TestLoadAllNodes:
    def test_ensures_constraints_first(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        assert mock_client.constraints_ensured

    def test_returns_seven_results(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        assert len(results) == 7
        expected_types = [
            "Narrator",
            "Hadith",
            "Collection",
            "Chain",
            "Grading",
            "HistoricalEvent",
            "Location",
        ]
        assert [r.node_type for r in results] == expected_types

    def test_full_load_with_all_data(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_narrators_canonical(
            curated_dir,
            [
                {"canonical_id": "nar:1", "name_en": "Narrator 1"},
            ],
        )
        write_hadiths(
            staging_dir,
            [
                {"source_id": "h-1", "grade": "sahih", "collection_name": "bukhari"},
            ],
        )
        write_collections(
            staging_dir,
            [
                {"collection_id": "bukhari", "name_en": "Bukhari"},
            ],
        )
        write_narrator_mentions(
            staging_dir,
            [
                {
                    "mention_id": "m1",
                    "source_hadith_id": "h-1",
                    "position_in_chain": 0,
                    "name_ar": "n",
                },
            ],
        )
        write_historical_events_yaml(
            curated_dir,
            [
                {"id": "evt:1", "name_en": "Event"},
            ],
        )
        write_locations_yaml(
            curated_dir,
            [
                {"id": "loc:1", "name_en": "Place"},
            ],
        )

        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        assert all(r.created + r.merged > 0 or r.node_type in ("Chain",) for r in results)
