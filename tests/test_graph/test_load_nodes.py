"""Tests for src.graph.load_nodes — Neo4j node loading with mock client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.graph.load_nodes import LoadResult, load_all_nodes
from src.parse.identity import chain_node_id
from tests.test_graph.conftest import (
    MockNeo4jClient,
    write_collections,
    write_hadiths,
    write_historical_events_yaml,
    write_locations_yaml,
    write_narrator_mentions,
    write_narrator_mentions_resolved,
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

    def test_narrator_carries_attestation(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#370: the attestation tag is SET on the Narrator node from the canonical row."""
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:bio-only",
                    "name_en": "Bio Only",
                    "mention_count": 0,
                    "attestation": "biographical_only",
                },
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        query, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        assert "n.attestation" in query
        assert isinstance(batch, list)
        assert batch[0]["attestation"] == "biographical_only"

    def test_narrator_attestation_defaults_to_unknown_when_absent(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A legacy canonical row without the attestation column loads as 'unknown'."""
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:legacy-attest", "name_en": "Legacy"}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        _, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        assert isinstance(batch, list)
        assert batch[0]["attestation"] == "unknown"

    def test_narrator_carries_over_merged_flag(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#445: a flagged canonical row SETs over_merged + note on the Narrator node."""
        write_narrators_canonical(
            curated_dir,
            [
                {
                    "canonical_id": "nar:over-merged",
                    "name_en": "Over Merged",
                    "over_merged": True,
                    "over_merge_note": "bare generic; betweenness inflated",
                },
            ],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        query, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        assert "n.over_merged" in query
        assert "n.over_merge_note" in query
        assert batch[0]["over_merged"] is True
        assert batch[0]["over_merge_note"] == "bare generic; betweenness inflated"

    def test_narrator_over_merged_defaults_false_when_absent(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A legacy/unflagged canonical row loads as over_merged=False (property present)."""
        write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:unflagged", "name_en": "Unflagged"}],
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        _, batch = next((q, b) for q, b in mock_client.calls if "MERGE (n:Narrator" in q)
        assert batch[0]["over_merged"] is False
        assert batch[0]["over_merge_note"] is None

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

    def _all_hadith_ids(self, mock_client: MockNeo4jClient) -> list[str]:
        """Every loaded Hadith id across ALL write batches (one batch per file)."""
        ids: list[str] = []
        for _query, batch in mock_client.calls:
            if (
                isinstance(batch, list)
                and batch
                and isinstance(batch[0], dict)
                and str(batch[0].get("id", "")).startswith("hdt:")
            ):
                ids.extend(str(r["id"]) for r in batch)
        return ids

    # A 3-token Arabic matn (meets the identity-token floor) shared by a curated
    # edition and a sanadset re-publication of the same tradition.
    _SHARED_MATN = "انما الاعمال بالنيات"
    _UNIQUE_MATN = "لا ضرر ولا ضرار"

    def test_cross_edition_dedup_curated_wins(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#220 / B2: a sanadset hadith duplicating a curated tradition is dropped
        (curated wins), while a sanadset-unique tradition still loads. No double-count."""
        # Curated edition (lk) — not a dedup source, populates the identity index.
        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": self._SHARED_MATN}],
            suffix="lk",
        )
        # sanadset: one row duplicates the curated matn (-> dropped), one is unique.
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "matn_ar": self._SHARED_MATN,
                },
                {
                    "source_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "matn_ar": self._UNIQUE_MATN,
                },
            ],
            suffix="sanadset",
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = self._all_hadith_ids(mock_client)
        # Curated kept; sanadset duplicate dropped; sanadset-unique kept.
        assert "hdt:lk:bukhari:1" in ids
        assert "hdt:sanadset:1:dup" not in ids
        assert "hdt:sanadset:1:uniq" in ids
        # The shared tradition appears exactly once — no double-count.
        assert ids.count("hdt:lk:bukhari:1") == 1
        hadith_result = next(r for r in results if r.node_type == "Hadith")
        assert hadith_result.skipped == 1

    def test_cross_edition_dedup_no_overlap_keeps_all(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """With no curated twin, every sanadset hadith loads (dedup is a no-op)."""
        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": self._SHARED_MATN}],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "matn_ar": self._UNIQUE_MATN,
                }
            ],
            suffix="sanadset",
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = self._all_hadith_ids(mock_client)
        assert "hdt:lk:bukhari:1" in ids
        assert "hdt:sanadset:1:uniq" in ids

    def test_cross_edition_dedup_only_curated_seeds_index(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """Two sanadset hadiths sharing a matn but with NO curated twin are both
        kept — the index is seeded from curated sources only, so sanadset never
        deduplicates against itself."""
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:a",
                    "source_corpus": "sanadset",
                    "matn_ar": self._SHARED_MATN,
                },
                {
                    "source_id": "sanadset:1:b",
                    "source_corpus": "sanadset",
                    "matn_ar": self._SHARED_MATN,
                },
            ],
            suffix="sanadset",
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = self._all_hadith_ids(mock_client)
        assert "hdt:sanadset:1:a" in ids
        assert "hdt:sanadset:1:b" in ids

    def test_cross_edition_dedup_short_matn_not_deduped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """A sub-threshold (too-short) matn carries no identity, so even an exact
        curated match does not drop the sanadset hadith — the conservative
        keep-when-unsure direction."""
        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": "متن"}],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [{"source_id": "sanadset:1:short", "source_corpus": "sanadset", "matn_ar": "متن"}],
            suffix="sanadset",
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = self._all_hadith_ids(mock_client)
        assert "hdt:sanadset:1:short" in ids

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
        # Chains are built from the *resolved* mention master in curated/, which
        # carries canonical_narrator_id — NOT the raw staging mentions (which
        # lack it). Reading staging produced hollow chains: the #723 defect.
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": "h-1",
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:n2",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h-1",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:n1",
                },
                {
                    "mention_id": "m3",
                    "hadith_id": "h-2",
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:n3",
                },
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        chain_result = results[3]
        assert chain_result.node_type == "Chain"
        assert chain_result.created + chain_result.merged == 2  # 2 distinct hadiths

        # Regression guard for #723 "chains empty": the chains must be populated
        # (non-hollow), with narrator_ids ordered by position_in_chain.
        chain_batches = [
            batch
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0]["id"].startswith("chn:")
        ]
        assert chain_batches, "no Chain MERGE batch was issued"
        by_id = {row["id"]: row for batch in chain_batches for row in batch}
        h1 = by_id[chain_node_id("h-1", 0)]
        assert h1["narrator_ids"] == ["nar:n1", "nar:n2"]  # ordered by position
        assert h1["chain_length"] == 2
        assert h1["is_complete"] is True

    def test_multi_chain_hadith_yields_one_chain_node_per_chain(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        """da#282: a hadith carrying two isnad-chains (distinct ``chain_index``,
        each numbered from position 0 — the lk ar/en shape) produces ONE Chain node
        per chain, each keyed ``chn:<hadith>-<chain_index>`` with only its own
        members — not a single flattened node interleaving both chains.
        """
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m0",
                    "hadith_id": "h-1",
                    "chain_index": 0,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:A",
                },
                {
                    "mention_id": "m1",
                    "hadith_id": "h-1",
                    "chain_index": 0,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:B",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": "h-1",
                    "chain_index": 1,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:C",
                },
                {
                    "mention_id": "m3",
                    "hadith_id": "h-1",
                    "chain_index": 1,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:D",
                },
            ],
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        chain_result = results[3]
        assert chain_result.created + chain_result.merged == 2  # two chains, one hadith

        chain_batches = [
            batch
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and str(batch[0]["id"]).startswith("chn:")
        ]
        by_id = {row["id"]: row for batch in chain_batches for row in batch}
        c0 = by_id[chain_node_id("h-1", 0)]
        c1 = by_id[chain_node_id("h-1", 1)]
        assert c0["narrator_ids"] == ["nar:A", "nar:B"]
        assert c0["chain_index"] == 0
        assert c1["narrator_ids"] == ["nar:C", "nar:D"]
        assert c1["chain_index"] == 1

    def test_chains_exclude_provenance_orphan_links(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # Regression: the narrator_mentions_resolved* glob also matches the curated
        # muhaddithat orphan-link file (da#228), whose rows carry `provenance` +
        # position_in_chain=0 and a REAL hadith_id (e.g. sunnah:bukhari:2). Those
        # orphan narrators are NARRATED-only links, NOT isnad chain members — they
        # must not be folded into Chain.narrator_ids (which would inflate
        # chain_length and re-introduce the #723 chain pollution). _load_chains must
        # skip provenance-bearing rows, mirroring the _load_transmitted_to guard.
        from src.resolve.muhaddithat_links import (
            build_muhaddithat_mention_links,
            canonical_id_for,
        )

        orphan_hid = "sunnah:bukhari:2"
        orphan_nid = canonical_id_for("عائشة بنت أبي بكر")

        # Real isnad chain for the same hadith the orphan-link attests.
        write_narrator_mentions_resolved(
            curated_dir,
            [
                {
                    "mention_id": "m1",
                    "hadith_id": orphan_hid,
                    "position_in_chain": 0,
                    "canonical_narrator_id": "nar:real-0",
                },
                {
                    "mention_id": "m2",
                    "hadith_id": orphan_hid,
                    "position_in_chain": 1,
                    "canonical_narrator_id": "nar:real-1",
                },
            ],
        )
        # Producer writes narrator_mentions_resolved_muhaddithat.parquet (provenance-
        # bearing), which the _load_chains glob also matches.
        build_muhaddithat_mention_links(curated_dir)

        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)

        chain_batches = [
            batch
            for _query, batch in mock_client.calls
            if isinstance(batch, list)
            and batch
            and "id" in batch[0]
            and batch[0]["id"].startswith("chn:")
        ]
        by_id = {row["id"]: row for batch in chain_batches for row in batch}
        chain = by_id[chain_node_id(orphan_hid, 0)]
        # The provenance-bearing orphan narrator is absent; only the two real isnad
        # mentions remain (pre-fix code would fold it in and report chain_length 3).
        assert orphan_nid not in chain["narrator_ids"]
        assert chain["narrator_ids"] == ["nar:real-0", "nar:real-1"]
        assert chain["chain_length"] == 2


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


def _hadith_ids_written(client: MockNeo4jClient) -> set[str]:
    """The set of Hadith node ids the loader actually wrote (from the MERGE batches)."""
    ids: set[str] = set()
    for query, batch in client.calls:
        if isinstance(batch, list) and "MERGE (n:Hadith {id: row.id})" in str(query):
            ids.update(item["id"] for item in batch)
    return ids


class TestBuildLoadedHadithIds:
    """da#373 — build_loaded_hadith_ids reproduces EXACTLY the Hadith.id set the node
    loader loads, so the edge loader can gate chain edges on the same kept set."""

    _SHARED_MATN = "انما الاعمال بالنيات"
    _UNIQUE_MATN = "لا ضرر ولا ضرار"

    def test_matches_node_loader_kept_set(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        from src.graph.load_nodes import build_loaded_hadith_ids

        write_hadiths(
            staging_dir,
            [{"source_id": "lk:bukhari:1", "source_corpus": "lk", "matn_ar": self._SHARED_MATN}],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "matn_ar": self._SHARED_MATN,
                },
                {
                    "source_id": "sanadset:1:uniq",
                    "source_corpus": "sanadset",
                    "matn_ar": self._UNIQUE_MATN,
                },
            ],
            suffix="sanadset",
        )
        # fawaz six-books duplicate — dropped by the composition gate (da#191).
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "fawaz:bukhari:1",
                    "source_corpus": "fawaz",
                    "collection_name": "bukhari",
                    "matn_ar": self._UNIQUE_MATN,
                }
            ],
            suffix="fawaz",
        )

        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        loaded_by_loader = _hadith_ids_written(mock_client)

        files = sorted(staging_dir.glob("hadiths_*.parquet"))
        built = build_loaded_hadith_ids(files)

        # The set the edge loader will consult is exactly the set the node loader loaded.
        assert built == loaded_by_loader
        assert "hdt:lk:bukhari:1" in built
        assert "hdt:sanadset:1:uniq" in built
        assert "hdt:sanadset:1:dup" not in built  # cross-edition deduped (da#220)
        assert "hdt:fawaz:bukhari:1" not in built  # composition-dropped (da#191)

    def test_punctuation_only_duplicate_is_deduped(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        # The sanadset copy differs from the curated lk edition ONLY by punctuation;
        # da#373's strip makes the matn identities collide, so it is dropped. Before
        # the strip it survived — this is the dead-gate defect at the loader level.
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "lk:bukhari:1",
                    "source_corpus": "lk",
                    "matn_ar": "انما الاعمال بالنيات",
                }
            ],
            suffix="lk",
        )
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sanadset:1:dup",
                    "source_corpus": "sanadset",
                    "matn_ar": "«انما الاعمال بالنيات».",
                }
            ],
            suffix="sanadset",
        )
        load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        ids = _hadith_ids_written(mock_client)
        assert "hdt:lk:bukhari:1" in ids
        assert "hdt:sanadset:1:dup" not in ids


class TestMalformedIdDiagnostic:
    """da#373 follow-up / da#355 — a quarantined double-prefixed staging row must
    record WHY (the ``doubled leading corpus`` marker), not a bare ``malformed`` tag.

    Unit-tier guard for the same contract the live
    ``tests/integration/test_source_id_collision.py`` asserts against a real Neo4j.
    The shared keep-decision (``_hadith_load_outcome``) swallows the
    ``DoubledCorpusPrefixError``, so ``_load_hadiths`` must re-mint and record its
    message verbatim in the non-strict (production ``_cmd_load``) path — a bare
    generic string would silently degrade da#355's "stop guessing, start recording"
    diagnostic, which is exactly the axis #373 is repairing.
    """

    def test_double_prefixed_row_records_marker_and_quarantines(
        self, mock_client: MockNeo4jClient, staging_dir: Path, curated_dir: Path
    ) -> None:
        write_hadiths(
            staging_dir,
            [
                {
                    "source_id": "sunnah:sunnah:bukhari:1:1:1",
                    "collection_name": "bukhari",
                    "source_corpus": "sunnah",
                },
                {
                    "source_id": "lk:bukhari:1:1",
                    "collection_name": "bukhari",
                    "source_corpus": "lk",
                },
            ],
            suffix="streaming",
        )
        results = load_all_nodes(mock_client, staging_dir, curated_dir, strict=False)
        hadith_result = next(r for r in results if r.node_type == "Hadith")
        assert hadith_result.skipped >= 1
        assert hadith_result.malformed_ids >= 1
        assert any("doubled leading corpus" in e for e in hadith_result.validation_errors), (
            hadith_result.validation_errors
        )
        # Only the well-formed row loads; the doubled id never becomes a node.
        assert _hadith_ids_written(mock_client) == {"hdt:lk:bukhari:1:1"}
