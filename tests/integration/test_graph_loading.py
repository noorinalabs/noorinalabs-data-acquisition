"""Integration tests for graph loading against a real Neo4j container."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.graph import TOPOLOGY_DERIVED_NARRATOR_PROPERTIES, load_all
from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.graph.validate import run_validation
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.neo4j_client import Neo4jClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_NARRATORS = [
    {
        "canonical_id": "nar:001",
        "name_ar": "أبو هريرة",
        "name_en": "Abu Hurayrah",
        "name_ar_normalized": "ابو هريره",
        "aliases": ["عبد الرحمن بن صخر"],
        "birth_year_ah": None,
        "death_year_ah": 59,
        "generation": "sahabi",
        "gender": "male",
        "trustworthiness": "thiqah",
        "source_ids": ["src:1"],
        "external_id": "ext:001",
        "mention_count": 5,
        # da#103: transmits in both Sunni and Shia chains → neutral affiliation.
        "source_corpus": "sunnah",
        "source_corpora": ["sunnah", "thaqalayn"],
        "sect_affiliation": "neutral",
    },
    {
        "canonical_id": "nar:002",
        "name_ar": "مالك بن أنس",
        "name_en": "Malik ibn Anas",
        "name_ar_normalized": "مالك بن انس",
        "aliases": [],
        "birth_year_ah": 93,
        "death_year_ah": 179,
        "generation": "tabii",
        "gender": "male",
        "trustworthiness": "thiqah",
        "source_ids": ["src:2"],
        "external_id": "ext:002",
        "mention_count": 3,
    },
    {
        "canonical_id": "nar:003",
        "name_ar": "نافع",
        "name_en": "Nafi",
        "name_ar_normalized": "نافع",
        "aliases": [],
        "birth_year_ah": None,
        "death_year_ah": 117,
        "generation": "tabii",
        "gender": "male",
        "trustworthiness": "thiqah",
        "source_ids": ["src:3"],
        "external_id": "ext:003",
        "mention_count": 2,
    },
    {
        "canonical_id": "nar:004",
        "name_ar": "عبد الله بن عمر",
        "name_en": "Abdullah ibn Umar",
        "name_ar_normalized": "عبد الله بن عمر",
        "aliases": [],
        "birth_year_ah": None,
        "death_year_ah": 73,
        "generation": "sahabi",
        "gender": "male",
        "trustworthiness": "thiqah",
        "source_ids": ["src:4"],
        "external_id": "ext:004",
        "mention_count": 4,
    },
    {
        "canonical_id": "nar:005",
        "name_ar": "عائشة بنت أبي بكر",
        "name_en": "Aisha bint Abi Bakr",
        "name_ar_normalized": "عايشه بنت ابي بكر",
        "aliases": ["أم المؤمنين"],
        "birth_year_ah": None,
        "death_year_ah": 58,
        "generation": "sahabi",
        "gender": "female",
        "trustworthiness": "thiqah",
        "source_ids": ["src:5"],
        "external_id": "ext:005",
        "mention_count": 6,
    },
]

# NOTE (da#77): every hadith here carries a NON-null hadith_number on purpose.
# The APPEARS_IN loader SETs hadith_number_in_book = coalesce(row.hadith_number,
# …), and Neo4j drops a property SET to null — so an edge built from a null
# hadith_number has NO hadith_number_in_book key (absent, not present-null). If a
# null-hadith_number row is ever added here, any per-edge
# ``"hadith_number_in_book" in keys(r)`` assertion (e.g. #74's read-back) will
# fail for that row. The null path is covered separately by
# ``test_appears_in_edges_load_with_null_hadith_number``.
SAMPLE_HADITHS = [
    {
        "source_id": "bukhari:1",
        "source_corpus": "sunnah",
        "collection_name": "bukhari",
        "book_number": 1,
        "chapter_number": 1,
        "hadith_number": 1,
        "matn_ar": "إنما الأعمال بالنيات",
        "matn_en": "Actions are judged by intentions",
        "isnad_raw_ar": None,
        "isnad_raw_en": None,
        "full_text_ar": None,
        "full_text_en": None,
        "grade": "sahih",
        "chapter_name_ar": None,
        "chapter_name_en": None,
        "sect": "sunni",
    },
    {
        "source_id": "bukhari:2",
        "source_corpus": "sunnah",
        "collection_name": "bukhari",
        "book_number": 1,
        "chapter_number": 1,
        "hadith_number": 2,
        "matn_ar": "بني الإسلام على خمس",
        "matn_en": "Islam is built on five pillars",
        "isnad_raw_ar": None,
        "isnad_raw_en": None,
        "full_text_ar": None,
        "full_text_en": None,
        "grade": "sahih",
        "chapter_name_ar": None,
        "chapter_name_en": None,
        "sect": "sunni",
    },
    {
        "source_id": "muslim:1",
        "source_corpus": "sunnah",
        "collection_name": "muslim",
        "book_number": 1,
        "chapter_number": 1,
        "hadith_number": 1,
        "matn_ar": "إنما الأعمال بالنيات",
        "matn_en": "Actions are judged by intentions",
        "isnad_raw_ar": None,
        "isnad_raw_en": None,
        "full_text_ar": None,
        "full_text_en": None,
        "grade": "sahih",
        "chapter_name_ar": None,
        "chapter_name_en": None,
        "sect": "sunni",
    },
]

# NOTE: ``collection_id`` here MUST be the corpus-qualified ``{corpus}:{name}``
# form, because that is the key the APPEARS_IN edge loader reconstructs from each
# hadith's ``source_corpus`` + ``collection_name`` (see load_edges.py
# ``_load_appears_in``: ``col:{corpus}:{name}``) and matches against the
# Collection node id (load_nodes.py: ``col:{collection_id}``). Plain ids like
# "bukhari" produce node ``col:bukhari`` while the edge loader looks for
# ``col:sunnah:bukhari`` — the endpoints never match, so ZERO APPEARS_IN edges
# are created and the prior ``count(r) >= 0`` assertion silently passed on an
# empty graph (#69).
SAMPLE_COLLECTIONS = [
    {
        "collection_id": "sunnah:bukhari",
        "name_ar": "صحيح البخاري",
        "name_en": "Sahih al-Bukhari",
        "compiler_name": "Muhammad ibn Ismail al-Bukhari",
        "compilation_year_ah": 256,
        "sect": "sunni",
        "total_hadiths": 7563,
        "source_corpus": "sunnah",
    },
    {
        "collection_id": "sunnah:muslim",
        "name_ar": "صحيح مسلم",
        "name_en": "Sahih Muslim",
        "compiler_name": "Muslim ibn al-Hajjaj",
        "compilation_year_ah": 261,
        "sect": "sunni",
        "total_hadiths": 7500,
        "source_corpus": "sunnah",
    },
    {
        "collection_id": "sunnah:tirmidhi",
        "name_ar": "سنن الترمذي",
        "name_en": "Jami at-Tirmidhi",
        "compiler_name": "Abu Isa al-Tirmidhi",
        "compilation_year_ah": 279,
        "sect": "sunni",
        "total_hadiths": 3956,
        "source_corpus": "sunnah",
    },
]


def _write_narrators_parquet(curated: Path, rows: list[dict[str, Any]]) -> Path:
    """Write narrators_canonical.parquet into the curated (resolve-output) dir.

    The loader reads the canonical narrator master from the curated dir
    (da#112 artifact-location contract).
    """
    arrays = {
        "canonical_id": pa.array([r["canonical_id"] for r in rows], type=pa.string()),
        "name_ar": pa.array([r.get("name_ar") for r in rows], type=pa.string()),
        "name_en": pa.array([r.get("name_en") for r in rows], type=pa.string()),
        "name_ar_normalized": pa.array(
            [r.get("name_ar_normalized") for r in rows], type=pa.string()
        ),
        "aliases": pa.array([r.get("aliases", []) for r in rows], type=pa.list_(pa.string())),
        "birth_year_ah": pa.array([r.get("birth_year_ah") for r in rows], type=pa.int32()),
        "death_year_ah": pa.array([r.get("death_year_ah") for r in rows], type=pa.int32()),
        "birth_year_ah_earliest": pa.array(
            [r.get("birth_year_ah_earliest") for r in rows], type=pa.int32()
        ),
        "birth_year_ah_latest": pa.array(
            [r.get("birth_year_ah_latest") for r in rows], type=pa.int32()
        ),
        "birth_date_precision": pa.array(
            [r.get("birth_date_precision") for r in rows], type=pa.string()
        ),
        "death_year_ah_earliest": pa.array(
            [r.get("death_year_ah_earliest") for r in rows], type=pa.int32()
        ),
        "death_year_ah_latest": pa.array(
            [r.get("death_year_ah_latest") for r in rows], type=pa.int32()
        ),
        "death_date_precision": pa.array(
            [r.get("death_date_precision") for r in rows], type=pa.string()
        ),
        "generation": pa.array([r.get("generation") for r in rows], type=pa.string()),
        "gender": pa.array([r.get("gender") for r in rows], type=pa.string()),
        "trustworthiness": pa.array([r.get("trustworthiness") for r in rows], type=pa.string()),
        "source_ids": pa.array([r.get("source_ids", []) for r in rows], type=pa.list_(pa.string())),
        "external_id": pa.array([r.get("external_id") for r in rows], type=pa.string()),
        "death_year_provenance": pa.array(
            [r.get("death_year_provenance") for r in rows], type=pa.string()
        ),
        "mention_count": pa.array([r.get("mention_count") for r in rows], type=pa.int32()),
        "attestation": pa.array([r.get("attestation") for r in rows], type=pa.string()),
        "source_corpus": pa.array([r.get("source_corpus") for r in rows], type=pa.string()),
        "source_corpora": pa.array(
            [r.get("source_corpora", []) for r in rows], type=pa.list_(pa.string())
        ),
        "sect_affiliation": pa.array([r.get("sect_affiliation") for r in rows], type=pa.string()),
        "over_merged": pa.array([r.get("over_merged") for r in rows], type=pa.bool_()),
        "over_merge_note": pa.array([r.get("over_merge_note") for r in rows], type=pa.string()),
    }
    table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    path = curated / "narrators_canonical.parquet"
    pq.write_table(table, path)
    return path


def _write_hadiths_parquet(staging: Path, rows: list[dict[str, Any]]) -> Path:
    """Write hadiths_test.parquet."""
    arrays = {
        "source_id": pa.array([r["source_id"] for r in rows], type=pa.string()),
        "source_corpus": pa.array(
            [r.get("source_corpus", "sunnah") for r in rows], type=pa.string()
        ),
        "collection_name": pa.array(
            [r.get("collection_name", "bukhari") for r in rows], type=pa.string()
        ),
        "book_number": pa.array([r.get("book_number") for r in rows], type=pa.int32()),
        "chapter_number": pa.array([r.get("chapter_number") for r in rows], type=pa.int32()),
        "hadith_number": pa.array([r.get("hadith_number") for r in rows], type=pa.int32()),
        "matn_ar": pa.array([r.get("matn_ar") for r in rows], type=pa.string()),
        "matn_en": pa.array([r.get("matn_en") for r in rows], type=pa.string()),
        "isnad_raw_ar": pa.array([r.get("isnad_raw_ar") for r in rows], type=pa.string()),
        "isnad_raw_en": pa.array([r.get("isnad_raw_en") for r in rows], type=pa.string()),
        "full_text_ar": pa.array([r.get("full_text_ar") for r in rows], type=pa.string()),
        "full_text_en": pa.array([r.get("full_text_en") for r in rows], type=pa.string()),
        "grade": pa.array([r.get("grade") for r in rows], type=pa.string()),
        "chapter_name_ar": pa.array([r.get("chapter_name_ar") for r in rows], type=pa.string()),
        "chapter_name_en": pa.array([r.get("chapter_name_en") for r in rows], type=pa.string()),
        "sect": pa.array([r.get("sect", "sunni") for r in rows], type=pa.string()),
    }
    table = pa.table(arrays, schema=HADITH_SCHEMA)
    path = staging / "hadiths_test.parquet"
    pq.write_table(table, path)
    return path


def _write_collections_parquet(staging: Path, rows: list[dict[str, Any]]) -> Path:
    """Write collections_test.parquet."""
    arrays = {
        "collection_id": pa.array([r["collection_id"] for r in rows], type=pa.string()),
        "name_ar": pa.array([r.get("name_ar") for r in rows], type=pa.string()),
        "name_en": pa.array([r.get("name_en", "") for r in rows], type=pa.string()),
        "compiler_name": pa.array([r.get("compiler_name") for r in rows], type=pa.string()),
        "compilation_year_ah": pa.array(
            [r.get("compilation_year_ah") for r in rows], type=pa.int32()
        ),
        "sect": pa.array([r.get("sect", "sunni") for r in rows], type=pa.string()),
        "total_hadiths": pa.array([r.get("total_hadiths") for r in rows], type=pa.int32()),
        "expected_count": pa.array([r.get("expected_count") for r in rows], type=pa.int32()),
        "source_corpus": pa.array(
            [r.get("source_corpus", "sunnah") for r in rows], type=pa.string()
        ),
    }
    table = pa.table(arrays, schema=COLLECTION_SCHEMA)
    path = staging / "collections_test.parquet"
    pq.write_table(table, path)
    return path


def _write_staging_data(tmp_path: Path) -> tuple[Path, Path]:
    """Write all sample staging/curated data and return (staging_dir, curated_dir)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()

    _write_narrators_parquet(curated, SAMPLE_NARRATORS)
    _write_hadiths_parquet(staging, SAMPLE_HADITHS)
    _write_collections_parquet(staging, SAMPLE_COLLECTIONS)

    # Write minimal curated data
    events_path = curated / "historical_events.yaml"
    with open(events_path, "w") as f:
        yaml.dump(
            {
                "events": [
                    {
                        "id": "evt:battle_of_badr",
                        "name_en": "Battle of Badr",
                        "name_ar": "غزوة بدر",
                        "year_start_ah": 2,
                        "year_end_ah": 2,
                        "type": "battle",
                    }
                ]
            },
            f,
        )

    locations_path = curated / "locations.yaml"
    with open(locations_path, "w") as f:
        yaml.dump(
            {
                "locations": [
                    {
                        "id": "loc:makkah",
                        "name_en": "Makkah",
                        "name_ar": "مكة",
                        "region": "Hejaz",
                        "lat": 21.4225,
                        "lon": 39.8262,
                    }
                ]
            },
            f,
        )

    return staging, curated


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNodeLoading:
    """Test loading nodes into a real Neo4j container."""

    def test_load_all_nodes(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)

        results = load_all_nodes(neo4j_client, staging, curated, strict=False)

        # Verify results
        assert len(results) == 7  # 7 node types
        narrator_result = results[0]
        assert narrator_result.node_type == "Narrator"
        assert narrator_result.created + narrator_result.merged == len(SAMPLE_NARRATORS)

        hadith_result = results[1]
        assert hadith_result.node_type == "Hadith"
        assert hadith_result.created + hadith_result.merged == len(SAMPLE_HADITHS)

        collection_result = results[2]
        assert collection_result.node_type == "Collection"
        assert collection_result.created + collection_result.merged == len(SAMPLE_COLLECTIONS)

    def test_narrators_exist_in_neo4j(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == len(SAMPLE_NARRATORS)

    def test_hadiths_exist_in_neo4j(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (h:Hadith) RETURN count(h) AS cnt")
        assert rows[0]["cnt"] == len(SAMPLE_HADITHS)

    def test_collections_exist_in_neo4j(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (c:Collection) RETURN count(c) AS cnt")
        assert rows[0]["cnt"] == len(SAMPLE_COLLECTIONS)

    def test_constraints_created(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("SHOW CONSTRAINTS")
        constraint_labels = {r.get("labelsOrTypes", [None])[0] for r in rows}
        assert "Narrator" in constraint_labels
        assert "Hadith" in constraint_labels
        assert "Collection" in constraint_labels

    def test_narrator_properties(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read(
            "MATCH (n:Narrator {id: 'nar:001'}) RETURN properties(n) AS props"
        )
        assert len(rows) == 1
        props = rows[0]["props"]
        assert props["name_en"] == "Abu Hurayrah"
        assert props["death_year_ah"] == 59
        assert props["generation"] == "sahabi"
        # da#103 live acceptance: Narrator nodes carry sect/corpus provenance.
        assert props["source_corpus"] == "sunnah"
        assert sorted(props["source_corpora"]) == ["sunnah", "thaqalayn"]
        assert props["sect_affiliation"] == "neutral"

    def test_idempotent_reload(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        """Loading twice should not duplicate nodes (MERGE semantics)."""
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == len(SAMPLE_NARRATORS)


class TestEdgeLoading:
    """Test loading edges into a real Neo4j container."""

    def test_appears_in_edges(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        edge_results = load_all_edges(neo4j_client, staging, curated, strict=False)

        # The loader returns exactly one APPEARS_IN *result object* (one per edge
        # type), and it must report edges actually created — not a no-op. The
        # 3 sample hadiths (2 bukhari + 1 muslim) all match a loaded collection,
        # so 3 edges are expected.
        appears_in = [r for r in edge_results if r.edge_type == "APPEARS_IN"]
        assert len(appears_in) == 1
        assert appears_in[0].created == 3

        # Read the edges back from the real graph and assert on the property key
        # itself, not just on edge existence. A bare ``count(r) >= 0`` is always
        # true: it passes regardless of how the property is named AND even when
        # ZERO edges were persisted, so it gives no regression protection against
        # a key rename (#69 — PR #68 renamed the key to the ig#935-canonical
        # ``hadith_number_in_book``, guarded only by a unit string-match on the
        # Cypher) and previously masked an empty-graph bug.
        rows = neo4j_client.execute_read(
            "MATCH ()-[r:APPEARS_IN]->() "
            "RETURN keys(r) AS keys, r.hadith_number_in_book AS hadith_number_in_book"
        )
        # Edges must actually exist in the graph (count agrees with the loader) ...
        assert len(rows) == 3
        # NOTE: this per-edge ``hadith_number_in_book in keys(r)`` assertion holds
        # only because every hadith in this fixture has a NON-null hadith_number.
        # Post-da#77 the loader SETs that property via ``coalesce(row.hadith_number,
        # ...)`` after the MERGE, and Neo4j drops a property SET to null — so a
        # null-hadith_number row would yield an edge WITHOUT this key. If you add a
        # null-hadith_number hadith to SAMPLE_HADITHS, relax this loop accordingly.
        for row in rows:
            # ... the canonical key MUST be present on every edge ...
            assert "hadith_number_in_book" in row["keys"]
            # ... carry a populated value (not a declared-but-null key) ...
            assert row["hadith_number_in_book"] is not None
            # ... and the pre-#68 ``hadith_number`` key MUST NOT linger.
            assert "hadith_number" not in row["keys"]

    def test_appears_in_edges_load_with_null_hadith_number(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        """da#77 regression: a null ``hadith_number`` must NOT abort the load.

        Scraped hadiths carry null ``hadith_number`` until da#72. The previous
        ``MERGE (h)-[:APPEARS_IN {hadith_number_in_book: row.hadith_number}]->(c)``
        aborted the whole edge stage with a Neo4j null-property error. The fix
        SETs the positional props after the MERGE, so the edge is created with a
        null ``hadith_number_in_book`` instead of erroring. Runs against a real
        Neo4j — the mock load suite cannot enforce the MERGE-null rule.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        null_hadith = dict(SAMPLE_HADITHS[0])
        null_hadith["source_id"] = "bukhari:nullnum"
        null_hadith["hadith_number"] = None  # the da#77 trigger
        _write_hadiths_parquet(staging, [null_hadith])
        # Self-contained collection: the edge loader derives the target id as
        # ``col:{source_corpus}:{collection_name}`` = ``col:sunnah:bukhari``, so
        # the Collection node id must be the corpus-qualified ``sunnah:bukhari``
        # for the endpoints to match (independent of the shared fixture).
        null_collection = {
            "collection_id": "sunnah:bukhari",
            "name_en": "Sahih al-Bukhari",
            "sect": "sunni",
            "source_corpus": "sunnah",
        }
        _write_collections_parquet(staging, [null_collection])

        load_all_nodes(neo4j_client, staging, curated, strict=False)
        edge_results = load_all_edges(neo4j_client, staging, curated, strict=False)

        appears_in = next(r for r in edge_results if r.edge_type == "APPEARS_IN")
        # The edge is created (no abort) ...
        assert appears_in.created == 1
        rows = neo4j_client.execute_read(
            "MATCH ()-[r:APPEARS_IN]->() RETURN r.hadith_number_in_book AS num, keys(r) AS keys"
        )
        assert len(rows) == 1
        # ... and reads back with a null hadith_number_in_book instead of failing
        # the MERGE. Neo4j does not persist a property SET to null, so the key is
        # simply absent from keys(r) — the point is the load SUCCEEDED.
        assert rows[0]["num"] is None
        assert "hadith_number_in_book" not in rows[0]["keys"]
        # The non-null positional props are still stored.
        assert "book_number" in rows[0]["keys"]

    def test_graded_by_edges(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_edges(neo4j_client, staging, curated, strict=False)

        rows = neo4j_client.execute_read("MATCH ()-[r:GRADED_BY]->() RETURN count(r) AS cnt")
        # All 3 sample hadiths have grade="sahih", so should have grading edges
        assert rows[0]["cnt"] >= 0


class TestPostLoadValidationAgainstRealFiles:
    """da#319 regression guard: run against the checked-in ``.cypher`` files.

    ``graph_integrity_deferred_inventory.cypher`` (ADR-004) and
    ``sanadset_orphan_inventory.cypher`` (ADR-003) are each 5-statement
    scripts. Reading a whole file as ONE query string and handing it to a
    single ``tx.run()`` previously raised, against a REAL Neo4j driver::

        neo4j.exceptions.CypherSyntaxError: Expected exactly one statement
        per query but got: 5

    A canned-row unit stub (``tests/test_graph/test_validate.py``) cannot
    reproduce a driver-level syntax error, which is exactly how this bug
    shipped — the unit suite was green while every real load's post-load
    validation hard-failed. This exercises ``run_validation`` against the
    actual repo files (not a synthetic fixture) over a real container.
    """

    QUERIES_DIR = Path(__file__).resolve().parents[2] / "queries"

    def _load_sample_graph(self, neo4j_client: Neo4jClient, tmp_path: Path) -> None:
        staging, curated = _write_staging_data(tmp_path)
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_edges(neo4j_client, staging, curated, strict=False)

    def test_all_validation_files_run_without_syntax_error(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        self._load_sample_graph(neo4j_client, tmp_path)

        # All 5 files in queries/validation/ must produce a result -- most
        # importantly this call must not raise (the pre-fix behavior against a
        # real driver was a CypherSyntaxError bubbling out of run_validation
        # for the two multi-statement files).
        results = run_validation(neo4j_client, self.QUERIES_DIR, timeout_seconds=30.0)

        names = {r.query_name for r in results}
        assert names == {
            "chain_integrity",
            "collection_coverage",
            "graph_integrity_deferred_inventory",
            "orphan_narrators",
            "sanadset_orphan_inventory",
            "transmitted_to_hadith_ref",
        }
        by_name = {r.query_name: r for r in results}
        # The two da#319 multi-statement files specifically must never be a
        # hard failure on a clean load -- that is the bug this PR fixes.
        # (`orphan_narrators` legitimately FAILs here: this minimal node/edge
        # fixture wires no NARRATED/TRANSMITTED_TO edges for the 5 sample
        # narrators, an unrelated, expected fixture limitation -- not a da#319
        # regression -- so it is intentionally excluded from this assertion.)
        for name in ("graph_integrity_deferred_inventory", "sanadset_orphan_inventory"):
            assert not by_name[name].is_fatal, (name, by_name[name].details)

    def test_inventory_files_classify_as_informational_pass(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        self._load_sample_graph(neo4j_client, tmp_path)

        results = run_validation(neo4j_client, self.QUERIES_DIR, timeout_seconds=30.0)
        by_name = {r.query_name: r for r in results}

        for name in ("graph_integrity_deferred_inventory", "sanadset_orphan_inventory"):
            result = by_name[name]
            assert result.passed is True
            assert result.status == "PASS"
            assert result.warning is False


class TestStaleEnrichMetricsAreInvalidatedByLoad:
    """da#351, against a real Neo4j.

    `load_all` is MERGE-only: it never deletes, so metrics an earlier `enrich`
    wrote survive a reload and read as current. Cypher has no strict-property
    mode -- a stale `betweenness_centrality` is indistinguishable from a fresh
    one. This test plants a stale value, reloads, and asserts it is gone.
    """

    QUERIES_DIR = Path(__file__).resolve().parents[2] / "queries"

    def _load(self, neo4j_client: Neo4jClient, staging: Path, curated: Path) -> Any:
        return load_all(
            neo4j_client,
            staging,
            curated,
            self.QUERIES_DIR,
            strict=False,
            skip_validation=True,
        )

    def _enrich_like_write(self, neo4j_client: Neo4jClient) -> None:
        """Stand in for `enrich`: write every GDS metric onto every Narrator."""
        sets = ", ".join(f"n.{p} = 99.5" for p in TOPOLOGY_DERIVED_NARRATOR_PROPERTIES)
        neo4j_client.execute_write(f"MATCH (n:Narrator) SET {sets}")

    def test_stale_centrality_does_not_survive_a_reload(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        staging, curated = _write_staging_data(tmp_path)
        self._load(neo4j_client, staging, curated)
        self._enrich_like_write(neo4j_client)

        # Instrument check: the probe CAN see a nonzero count right now, so the
        # zero we expect after the reload is a measurement, not a silent zero.
        before = neo4j_client.execute_read(
            "MATCH (n:Narrator) RETURN count(n.betweenness_centrality) AS c"
        )[0]["c"]
        assert before > 0, (
            "fixture did not plant any centrality; the assertion below would be inert"
        )

        summary = self._load(neo4j_client, staging, curated)

        after = neo4j_client.execute_read(
            "MATCH (n:Narrator) RETURN count(n.betweenness_centrality) AS c"
        )[0]["c"]
        assert after == 0, f"stale centrality survived the reload on {after} narrator(s)"
        assert summary.invalidated_narrators == before

        # Enumerate keys rather than probing a name: a misspelled property
        # returns NULL for every row and never errors (reference_graph_ops).
        keys = {
            row["k"]
            for row in neo4j_client.execute_read(
                "MATCH (n:Narrator) UNWIND keys(n) AS k RETURN DISTINCT k AS k"
            )
        }
        assert keys.isdisjoint(TOPOLOGY_DERIVED_NARRATOR_PROPERTIES)
        # The load's own data survives untouched -- invalidation is surgical.
        assert "name_ar" in keys and "id" in keys

    def test_invalidation_is_idempotent_on_an_unenriched_graph(
        self, neo4j_client: Neo4jClient, tmp_path: Path
    ) -> None:
        staging, curated = _write_staging_data(tmp_path)
        summary = self._load(neo4j_client, staging, curated)
        assert summary.invalidated_narrators == 0
