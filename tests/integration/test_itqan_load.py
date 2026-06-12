"""Live-Neo4j proof that Itqan narrator bios load as Narrator nodes (da#92a).

Exercises the REAL slice of the pipeline end-to-end against a live ``neo4j:5``
container:

    parse (itqan.run) -> promote (bio_promote) -> load (load_all_nodes)

and asserts the Narrator nodes land with canonical ``nar:`` ids, their Itqan
``external_id`` provenance tag, and the jarh-wa-ta'dil trustworthiness grade.

Requires Docker; the ``neo4j_client`` fixture ``pytest.skip``s when Docker is
unreachable (see ``tests/integration/conftest.py``), so this degrades to a clean
SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.graph.load_nodes import load_all_nodes
from src.parse import itqan
from src.resolve.bio_promote import promote_bios_to_canonical

FIXTURE = Path(__file__).parent.parent / "test_parse" / "fixtures" / "itqan_rijal_sample.json"


@pytest.mark.integration
class TestItqanNarratorLoadLive:
    def test_itqan_bios_load_as_narrator_nodes(self, neo4j_client, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "itqan"
        raw.mkdir(parents=True)
        shutil.copy(FIXTURE, raw / "profiles_sample.json")
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        # parse -> narrators_bio_itqan.parquet
        itqan.run(tmp_path / "raw", staging)
        # promote -> narrators_canonical.parquet IN curated (where the loader reads
        # it — da#112 artifact-location contract)
        canonical = promote_bios_to_canonical(staging, curated, sources={"itqan"})
        assert canonical is not None
        expected = pq.read_table(canonical).num_rows
        assert expected >= 20  # the 24-profile fixture, minus any name collisions

        # load into the live graph
        results = load_all_nodes(neo4j_client, staging, curated, strict=False)
        narrator_result = next(r for r in results if r.node_type == "Narrator")
        assert not narrator_result.validation_errors

        count = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS n")[0]["n"]
        assert count == expected, "every canonical Itqan narrator should be a Narrator node"

        nodes = neo4j_client.execute_read(
            "MATCH (n:Narrator) "
            "RETURN n.id AS id, n.external_id AS ext, n.trustworthiness AS trust, "
            "n.source_ids AS source_ids"
        )
        # Correct ids: canonical nar: prefix on every node.
        assert all(n["id"].startswith("nar:") for n in nodes)
        # Provenance tagging: the Itqan profile id is carried as external_id.
        assert all(n["ext"] for n in nodes)
        # Grading made it onto the graph: the fixture's three real jarh tiers.
        trust = {n["trust"] for n in nodes}
        assert {"matruk", "kadhdhab", "thiqa"} <= trust

        # Node-level provenance: every node carries an itqan: source_id, so the
        # corpus is auditable AND removable on the graph itself (the property the
        # owner's licensing clearance relies on).
        assert all(any(s.startswith("itqan:") for s in n["source_ids"]) for n in nodes)
        removable = neo4j_client.execute_read(
            "MATCH (n:Narrator) WHERE any(s IN n.source_ids WHERE s STARTS WITH 'itqan:') "
            "RETURN count(n) AS n"
        )[0]["n"]
        assert removable == count, "every Itqan narrator must be selectable for removal"

    def test_reload_is_idempotent(self, neo4j_client, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "itqan"
        raw.mkdir(parents=True)
        shutil.copy(FIXTURE, raw / "profiles_sample.json")
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        itqan.run(tmp_path / "raw", staging)
        promote_bios_to_canonical(staging, curated, sources={"itqan"})

        load_all_nodes(neo4j_client, staging, curated, strict=False)
        first = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS n")[0]["n"]
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        second = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS n")[0]["n"]
        assert first == second  # MERGE on canonical_id — no duplicate nodes
