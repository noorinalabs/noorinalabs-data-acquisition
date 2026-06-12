"""Live-Neo4j end-to-end proof for the muhaddithat corpus (da#90, epic #81).

Exercises the REAL orchestrated slice of the pipeline against a live ``neo4j:5``
container:

    parse (muhaddithat.run) -> resolve (run_all) -> load (load_all_nodes +
    _load_studied_under)

and asserts BOTH halves of the acceptance criteria that earlier breaks masked:

* **Narrator nodes materialize** with the ``muhaddithat`` ``source_corpus`` and a
  ``neutral`` ``sect_affiliation`` — the **both-sects** tag. muhaddithat is a
  cross-tradition corpus (no per-narrator sect in the source; spans Sunni *and*
  Shia female narrators), so a muhaddithat-only narrator must be ``neutral`` —
  not the pre-#90 ``sunni`` guess and not ``unknown``.
* **STUDIED_UNDER edges fire** between two loaded Narrator nodes. The fixture
  mirrors the real upstream layout (a Latin ``displayname`` *and* a separate
  ``arabicname``); before #90 the edge map carried the Latin name while
  ``bio_promote`` keyed the node on the Arabic name, so every endpoint missed and
  zero edges landed. We assert canonical ids reconcile and real edges connect.

No canonical id is hand-injected — the proof drives the same parse→resolve→load
path production runs, so a regression in id reconciliation re-breaks this test
rather than being papered over (the anti-pattern that hid the earlier gaps).

Requires Docker; the ``neo4j_client`` fixture ``pytest.skip``s when Docker is
unreachable (see ``tests/integration/conftest.py``), so this degrades to a clean
SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.graph.load_edges import _load_studied_under
from src.graph.load_nodes import load_all_nodes
from src.parse import muhaddithat
from src.resolve import run_all

# Mirrors the real muhaddithat/isnad-datasets upstream layout: narrators carry a
# Latin ``displayname`` AND a dedicated ``arabicname`` (the column bio_promote
# mints the canonical identity from); hadiths carry a comma-separated ``isnad``
# chain of narrator ids. Female narrators recognized across both traditions.
_NARRATORS_CSV = (
    "id,displayname,arabicname,gender,generation,info\n"
    "1,Prophet Muhammad,محمد رسول الله,male,,\n"
    "2,Aisha bint Abi Bakr,عائشة بنت أبي بكر,female,,Umm al-Mu'minin\n"
    "3,Fatima bint al-Husayn,فاطمة بنت الحسين,female,,\n"
    "4,Umm Salama,أم سلمة,female,,Umm al-Mu'minin\n"
)
_HADITHS_CSV = (
    "id,URL,isnad,notes\n"
    '1,https://sunnah.com/a,"1, 2",\n'
    '2,https://sunnah.com/b,"2, 3",\n'
    '3,https://sunnah.com/c,"1, 4, 3",\n'
)


def _seed_muhaddithat(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the muhaddithat raw CSVs; return (raw_root, staging, curated)."""
    raw = tmp_path / "raw" / "muhaddithat"
    raw.mkdir(parents=True)
    (raw / "variousnarrators.csv").write_text(_NARRATORS_CSV, encoding="utf-8")
    (raw / "hadiths.csv").write_text(_HADITHS_CSV, encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()
    return tmp_path / "raw", staging, curated


@pytest.mark.integration
class TestMuhaddithatLightupLive:
    def test_narrators_load_with_neutral_sect_and_corpus(
        self, neo4j_client, tmp_path: Path
    ) -> None:
        raw_root, staging, curated = _seed_muhaddithat(tmp_path)

        # parse -> staging bio + edge parquet
        muhaddithat.run(raw_root, staging)
        # orchestrated resolve (NER->disambiguate->bio_promote->dedup+parallels);
        # muhaddithat has no isnad text so NER/disambiguate no-op and bio_promote
        # carries the narrators to canonical (the #117 wiring).
        run_all(raw_root, staging, curated)

        canonical = curated / "narrators_canonical.parquet"
        assert canonical.exists()
        expected = pq.read_table(canonical).num_rows
        assert expected == 4  # every fixture narrator promoted

        results = load_all_nodes(neo4j_client, staging, curated, strict=False)
        narrator_result = next(r for r in results if r.node_type == "Narrator")
        assert not narrator_result.validation_errors

        nodes = neo4j_client.execute_read(
            "MATCH (n:Narrator) "
            "RETURN n.id AS id, n.source_corpus AS corpus, "
            "n.sect_affiliation AS sect"
        )
        assert len(nodes) == expected, "every canonical muhaddithat narrator is a node"
        # Correct canonical ids on every node.
        assert all(n["id"].startswith("nar:") for n in nodes)
        # Provenance: the muhaddithat corpus tag landed on the graph.
        assert {n["corpus"] for n in nodes} == {"muhaddithat"}
        # Both-sects tag: cross-tradition corpus → neutral, never sunni/unknown.
        assert {n["sect"] for n in nodes} == {"neutral"}, (
            "muhaddithat narrators must be neutral (both sects), not a one-sided guess"
        )

    def test_chains_load_as_studied_under_edges(self, neo4j_client, tmp_path: Path) -> None:
        raw_root, staging, curated = _seed_muhaddithat(tmp_path)

        muhaddithat.run(raw_root, staging)
        run_all(raw_root, staging, curated)
        load_all_nodes(neo4j_client, staging, curated, strict=False)

        # The edge endpoints must reconcile to the SAME canonical ids the
        # bio-promoted nodes carry (the Arabic-name fix). Before #90 this was 0.
        result = _load_studied_under(neo4j_client, staging)
        assert result.created > 0, "muhaddithat chains must connect loaded narrators"
        assert result.missing_endpoints == 0, (
            "no edge endpoint may miss — Arabic-name reconciliation must hold"
        )

        edges = neo4j_client.execute_read(
            "MATCH (:Narrator)-[r:STUDIED_UNDER]->(:Narrator) RETURN count(r) AS n"
        )[0]["n"]
        assert edges == result.created

    def test_reload_is_idempotent(self, neo4j_client, tmp_path: Path) -> None:
        raw_root, staging, curated = _seed_muhaddithat(tmp_path)

        muhaddithat.run(raw_root, staging)
        run_all(raw_root, staging, curated)

        load_all_nodes(neo4j_client, staging, curated, strict=False)
        _load_studied_under(neo4j_client, staging)
        first_n = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS n")[0]["n"]
        first_e = neo4j_client.execute_read("MATCH ()-[r:STUDIED_UNDER]->() RETURN count(r) AS n")[
            0
        ]["n"]

        load_all_nodes(neo4j_client, staging, curated, strict=False)
        _load_studied_under(neo4j_client, staging)
        second_n = neo4j_client.execute_read("MATCH (n:Narrator) RETURN count(n) AS n")[0]["n"]
        second_e = neo4j_client.execute_read("MATCH ()-[r:STUDIED_UNDER]->() RETURN count(r) AS n")[
            0
        ]["n"]

        assert first_n == second_n  # MERGE on canonical_id — no duplicate nodes
        assert first_e == second_e  # MERGE on the relationship — no duplicate edges
