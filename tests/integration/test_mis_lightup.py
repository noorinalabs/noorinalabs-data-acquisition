"""Live-Neo4j proof that MIS contributes multi-isnad edges only — no Hadith nodes
(da#97; composition per da#191).

Exercises the slice end-to-end against a live ``neo4j:5`` container:

    parse (mis.run) -> load_all_nodes (composition drops mis Hadith) -> load edges

and asserts (a) MIS loads ZERO Hadith nodes — its Sahih Muslim matn duplicates
the ``lk`` canonical edition, so the canonical composition (da#191) keeps MIS for
its multi-isnad CHAINS only — and (b) the **multiplicity** still survives all the
way to graph relationships — a hadith with three isnads yields the chain-specific
edges (``A->D``, ``E->B``) that would be gone had the parallel asanid been
collapsed.

The network-edge → graph wiring for non-muhaddithat sources is not yet in the
production ``load_edges`` loader (it is hardcoded to
``network_edges_muhaddithat.parquet``); this test loads MIS's own
``network_edges_mis.parquet`` rows into the live graph directly, which is the
honest E2E proof for the adapter while that wiring decision (a graph-layer
follow-up) is made.

Requires Docker; the ``neo4j_client`` fixture ``pytest.skip``s when Docker is
unreachable (see ``tests/integration/conftest.py``), so this degrades to a clean
SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from src.graph.load_nodes import load_all_nodes
from src.parse import mis
from src.parse.identity import narrator_node_id

_LOAD_EDGES = """\
UNWIND $rows AS row
MERGE (a:Narrator {id: row.from_id})
MERGE (b:Narrator {id: row.to_id})
CREATE (a)-[:ISNAD_TRANSMITTED {hadith_id: row.hadith_id}]->(b)
"""


def _write_xlsx(path: Path, header: list[str], rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(str(path))


def _make_mis_fixture(raw_dir: Path) -> None:
    """Hadith 1 = three isnads (A/B/E), hadith 2 = one isnad."""
    mis_dir = raw_dir / "mis"
    mis_dir.mkdir(parents=True)
    _write_xlsx(
        mis_dir / "Hadith_SahihMuslim_CoreInfo.xlsx",
        ["HadithNo", "BookNo", "Matn"],
        [[1, 1, "متن ١"], [2, 1, "متن ٢"]],
    )
    _write_xlsx(
        mis_dir / "Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.xlsx",
        ["HadithNo", "IsnadNo", "Position", "Narrator"],
        [
            [1, "A", 1, "A"],
            [1, "A", 2, "B"],
            [1, "A", 3, "C"],
            [1, "B", 1, "A"],
            [1, "B", 2, "D"],
            [1, "B", 3, "C"],
            [1, "E", 1, "E"],
            [1, "E", 2, "B"],
            [1, "E", 3, "C"],
            [2, "1", 1, "X"],
            [2, "1", 2, "Y"],
        ],
    )


@pytest.mark.integration
class TestMisLoadLive:
    def test_mis_loads_no_hadith_nodes_but_multi_isnad_edges_survive(
        self, neo4j_client, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        staging = tmp_path / "staging"
        staging.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()
        _make_mis_fixture(raw)

        _hadiths_path, edges_path = mis.run(raw, staging)

        # --- MIS contributes NO Hadith nodes (composition, da#191) ----------
        # Its Sahih Muslim matn duplicates the lk canonical edition, so the node
        # loader drops every mis Hadith; MIS is kept for its multi-isnad chains.
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        mis_hadith_nodes = neo4j_client.execute_read(
            "MATCH (h:Hadith {source_corpus: 'mis'}) RETURN count(h) AS n"
        )[0]["n"]
        assert mis_hadith_nodes == 0

        # --- Multi-isnad edges into the live graph --------------------------
        edge_rows = pq.read_table(edges_path).to_pylist()
        graph_rows = [
            {
                "from_id": narrator_node_id(r["from_external_id"] or r["from_narrator_name"]),
                "to_id": narrator_node_id(r["to_external_id"] or r["to_narrator_name"]),
                "hadith_id": r["hadith_id"],
            }
            for r in edge_rows
        ]
        neo4j_client.execute_write(_LOAD_EDGES, {"rows": graph_rows})

        # Every staging edge became a real relationship — nothing collapsed.
        total = neo4j_client.execute_read(
            "MATCH ()-[t:ISNAD_TRANSMITTED]->() RETURN count(t) AS n"
        )[0]["n"]
        assert total == len(edge_rows) == 7

        # The three isnads of hadith 1 survive as distinct directed pairs,
        # including the chain-specific links that only exist un-collapsed.
        h1_id = "mis:sahih_muslim:1:1"
        pairs = neo4j_client.execute_read(
            "MATCH (a:Narrator)-[t:ISNAD_TRANSMITTED {hadith_id: $h}]->(b:Narrator) "
            "RETURN DISTINCT a.id AS frm, b.id AS to",
            {"h": h1_id},
        )
        pair_set = {(p["frm"], p["to"]) for p in pairs}
        assert pair_set == {
            (narrator_node_id(a), narrator_node_id(b))
            for a, b in [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("E", "B")]
        }
        assert (narrator_node_id("A"), narrator_node_id("D")) in pair_set
        assert (narrator_node_id("E"), narrator_node_id("B")) in pair_set
