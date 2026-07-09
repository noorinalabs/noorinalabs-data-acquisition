"""Live-Neo4j end-to-end light-up for the sanadset narrator keystone (da#89).

Exercises the REAL pipeline — ``parse → ner → disambiguate → load`` — against a
live ``neo4j:5`` container and proves the keystone narrators for
isnad-graph#963 / #965 actually land in the graph:

1. **Narrator nodes land** — disambiguation produces canonical narrators that
   load as ``Narrator`` nodes with ``nar:``-prefixed ids (via the da#82
   :func:`narrator_node_id` helper) and searchable Arabic names.
2. **Hadith nodes carry sect + source_corpus** — every sanadset Hadith is tagged
   ``sect="sunni"`` and ``source_corpus="sanadset"`` and gets a single-corpus
   ``hdt:sanadset:...`` id (no da#82 double-prefix).
3. **Edges land** — ``APPEARS_IN`` hadith→collection edges materialize against a
   sanadset Collection that is itself ``sect="sunni"`` / ``source_corpus="sanadset"``.

Only the narrator-resolution path (NER + disambiguate) is driven directly; the
heavy embedding-based dedup step (parallel-link detection) is orthogonal to the
narrator keystone and is covered by its own tests, so it is intentionally not
run here.

Requires Docker; the ``neo4j_client`` fixture ``pytest.skip``s when Docker is
unreachable (see ``tests/integration/conftest.py``), so this degrades to a clean
SKIP off-CI rather than a red ERROR.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.graph.load_edges import load_all_edges
from src.graph.load_nodes import load_all_nodes
from src.parse import sanadset
from src.resolve import disambiguate, ner

# A NAR-tagged sanadset CSV. The narrator names below are mirrored exactly in the
# bio CSV so disambiguation resolves them via the deterministic exact-match stage.
_BOOK_BUKHARI = "صحيح البخاري"
_BOOK_MUSLIM = "صحيح مسلم"
_SANADSET_CSV = (
    "Hadith,Book,Num_hadith,grade\n"
    '"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>أبو هريرة</NAR></SANAD>'
    f'<MATN>إنما الأعمال بالنيات</MATN>",{_BOOK_BUKHARI},1,Sahih\n'
    '"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>أنس بن مالك</NAR></SANAD>'
    f'<MATN>لا ضرر ولا ضرار</MATN>",{_BOOK_BUKHARI},2,Hasan\n'
    f'"<SANAD>No SANAD</SANAD><MATN>بعض المتن هنا</MATN>",{_BOOK_MUSLIM},1,\n'
)

# Bios whose names match the NAR mentions above (exact-match → resolved narrator).
_BIOS_CSV = "name,death_year\nمالك بن أنس,179\nأبو هريرة,59\nأنس بن مالك,93\n"

_EXPECTED_HADITHS = 3
# One Collection per distinct ``Book`` name: البخاري (rows 1-2) and مسلم (row 3).
_EXPECTED_COLLECTIONS = 2


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Path]:
    """Lay out a raw sanadset corpus (hadith CSV + narrator bios) on disk."""
    raw_dir = tmp_path / "sanadset"
    narrators_dir = raw_dir / "narrators"
    narrators_dir.mkdir(parents=True)
    (raw_dir / "bukhari.csv").write_text(_SANADSET_CSV, encoding="utf-8")
    (narrators_dir / "bios.csv").write_text(_BIOS_CSV, encoding="utf-8")
    return {
        "raw": raw_dir,
        "staging": _mkdir(tmp_path / "staging"),
        "curated": _mkdir(tmp_path / "curated"),
        "resolve": _mkdir(tmp_path / "resolve"),
    }


def _mkdir(p: Path) -> Path:
    p.mkdir()
    return p


@pytest.mark.integration
class TestSanadsetLightupLive:
    def test_sanadset_narrators_and_hadiths_land_in_graph(
        self, neo4j_client, sources: dict[str, Path]
    ) -> None:
        raw, staging, curated, resolve_out = (
            sources["raw"],
            sources["staging"],
            sources["curated"],
            sources["resolve"],
        )

        # --- parse: raw CSV -> staging Parquet (hadiths + mentions + bios) ------
        outputs = sanadset.parse_sanadset(raw_dir=raw, staging_dir=staging)
        assert "narrators_bio" in outputs, "bio parse must run for narrator resolution"

        # --- resolve (narrator path): NER consolidation -> disambiguation -------
        ner.run(staging, resolve_out)
        disambiguate.run(staging, resolve_out)
        canonical = resolve_out / "narrators_canonical.parquet"
        assert canonical.exists(), "disambiguation must produce canonical narrators"
        assert pq.read_table(canonical).num_rows >= 1, "expected >=1 resolved narrator"
        # The loader reads narrators_canonical.parquet from the curated
        # (resolve-output) dir — da#112 artifact-location contract. Stage the
        # resolve output into curated for the load step.
        shutil.copy(canonical, curated / "narrators_canonical.parquet")

        # The sanadset Collection that APPEARS_IN (hadith->collection) needs is now
        # emitted by the REAL parser path (collections_sanadset.parquet, da#219/B1)
        # — NOT fabricated out of band. Before B1 this test injected a Collection
        # via ``write_collections``, which masked the production gap that the parser
        # emitted no collections file at all (ADR-003; cf. org memory
        # "test-mock injection masks production failure"). Assert the parser
        # produced it so this test exercises the path production runs.
        assert "collections" in outputs, "B1: parser must emit collections_sanadset.parquet"
        assert (staging / "collections_sanadset.parquet").exists()

        # --- load: real loaders into the live neo4j:5 container -----------------
        load_all_nodes(neo4j_client, staging, curated, strict=False)
        load_all_edges(neo4j_client, staging, curated, strict=False)

        # 1. Narrator nodes landed with nar: ids and searchable Arabic names.
        narrators = neo4j_client.execute_read(
            "MATCH (n:Narrator) RETURN n.id AS id, n.name_ar AS name_ar"
        )
        assert len(narrators) >= 1
        assert all(r["id"].startswith("nar:") for r in narrators)
        assert any(r["name_ar"] for r in narrators), "narrators must carry a searchable name"

        # 2. Hadith nodes carry sect + source_corpus and a single-corpus id.
        hadiths = neo4j_client.execute_read(
            "MATCH (h:Hadith) RETURN h.id AS id, h.sect AS sect, h.source_corpus AS source_corpus"
        )
        assert len(hadiths) == _EXPECTED_HADITHS
        assert all(h["sect"] == "sunni" for h in hadiths)
        assert all(h["source_corpus"] == "sanadset" for h in hadiths)
        # da#82: ids are hdt:sanadset:... exactly once (no doubled corpus).
        assert all(h["id"].startswith("hdt:sanadset:") for h in hadiths)
        assert not any(h["id"].startswith("hdt:sanadset:sanadset:") for h in hadiths)

        # 3. Collection nodes + APPEARS_IN edges landed with correct tagging.
        # ONE Collection PER BOOK (da#353): the join key is each hadith row's own
        # ``Book`` name-digest, so البخاري and مسلم are distinct collections. This
        # previously asserted a single Collection — but only because the fixture
        # carried no ``Book`` column, so every row fell back to the CSV stem
        # ("bukhari") and collapsed into one. The old assertion was passing
        # through the fallback, not through the name join it claimed to exercise.
        collections = neo4j_client.execute_read(
            "MATCH (c:Collection) RETURN c.sect AS sect, c.source_corpus AS source_corpus"
        )
        assert len(collections) == _EXPECTED_COLLECTIONS
        assert all(c["sect"] == "sunni" for c in collections)
        assert all(c["source_corpus"] == "sanadset" for c in collections)

        appears_in = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS n"
        )
        assert appears_in[0]["n"] == _EXPECTED_HADITHS
