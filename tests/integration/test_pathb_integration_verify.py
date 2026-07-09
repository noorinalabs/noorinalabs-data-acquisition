"""Path B composed integration verification — da#202 (the parent / integration-verify).

da#202 is the headline production data-quality defect: ``sanadset`` was bulk-loaded
as ~650,986 Hadith nodes with **0 collections** (≈85% orphan/unlinked), a
**polluted narrator table** (raw isnad strings minted as Narrator nodes), and
sparse chains. Three Path B children fixed the three mechanisms independently:

* **B1 (da#219)** — ``parse_sanadset`` now emits ``collections_sanadset.parquet``
  so every sanadset Hadith has a Collection node and its ``APPEARS_IN`` edge loads
  (no orphans).
* **B2 (da#220)** — the graph node loader deduplicates a ``sanadset`` Hadith whose
  normalized matn matches an already-loaded *curated* tradition (curated wins).
* **B3 (da#221)** — the ``<NAR>`` firehose is re-segmented + pollution-filtered so
  the narrator mentions feeding the graph are genuine narrators, not honorifics /
  transmission verbs / English fragments / whole-chain blobs.

This module is the *composed* verification the parent issue asks for: it proves the
three fixes work **together end-to-end** on one representative ``sanadset``-shaped
fixture, rather than each in isolation (each already has its own unit tests).

Scope note — runtime-gate scoping (memory ``feedback_runtime_gate_scoping``)
---------------------------------------------------------------------------
This does NOT re-run the literal ~650k-row production load: that needs prod-only
state (the full corpus + a live Neo4j) and is a **deploy-time** activity, not PR
acceptance. The production re-run + verification (orphan count drops from 650k,
narrator-table count drops per the B3 baseline) is tracked as a deploy-time
follow-up (see the PR body). Here we compose the three mechanisms on a fixture.

Two complementary tests (mirroring how the existing load-layer tests run —
``tests/test_graph`` runs the loaders against a mock client, ``tests/integration``
against a live ``neo4j:5`` container):

* :meth:`test_pathb_compose_parse_and_dedup` — **Neo4j-free**, unmarked so it runs
  in the hard unit job for always-on signal. Proves B1 linkage (orphan-free) +
  B3 narrator cleanliness on the real parse output, and B2 dedup by inspecting the
  exact Hadith batch the real loader sends (curated wins) via a mock client.
* :meth:`test_pathb_compose_live_graph` — **Neo4j-gated** (``@pytest.mark.integration``,
  skips cleanly without Docker). Loads the composed fixture into a live graph and
  asserts B1 edges + zero orphans and B2 dedup on the materialized graph.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parse import sanadset
from src.parse.identity import collection_node_id
from src.parse.sanadset import _is_narrator_like
from tests.test_graph.conftest import MockNeo4jClient, write_collections, write_hadiths

# A canonical tradition carried by a *curated* source (``lk``). Its sanadset twin
# (row 1 below) shares this exact matn, so B2 must drop the sanadset copy at load
# and keep this curated one ("curated wins"). 3 normalized tokens — exactly the
# minimum for a reliable cross-edition identity (``_MIN_IDENTITY_TOKENS``).
_SHARED_MATN = "إنما الأعمال بالنيات"

# A representative sanadset CSV:
#   row 1 — matn duplicates the curated ``lk`` tradition above (B2 dedup target),
#           clean two-narrator chain;
#   row 2 — a unique tradition, clean two-narrator chain (survives dedup);
#   row 3 — a unique tradition whose <NAR> firehose interleaves genuine narrators
#           with the full pollution menagerie B3 must drop: a bare transmission
#           verb (قال), a whole honorific phrase (رضي الله عنه), an English
#           fragment (He said), and a Latin transliteration (Malik ibn Anas).
# The header is the REAL production one — ``Hadith,Book,Num_hadith`` — where
# ``Book`` is the book's Arabic NAME (not an int id) and ``Num_hadith`` an in-book
# ordinal that REPEATS across books (rows 1 and 3 are both number 1). The former
# fixture used a synthetic ``hadith_id,book_id,hadith`` header that exists in no
# edition of the corpus, which is what let the books.csv-absent identity collapse
# (da#353 review) hide behind a green suite.
_BOOK_BUKHARI = "صحيح البخاري"
_BOOK_MUSLIM = "صحيح مسلم"
_SANADSET_CSV = (
    "Hadith,Book,Num_hadith,grade\n"
    '"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>أبو هريرة</NAR></SANAD>'
    f'<MATN>{_SHARED_MATN}</MATN>",{_BOOK_BUKHARI},1,Sahih\n'
    '"<SANAD><NAR>أنس بن مالك</NAR> عن <NAR>مالك بن أنس</NAR></SANAD>'
    f'<MATN>لا ضرر ولا ضرار</MATN>",{_BOOK_BUKHARI},2,Hasan\n'
    '"<SANAD><NAR>مالك بن أنس</NAR> عن <NAR>قال</NAR> <NAR>رضي الله عنه</NAR> '
    "<NAR>He said</NAR> <NAR>Malik ibn Anas</NAR> عن <NAR>أبو هريرة</NAR></SANAD>"
    f'<MATN>هذا متن آخر مختلف</MATN>",{_BOOK_MUSLIM},1,Sahih\n'
)

# books.csv (B1): supplies Collection METADATA keyed on the book NAME digest — the
# same key the hadith row's own ``Book`` column yields. It never supplies the join
# key itself, so its absence degrades metadata, not identity (da#353 review).
_BOOKS_CSV = f"book_id,name,author\n1,{_BOOK_BUKHARI},البخاري\n2,{_BOOK_MUSLIM},مسلم\n"

# Pollution that MUST NOT survive B3 re-segmentation into narrator mentions.
_POLLUTION = ("قال", "رضي الله عنه", "He said", "Malik ibn Anas")
# Genuine narrators that MUST survive.
_GENUINE = {"مالك بن أنس", "أبو هريرة", "أنس بن مالك"}


def _lay_out_sanadset(tmp_path: Path) -> Path:
    """Write the representative raw sanadset corpus (hadith CSV + books.csv)."""
    raw_dir = tmp_path / "sanadset"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sanadset.csv").write_text(_SANADSET_CSV, encoding="utf-8")
    (raw_dir / "books.csv").write_text(_BOOKS_CSV, encoding="utf-8")
    return raw_dir


def _stage_curated_lk_twin(staging_dir: Path) -> None:
    """Stage a curated ``lk`` Hadith (+ its Collection) sharing row 1's matn.

    ``lk`` is a curated (non-dedup) source, so its identity occupies the
    cross-edition index against which the sanadset twin is dropped.
    """
    write_hadiths(
        staging_dir,
        [
            {
                "source_id": "lk:bukhari:1:1",
                "source_corpus": "lk",
                "collection_name": "bukhari",
                "matn_ar": _SHARED_MATN,
                "grade": "Sahih",
            }
        ],
        suffix="lk",
    )
    write_collections(
        staging_dir,
        [
            {
                "collection_id": "lk:bukhari",
                "name_en": "Sahih al-Bukhari",
                "source_corpus": "lk",
                "sect": "sunni",
            }
        ],
        suffix="lk",
    )


def _read_mentions(staging_dir: Path) -> list[str]:
    """Narrator ``name_ar`` values the B3-filtered parse emitted for sanadset."""
    table = pq.read_table(staging_dir / "narrator_mentions_sanadset.parquet")
    return [n for n in table.column("name_ar").to_pylist() if n is not None]


def _assert_b3_narrator_cleanliness(staging_dir: Path) -> None:
    """B3: re-segmented mentions are genuine narrators, free of firehose pollution."""
    names = _read_mentions(staging_dir)
    assert names, "expected genuine narrator mentions from the fixture chains"
    # Genuine narrators survived (incl. the clean names from the polluted row 3).
    assert set(names) <= _GENUINE, f"unexpected (polluted?) narrator names: {set(names) - _GENUINE}"
    assert _GENUINE <= set(names), f"a genuine narrator was dropped: {_GENUINE - set(names)}"
    # None of the pollution classes leaked through as a narrator mention.
    for junk in _POLLUTION:
        assert junk not in names, f"B3 leaked pollution into the narrator table: {junk!r}"
    # Every emitted mention independently passes the narrator-likeness predicate —
    # the polluted firehose never mints a non-narrator node (ADR-003 headline).
    assert all(_is_narrator_like(n) for n in names)


def _assert_b1_orphan_free(staging_dir: Path) -> None:
    """B1: every parsed sanadset Hadith's APPEARS_IN endpoint resolves to a Collection.

    Rebuilds the exact Collection node id ``load_edges._load_appears_in`` MATCHes
    on (``col:<corpus>:<collection_name>``) and asserts the parser emitted a
    Collection for every one — so no sanadset Hadith can load as an orphan (the
    650k-orphan defect), without standing up Neo4j.
    """
    hadiths = pq.read_table(staging_dir / "hadiths_sanadset.parquet").to_pylist()
    collections = pq.read_table(staging_dir / "collections_sanadset.parquet").to_pylist()
    emitted = {collection_node_id(c["collection_id"]) for c in collections}
    assert emitted, "B1: parser must emit at least one sanadset Collection"
    for h in hadiths:
        endpoint = collection_node_id(f"{h['source_corpus']}:{h['collection_name']}")
        assert endpoint in emitted, f"orphan Hadith — no Collection for {endpoint}"


def _loaded_hadith_rows(client: MockNeo4jClient) -> list[dict[str, object]]:
    """Every row the Hadith MERGE actually sent to the loader (post-dedup)."""
    rows: list[dict[str, object]] = []
    for query, batch in client.calls:
        if "MERGE (n:Hadith" in query and isinstance(batch, list):
            rows.extend(batch)
    return rows


class TestPathBIntegrationVerify:
    """da#202: B1 + B2 + B3 composed end-to-end on a representative fixture."""

    def test_pathb_compose_parse_and_dedup(self, tmp_path: Path) -> None:
        """Neo4j-free composition: B1 linkage + B3 cleanliness + B2 dedup.

        Runs in the hard unit job (no ``integration`` marker) for always-on signal
        even where Docker is unavailable.
        """
        from src.graph.load_nodes import load_all_nodes  # local: keep import graph flat

        raw_dir = _lay_out_sanadset(tmp_path)
        staging_dir = tmp_path / "staging"
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # --- B1 + B3: real parse of the sanadset fixture -----------------------
        outputs = sanadset.parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)
        assert "collections" in outputs, "B1: parser must emit collections_sanadset.parquet"
        _assert_b1_orphan_free(staging_dir)
        _assert_b3_narrator_cleanliness(staging_dir)

        # --- B2: stage a curated twin, then load through the REAL node loader ---
        _stage_curated_lk_twin(staging_dir)
        client = MockNeo4jClient()
        load_all_nodes(client, staging_dir, curated_dir, strict=False)

        loaded = _loaded_hadith_rows(client)
        # The sanadset duplicate of the curated tradition was dropped at load:
        # exactly one Hadith carries the shared matn, and it is the curated one.
        twins = [r for r in loaded if r.get("matn_ar") == _SHARED_MATN]
        assert len(twins) == 1, "B2: shared matn must be loaded exactly once (no double-count)"
        assert twins[0]["source_corpus"] == "lk", "B2: curated edition must win the dedup"
        # The two *unique* sanadset traditions still load (dedup is exact-matn only).
        sanadset_loaded = [r for r in loaded if r.get("source_corpus") == "sanadset"]
        assert len(sanadset_loaded) == 2, "B2: only the exact-matn duplicate is dropped"

    @pytest.mark.integration
    def test_pathb_compose_live_graph(self, neo4j_client, tmp_path: Path) -> None:
        """Neo4j-gated composition: B1 edges + zero orphans and B2 dedup in the graph.

        Skips cleanly when Docker is unavailable (``tests/integration/conftest``),
        matching ``test_sanadset_lightup.py``.
        """
        from src.graph.load_edges import load_all_edges
        from src.graph.load_nodes import load_all_nodes

        raw_dir = _lay_out_sanadset(tmp_path)
        staging_dir = tmp_path / "staging"
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        sanadset.parse_sanadset(raw_dir=raw_dir, staging_dir=staging_dir)
        # B3 cleanliness holds on the same composed fixture (mentions feed the graph).
        _assert_b3_narrator_cleanliness(staging_dir)

        _stage_curated_lk_twin(staging_dir)
        load_all_nodes(neo4j_client, staging_dir, curated_dir, strict=False)
        load_all_edges(neo4j_client, staging_dir, curated_dir, strict=False)

        # --- B1: collections exist and NO Hadith is an orphan ------------------
        collections = neo4j_client.execute_read(
            "MATCH (c:Collection) RETURN c.id AS id, c.source_corpus AS corpus"
        )
        corpora = {c["corpus"] for c in collections}
        assert "sanadset" in corpora, "B1: sanadset Collection nodes must load"
        orphans = neo4j_client.execute_read(
            "MATCH (h:Hadith) WHERE NOT (h)-[:APPEARS_IN]->(:Collection) RETURN count(h) AS n"
        )
        assert orphans[0]["n"] == 0, "B1: every loaded Hadith must be collection-linked"
        appears_in = neo4j_client.execute_read(
            "MATCH (:Hadith)-[r:APPEARS_IN]->(:Collection) RETURN count(r) AS n"
        )
        # 2 surviving sanadset + 1 curated lk = 3 collection-linked hadiths.
        assert appears_in[0]["n"] == 3

        # --- B2: the shared tradition is present once, from the curated source --
        twins = neo4j_client.execute_read(
            "MATCH (h:Hadith) WHERE h.matn_ar = $m RETURN h.source_corpus AS corpus",
            {"m": _SHARED_MATN},
        )
        assert len(twins) == 1, "B2: shared matn must exist exactly once (curated wins)"
        assert twins[0]["corpus"] == "lk"
        sanadset_hadiths = neo4j_client.execute_read(
            "MATCH (h:Hadith {source_corpus: 'sanadset'}) RETURN count(h) AS n"
        )
        assert sanadset_hadiths[0]["n"] == 2, "B2: only the exact-matn duplicate is deduped"
