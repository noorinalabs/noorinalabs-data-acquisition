"""Tests for the Multi-IsnadSet (MIS) parser (da#97).

The headline property is **multi-isnad multiplicity**: a single hadith with N
parallel isnad chains must yield N distinct edge-chains in staging, never a
single collapsed chain. The sequence-layout tests assert chain-specific edges
(links that exist only because the parallel asanid were kept apart) survive into
``network_edges_mis.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from src.parse.identity import validate_source_id
from src.parse.mis import run
from src.parse.schemas import HADITH_SCHEMA, NETWORK_EDGE_SCHEMA


def _write_xlsx(path: Path, header: list[str], rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(str(path))


def _make_sequence_data(raw_dir: Path) -> None:
    """CoreInfo + sequence-layout DetailsInfo (.xlsx).

    Hadith 1 carries THREE isnads, hadith 2 carries ONE. The three chains of
    hadith 1 share endpoints but diverge in the middle, so chain-specific edges
    (A->D, E->B) only exist if the chains are kept distinct.
    """
    mis_dir = raw_dir / "mis"
    mis_dir.mkdir(parents=True)

    _write_xlsx(
        mis_dir / "Hadith_SahihMuslim_CoreInfo.xlsx",
        ["HadithNo", "BookNo", "Matn"],
        [
            [1, 1, "متن الحديث الأول"],
            [2, 1, "متن الحديث الثاني"],
        ],
    )

    # (HadithNo, IsnadNo, Position, Narrator)
    detail_rows: list[list[object]] = [
        # hadith 1, isnad A: A -> B -> C
        [1, "A", 1, "A"],
        [1, "A", 2, "B"],
        [1, "A", 3, "C"],
        # hadith 1, isnad B: A -> D -> C
        [1, "B", 1, "A"],
        [1, "B", 2, "D"],
        [1, "B", 3, "C"],
        # hadith 1, isnad E: E -> B -> C
        [1, "E", 1, "E"],
        [1, "E", 2, "B"],
        [1, "E", 3, "C"],
        # hadith 2, single isnad: X -> Y
        [2, "1", 1, "X"],
        [2, "1", 2, "Y"],
    ]
    _write_xlsx(
        mis_dir / "Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.xlsx",
        ["HadithNo", "IsnadNo", "Position", "Narrator"],
        detail_rows,
    )


def _make_edge_data(raw_dir: Path) -> None:
    """CoreInfo (.xlsx) + explicit edge-layout DetailsInfo (.csv)."""
    mis_dir = raw_dir / "mis"
    mis_dir.mkdir(parents=True)

    _write_xlsx(
        mis_dir / "Hadith_SahihMuslim_CoreInfo.xlsx",
        ["HadithNo", "BookNo", "Matn"],
        [[1, 2, "متن"]],
    )

    csv = "HadithNo,IsnadNo,FromNarrator,ToNarrator\n"
    csv += "1,A,A,B\n1,A,B,C\n1,B,A,D\n1,B,D,C\n"
    (mis_dir / "Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.csv").write_text(
        csv, encoding="utf-8"
    )


class TestMisSequenceLayout:
    def test_produces_both_parquet_files(self, tmp_path: Path) -> None:
        _make_sequence_data(tmp_path / "raw")
        hadiths_path, edges_path = run(tmp_path / "raw", tmp_path / "staging")
        assert hadiths_path.exists()
        assert edges_path.exists()

    def test_hadith_schema_and_tagging(self, tmp_path: Path) -> None:
        _make_sequence_data(tmp_path / "raw")
        hadiths_path, _ = run(tmp_path / "raw", tmp_path / "staging")

        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows == 2
        assert set(table.column("source_corpus").to_pylist()) == {"mis"}
        assert set(table.column("sect").to_pylist()) == {"sunni"}
        # source_id grammar is corpus-namespaced and contract-valid.
        for sid in table.column("source_id").to_pylist():
            assert validate_source_id(sid) == [], sid
            assert sid.startswith("mis:sahih_muslim:")

    def test_edge_schema_and_tagging(self, tmp_path: Path) -> None:
        _make_sequence_data(tmp_path / "raw")
        _, edges_path = run(tmp_path / "raw", tmp_path / "staging")

        table = pq.read_table(edges_path)
        assert table.schema == NETWORK_EDGE_SCHEMA
        assert set(table.column("source").to_pylist()) == {"mis"}
        # MIS isnad edges declare TRANSMITTED_TO so the graph loader never routes
        # them to STUDIED_UNDER (da#133).
        assert set(table.column("relation").to_pylist()) == {"TRANSMITTED_TO"}

    def test_multiplicity_preserved(self, tmp_path: Path) -> None:
        """The three isnads of hadith 1 must survive as three distinct chains."""
        _make_sequence_data(tmp_path / "raw")
        _, edges_path = run(tmp_path / "raw", tmp_path / "staging")
        table = pq.read_table(edges_path)
        rows = table.to_pylist()

        # Total edges = 2 (isnad A) + 2 (isnad B) + 2 (isnad E) + 1 (hadith 2) = 7.
        assert len(rows) == 7

        h1_id = "mis:sahih_muslim:1:1"
        h1_pairs = {
            (r["from_narrator_name"], r["to_narrator_name"])
            for r in rows
            if r["hadith_id"] == h1_id
        }
        # Five distinct directed pairs across the three chains.
        assert h1_pairs == {("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("E", "B")}
        # Chain-specific edges that ONLY exist because the parallel asanid were
        # not collapsed into one sequence:
        assert ("A", "D") in h1_pairs
        assert ("E", "B") in h1_pairs
        # Every hadith-1 edge points at the same hadith node as its HADITH row.
        assert all(
            r["hadith_id"] == h1_id
            for r in rows
            if r["from_narrator_name"] in {"A", "B", "C", "D", "E"}
            and r["to_narrator_name"] in {"A", "B", "C", "D", "E"}
        )

    def test_edges_join_to_hadith_source_id(self, tmp_path: Path) -> None:
        _make_sequence_data(tmp_path / "raw")
        hadiths_path, edges_path = run(tmp_path / "raw", tmp_path / "staging")
        hadith_ids = set(pq.read_table(hadiths_path).column("source_id").to_pylist())
        edge_hadith_ids = set(pq.read_table(edges_path).column("hadith_id").to_pylist())
        # Every edge references a real hadith node id.
        assert edge_hadith_ids <= hadith_ids


class TestMisEdgeLayout:
    def test_explicit_edges_csv(self, tmp_path: Path) -> None:
        _make_edge_data(tmp_path / "raw")
        hadiths_path, edges_path = run(tmp_path / "raw", tmp_path / "staging")

        edges = pq.read_table(edges_path).to_pylist()
        pairs = {(e["from_narrator_name"], e["to_narrator_name"]) for e in edges}
        assert pairs == {("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")}
        assert all(e["hadith_id"] == "mis:sahih_muslim:2:1" for e in edges)
        # CoreInfo book number (2) flows into the source_id.
        assert pq.read_table(hadiths_path).column("source_id").to_pylist() == [
            "mis:sahih_muslim:2:1"
        ]


def _make_current_mendeley_data(raw_dir: Path) -> None:
    """The **current** Mendeley v2 three-file graph export (da#144).

    Mirrors the real dataset shape:
      * ``1_…HadithContent.xlsx``        — core hadith rows (HadithContent cols).
      * ``2_…Narrators=Nodes for Graph.xlsx`` — narrator NODE list (decoy: its
        filename contains "narrator", so it must NOT be picked as the detail file).
      * ``3_…Isnad=Edges for Graph.xlsx`` — explicit ``source*``/``target*`` edge
        rows keyed by ``(BookNo, HadithNo, SanadNo)``.

    Hadith 1 carries TWO sanads (1, 2) that share endpoints but diverge in the
    middle; hadith 2 a single sanad. The chain-specific edges only survive if the
    parallel asanid are kept apart.
    """
    mis_dir = raw_dir / "mis"
    mis_dir.mkdir(parents=True)

    _write_xlsx(
        mis_dir / "1_Hadith_SahihMuslim_HadithContent.xlsx",
        [
            "BookNo",
            "BookTitleEnAr",
            "HadithNo",
            "ChapterNo",
            "UnitNo",
            "HadithText_Mushakkal",
            "HadithText_GhairMushakkal",
            "SanadCount",
        ],
        [
            [2, "Sahih Muslim", 1, 1, 1, "مَتْنٌ مُشَكَّل", "متن", 2],
            [2, "Sahih Muslim", 2, 1, 2, "مَتْنٌ ثَانٍ", "متن ثان", 1],
        ],
    )

    # Decoy node list — filename contains "narrator" but is NOT the edge file.
    _write_xlsx(
        mis_dir / "2_Hadith_SahihMuslim_Narrators=Nodes for Graph.xlsx",
        ["Label", "NarratorID_Mapped", "NarratorNameEn-Mapped"],
        [["A (0)", 1, "A"], ["B (0)", 2, "B"]],
    )

    # (BookNo, HadithNo, SanadNo, sourceNarratorID, sourceNarratorNameEn,
    #  targetNarratorID, targetNarratorNameEn)
    edge_rows: list[list[object]] = [
        # hadith 1, sanad 1: A -> B -> C
        [2, 1, 1, 10, "A", 11, "B"],
        [2, 1, 1, 11, "B", 12, "C"],
        # hadith 1, sanad 2: A -> D -> C  (chain-specific A->D, D->C)
        [2, 1, 2, 10, "A", 13, "D"],
        [2, 1, 2, 13, "D", 12, "C"],
        # hadith 2, single sanad: X -> Y
        [2, 2, 1, 20, "X", 21, "Y"],
    ]
    _write_xlsx(
        mis_dir / "3_Hadith_SahihMuslim_Isnad=Edges for Graph.xlsx",
        [
            "BookNo",
            "HadithNo",
            "SanadNo",
            "sourceNarratorID",
            "sourceNarratorNameEn",
            "targetNarratorID",
            "targetNarratorNameEn",
        ],
        edge_rows,
    )


class TestMisCurrentMendeleyShape:
    """da#144 — parse the current three-file Mendeley graph export."""

    def test_selects_content_and_edges_not_nodes(self, tmp_path: Path) -> None:
        """The detail file resolves to the *Edges* workbook, never the *Nodes* one."""
        from src.parse.mis import _CORE_TOKENS, _DETAIL_TOKENS, _find_file

        _make_current_mendeley_data(tmp_path / "raw")
        mis_dir = tmp_path / "raw" / "mis"
        core = _find_file(mis_dir, _CORE_TOKENS)
        detail = _find_file(mis_dir, _DETAIL_TOKENS)
        assert core is not None and "HadithContent" in core.name
        assert detail is not None and "Edges" in detail.name
        assert "Nodes" not in detail.name  # the decoy must lose

    def test_hadith_schema_matn_and_tagging(self, tmp_path: Path) -> None:
        _make_current_mendeley_data(tmp_path / "raw")
        hadiths_path, _ = run(tmp_path / "raw", tmp_path / "staging")

        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows == 2
        assert set(table.column("source_corpus").to_pylist()) == {"mis"}
        assert set(table.column("sect").to_pylist()) == {"sunni"}
        # The vocalized HadithText_Mushakkal column feeds matn_ar.
        assert table.column("matn_ar").to_pylist() == ["مَتْنٌ مُشَكَّل", "مَتْنٌ ثَانٍ"]
        for sid in table.column("source_id").to_pylist():
            assert validate_source_id(sid) == [], sid
            assert sid.startswith("mis:sahih_muslim:")
        # BookNo flows into the source_id grammar.
        assert "mis:sahih_muslim:2:1" in table.column("source_id").to_pylist()

    def test_edge_schema_relation_and_external_ids(self, tmp_path: Path) -> None:
        _make_current_mendeley_data(tmp_path / "raw")
        _, edges_path = run(tmp_path / "raw", tmp_path / "staging")

        table = pq.read_table(edges_path)
        assert table.schema == NETWORK_EDGE_SCHEMA
        assert set(table.column("source").to_pylist()) == {"mis"}
        # isnad-transmission edges → TRANSMITTED_TO, never STUDIED_UNDER (da#133).
        assert set(table.column("relation").to_pylist()) == {"TRANSMITTED_TO"}
        # The source/target narrator IDs flow into the external-id columns.
        assert all(v is not None for v in table.column("from_external_id").to_pylist())
        assert all(v is not None for v in table.column("to_external_id").to_pylist())
        ab = next(
            r
            for r in table.to_pylist()
            if r["from_narrator_name"] == "A" and r["to_narrator_name"] == "B"
        )
        assert ab["from_external_id"] == "10"
        assert ab["to_external_id"] == "11"

    def test_multiplicity_preserved_across_sanads(self, tmp_path: Path) -> None:
        """Hadith 1's two sanads survive as distinct chains (A->D, D->C exist)."""
        _make_current_mendeley_data(tmp_path / "raw")
        _, edges_path = run(tmp_path / "raw", tmp_path / "staging")
        rows = pq.read_table(edges_path).to_pylist()

        # 2 (sanad 1) + 2 (sanad 2) + 1 (hadith 2) = 5 edges.
        assert len(rows) == 5
        h1_id = "mis:sahih_muslim:2:1"
        h1_pairs = {
            (r["from_narrator_name"], r["to_narrator_name"])
            for r in rows
            if r["hadith_id"] == h1_id
        }
        assert h1_pairs == {("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")}
        # Chain-specific edges that only exist because the asanid were not collapsed.
        assert ("A", "D") in h1_pairs
        assert ("D", "C") in h1_pairs

    def test_edges_join_to_hadith_source_id(self, tmp_path: Path) -> None:
        _make_current_mendeley_data(tmp_path / "raw")
        hadiths_path, edges_path = run(tmp_path / "raw", tmp_path / "staging")
        hadith_ids = set(pq.read_table(hadiths_path).column("source_id").to_pylist())
        edge_hadith_ids = set(pq.read_table(edges_path).column("hadith_id").to_pylist())
        assert edge_hadith_ids <= hadith_ids


class TestMisErrors:
    def test_missing_source_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError):
            run(tmp_path / "raw", tmp_path / "staging")

    def test_missing_detail_file_raises(self, tmp_path: Path) -> None:
        mis_dir = tmp_path / "raw" / "mis"
        mis_dir.mkdir(parents=True)
        _write_xlsx(mis_dir / "Hadith_SahihMuslim_CoreInfo.xlsx", ["HadithNo"], [[1]])
        with pytest.raises(FileNotFoundError):
            run(tmp_path / "raw", tmp_path / "staging")
