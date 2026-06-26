"""Tests for ``src.resolve.muhaddithat_links`` — curated orphan mention-links (da#228).

Pins the ADR-004 item-#3 contract: the 8 bio-only muhaddithat orphans are
mention-linked **with first-class provenance**, **exactly** the named set (no
bulk-link), **evidence-anchored** to the canonical id ``bio_promote`` mints, and
**never fabricated** for a narrator that was not promoted.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parse.identity import make_canonical_id
from src.resolve.muhaddithat_links import (
    MUHADDITHAT_ORPHAN_LINKS,
    OrphanLink,
    build_muhaddithat_mention_links,
    canonical_id_for,
)
from src.resolve.schemas import MUHADDITHAT_MENTION_LINKS_SCHEMA
from src.utils.arabic import normalize_arabic


def _write_canonical(curated: Path, canonical_ids: list[str]) -> Path:
    """Write a minimal narrators_canonical.parquet carrying only ``canonical_id``."""
    from tests.test_graph.conftest import write_narrators_canonical

    return write_narrators_canonical(curated, [{"canonical_id": cid} for cid in canonical_ids])


class TestCuratedSet:
    def test_exactly_eight_links(self) -> None:
        assert len(MUHADDITHAT_ORPHAN_LINKS) == 8

    def test_every_link_carries_nonempty_provenance(self) -> None:
        # Provenance is the whole point — an orphan-link with no source of the link
        # is the "auto-link blindly" failure ADR-004 forbids.
        for link in MUHADDITHAT_ORPHAN_LINKS:
            assert link.provenance.strip(), f"{link.name_en} has empty provenance"

    def test_links_target_distinct_hadith(self) -> None:
        # Distinct hadith per link so the position-0 NARRATED reduction emits one
        # edge per narrator (two links on one hadith would collapse to one edge).
        hadith_ids = [link.hadith_id for link in MUHADDITHAT_ORPHAN_LINKS]
        assert len(set(hadith_ids)) == len(hadith_ids)

    def test_links_target_distinct_narrators(self) -> None:
        cids = [canonical_id_for(link.name_ar) for link in MUHADDITHAT_ORPHAN_LINKS]
        assert len(set(cids)) == len(cids)

    def test_canonical_id_routes_through_bio_promote_identity(self) -> None:
        # The link MUST resolve to the SAME id bio_promote mints, or it attaches to
        # no node. canonical_id_for must equal make_canonical_id(normalize_arabic()).
        for link in MUHADDITHAT_ORPHAN_LINKS:
            expected = make_canonical_id(normalize_arabic(link.name_ar))
            assert canonical_id_for(link.name_ar) == expected
            assert expected.startswith("nar:")


class TestBuildLinks:
    def test_emits_one_row_per_link_unguarded(self, tmp_path: Path) -> None:
        curated = tmp_path / "curated"
        curated.mkdir()
        path = build_muhaddithat_mention_links(curated)
        assert path is not None
        assert path.name == "narrator_mentions_resolved_muhaddithat.parquet"
        # Filename rides the resolved-mentions glob the NARRATED loader reads.
        assert path.name.startswith("narrator_mentions_resolved")

        table = pq.read_table(path)
        assert table.num_rows == len(MUHADDITHAT_ORPHAN_LINKS)
        assert table.schema.equals(MUHADDITHAT_MENTION_LINKS_SCHEMA)

    def test_every_row_carries_provenance_and_canonical_id(self, tmp_path: Path) -> None:
        curated = tmp_path / "curated"
        curated.mkdir()
        path = build_muhaddithat_mention_links(curated)
        assert path is not None
        rows = pq.read_table(path).to_pylist()

        by_provenance = {r["provenance"] for r in rows}
        assert by_provenance == {link.provenance for link in MUHADDITHAT_ORPHAN_LINKS}

        for r in rows:
            assert r["provenance"]  # non-null, non-empty
            assert r["canonical_narrator_id"].startswith("nar:")
            assert r["position_in_chain"] == 0
            assert r["source_corpus"] == "muhaddithat"

    def test_provenance_column_is_non_nullable(self) -> None:
        field = MUHADDITHAT_MENTION_LINKS_SCHEMA.field("provenance")
        assert not field.nullable

    def test_resolved_canonical_ids_match_curated(self, tmp_path: Path) -> None:
        curated = tmp_path / "curated"
        curated.mkdir()
        path = build_muhaddithat_mention_links(curated)
        assert path is not None
        rows = pq.read_table(path).to_pylist()
        emitted = {r["canonical_narrator_id"] for r in rows}
        expected = {canonical_id_for(link.name_ar) for link in MUHADDITHAT_ORPHAN_LINKS}
        assert emitted == expected


class TestEvidenceAnchorGuard:
    def test_only_emits_links_for_promoted_narrators(self, tmp_path: Path) -> None:
        # Canonical master carries only the FIRST three orphans — the other five
        # were never promoted, so their links must be dropped (not fabricated).
        curated = tmp_path / "curated"
        curated.mkdir()
        present = MUHADDITHAT_ORPHAN_LINKS[:3]
        canonical_path = _write_canonical(
            curated, [canonical_id_for(link.name_ar) for link in present]
        )

        path = build_muhaddithat_mention_links(curated, canonical_path=canonical_path)
        assert path is not None
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == 3
        emitted = {r["canonical_narrator_id"] for r in rows}
        assert emitted == {canonical_id_for(link.name_ar) for link in present}

    def test_returns_none_when_no_narrator_present(self, tmp_path: Path) -> None:
        curated = tmp_path / "curated"
        curated.mkdir()
        canonical_path = _write_canonical(curated, ["nar:unrelated-id"])
        path = build_muhaddithat_mention_links(curated, canonical_path=canonical_path)
        assert path is None
        assert not (curated / "narrator_mentions_resolved_muhaddithat.parquet").exists()

    def test_guard_skipped_when_no_canonical_master(self, tmp_path: Path) -> None:
        curated = tmp_path / "curated"
        curated.mkdir()
        missing = curated / "narrators_canonical.parquet"
        # canonical_path points at a non-existent file → guard skipped, all emitted.
        path = build_muhaddithat_mention_links(curated, canonical_path=missing)
        assert path is not None
        assert pq.read_table(path).num_rows == len(MUHADDITHAT_ORPHAN_LINKS)

    def test_no_bulk_link_beyond_named_set(self, tmp_path: Path) -> None:
        # An injected single-entry set proves the producer emits ONLY what it is
        # given — there is no scan that could add unnamed narrators.
        curated = tmp_path / "curated"
        curated.mkdir()
        one = (
            OrphanLink(
                name_ar="فاطمة بنت قيس",
                name_en="Fatima bint Qays",
                hadith_id="sunnah:muslim:1480b",
                transmission_method="haddathana",
                provenance="test-only attestation",
            ),
        )
        path = build_muhaddithat_mention_links(curated, links=one)
        assert path is not None
        assert pq.read_table(path).num_rows == 1


@pytest.mark.parametrize("link", MUHADDITHAT_ORPHAN_LINKS, ids=lambda link: link.name_en)
def test_each_link_is_well_formed(link: OrphanLink) -> None:
    assert link.name_ar.strip()
    assert link.name_en.strip()
    assert link.hadith_id.strip()
    assert link.transmission_method.strip()
    assert link.provenance.strip()
