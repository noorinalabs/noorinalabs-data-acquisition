"""Tests for the Itqan rijal narrator-bio parser (da#92a)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parse import itqan
from src.parse.itqan import (
    _GRADE_TO_TRUST,
    _build_aliases,
    _build_edges,
    _clean,
    _extract_death_year,
    _generation,
    _id_list,
    _iter_profiles,
    _parse_profile,
)
from src.parse.schemas import (
    NARRATOR_ALIAS_SCHEMA,
    NARRATOR_BIO_SCHEMA,
    NETWORK_EDGE_SCHEMA,
)

FIXTURE = Path(__file__).parent / "fixtures" / "itqan_rijal_sample.json"


def _output(outputs: list[Path], stem: str) -> Path:
    """Pick the staging file with the given stem from a parser's output list."""
    return next(p for p in outputs if p.name == f"{stem}.parquet")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw/itqan dir holding the sample profiles as a profiles_*.json bucket."""
    d = tmp_path / "raw" / "itqan"
    d.mkdir(parents=True)
    shutil.copy(FIXTURE, d / "profiles_sample.json")
    return tmp_path / "raw"


class TestPureMappings:
    def test_every_grade_bucket_maps_to_a_trust_tier(self) -> None:
        # All seven Itqan grade buckets must have a trustworthiness mapping.
        for grade in (
            "reliable",
            "mostly_reliable",
            "weak",
            "abandoned",
            "fabricator",
            "companion",
            "unknown",
        ):
            assert grade in _GRADE_TO_TRUST

    def test_jarh_tiers(self) -> None:
        assert _GRADE_TO_TRUST["reliable"] == "thiqa"
        assert _GRADE_TO_TRUST["mostly_reliable"] == "saduq"
        assert _GRADE_TO_TRUST["weak"] == "daif"
        assert _GRADE_TO_TRUST["abandoned"] == "matruk"
        assert _GRADE_TO_TRUST["fabricator"] == "kadhdhab"
        assert _GRADE_TO_TRUST["unknown"] == "unknown"

    def test_clean_treats_dash_as_null(self) -> None:
        assert _clean("-") is None
        assert _clean("  ") is None
        assert _clean(None) is None
        assert _clean("البصرة") == "البصرة"

    def test_extract_death_year_single(self) -> None:
        assert _extract_death_year("168 هـ") == 168

    def test_extract_death_year_range_takes_first(self) -> None:
        assert _extract_death_year("بين 161 هـ إلى 170 هـ") == 161

    def test_extract_death_year_missing(self) -> None:
        assert _extract_death_year("-") is None
        assert _extract_death_year("بعد عصر التابعين") is None

    def test_extract_death_year_rejects_absurd(self) -> None:
        assert _extract_death_year("3200") is None

    def test_generation_companion_is_sahabi(self) -> None:
        assert _generation({"tabaqat": "-"}, "companion") == "sahabi"

    def test_generation_from_tabaqat(self) -> None:
        assert _generation({"tabaqat": "السادسة"}, "weak") == "taba_tabii"
        assert _generation({"tabaqat": "الثالثة"}, "reliable") == "tabii"

    def test_generation_unknown_tabaqat_is_none(self) -> None:
        assert _generation({"tabaqat": "-"}, "weak") is None


class TestParseProfile:
    def test_row_shape_and_tagging(self) -> None:
        profile = {
            "id": 320,
            "full_name": "سعيد بن سماك بن حرب",
            "kunya": "أبي خزامة",
            "grade_en": "abandoned",
            "grade_ar": "متروك الحديث",
            "death": "168 هـ",
            "tabaqat": "-",
            "city": "البصرة",
            "nasab": "الذهلي",
            "laqab": "-",
        }
        row = _parse_profile("320", profile)
        assert row is not None
        assert row["bio_id"] == "itqan:320"
        assert row["source"] == "itqan"
        assert row["external_id"] == "320"
        assert row["name_ar"] == "سعيد بن سماك بن حرب"
        assert row["name_ar_normalized"]  # normalized, non-empty
        assert row["trustworthiness"] == "matruk"
        assert row["death_year_ah"] == 168
        assert row["birth_location"] == "البصرة"
        assert row["nisba"] == "الذهلي"
        assert row["laqab"] is None  # dash placeholder
        assert row["bio_text"] == "متروك الحديث"

    def test_profile_without_name_is_dropped(self) -> None:
        assert _parse_profile("1", {"id": 1, "full_name": "-", "grade_en": "weak"}) is None


class TestRunOverFixture:
    def test_produces_schema_conformant_parquet(self, raw_dir: Path, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        out = itqan.run(raw_dir, staging)
        # run() now emits bios + edges + aliases.
        assert {p.name for p in out} == {
            "narrators_bio_itqan.parquet",
            "network_edges_itqan.parquet",
            "narrator_aliases_itqan.parquet",
        }

        table = pq.read_table(_output(out, "narrators_bio_itqan"))
        assert table.schema.equals(NARRATOR_BIO_SCHEMA)

        with FIXTURE.open(encoding="utf-8") as f:
            expected = len(json.load(f))
        assert table.num_rows == expected

        rows = table.to_pylist()
        # Every row tagged to the itqan source with a namespaced, unique bio_id.
        assert {r["source"] for r in rows} == {"itqan"}
        bio_ids = [r["bio_id"] for r in rows]
        assert all(b.startswith("itqan:") for b in bio_ids)
        assert len(set(bio_ids)) == len(bio_ids)

        # The three real grade buckets in the fixture exercise three jarh tiers.
        trust = {r["trustworthiness"] for r in rows}
        assert {"matruk", "kadhdhab", "thiqa"} <= trust

    def test_idempotent_dedup_on_repeated_bucket(self, raw_dir: Path, tmp_path: Path) -> None:
        # A second bucket file with overlapping ids must not double-count.
        shutil.copy(FIXTURE, raw_dir / "itqan" / "profiles_dup.json")
        staging = tmp_path / "staging"
        staging.mkdir()
        out = itqan.run(raw_dir, staging)
        table = pq.read_table(_output(out, "narrators_bio_itqan"))
        with FIXTURE.open(encoding="utf-8") as f:
            expected = len(json.load(f))
        assert table.num_rows == expected


class TestEdgeExtraction:
    def test_id_list_coerces_ints_and_strs(self) -> None:
        assert _id_list([2325, 7019]) == ["2325", "7019"]
        assert _id_list(["1", "2"]) == ["1", "2"]
        assert _id_list(None) == []
        assert _id_list([]) == []

    def test_build_edges_orients_student_to_teacher(self) -> None:
        # P=1 has teacher 2 and student 3 -> edges (1->2) and (3->1).
        profiles = [
            ("1", {"full_name": "الف", "teachers": [2], "students": [3]}),
            ("2", {"full_name": "باء"}),
            ("3", {"full_name": "جيم"}),
        ]
        id_to_name = {"1": "الف", "2": "باء", "3": "جيم"}
        edges = _build_edges(profiles, id_to_name)
        pairs = {(e["from_external_id"], e["to_external_id"]) for e in edges}
        assert pairs == {("1", "2"), ("3", "1")}
        by_pair = {(e["from_external_id"], e["to_external_id"]): e for e in edges}
        # from = student, to = teacher; names resolved from the id map.
        assert by_pair[("1", "2")]["from_narrator_name"] == "الف"
        assert by_pair[("1", "2")]["to_narrator_name"] == "باء"
        # rijal edges are not hadith-scoped.
        assert all(e["hadith_id"] is None for e in edges)
        assert all(e["source"] == "itqan" for e in edges)

    def test_build_edges_dedups_reciprocal_perspectives(self) -> None:
        # 1 lists 2 as teacher AND 2 lists 1 as student -> the SAME edge (1->2).
        profiles = [
            ("1", {"full_name": "الف", "teachers": [2]}),
            ("2", {"full_name": "باء", "students": [1]}),
        ]
        edges = _build_edges(profiles, {"1": "الف", "2": "باء"})
        assert len(edges) == 1
        assert (edges[0]["from_external_id"], edges[0]["to_external_id"]) == ("1", "2")

    def test_build_edges_skips_self_loops(self) -> None:
        edges = _build_edges([("1", {"full_name": "الف", "teachers": [1]})], {"1": "الف"})
        assert edges == []

    def test_run_emits_schema_valid_nonempty_edges(self, raw_dir: Path, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        out = itqan.run(raw_dir, staging)
        table = pq.read_table(_output(out, "network_edges_itqan"))
        assert table.schema.equals(NETWORK_EDGE_SCHEMA)
        assert table.num_rows > 0
        rows = table.to_pylist()
        assert {r["source"] for r in rows} == {"itqan"}
        # Provenance: both endpoints carry the source profile id.
        assert all(r["from_external_id"] and r["to_external_id"] for r in rows)
        # Studentship edges declare STUDIED_UNDER for relation-keyed loading (da#133).
        assert {r["relation"] for r in rows} == {"STUDIED_UNDER"}


class TestAliasExtraction:
    def test_build_aliases_drops_primary_spelling_and_dups(self) -> None:
        profile = {
            "full_name": "سعيد بن سماك",
            "namings": [
                "سعيد بن سماك",  # == primary -> dropped
                "ابن سماك",
                "ابن سماك",  # duplicate -> collapsed
                "-",  # placeholder -> dropped
            ],
        }
        rows = _build_aliases([("320", profile)])
        variants = [r["alias"] for r in rows]
        assert variants == ["ابن سماك"]
        assert rows[0]["bio_id"] == "itqan:320"
        assert rows[0]["source"] == "itqan"
        assert rows[0]["alias_normalized"]

    def test_build_aliases_keys_to_canonical_name(self) -> None:
        # canonical_name_ar_normalized must equal the bio's name_ar_normalized so
        # bio_promote can recompute the same nar: id.
        from src.utils.arabic import normalize_arabic

        profile = {"full_name": "محمد", "namings": ["أبو القاسم"]}
        rows = _build_aliases([("5", profile)])
        assert rows[0]["canonical_name_ar_normalized"] == normalize_arabic("محمد")

    def test_build_aliases_skips_profile_without_name(self) -> None:
        assert _build_aliases([("9", {"full_name": "-", "namings": ["x"]})]) == []

    def test_run_emits_schema_valid_nonempty_aliases(self, raw_dir: Path, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        out = itqan.run(raw_dir, staging)
        table = pq.read_table(_output(out, "narrator_aliases_itqan"))
        assert table.schema.equals(NARRATOR_ALIAS_SCHEMA)
        assert table.num_rows > 0
        rows = table.to_pylist()
        assert {r["source"] for r in rows} == {"itqan"}
        assert all(r["bio_id"].startswith("itqan:") for r in rows)


class TestIterProfiles:
    def test_first_occurrence_wins_across_buckets(self, raw_dir: Path) -> None:
        # Duplicate the fixture as a second bucket; profile ids must not double.
        shutil.copy(FIXTURE, raw_dir / "itqan" / "profiles_dup.json")
        buckets = sorted((raw_dir / "itqan").glob("profiles_*.json"))
        profiles = _iter_profiles(buckets)
        with FIXTURE.open(encoding="utf-8") as f:
            expected = len(json.load(f))
        assert len(profiles) == expected
        assert len({pid for pid, _ in profiles}) == expected
