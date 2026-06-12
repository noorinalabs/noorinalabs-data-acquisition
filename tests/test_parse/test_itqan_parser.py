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
    _clean,
    _extract_death_year,
    _generation,
    _parse_profile,
)
from src.parse.schemas import NARRATOR_BIO_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "itqan_rijal_sample.json"


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
        assert out.name == "narrators_bio_itqan.parquet"

        table = pq.read_table(out)
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
        table = pq.read_table(out)
        with FIXTURE.open(encoding="utf-8") as f:
            expected = len(json.load(f))
        assert table.num_rows == expected
