"""Tests for src.resolve.schemas — Phase 2 resolve output schemas."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from src.resolve.schemas import (
    AMBIGUOUS_NARRATORS_SCHEMA,
    NARRATOR_MENTIONS_RESOLVED_SCHEMA,
    NARRATORS_CANONICAL_SCHEMA,
    PARALLEL_LINKS_SCHEMA,
)

_ALL_SCHEMAS = [
    ("NARRATOR_MENTIONS_RESOLVED_SCHEMA", NARRATOR_MENTIONS_RESOLVED_SCHEMA),
    ("NARRATORS_CANONICAL_SCHEMA", NARRATORS_CANONICAL_SCHEMA),
    ("AMBIGUOUS_NARRATORS_SCHEMA", AMBIGUOUS_NARRATORS_SCHEMA),
    ("PARALLEL_LINKS_SCHEMA", PARALLEL_LINKS_SCHEMA),
]


class TestEmptyTables:
    @pytest.mark.parametrize("name,schema", _ALL_SCHEMAS, ids=[s[0] for s in _ALL_SCHEMAS])
    def test_empty_table_creation(self, name: str, schema: pa.Schema) -> None:
        table = schema.empty_table()
        assert table.num_rows == 0
        assert table.schema.equals(schema)


class TestSampleData:
    def test_narrator_mentions_resolved(self) -> None:
        data = {
            "mention_id": pa.array(["m-1"], type=pa.string()),
            "hadith_id": pa.array(["h-1"], type=pa.string()),
            "source_corpus": pa.array(["sunnah"], type=pa.string()),
            "position_in_chain": pa.array([0], type=pa.int32()),
            "name_raw": pa.array(["Abu Hurayra"], type=pa.string()),
            "name_normalized": pa.array(["abu hurayra"], type=pa.string()),
            "canonical_narrator_id": pa.array([None], type=pa.string()),
            "transmission_method": pa.array(["haddathana"], type=pa.string()),
            "confidence": pa.array([None], type=pa.float32()),
        }
        table = pa.table(data, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA)
        assert table.num_rows == 1
        assert table.schema.equals(NARRATOR_MENTIONS_RESOLVED_SCHEMA)

    def test_narrators_canonical(self) -> None:
        data = {
            "canonical_id": pa.array(["c-1"], type=pa.string()),
            "name_ar": pa.array(["\u0639\u0644\u064a"], type=pa.string()),
            "name_en": pa.array(["Ali"], type=pa.string()),
            "name_ar_normalized": pa.array(["\u0639\u0644\u064a"], type=pa.string()),
            "aliases": pa.array([["Ali ibn Abi Talib"]], type=pa.list_(pa.string())),
            "birth_year_ah": pa.array([None], type=pa.int32()),
            "death_year_ah": pa.array([40], type=pa.int32()),
            "birth_year_ah_earliest": pa.array([None], type=pa.int32()),
            "birth_year_ah_latest": pa.array([None], type=pa.int32()),
            "birth_date_precision": pa.array(["unknown"], type=pa.string()),
            "death_year_ah_earliest": pa.array([38], type=pa.int32()),
            "death_year_ah_latest": pa.array([41], type=pa.int32()),
            "death_date_precision": pa.array(["range"], type=pa.string()),
            "generation": pa.array(["sahabi"], type=pa.string()),
            "gender": pa.array(["male"], type=pa.string()),
            "trustworthiness": pa.array(["thiqa"], type=pa.string()),
            "source_ids": pa.array([["bio-1"]], type=pa.list_(pa.string())),
            "external_id": pa.array([None], type=pa.string()),
            "death_year_provenance": pa.array(["corroborated"], type=pa.string()),
            "mention_count": pa.array([5], type=pa.int32()),
            "source_corpus": pa.array(["sunnah"], type=pa.string()),
            "source_corpora": pa.array([["sunnah", "thaqalayn"]], type=pa.list_(pa.string())),
            "sect_affiliation": pa.array(["neutral"], type=pa.string()),
        }
        table = pa.table(data, schema=NARRATORS_CANONICAL_SCHEMA)
        assert table.num_rows == 1
        assert {"source_corpus", "source_corpora", "sect_affiliation"} <= set(
            NARRATORS_CANONICAL_SCHEMA.names
        )


class TestNarratorModelMirror:
    """The canonical schema must mirror the date-bound fields on the Narrator
    model (da#161) 1:1 so model and schema cannot drift (da#162)."""

    _DATE_FIELDS = (
        "birth_year_ah_earliest",
        "birth_year_ah_latest",
        "birth_date_precision",
        "death_year_ah_earliest",
        "death_year_ah_latest",
        "death_date_precision",
    )

    def test_schema_carries_all_model_date_fields(self) -> None:
        from src.models.narrator import Narrator

        names = set(NARRATORS_CANONICAL_SCHEMA.names)
        for field in self._DATE_FIELDS:
            assert field in Narrator.model_fields, f"{field} missing from model"
            assert field in names, f"{field} missing from canonical schema"

    def test_date_fields_nullable(self) -> None:
        for field in self._DATE_FIELDS:
            assert NARRATORS_CANONICAL_SCHEMA.field(field).nullable, (
                f"{field} must be nullable (producers leave it None via r.get)"
            )

    def test_roundtrips_with_model(self) -> None:
        """A Narrator carrying populated date bounds + precision projects onto the
        canonical schema and reads back the same values."""
        from src.models.enums import (
            DatePrecision,
            Gender,
            NarratorGeneration,
            SectAffiliation,
            TrustworthinessGrade,
        )
        from src.models.narrator import Narrator

        narrator = Narrator(
            id="nar:ali-001",
            name_ar="علي",
            name_en="Ali",
            death_year_ah=40,
            death_year_ah_earliest=38,
            death_year_ah_latest=41,
            death_date_precision=DatePrecision.RANGE,
            generation=NarratorGeneration.SAHABI,
            gender=Gender.MALE,
            sect_affiliation=SectAffiliation.NEUTRAL,
            trustworthiness_consensus=TrustworthinessGrade.THIQA,
        )
        # Project the model's date-bound fields onto a single-row canonical table,
        # exactly as the producers do via ``{f.name: [r.get(f.name)] ...}``.
        record: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
        record["canonical_id"] = narrator.id
        for field in self._DATE_FIELDS:
            value = getattr(narrator, field)
            record[field] = value.value if isinstance(value, DatePrecision) else value
        arrays = {f.name: [record[f.name]] for f in NARRATORS_CANONICAL_SCHEMA}
        table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)

        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["death_year_ah_earliest"] == 38
        assert row["death_year_ah_latest"] == 41
        assert row["death_date_precision"] == "range"
        assert row["birth_date_precision"] == "unknown"
        assert row["birth_year_ah_earliest"] is None

    def test_ambiguous_narrators(self) -> None:
        data = {
            "mention_id": pa.array(["m-1"], type=pa.string()),
            "mention_text": pa.array(["ambiguous name"], type=pa.string()),
            "source_corpus": pa.array(["sunnah"], type=pa.string()),
            "candidate_1_id": pa.array(["c-1"], type=pa.string()),
            "candidate_1_name": pa.array(["Name A"], type=pa.string()),
            "candidate_1_score": pa.array([0.65], type=pa.float32()),
            "candidate_1_stage": pa.array(["fuzzy"], type=pa.string()),
            "candidate_2_id": pa.array([None], type=pa.string()),
            "candidate_2_name": pa.array([None], type=pa.string()),
            "candidate_2_score": pa.array([None], type=pa.float32()),
            "candidate_2_stage": pa.array([None], type=pa.string()),
            "candidate_3_id": pa.array([None], type=pa.string()),
            "candidate_3_name": pa.array([None], type=pa.string()),
            "candidate_3_score": pa.array([None], type=pa.float32()),
            "candidate_3_stage": pa.array([None], type=pa.string()),
        }
        table = pa.table(data, schema=AMBIGUOUS_NARRATORS_SCHEMA)
        assert table.num_rows == 1

    def test_parallel_links(self) -> None:
        data = {
            "hadith_id_a": pa.array(["h-1"], type=pa.string()),
            "hadith_id_b": pa.array(["h-2"], type=pa.string()),
            "similarity_score": pa.array([0.92], type=pa.float32()),
            "variant_type": pa.array(["verbatim"], type=pa.string()),
            "cross_sect": pa.array([False], type=pa.bool_()),
        }
        table = pa.table(data, schema=PARALLEL_LINKS_SCHEMA)
        assert table.num_rows == 1
