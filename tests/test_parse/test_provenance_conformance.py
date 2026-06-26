"""Provenance-tagging conformance gate (da#83, T0-B).

Every parser must emit ``sect`` / ``source_corpus`` / ``source`` provenance that
is non-empty and in the enum value-domain. A non-nullable schema column already
rejects a *null* value at construction — the gap this gate closes is the
**empty string** (which passes a non-nullable column) and the **wrong-domain**
value (a ``source_corpus`` typo, a ``sect`` outside the enum) that nothing else
catches. The invariant is enforced fail-fast in
:func:`src.parse.base.write_parquet`, so it holds for every parser and every
future parser, and these tests assert that contract directly.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.parse as parse_pkg
from src.models.enums import Sect, SourceCorpus
from src.parse.base import provenance_violations, write_parquet
from src.parse.schemas import (
    COLLECTION_SCHEMA,
    HADITH_SCHEMA,
    NARRATOR_BIO_SCHEMA,
    NARRATOR_MENTION_SCHEMA,
    NETWORK_EDGE_SCHEMA,
)

# A canonical schema is "provenance-bearing" iff it carries one of these columns.
_PROVENANCE_COLUMNS = frozenset({"sect", "source_corpus", "source"})
_ALL_SCHEMA_NAMES = frozenset(
    {
        "HADITH_SCHEMA",
        "NARRATOR_MENTION_SCHEMA",
        "NARRATOR_BIO_SCHEMA",
        "COLLECTION_SCHEMA",
        "NETWORK_EDGE_SCHEMA",
    }
)

# Parse modules that are framework, not per-source parsers — excluded from the
# "every parser emits a provenance-bearing schema" coverage assertion.
_NON_PARSER_MODULES = frozenset(
    {
        "__init__",
        "base",
        "schemas",
        "validate",
        "narrator_extraction",
        "identity",
        # da#191: declares the canonical corpus composition (which source/collection
        # pairs load) — not a per-source parser, emits no parquet.
        "composition",
        # da#230: curated sourced name_ar/expected_count fill applied by parsers —
        # a metadata table, not a per-source parser, emits no parquet.
        "collection_metadata",
        # da#229: shared in-book-ordinal derivation helper — mutates a parser's
        # records in place, emits no schema/parquet of its own.
        "in_book_ordinal",
        # da#164: pure life-date notation parser used BY the source parsers to fill
        # the bound/precision columns — emits no schema/parquet of its own.
        "narrator_dates",
    }
)


def _valid_hadith_table(**overrides: object) -> pa.Table:
    """A 1-row HADITH_SCHEMA table with valid provenance, castable to schema."""
    row: dict[str, object] = {field.name: None for field in HADITH_SCHEMA}
    row.update(
        source_id="sunnah:bukhari:1:1:1",
        source_corpus="sunnah",
        collection_name="bukhari",
        sect="sunni",
    )
    row.update(overrides)
    return pa.table({k: [v] for k, v in row.items()}).cast(HADITH_SCHEMA)


class TestProvenanceValidatorClean:
    """Valid provenance produces no violations."""

    def test_valid_hadith_provenance(self) -> None:
        table = pa.table({"sect": ["sunni", "shia"], "source_corpus": ["sunnah", "thaqalayn"]})
        assert provenance_violations(table) == []

    def test_valid_bio_source(self) -> None:
        table = pa.table({"source": ["muhaddithat", "kaggle_narrators"]})
        assert provenance_violations(table) == []

    def test_table_without_provenance_columns_is_clean(self) -> None:
        # A non-canonical intermediate (no provenance columns) is not checked.
        table = pa.table({"some_metric": [1, 2, 3]})
        assert provenance_violations(table) == []

    def test_every_enum_value_accepted(self) -> None:
        table = pa.table(
            {
                "sect": [s.value for s in Sect],
                "source_corpus": [s.value for s in list(SourceCorpus)[: len(list(Sect))]],
            }
        )
        assert provenance_violations(table) == []


class TestProvenanceValidatorCatchesGaps:
    """The empty-string / wrong-domain gaps a non-nullable schema misses."""

    def test_empty_string_sect_flagged(self) -> None:
        table = pa.table({"sect": ["sunni", ""], "source_corpus": ["sunnah", "sunnah"]})
        problems = provenance_violations(table)
        assert any("sect" in p and "empty" in p for p in problems)

    def test_whitespace_only_sect_flagged(self) -> None:
        table = pa.table({"sect": ["   "], "source_corpus": ["sunnah"]})
        problems = provenance_violations(table)
        assert any("sect" in p and "empty" in p for p in problems)

    def test_wrong_domain_sect_flagged(self) -> None:
        # Right idea, wrong casing/domain — a non-nullable column would accept it.
        table = pa.table({"sect": ["Sunni"], "source_corpus": ["sunnah"]})
        problems = provenance_violations(table)
        assert any("sect" in p and "outside" in p for p in problems)

    def test_wrong_domain_source_corpus_flagged(self) -> None:
        table = pa.table({"sect": ["sunni"], "source_corpus": ["sunna"]})
        problems = provenance_violations(table)
        assert any("source_corpus" in p and "outside" in p for p in problems)

    def test_empty_source_flagged(self) -> None:
        table = pa.table({"source": ["muhaddithat", ""]})
        problems = provenance_violations(table)
        assert any("source" in p and "empty" in p for p in problems)

    def test_null_provenance_flagged(self) -> None:
        # A null in a column the schema declares NON-nullable is flagged (the
        # parser-staging contract: sect/source_corpus are mandatory tags).
        schema = pa.schema(
            [
                pa.field("sect", pa.string(), nullable=False),
                pa.field("source_corpus", pa.string(), nullable=False),
            ]
        )
        table = pa.table(
            {"sect": ["sunni", None], "source_corpus": ["sunnah", "sunnah"]}, schema=schema
        )
        problems = provenance_violations(table)
        assert any("sect" in p and "null" in p for p in problems)

    def test_null_allowed_in_nullable_column(self) -> None:
        # A resolve-derived NULLABLE provenance column (e.g. narrators_canonical
        # ``source_corpus``, da#103) permits null — "no attributable corpus" — but
        # empty-string / wrong-domain are still caught on the values that ARE present.
        schema = pa.schema([pa.field("source_corpus", pa.string(), nullable=True)])
        clean = pa.table({"source_corpus": ["sunnah", None]}, schema=schema)
        assert provenance_violations(clean) == []
        bad = pa.table({"source_corpus": ["sunna", None]}, schema=schema)
        assert any("source_corpus" in p and "outside" in p for p in provenance_violations(bad))


class TestWriteParquetEnforces:
    """write_parquet is the fail-fast chokepoint for the provenance contract."""

    def test_valid_table_writes(self, tmp_path: Path) -> None:
        out = write_parquet(_valid_hadith_table(), tmp_path / "hadiths_x.parquet")
        assert out.exists()
        assert pq.read_table(out).num_rows == 1

    def test_empty_sect_is_rejected(self, tmp_path: Path) -> None:
        bad = _valid_hadith_table(sect="")
        with pytest.raises(ValueError, match="provenance conformance failed"):
            write_parquet(bad, tmp_path / "hadiths_bad.parquet")

    def test_wrong_domain_corpus_is_rejected(self, tmp_path: Path) -> None:
        bad = _valid_hadith_table(source_corpus="sunna")
        with pytest.raises(ValueError, match="source_corpus"):
            write_parquet(bad, tmp_path / "hadiths_bad.parquet")

    def test_rejected_table_is_not_written(self, tmp_path: Path) -> None:
        dest = tmp_path / "hadiths_bad.parquet"
        with pytest.raises(ValueError):
            write_parquet(_valid_hadith_table(sect=""), dest)
        assert not dest.exists()


class TestEverySchemaIsProvenanceBearing:
    """Each canonical staging schema carries a provenance column."""

    @pytest.mark.parametrize(
        "schema",
        [
            HADITH_SCHEMA,
            NARRATOR_MENTION_SCHEMA,
            NARRATOR_BIO_SCHEMA,
            COLLECTION_SCHEMA,
            NETWORK_EDGE_SCHEMA,
        ],
    )
    def test_schema_has_provenance_column(self, schema: pa.Schema) -> None:
        assert _PROVENANCE_COLUMNS & {f.name for f in schema}, (
            f"schema {schema} carries no provenance column ({sorted(_PROVENANCE_COLUMNS)})"
        )


class TestEveryParserEmitsProvenanceSchema:
    """A per-source parser must emit at least one provenance-bearing schema.

    Guards against a future parser that writes raw, untagged staging data: every
    module under ``src/parse`` that is a per-source parser must reference one of
    the canonical schemas (all of which are provenance-bearing).
    """

    def _parser_modules(self) -> list[str]:
        names = [
            m.name
            for m in pkgutil.iter_modules(parse_pkg.__path__)
            if not m.ispkg and m.name not in _NON_PARSER_MODULES
        ]
        assert names, "no per-source parser modules discovered under src/parse"
        return names

    def test_parser_modules_reference_a_canonical_schema(self) -> None:
        offenders: list[str] = []
        for name in self._parser_modules():
            module = importlib.import_module(f"src.parse.{name}")
            source = inspect.getsource(module)
            if not any(schema_name in source for schema_name in _ALL_SCHEMA_NAMES):
                offenders.append(name)
        assert not offenders, (
            f"parser modules emit no provenance-bearing canonical schema: {offenders}"
        )
