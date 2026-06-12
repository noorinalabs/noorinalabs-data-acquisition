"""Parquet staging table schemas.

These define the intermediate representation between raw source files and
final graph nodes. Each source parser outputs one or more of these tables.
Entity resolution in Phase 2 will consume these and produce canonical records.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = [
    "HADITH_SCHEMA",
    "NARRATOR_MENTION_SCHEMA",
    "NARRATOR_BIO_SCHEMA",
    "NARRATOR_ALIAS_SCHEMA",
    "COLLECTION_SCHEMA",
    "NETWORK_EDGE_SCHEMA",
]

# ``source_id`` grammar and the graph node-id rules derived from it are the
# canonical identity contract — see :mod:`src.parse.identity` (``<corpus>:
# <collection>[:<part>...]``, corpus exactly once). ``hadith_number`` is the
# IN-BOOK ordinal that flows to ``APPEARS_IN.hadith_number_in_book`` (da#77) —
# NOT the collection-wide reference number.
HADITH_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_corpus", pa.string(), nullable=False),
        pa.field("collection_name", pa.string(), nullable=False),
        pa.field("book_number", pa.int32(), nullable=True),
        pa.field("chapter_number", pa.int32(), nullable=True),
        pa.field("hadith_number", pa.int32(), nullable=True),
        pa.field("matn_ar", pa.string(), nullable=True),
        pa.field("matn_en", pa.string(), nullable=True),
        pa.field("isnad_raw_ar", pa.string(), nullable=True),
        pa.field("isnad_raw_en", pa.string(), nullable=True),
        pa.field("full_text_ar", pa.string(), nullable=True),
        pa.field("full_text_en", pa.string(), nullable=True),
        pa.field("grade", pa.string(), nullable=True),
        pa.field("chapter_name_ar", pa.string(), nullable=True),
        pa.field("chapter_name_en", pa.string(), nullable=True),
        pa.field("sect", pa.string(), nullable=False),
    ]
)

NARRATOR_MENTION_SCHEMA = pa.schema(
    [
        pa.field("mention_id", pa.string(), nullable=False),
        pa.field("source_hadith_id", pa.string(), nullable=False),
        pa.field("source_corpus", pa.string(), nullable=False),
        pa.field("position_in_chain", pa.int32(), nullable=False),
        pa.field("name_ar", pa.string(), nullable=True),
        pa.field("name_en", pa.string(), nullable=True),
        pa.field("name_ar_normalized", pa.string(), nullable=True),
        # Values should be compatible with TransmissionMethod enum in src/models/enums.py
        # or raw Arabic phrases for Phase 2 mapping
        pa.field("transmission_method", pa.string(), nullable=True),
    ]
)

NARRATOR_BIO_SCHEMA = pa.schema(
    [
        pa.field("bio_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("name_ar", pa.string(), nullable=True),
        pa.field("name_en", pa.string(), nullable=True),
        pa.field("name_ar_normalized", pa.string(), nullable=True),
        pa.field("name_en_normalized", pa.string(), nullable=True),
        pa.field("kunya", pa.string(), nullable=True),
        pa.field("nisba", pa.string(), nullable=True),
        pa.field("laqab", pa.string(), nullable=True),
        pa.field("birth_year_ah", pa.int32(), nullable=True),
        pa.field("death_year_ah", pa.int32(), nullable=True),
        pa.field("birth_location", pa.string(), nullable=True),
        pa.field("death_location", pa.string(), nullable=True),
        pa.field("generation", pa.string(), nullable=True),
        pa.field("gender", pa.string(), nullable=True),
        pa.field("trustworthiness", pa.string(), nullable=True),
        pa.field("bio_text", pa.string(), nullable=True),
        pa.field("external_id", pa.string(), nullable=True),
    ]
)

COLLECTION_SCHEMA = pa.schema(
    [
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("name_ar", pa.string(), nullable=True),
        pa.field("name_en", pa.string(), nullable=False),
        pa.field("compiler_name", pa.string(), nullable=True),
        pa.field("compilation_year_ah", pa.int32(), nullable=True),
        pa.field("sect", pa.string(), nullable=False),
        pa.field("total_hadiths", pa.int32(), nullable=True),
        pa.field("source_corpus", pa.string(), nullable=False),
    ]
)

NETWORK_EDGE_SCHEMA = pa.schema(
    [
        pa.field("from_narrator_name", pa.string(), nullable=False),
        pa.field("to_narrator_name", pa.string(), nullable=False),
        pa.field("hadith_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("from_external_id", pa.string(), nullable=True),
        pa.field("to_external_id", pa.string(), nullable=True),
    ]
)

# One row per (narrator, distinct name-variant). Produced by rijal-style sources
# that carry alternate spellings of a narrator's name (e.g. Itqan ``namings`` —
# da#94). ``canonical_name_ar_normalized`` is the SAME normalized form the bio
# carries as ``name_ar_normalized`` and from which ``identity.make_canonical_id``
# mints the ``nar:`` id, so the resolve stage (``bio_promote``) can attach each
# variant to the right canonical Narrator without re-parsing the source. The
# variant is stored both raw (``alias``) and Arabic-normalized (``alias_normalized``)
# — the normalized form is what a future matcher keys on, the raw form preserves
# the source spelling for provenance/display.
NARRATOR_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("bio_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("canonical_name_ar_normalized", pa.string(), nullable=False),
        pa.field("alias", pa.string(), nullable=False),
        pa.field("alias_normalized", pa.string(), nullable=False),
    ]
)
