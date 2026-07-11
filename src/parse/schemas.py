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
    "EDGE_RELATION_STUDIED_UNDER",
    "EDGE_RELATION_TRANSMITTED_TO",
    "VALID_EDGE_RELATIONS",
]

# ``NETWORK_EDGE_SCHEMA.relation`` values (da#133). The graph loader routes a
# network edge to a Neo4j relationship type by this DECLARED relation rather than
# guessing it from the producing file's name: a studentship producer (muhaddithat,
# itqan) emits ``STUDIED_UNDER``; an isnad-transmission producer (mis) emits
# ``TRANSMITTED_TO``. There is deliberately NO default relation (da#157): a row
# that declares none is REFUSED by the loader, never silently routed onto the
# studentship allowlist. Every producer MUST set ``relation`` explicitly.
EDGE_RELATION_STUDIED_UNDER = "STUDIED_UNDER"
EDGE_RELATION_TRANSMITTED_TO = "TRANSMITTED_TO"
VALID_EDGE_RELATIONS = frozenset({EDGE_RELATION_STUDIED_UNDER, EDGE_RELATION_TRANSMITTED_TO})

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
        # Which isnad-chain WITHIN this hadith the mention belongs to (da#282).
        # ``position_in_chain`` alone is ambiguous once a single ``source_hadith_id``
        # carries more than one transmission sequence — e.g. ``lk`` emits both an
        # Arabic and an English isnad for one hadith, each numbered from position 0,
        # so grouping by hadith and sorting by position interleaves two chains and
        # fabricates cross-chain narrator adjacencies. ``chain_index`` is the
        # per-hadith chain ordinal (0 for a single-isnad hadith; 0/1 for lk's ar/en
        # lanes) that the graph loaders group on so a chain's edges are built WITHIN
        # the chain, never across a hadith's chains. It also matches
        # ``Chain.chain_index`` / ``identity.chain_node_id(hadith_id, chain_index)``.
        # Nullable: a producer that does not set it (or a legacy file) reads back as
        # None, which every consumer coalesces to 0 (single chain) — so single-isnad
        # data is unaffected.
        pa.field("chain_index", pa.int32(), nullable=True),
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
        # Date uncertainty bounds + precision parsed from the source's free-text
        # life-date notation (da#164). The *_year_ah point estimates above stay the
        # "best single value"; these express the attested span/openness ("بين 161
        # إلى 170", "بعد 130", "circa 150"). All nullable: a source silent on dates
        # leaves the three int bounds None and the precision string "unknown" (a
        # concrete value — never null — so no downstream DatePrecision(None) can
        # arise). The resolve stage (bio_promote / disambiguate, da#165) carries
        # these onto NARRATORS_CANONICAL_SCHEMA and reconciles across sources.
        pa.field("birth_year_ah_earliest", pa.int32(), nullable=True),
        pa.field("birth_year_ah_latest", pa.int32(), nullable=True),
        pa.field("birth_date_precision", pa.string(), nullable=True),
        pa.field("death_year_ah_earliest", pa.int32(), nullable=True),
        pa.field("death_year_ah_latest", pa.int32(), nullable=True),
        pa.field("death_date_precision", pa.string(), nullable=True),
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
        # Canonical expected hadith count from sourced metadata (da#230) — the
        # authoritative reference total used by collection_coverage.cypher,
        # distinct from ``total_hadiths`` (the source's own reported count).
        # Filled by :func:`src.parse.collection_metadata.apply_collection_metadata`;
        # null when no sourced value exists (never guessed).
        pa.field("expected_count", pa.int32(), nullable=True),
        pa.field("source_corpus", pa.string(), nullable=False),
    ]
)

# ``relation`` declares which graph relationship the row's narrator pair forms
# (:data:`EDGE_RELATION_STUDIED_UNDER` vs :data:`EDGE_RELATION_TRANSMITTED_TO`) so
# the loader keys on the data, not on the filename (da#133). Every producer MUST
# set it: the loader REFUSES a row that declares no relation rather than defaulting
# it to STUDIED_UNDER (da#157). The field stays physically nullable so a malformed
# (relation-less) parquet can still be read — and then loudly rejected — rather
# than failing opaquely at read time.
NETWORK_EDGE_SCHEMA = pa.schema(
    [
        pa.field("from_narrator_name", pa.string(), nullable=False),
        pa.field("to_narrator_name", pa.string(), nullable=False),
        pa.field("hadith_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("from_external_id", pa.string(), nullable=True),
        pa.field("to_external_id", pa.string(), nullable=True),
        pa.field("relation", pa.string(), nullable=True),
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
