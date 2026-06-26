"""Phase 2 entity resolution output schemas."""

from __future__ import annotations

import pyarrow as pa

__all__ = [
    "RESOLVE_SCHEMA_VERSION",
    "NARRATOR_MENTIONS_RESOLVED_SCHEMA",
    "NARRATORS_CANONICAL_SCHEMA",
    "AMBIGUOUS_NARRATORS_SCHEMA",
    "PARALLEL_LINKS_SCHEMA",
]

RESOLVE_SCHEMA_VERSION = 1

NARRATOR_MENTIONS_RESOLVED_SCHEMA = pa.schema(
    [
        pa.field("mention_id", pa.string(), nullable=False),
        pa.field("hadith_id", pa.string(), nullable=False),
        pa.field("source_corpus", pa.string(), nullable=False),
        pa.field("position_in_chain", pa.int32(), nullable=False),
        pa.field("name_raw", pa.string(), nullable=True),
        pa.field("name_normalized", pa.string(), nullable=True),
        pa.field("canonical_narrator_id", pa.string(), nullable=True),
        pa.field("transmission_method", pa.string(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
    ]
)

NARRATORS_CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("canonical_id", pa.string(), nullable=False),
        pa.field("name_ar", pa.string(), nullable=True),
        pa.field("name_en", pa.string(), nullable=True),
        pa.field("name_ar_normalized", pa.string(), nullable=True),
        pa.field("aliases", pa.list_(pa.string()), nullable=True),
        pa.field("birth_year_ah", pa.int32(), nullable=True),
        pa.field("death_year_ah", pa.int32(), nullable=True),
        # Date uncertainty bounds + precision — mirror Narrator (da#161) 1:1 so the
        # model and this canonical schema cannot drift. The *_year_ah fields above
        # stay the point estimate; these express the real-world uncertainty of
        # classical dating. All nullable (PyArrow fields carry no column default):
        # existing canonical producers leave them None via ``r.get(name)``, and the
        # populating stages (da#164/#165/#166) fill them downstream. The
        # ``*_date_precision`` columns carry the DatePrecision StrEnum value as a
        # string ("unknown" when unset, matching the model's default), alongside the
        # other string-enum columns (generation/gender/trustworthiness) below.
        pa.field("birth_year_ah_earliest", pa.int32(), nullable=True),
        pa.field("birth_year_ah_latest", pa.int32(), nullable=True),
        pa.field("birth_date_precision", pa.string(), nullable=True),
        pa.field("death_year_ah_earliest", pa.int32(), nullable=True),
        pa.field("death_year_ah_latest", pa.int32(), nullable=True),
        pa.field("death_date_precision", pa.string(), nullable=True),
        pa.field("generation", pa.string(), nullable=True),
        pa.field("gender", pa.string(), nullable=True),
        pa.field("trustworthiness", pa.string(), nullable=True),
        pa.field("source_ids", pa.list_(pa.string()), nullable=True),
        pa.field("external_id", pa.string(), nullable=True),
        pa.field("mention_count", pa.int32(), nullable=True),
        # Sect/corpus provenance on the canonical narrator (da#103, epic #81).
        # ``source_corpus`` is the scalar primary corpus (mirrors the Hadith /
        # Collection node property); ``source_corpora`` is the full multi-source
        # set; ``sect_affiliation`` is the resolution-time SectAffiliation derived
        # from those corpora (see src/resolve/sect_affiliation.py). All nullable so
        # other narrators_canonical producers (#109/#117) need not populate them.
        pa.field("source_corpus", pa.string(), nullable=True),
        pa.field("source_corpora", pa.list_(pa.string()), nullable=True),
        pa.field("sect_affiliation", pa.string(), nullable=True),
    ]
)

AMBIGUOUS_NARRATORS_SCHEMA = pa.schema(
    [
        pa.field("mention_id", pa.string(), nullable=False),
        pa.field("mention_text", pa.string(), nullable=False),
        pa.field("source_corpus", pa.string(), nullable=False),
        pa.field("candidate_1_id", pa.string(), nullable=True),
        pa.field("candidate_1_name", pa.string(), nullable=True),
        pa.field("candidate_1_score", pa.float32(), nullable=True),
        pa.field("candidate_1_stage", pa.string(), nullable=True),
        pa.field("candidate_2_id", pa.string(), nullable=True),
        pa.field("candidate_2_name", pa.string(), nullable=True),
        pa.field("candidate_2_score", pa.float32(), nullable=True),
        pa.field("candidate_2_stage", pa.string(), nullable=True),
        pa.field("candidate_3_id", pa.string(), nullable=True),
        pa.field("candidate_3_name", pa.string(), nullable=True),
        pa.field("candidate_3_score", pa.float32(), nullable=True),
        pa.field("candidate_3_stage", pa.string(), nullable=True),
    ]
)

PARALLEL_LINKS_SCHEMA = pa.schema(
    [
        pa.field("hadith_id_a", pa.string(), nullable=False),
        pa.field("hadith_id_b", pa.string(), nullable=False),
        pa.field("similarity_score", pa.float32(), nullable=False),
        pa.field("variant_type", pa.string(), nullable=False),
        pa.field("cross_sect", pa.bool_(), nullable=False),
    ]
)
