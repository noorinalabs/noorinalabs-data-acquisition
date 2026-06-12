"""Tests for the canonical entity-identity contract (da#82).

These lock the two historical identity hazards out:
1. double-prefix (``hdt:sunnah:sunnah:...``) — both ingest paths must converge
   on one id per hadith.
2. cross-source collision-safety — ids are corpus-namespaced and unique across
   all sources.
"""

from __future__ import annotations

import pytest

from src.models.enums import SourceCorpus
from src.parse.base import generate_source_id
from src.parse.identity import (
    SOURCE_CORPORA,
    bare_source_id,
    chain_node_id,
    collection_node_id,
    grading_node_id,
    hadith_node_id,
    is_double_prefixed,
    narrator_node_id,
    validate_source_id,
)


class TestSourceCorpora:
    def test_corpora_derived_from_enum(self) -> None:
        # Single source of truth — never drifts from the enum.
        assert SOURCE_CORPORA == {c.value for c in SourceCorpus}

    def test_known_corpora_present(self) -> None:
        for corpus in ("lk", "sanadset", "thaqalayn", "sunnah", "fawaz"):
            assert corpus in SOURCE_CORPORA


class TestGenerateSourceId:
    def test_basic_format(self) -> None:
        assert generate_source_id("lk", "bukhari", 1, 1) == "lk:bukhari:1:1"

    def test_collection_only(self) -> None:
        assert generate_source_id("sunnah", "bukhari") == "sunnah:bukhari"

    def test_unknown_corpus_raises(self) -> None:
        # An unknown corpus is a collision hazard — fail fast.
        with pytest.raises(ValueError, match="unknown source corpus"):
            generate_source_id("notacorpus", "bukhari", 1)

    def test_every_enum_corpus_accepted(self) -> None:
        for corpus in SOURCE_CORPORA:
            assert generate_source_id(corpus, "coll", 1).startswith(f"{corpus}:")


class TestHadithNodeId:
    def test_adds_single_prefix(self) -> None:
        assert hadith_node_id("sunnah:bukhari:1:1:1") == "hdt:sunnah:bukhari:1:1:1"

    def test_idempotent_on_hdt_prefix(self) -> None:
        once = hadith_node_id("sunnah:bukhari:1:1:1")
        assert hadith_node_id(once) == once

    def test_collapses_doubled_corpus_bare(self) -> None:
        # The main#139 streaming-path shape, pre-``hdt:``.
        assert hadith_node_id("sunnah:sunnah:bukhari:1:1:1") == "hdt:sunnah:bukhari:1:1:1"

    def test_collapses_doubled_corpus_with_prefix(self) -> None:
        assert hadith_node_id("hdt:sunnah:sunnah:bukhari:1:1:1") == "hdt:sunnah:bukhari:1:1:1"

    def test_collapses_triple_corpus(self) -> None:
        assert hadith_node_id("sunnah:sunnah:sunnah:bukhari:1") == "hdt:sunnah:bukhari:1"

    def test_does_not_collapse_legit_repeat_deeper(self) -> None:
        # A repeated *non-leading* segment that is not a doubled corpus is kept.
        assert hadith_node_id("lk:bukhari:1:1") == "hdt:lk:bukhari:1:1"

    def test_does_not_collapse_non_corpus_lead(self) -> None:
        # If the leading segment is not a known corpus, never collapse.
        assert hadith_node_id("h-1:h-1:x") == "hdt:h-1:h-1:x"


class TestBatchVsStreamingConverge:
    """The keystone guarantee: the same hadith -> exactly one node id."""

    def test_two_paths_same_id(self) -> None:
        source_id = generate_source_id("sunnah", "bukhari", 1, 1, 1)
        # Batch loader shape: hdt: + source_id (corpus once).
        batch_id = hadith_node_id(source_id)
        # Streaming/normalize bug shape: corpus re-injected before prefixing.
        streaming_buggy = f"{SourceCorpus.SUNNAH.value}:{source_id}"
        streaming_id = hadith_node_id(streaming_buggy)
        assert batch_id == streaming_id

    def test_naive_concat_would_diverge(self) -> None:
        # Proves the helper is load-bearing: naive prefixing yields TWO ids.
        source_id = "sunnah:bukhari:1:1:1"
        naive_batch = f"hdt:{source_id}"
        naive_streaming = f"hdt:{SourceCorpus.SUNNAH.value}:{source_id}"
        assert naive_batch != naive_streaming
        # ...but routed through the canonical helper they converge.
        assert hadith_node_id(naive_batch) == hadith_node_id(naive_streaming)


class TestOtherNodeIds:
    def test_collection_node_id(self) -> None:
        assert collection_node_id("sunnah:bukhari") == "col:sunnah:bukhari"
        assert collection_node_id("col:sunnah:bukhari") == "col:sunnah:bukhari"

    def test_narrator_node_id(self) -> None:
        assert narrator_node_id("abu-hurayra-001") == "nar:abu-hurayra-001"
        assert narrator_node_id("nar:abu-hurayra-001") == "nar:abu-hurayra-001"

    def test_chain_node_id(self) -> None:
        assert chain_node_id("sunnah:bukhari:1:1:1", 0) == "chn:sunnah:bukhari:1:1:1-0"
        assert chain_node_id("hdt:sunnah:bukhari:1:1:1", 2) == "chn:sunnah:bukhari:1:1:1-2"
        assert chain_node_id("chn:already-0") == "chn:already-0"

    def test_grading_node_id(self) -> None:
        assert grading_node_id("lk:bukhari:1:1") == "grd:lk:bukhari:1:1"
        # Grading and Hadith id derive from the SAME bare source_id.
        assert grading_node_id("hdt:lk:bukhari:1:1") == "grd:lk:bukhari:1:1"

    def test_bare_source_id(self) -> None:
        assert bare_source_id("hdt:sunnah:bukhari:1") == "sunnah:bukhari:1"
        assert bare_source_id("sunnah:sunnah:bukhari:1") == "sunnah:bukhari:1"


class TestCrossSourceCollisionSafety:
    def test_same_coords_distinct_per_corpus(self) -> None:
        # Same collection/book/chapter/num in every corpus -> distinct ids.
        ids = {hadith_node_id(generate_source_id(c, "bukhari", 1, 1, 1)) for c in SOURCE_CORPORA}
        assert len(ids) == len(SOURCE_CORPORA)


class TestIsDoublePrefixed:
    def test_detects_doubled(self) -> None:
        assert is_double_prefixed("hdt:sunnah:sunnah:bukhari:1")
        assert is_double_prefixed("sunnah:sunnah:bukhari:1")

    def test_clean_is_not_flagged(self) -> None:
        assert not is_double_prefixed("hdt:sunnah:bukhari:1")
        assert not is_double_prefixed("hdt:lk:bukhari:1:1")


class TestValidateSourceId:
    def test_valid(self) -> None:
        assert validate_source_id("sunnah:bukhari:1:1:1") == []

    def test_empty(self) -> None:
        assert validate_source_id("") == ["source_id is empty"]

    def test_unknown_corpus(self) -> None:
        problems = validate_source_id("xyz:bukhari:1")
        assert any("not a known corpus" in p for p in problems)

    def test_no_collection_segment(self) -> None:
        problems = validate_source_id("sunnah")
        assert any("no collection segment" in p for p in problems)

    def test_doubled_corpus_flagged(self) -> None:
        problems = validate_source_id("sunnah:sunnah:bukhari:1")
        assert any("doubled leading corpus" in p for p in problems)

    def test_empty_segment_flagged(self) -> None:
        problems = validate_source_id("sunnah::1")
        assert any("empty segment" in p for p in problems)
