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
    DoubledCorpusPrefixError,
    bare_source_id,
    chain_node_id,
    collection_node_id,
    grading_node_id,
    hadith_node_id,
    is_double_prefixed,
    make_canonical_id,
    make_discriminated_canonical_id,
    narrator_node_id,
    validate_source_id,
)
from src.utils.arabic import normalize_arabic


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

    def test_doubled_corpus_raises_bare(self) -> None:
        # The main#139 streaming-path shape, pre-``hdt:``. No producer emits this
        # any more (da#353), so its arrival is a producer defect -> raise (da#355).
        with pytest.raises(DoubledCorpusPrefixError, match="doubled leading corpus"):
            hadith_node_id("sunnah:sunnah:bukhari:1:1:1")

    def test_doubled_corpus_raises_with_prefix(self) -> None:
        with pytest.raises(DoubledCorpusPrefixError, match="doubled leading corpus"):
            hadith_node_id("hdt:sunnah:sunnah:bukhari:1:1:1")

    def test_triple_corpus_raises(self) -> None:
        with pytest.raises(DoubledCorpusPrefixError):
            hadith_node_id("sunnah:sunnah:sunnah:bukhari:1")

    def test_corpus_named_collection_is_not_rewritten(self) -> None:
        """da#355: ``corpus:collection`` where collection == corpus is VALID grammar.

        The old ``_collapse_double_corpus`` could not tell this apart from a
        genuinely double-prefixed id and silently DROPPED the collection segment,
        rewriting a valid identifier. It must now raise rather than guess — the
        one thing it must never do is return ``hdt:lk:1``.
        """
        with pytest.raises(DoubledCorpusPrefixError):
            hadith_node_id("lk:lk:1")

    def test_chain_and_grading_ids_also_raise(self) -> None:
        # Every hadith-derived id routes through ``bare_source_id``.
        with pytest.raises(DoubledCorpusPrefixError):
            chain_node_id("lk:lk:1")
        with pytest.raises(DoubledCorpusPrefixError):
            grading_node_id("lk:lk:1")

    def test_error_is_a_valueerror(self) -> None:
        # Subclasses ValueError so existing broad handlers/tests keep working.
        assert issubclass(DoubledCorpusPrefixError, ValueError)

    def test_does_not_flag_legit_repeat_deeper(self) -> None:
        # A repeated *non-leading* segment that is not a doubled corpus is kept.
        assert hadith_node_id("lk:bukhari:1:1") == "hdt:lk:bukhari:1:1"

    def test_does_not_flag_non_corpus_lead(self) -> None:
        # If the leading segment is not a known corpus, never flag.
        assert hadith_node_id("h-1:h-1:x") == "hdt:h-1:h-1:x"


class TestBatchVsStreamingConverge:
    """The keystone guarantee: the same hadith -> exactly one node id.

    Both paths converge because both build the id through this module (ig#63 /
    ig#72 fixed the streaming producer). The convergence is NOT achieved by
    repairing a doubled corpus after the fact: since da#355 that shape RAISES, so
    a producer that re-injects the corpus fails loudly instead of being silently
    normalized onto the other path's id.
    """

    def test_two_paths_same_id(self) -> None:
        source_id = generate_source_id("sunnah", "bukhari", 1, 1, 1)
        # Both paths route through the ONE prefixing rule, from the same source_id.
        batch_id = hadith_node_id(source_id)
        streaming_id = hadith_node_id(source_id)
        assert batch_id == streaming_id == "hdt:sunnah:bukhari:1:1:1"
        # Idempotent: re-canonicalizing an already-canonical id is a no-op, so a
        # path that hands an already-prefixed id back in still converges.
        assert hadith_node_id(batch_id) == batch_id

    def test_streaming_double_prefix_bug_now_fails_loudly(self) -> None:
        # The main#139 shape: corpus re-injected before prefixing. Previously this
        # was silently collapsed onto the batch id; now it is a producer defect.
        source_id = generate_source_id("sunnah", "bukhari", 1, 1, 1)
        streaming_buggy = f"{SourceCorpus.SUNNAH.value}:{source_id}"
        with pytest.raises(DoubledCorpusPrefixError):
            hadith_node_id(streaming_buggy)

    def test_naive_concat_would_diverge(self) -> None:
        # Proves the helper is load-bearing: naive prefixing yields TWO ids...
        source_id = "sunnah:bukhari:1:1:1"
        naive_batch = f"hdt:{source_id}"
        naive_streaming = f"hdt:{SourceCorpus.SUNNAH.value}:{source_id}"
        assert naive_batch != naive_streaming
        # ...and the canonical helper no longer papers over the divergence: it
        # accepts the correct one and rejects the doubled one.
        assert hadith_node_id(naive_batch) == naive_batch
        with pytest.raises(DoubledCorpusPrefixError):
            hadith_node_id(naive_streaming)


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

    def test_bare_source_id_strips_hdt_prefix(self) -> None:
        # The ``hdt:`` strip is genuine, unambiguous prefix-stripping and stays
        # idempotent (da#355 keeps this; only the lossy collapse is removed).
        assert bare_source_id("hdt:sunnah:bukhari:1") == "sunnah:bukhari:1"
        assert bare_source_id("sunnah:bukhari:1") == "sunnah:bukhari:1"
        assert bare_source_id(bare_source_id("hdt:sunnah:bukhari:1")) == "sunnah:bukhari:1"

    def test_bare_source_id_rejects_doubled_corpus(self) -> None:
        with pytest.raises(DoubledCorpusPrefixError):
            bare_source_id("sunnah:sunnah:bukhari:1")


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


class TestMakeDiscriminatedCanonicalId:
    """The da#337 discriminated-id helper — sibling of ``make_canonical_id``."""

    def test_empty_discriminator_is_byte_identical_to_make_canonical_id(self) -> None:
        # The backward-compat contract: no discriminator == today's id, exactly.
        name = "محمد بن اسماعيل"
        assert make_discriminated_canonical_id(name) == make_canonical_id(name)
        assert make_discriminated_canonical_id(name, "") == make_canonical_id(name)

    def test_deterministic_on_repeat(self) -> None:
        name = "ابو عبد الله"
        assert make_discriminated_canonical_id(name, "d161") == make_discriminated_canonical_id(
            name, "d161"
        )
        # And the empty-discriminator path is equally stable across calls.
        assert make_discriminated_canonical_id(name) == make_discriminated_canonical_id(name)

    def test_distinct_discriminators_yield_distinct_ids(self) -> None:
        name = "سفيان"
        id_thawri = make_discriminated_canonical_id(name, "d161")
        id_ibn_uyayna = make_discriminated_canonical_id(name, "d198")
        assert id_thawri != id_ibn_uyayna
        # Both diverge from the undiscriminated collapse the split is fixing.
        assert id_thawri != make_canonical_id(name)
        assert id_ibn_uyayna != make_canonical_id(name)

    def test_same_name_and_discriminator_same_id(self) -> None:
        assert make_discriminated_canonical_id("سفيان", "kufa") == make_discriminated_canonical_id(
            "سفيان", "kufa"
        )

    def test_result_carries_narrator_prefix(self) -> None:
        assert make_discriminated_canonical_id("سفيان", "d161").startswith("nar:")

    def test_separator_disambiguates_name_discriminator_boundary(self) -> None:
        # The unit separator makes the name↔discriminator boundary unambiguous, so
        # two DIFFERENT splits of the same character run do NOT collide the way a
        # bare "name + discriminator" concatenation would ("ab"+"c" == "a"+"bc").
        assert make_discriminated_canonical_id("ab", "c") != make_discriminated_canonical_id(
            "a", "bc"
        )

    def test_discriminated_never_collides_with_a_real_normalized_name(self) -> None:
        # The separator is collision-safe against undiscriminated ids because a
        # normalized name can NEVER contain U+001F — normalize_arabic maps it to a
        # space (Python ``\s`` matches U+001F), so no real name reproduces a
        # discriminated key ``name + \x1f + disc``. Demonstrated on genuine
        # normalized inputs: a discriminated id differs from the plain id of its
        # base name AND from the plain id of the name+discriminator run.
        base = normalize_arabic("سفيان")
        disc = make_discriminated_canonical_id(base, "d161")
        assert disc != make_canonical_id(base)
        assert disc != make_canonical_id(normalize_arabic("سفيان d161"))

    def test_separator_edge_case_is_a_precondition_violation(self) -> None:
        # KNOWN, FLAGGED edge case (see PR notes): the ONLY way the empty-disc path
        # collides with a discriminated id is a synthetic "name" that literally
        # contains U+001F — which violates the pre-normalized-input precondition
        # (normalize_arabic strips it) and so cannot arise in production. Asserting
        # the honest behavior here: the empty-disc path is byte-identical to
        # make_canonical_id, unconditionally (the backward-compat contract).
        assert make_discriminated_canonical_id("a\x1fb", "") == make_canonical_id("a\x1fb")
