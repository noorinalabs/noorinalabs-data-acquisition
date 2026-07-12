"""Tests for the canonical entity-identity contract (da#82).

These lock the two historical identity hazards out:
1. double-prefix (``hdt:sunnah:sunnah:...``) — both ingest paths must converge
   on one id per hadith.
2. cross-source collision-safety — ids are corpus-namespaced and unique across
   all sources.
"""

from __future__ import annotations

import uuid

import pytest

from src.models.enums import SourceCorpus
from src.parse.base import generate_source_id
from src.parse.identity import (
    CANONICAL_NAMESPACE,
    SOURCE_CORPORA,
    DoubledCorpusPrefixError,
    bare_source_id,
    canonical_key,
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
from src.parse.name_quality import clean_narrator_name
from src.utils.arabic import canonical_surface, normalize_arabic


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

    def test_collection_equal_to_corpus_raises_at_the_producer(self) -> None:
        # da#355 producer gate. `collection == corpus` is forbidden BECAUSE it is
        # ambiguous: nothing in `sanadset:sanadset:1` distinguishes a doubled corpus
        # from a corpus-named collection. Catching it here costs a parse re-run;
        # catching it at load costs a partially-written graph after a 7.5h resolve.
        # This is exactly the da#353 CSV-stem fallback, made unproducible.
        with pytest.raises(DoubledCorpusPrefixError, match="doubled leading corpus"):
            generate_source_id("sanadset", "sanadset", 645817)

    def test_every_corpus_rejects_its_own_name_as_collection(self) -> None:
        for corpus in SOURCE_CORPORA:
            with pytest.raises(DoubledCorpusPrefixError):
                generate_source_id(corpus, corpus, 1)

    def test_empty_collection_raises(self) -> None:
        # An empty segment collapses `corpus::1` -> an id whose collection is
        # unrecoverable by the `<corpus>:<collection>` grammar the edge loader
        # parses (composition.is_canonical_hadith_id).
        with pytest.raises(ValueError, match="empty segment"):
            generate_source_id("lk", "", 1)

    def test_validate_source_id_has_a_production_caller(self) -> None:
        # Kwesi's TechDebt note on #359: validate_source_id had no non-test caller,
        # so the contract it encodes was never enforced anywhere it mattered.
        # generate_source_id is now that caller — this test pins the wiring, not
        # merely the behavior, so removing the call fails here and not silently.
        assert validate_source_id("sanadset:sanadset:1") != []
        with pytest.raises(DoubledCorpusPrefixError):
            generate_source_id("sanadset", "sanadset", 1)


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


def _pre_da371_id(name: str) -> str:
    """The id ``make_canonical_id`` produced BEFORE da#371 — ``canonical_surface`` only,
    no ``clean_narrator_name`` trim. Used as the frozen baseline for the negative
    control: an already-cleaned name must map to the SAME id under the new authority.
    """
    return narrator_node_id(str(uuid.uuid5(CANONICAL_NAMESPACE, canonical_surface(name))))


# Real narrator names that are ALREADY a fixed point of the declared writer
# ``clean_narrator_name(normalize_arabic(·))`` — the clean, folded majority (~99.5% of
# the corpus). Adding the da#371 trim to the id authority must be a NO-OP for these.
_ALREADY_CLEAN_NAMES = (
    "ابو هريره",
    "محمد بن اسماعيل البخاري",
    "عبد الله بن عمر",
    "مالك بن انس",
    "سفيان بن عيينه",
    "عائشه",
    "انس بن مالك",
    # زكريا (Zakariyya) — its trailing alif is part of the name, not an accusative
    # ending. It is deliberately EXCLUDED from the da#376 `_ACCUSATIVE_STEMS`
    # lexicon; including it here proves the trim path does NOT re-introduce the
    # naive "strip a trailing alif" corruption that would fold it to `زكري`.
    "زكريا",
)


class TestNameNormalizedDrift:
    """da#371 — one authority for the canonical normalized form.

    ``name_ar_normalized`` was *written* as ``clean_narrator_name(normalize_arabic(x))``
    but the id surface (``canonical_surface`` = normalize + da#376 fold) never applied
    the trim, so minting an id from a raw/uncleaned name disagreed with the
    cleaned-column path — two canonical ids for one name, and STUDIED_UNDER edges
    silently dropped when an endpoint resolved to the un-trimmed id. Option A folds the
    trim INTO :func:`make_canonical_id` (via :func:`canonical_key`) so all minting sites
    converge. da#356/da#376 had already closed the normalization/format-mark axis; these
    lock the residual clean axis and guard against a collateral whole-corpus re-key.
    """

    # ---- POSITIVE (red-first: these FAIL on pre-da#371 code) ----------------

    def test_clean_axis_raw_and_cleaned_name_mint_one_id(self) -> None:
        # The issue's own non-Arabic row: clean_narrator_name drops the '.', which
        # canonical_surface preserves. Pre-fix these minted two distinct nar: ids.
        assert make_canonical_id("Khabbab b. al-Aratt") == make_canonical_id("Khabbab b al-Aratt")

    def test_clean_axis_arabic_honorific_and_bare_mint_one_id(self) -> None:
        # An eulogy-suffixed spelling and the bare name are one narrator; the trim is
        # what makes them converge on a single id.
        with_eulogy = "خباب بن الأرت رضي الله عنه"
        bare = "خباب بن الأرت"
        assert make_canonical_id(with_eulogy) == make_canonical_id(bare)

    def test_studied_under_endpoint_attaches_to_the_promoted_node(self) -> None:
        # The load-bearing harm: _studied_under_endpoint resolved an edge endpoint via
        # make_canonical_id(normalize_arabic(name)) — un-trimmed — while the node it
        # must land on was minted by bio_promote from the CLEANED form. Pre-fix the ids
        # differ and the STUDIED_UNDER edge silently drops. After da#371 they agree, so
        # the edge attaches.
        from src.graph.load_edges import _studied_under_endpoint

        name = "خباب بن الأرت رضي الله عنه"
        promoted_node_id = make_canonical_id(clean_narrator_name(normalize_arabic(name)))
        endpoint = _studied_under_endpoint(name, None)
        assert endpoint == promoted_node_id

    def test_bio_promote_and_load_edges_rules_converge(self) -> None:
        # The two relocated minting sites the issue's mechanism moved to: bio_promote
        # cleans, load_edges did not. Both now route through the same authority.
        name = "'Abdur Rahman ( عبد الرحمن ( رضي الله عنه"
        bio_promote_rule = make_canonical_id(clean_narrator_name(normalize_arabic(name)))
        load_edges_rule = make_canonical_id(normalize_arabic(name))
        assert bio_promote_rule == load_edges_rule

    # ---- NEGATIVE CONTROL (must be GREEN before AND after — no collateral re-key) --

    def test_already_clean_names_are_not_rekeyed(self) -> None:
        # The failure mode to guard: silently re-keying the whole corpus. For a name
        # that is already a fixed point of the writer, the added trim must be a no-op,
        # so canonical_key == canonical_surface and the id is byte-identical to the
        # frozen pre-da#371 baseline.
        for name in _ALREADY_CLEAN_NAMES:
            assert canonical_key(name) == canonical_surface(name), name
            assert make_canonical_id(name) == _pre_da371_id(name), name

    def test_normalization_axis_still_one_id(self) -> None:
        # da#376 regression guard: the issue's ORIGINAL headline repro (format marks +
        # colon) must still collapse to one id under the composed authority.
        a = make_canonical_id("ابو داود ‏:‏ وكذلك")
        b = make_canonical_id(normalize_arabic("ابو داود : وكذلك"))
        assert a == b

    def test_da376_case_fold_preserved(self) -> None:
        # The kunya case endings still fold to one Abu Hurayra.
        assert make_canonical_id("ابي هريره") == make_canonical_id("ابو هريره")
        assert make_canonical_id("ابا هريره") == make_canonical_id("ابو هريره")

    def test_canonical_corruption_traps_never_false_merge(self) -> None:
        # On a wipe-coupled, wholesale-re-minted identity authority an عمرو→عمر
        # (ʿAmr→ʿUmar) false-merge is catastrophic and irreversible, so the guard
        # lives in the suite, not in a reviewer's read. clean_narrator_name is
        # token-level and the da#376 fold abstains by lexicon, so two distinct men
        # keep two ids: `عمرو` (ʿAmr) never collapses onto `عمر` (ʿUmar) — not the
        # base name, and not its accusative `عمرا` (the exact naive-alif-strip trap).
        assert make_canonical_id("عمرو") != make_canonical_id("عمر")
        assert make_canonical_id("عمرا") != make_canonical_id("عمر")
        # And Zakariyya's own name is not mistaken for a stem+alif inflection.
        assert make_canonical_id("زكريا") != make_canonical_id("زكري")

    # ---- Contract properties of the composed key ----------------------------

    def test_canonical_key_is_idempotent(self) -> None:
        for name in ("خباب بن الأرت رضي الله عنه", "Khabbab b. al-Aratt", *_ALREADY_CLEAN_NAMES):
            once = canonical_key(name)
            assert canonical_key(once) == once, name

    def test_pollution_reject_falls_back_not_collapse(self) -> None:
        # clean_narrator_name rejects a pure-honorific span (returns None). The key must
        # fall back to the pre-trim surface, NOT collapse every rejected name onto the
        # single empty-string id.
        assert make_canonical_id("رضي الله عنه") != make_canonical_id("")
        assert make_canonical_id("صلى الله عليه وسلم") != make_canonical_id("رضي الله عنه")

    def test_empty_name_is_stable(self) -> None:
        assert canonical_key("") == ""
        assert make_canonical_id("").startswith("nar:")

    def test_discriminated_name_half_is_also_trimmed(self) -> None:
        # The da#337 discriminated split must trim the NAME half through the same
        # authority, so a raw and a cleaned spelling under the same discriminator agree.
        raw = "خباب بن الأرت رضي الله عنه"
        cleaned = clean_narrator_name(normalize_arabic(raw))
        assert make_discriminated_canonical_id(raw, "d37") == make_discriminated_canonical_id(
            cleaned, "d37"
        )


class TestResidualNoiseFold:
    """da#434 — two residual letter-level noise classes at the mint surface.

    ``normalize_arabic`` folds ؤ ئ → ء but leaves two orthographic-noise classes that
    otherwise fragment one name across canonical ids: a dotless final ya (alif-maqsura
    ى) and the standalone hamza ء left over after the ؤئ→ء fold. :func:`canonical_surface`
    now folds ى→ي and drops ء so the surface forms of one name converge on one key. This
    is pure normalization — never an identity merge of distinct people (Kavitha's
    blast-radius: 192 clusters, ~1,689 mentions move, 0 distinct-person collisions).
    """

    # ---- POSITIVE: the two folds change the mint key --------------------------

    def test_alif_maqsura_folds_to_ya_in_the_key(self) -> None:
        # A final ى (alif-maqsura) folds to ي, so a dotless and a dotted spelling of
        # the same name mint one id. normalize_arabic leaves ى untouched, so pre-fix
        # these were two distinct nar: ids.
        assert canonical_surface("موسى") == canonical_surface("موسي")
        assert make_canonical_id("موسى") == make_canonical_id("موسي")

    def test_standalone_hamza_dropped_in_the_key(self) -> None:
        # The issue's own example: عاءشه and عائشة differ only by the standalone hamza
        # (after normalize_arabic folds ئ→ء and ة→ه). Dropping ء collapses them.
        assert canonical_surface("عاءشه") == canonical_surface("عائشة")
        assert make_canonical_id("عاءشه") == make_canonical_id("عائشة")

    # ---- CONTROL: genuinely different names still mint different ids ----------

    def test_distinct_names_sharing_only_the_folded_char_stay_distinct(self) -> None:
        # موسى (Musa) and عيسى (Isa) share only the trailing ى; the fold must not
        # collapse them. This is the guard that the fold is normalization, not merge.
        assert make_canonical_id("موسى") != make_canonical_id("عيسى")

    def test_fold_is_idempotent(self) -> None:
        # ي is not itself a fold trigger and no ء survives, so the folded surface is a
        # fixed point — an already-folded name does not move.
        for name in ("موسى", "عائشة", "يحيى", "عطاء"):
            once = canonical_surface(name)
            assert canonical_surface(once) == once, name
