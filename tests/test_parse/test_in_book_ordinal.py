"""Tests for the per-source in-book ordinal derivation (da#229 / ADR-004 item #1).

The contract under test:

* a 1-based positional ordinal is derived ONLY for order-reliable sources,
* ONLY for a ``(collection, book)`` group whose ordinal is entirely null,
* never overwriting a source-provided ordinal, and
* leaving every other null an explicit null (the no-fabrication floor).
"""

from __future__ import annotations

from typing import Any

from src.models.enums import SourceCorpus
from src.parse.in_book_ordinal import (
    ORDER_RELIABLE_CORPORA,
    derive_in_book_ordinals,
    is_order_reliable,
)

RELIABLE = "sunnah"
UNRELIABLE = "sanadset"


def _rec(collection: str, book: int | None, number: int | None, **extra: Any) -> dict[str, Any]:
    return {"collection_name": collection, "book_number": book, "hadith_number": number, **extra}


class TestRegistry:
    def test_reliable_corpora_are_known_source_corpus_values(self) -> None:
        valid = {c.value for c in SourceCorpus}
        assert ORDER_RELIABLE_CORPORA <= valid

    def test_sunnah_is_reliable(self) -> None:
        assert is_order_reliable(RELIABLE)

    def test_sanadset_is_not_reliable(self) -> None:
        # A source not vetted for in-book row order must stay null-by-default.
        assert not is_order_reliable(UNRELIABLE)

    def test_unknown_corpus_is_not_reliable(self) -> None:
        assert not is_order_reliable("not-a-corpus")


class TestDeriveReliable:
    def test_fully_null_book_gets_sequential_ordinals(self) -> None:
        records = [
            _rec("musnad-ahmad", 1, None),
            _rec("musnad-ahmad", 1, None),
            _rec("musnad-ahmad", 1, None),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert derived == 3
        assert [r["hadith_number"] for r in records] == [1, 2, 3]

    def test_per_book_ordinal_restarts(self) -> None:
        records = [
            _rec("musnad-ahmad", 1, None),
            _rec("musnad-ahmad", 1, None),
            _rec("musnad-ahmad", 2, None),
            _rec("musnad-ahmad", 2, None),
            _rec("musnad-ahmad", 2, None),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert derived == 5
        assert [r["hadith_number"] for r in records] == [1, 2, 1, 2, 3]

    def test_interleaved_books_group_by_key_not_position(self) -> None:
        # Rows for the same book are numbered in first-seen order even when other
        # books' rows are interleaved between them.
        records = [
            _rec("c", 1, None),
            _rec("c", 2, None),
            _rec("c", 1, None),
            _rec("c", 2, None),
        ]
        derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert [r["hadith_number"] for r in records] == [1, 1, 2, 2]

    def test_separate_collections_do_not_share_a_sequence(self) -> None:
        records = [
            _rec("coll-a", 1, None),
            _rec("coll-b", 1, None),
        ]
        derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert [r["hadith_number"] for r in records] == [1, 1]

    def test_other_fields_are_preserved(self) -> None:
        records = [_rec("c", 1, None, matn_ar="نص", source_id="sunnah:c:1:0:0")]
        derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert records[0]["matn_ar"] == "نص"
        assert records[0]["source_id"] == "sunnah:c:1:0:0"
        assert records[0]["hadith_number"] == 1


class TestNoFabrication:
    def test_unreliable_source_leaves_nulls(self) -> None:
        records = [
            _rec("x", 1, None),
            _rec("x", 1, None),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=UNRELIABLE)
        assert derived == 0
        assert [r["hadith_number"] for r in records] == [None, None]

    def test_partially_numbered_book_is_left_untouched(self) -> None:
        # Ambiguous: we cannot know where the gap falls in the 1..N sequence, so
        # the null is left null rather than fabricated into a partial sequence.
        records = [
            _rec("c", 1, 5),
            _rec("c", 1, None),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert derived == 0
        assert [r["hadith_number"] for r in records] == [5, None]

    def test_existing_ordinals_are_not_overwritten(self) -> None:
        records = [
            _rec("c", 1, 10),
            _rec("c", 1, 11),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert derived == 0
        assert [r["hadith_number"] for r in records] == [10, 11]

    def test_mixed_books_derive_only_the_fully_null_one(self) -> None:
        records = [
            _rec("c", 1, None),  # book 1: fully null -> derived
            _rec("c", 1, None),
            _rec("c", 2, 7),  # book 2: numbered -> untouched
            _rec("c", 2, None),
        ]
        derived = derive_in_book_ordinals(records, source_corpus=RELIABLE)
        assert derived == 2
        assert [r["hadith_number"] for r in records] == [1, 2, 7, None]

    def test_empty_records_is_a_noop(self) -> None:
        assert derive_in_book_ordinals([], source_corpus=RELIABLE) == 0
