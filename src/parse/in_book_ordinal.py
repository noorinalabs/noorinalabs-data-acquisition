"""Derive the per-source in-book ordinal (``hadith_number``) where reliable.

da#229 / ADR-004 item #1. The staging ``hadith_number`` column is the IN-BOOK
ordinal that flows to ``APPEARS_IN.hadith_number_in_book`` (da#77). For some
sources it is legitimately null because the source only carries a
collection-wide reference number — the explicit-null / no-fabrication contract
(ADR-004 item #1) is the floor and must NOT be violated where the ordinal is
genuinely unknown.

Where a source's PARSE-LANE ROW ORDER faithfully reproduces the in-book
sequence (the acquirer/parser walks the source in reading order within a book),
the position of a hadith on its book page *is* its in-book ordinal, so deriving
a 1-based positional ordinal is a sound recovery of a real value rather than a
fabrication. This module encodes *which* sources are trustworthy for that
derivation (:data:`ORDER_RELIABLE_CORPORA`) and applies it conservatively,
leaving the explicit null everywhere it cannot be derived honestly.

Conservatism rules (so this never fabricates):

* **Source gate.** Only corpora in :data:`ORDER_RELIABLE_CORPORA` are derived;
  for any other source every null stays null (the ADR-004 floor).
* **Whole-book gate.** Within a reliable source, an ordinal is derived only for
  a ``(collection, book)`` group whose ordinal is *entirely* unknown. A
  partially numbered book is ambiguous — we cannot know where the missing rows
  fall in the ``1..N`` sequence — so its nulls are left null rather than risk a
  derived value that collides with the source's own numbering.
* **No overwrite.** A record that already carries a non-null ordinal (a
  source-provided value) is never touched.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ORDER_RELIABLE_CORPORA",
    "is_order_reliable",
    "derive_in_book_ordinals",
]

# Source corpora (``SourceCorpus`` values) whose parse-lane row order faithfully
# reproduces the in-book sequence, so a 1-based positional ordinal per
# ``(collection, book)`` is a sound derivation rather than a fabrication.
#
# Inclusion criterion (deliberately strict): the acquirer/parser must walk the
# source in reading order within a book, so a row's position == its in-book
# sequence number.
#   * ``sunnah`` — the sunnah.com scraper (``src/acquire/sunnah_scraper.py``
#     ``_scrape_book_page``) walks each book page's hadith containers in DOM /
#     reading order, and the parser (``src/parse/sunnah_scraped.py``) preserves
#     that order, so a hadith's position on its book page IS its in-book ordinal.
#     The scraper already extracts an explicit in-book ordinal where the page
#     labels one; this derivation only recovers the ordinal for a book page that
#     carries no per-hadith label at all.
#
# Everything NOT listed is treated as order-UNRELIABLE: a missing ordinal stays
# an explicit null (the ADR-004 no-fabrication floor). Notable exclusions and
# why they stay null:
#   * sources that expose only a collection-wide reference number — the in-book
#     ordinal is genuinely unknown, not merely unlabelled;
#   * CSV/JSON dumps with no guaranteed in-book row ordering;
#   * rijal/bio sources, which have no in-book hadith sequence at all.
ORDER_RELIABLE_CORPORA: frozenset[str] = frozenset({"sunnah"})


def is_order_reliable(source_corpus: str) -> bool:
    """Whether *source_corpus*'s row order reflects the in-book sequence.

    A ``True`` result means a positional ordinal derived from row order is a
    faithful recovery of the in-book number; ``False`` means the ordinal must be
    left null where the source does not provide it (ADR-004 item #1).
    """
    return source_corpus in ORDER_RELIABLE_CORPORA


def derive_in_book_ordinals(
    records: list[dict[str, Any]],
    *,
    source_corpus: str,
    collection_key: str = "collection_name",
    book_key: str = "book_number",
    ordinal_key: str = "hadith_number",
) -> int:
    """Fill the in-book ordinal from row order for an order-reliable source.

    *records* is mutated in place: for an order-reliable *source_corpus*, each
    ``(collection, book)`` group whose ordinal is entirely null is assigned a
    1-based positional ordinal in first-seen row order. For an unreliable source,
    or for a book that is already wholly/partially numbered, nothing is changed —
    the explicit-null floor (ADR-004 item #1) is preserved.

    Returns the number of records whose ordinal was derived (``0`` when the
    source is unreliable or no group was eligible).
    """
    if not is_order_reliable(source_corpus):
        # Null-when-unreliable: the explicit-null floor (ADR-004 item #1) stands.
        # We do NOT invent an ordinal for a source whose ordering we cannot trust.
        return 0

    # Group record indices by (collection, book), preserving first-seen order so
    # the derived ordinal follows the source's reading order.
    groups: OrderedDict[tuple[Any, Any], list[int]] = OrderedDict()
    for idx, rec in enumerate(records):
        key = (rec.get(collection_key), rec.get(book_key))
        groups.setdefault(key, []).append(idx)

    derived = 0
    filled_books = 0
    for idxs in groups.values():
        # Whole-book gate: derive only for a book whose ordinal is ENTIRELY
        # unknown. A book that already carries any ordinal is ambiguous (we
        # cannot know where the gaps fall in the 1..N sequence), so leave its
        # nulls null rather than fabricate a colliding sequence.
        if any(records[i].get(ordinal_key) is not None for i in idxs):
            continue
        for position, i in enumerate(idxs, start=1):
            records[i][ordinal_key] = position
            derived += 1
        filled_books += 1

    if derived:
        logger.info(
            "in_book_ordinal_derived",
            source_corpus=source_corpus,
            derived=derived,
            books=filled_books,
        )
    return derived
