"""Curated, *sourced* supplementary metadata for hadith Collections.

Some upstream sources emit ``collections_*`` rows with a blank/null ``name_ar``
or carry no authoritative hadith count (e.g. ``riyadussalihin`` from the
Sunnah.com API / scraper). Per ADR-004 item #5 (da#153) the load layer must NOT
fabricate an Arabic name or guess a count — it leaves the field null rather than
inventing one.

This module is the *sourced* fill for those gaps (da#230). It is a hand-curated
table where **every value carries an explicit provenance citation** (`source`).
Parsers call :func:`apply_collection_metadata` on each emitted collection row to:

* fill ``name_ar`` **only when it is currently blank/None** and a sourced Arabic
  title exists — a value already supplied by the source is never overridden, and
* populate ``expected_count`` — the *canonical* number of hadiths the collection
  is expected to contain (read by ``queries/validation/collection_coverage.cypher``
  to measure load completeness). This is distinct from ``total_hadiths``, which
  is whatever the source file happened to report.

Adding a collection here is a deliberate, auditable act: supply the Arabic title,
the expected count, and a citation for both. Unknown slugs are left untouched
(``expected_count`` stays ``None``) — absence of a curated entry is *not* a guess.

``expected_count`` provenance for ``riyadussalihin`` (1896) is corroborated by
this repo's own project memory ``project_sunnah_scraper_truncation`` ("riyad lost
679/1896"); the others trace to the sunnah.com collection reference ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CollectionMetadata",
    "COLLECTION_METADATA",
    "lookup",
    "apply_collection_metadata",
]


@dataclass(frozen=True)
class CollectionMetadata:
    """A sourced metadata entry for one collection slug.

    ``source`` is the provenance citation — where ``name_ar`` and
    ``expected_count`` came from — so each enriched value is auditable rather
    than an unsourced guess.
    """

    name_ar: str | None
    expected_count: int | None
    source: str


# Keyed by collection *slug* — the trailing ``:``-segment of ``collection_id``
# (e.g. ``"sunnah:riyadussalihin"`` -> ``"riyadussalihin"``), lower-cased.
COLLECTION_METADATA: dict[str, CollectionMetadata] = {
    "riyadussalihin": CollectionMetadata(
        name_ar="رياض الصالحين",
        expected_count=1896,
        source="https://sunnah.com/riyadussalihin",
    ),
    "nawawi40": CollectionMetadata(
        name_ar="الأربعون النووية",
        expected_count=42,
        source="https://sunnah.com/nawawi40",
    ),
    "qudsi40": CollectionMetadata(
        name_ar="الأربعون القدسية",
        expected_count=40,
        source="https://sunnah.com/qudsi40",
    ),
}


def _slug_of(row: dict[str, Any]) -> str:
    """Derive the lower-cased collection slug from a COLLECTION_SCHEMA row."""
    cid = str(row.get("collection_id") or "")
    if ":" in cid:
        slug = cid.rsplit(":", 1)[-1]
    else:
        slug = cid or str(row.get("name_en") or "")
    return slug.strip().lower()


def lookup(slug: str) -> CollectionMetadata | None:
    """Return the sourced metadata for a collection slug, or ``None``."""
    return COLLECTION_METADATA.get(slug.strip().lower())


def apply_collection_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a collection row with sourced fields filled.

    Always ensures the ``expected_count`` key is present (``None`` when the slug
    is unknown) so the row satisfies the extended ``COLLECTION_SCHEMA``. Fills
    ``name_ar`` and ``expected_count`` from :data:`COLLECTION_METADATA` only when
    the row's own value is missing — source-supplied values are preserved.
    """
    enriched = dict(row)
    enriched.setdefault("expected_count", None)

    meta = COLLECTION_METADATA.get(_slug_of(enriched))
    if meta is None:
        return enriched

    current_name = str(enriched.get("name_ar") or "").strip()
    if not current_name and meta.name_ar:
        enriched["name_ar"] = meta.name_ar

    if enriched.get("expected_count") is None and meta.expected_count is not None:
        enriched["expected_count"] = meta.expected_count

    return enriched
