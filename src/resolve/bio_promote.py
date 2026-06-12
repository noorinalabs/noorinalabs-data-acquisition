"""Promote narrator BIO rows directly to canonical Narrator records.

The mention-driven disambiguator (``src/resolve/disambiguate.py``) only emits a
canonical narrator when an isnad *mention* resolves to a bio candidate — with
zero mentions it returns nothing. A **profile-only** source (a rijal database
such as Itqan that contributes biographies but no isnad chains in its narrators
slice) would therefore produce **no** Narrator nodes at all.

This module bridges that gap. Each bio becomes — or updates — a canonical
narrator keyed by the SAME ``nar:<uuid5(normalized-name)>`` identity the
disambiguator uses (``src.parse.identity.make_canonical_id``), so the two paths
converge: a later mention-driven run dedups onto the very same node, and the
graph loader (``load_nodes._load_narrators``) ingests the output unchanged.

It is **merge-safe**: when ``narrators_canonical.parquet`` already exists (e.g.
from a disambiguator run), it is loaded and unioned by ``canonical_id`` rather
than overwritten, so the bio-direct and mention-driven outputs compose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.base import safe_str, write_parquet
from src.parse.identity import make_canonical_id
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["promote_bios_to_canonical"]

# Scalar bio fields back-filled onto an existing canonical record when missing.
_BACKFILL_FIELDS = (
    "name_en",
    "birth_year_ah",
    "death_year_ah",
    "generation",
    "gender",
    "trustworthiness",
    "external_id",
)


def _load_existing_canonical(path: Path) -> dict[str, dict[str, Any]]:
    """Load an existing narrators_canonical.parquet into a canonical_id -> row map."""
    if not path.exists():
        return {}
    table = pq.read_table(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        cid = row.get("canonical_id")
        if cid:
            row.setdefault("source_ids", [])
            row.setdefault("aliases", [])
            rows[cid] = row
    return rows


def promote_bios_to_canonical(
    staging_dir: Path,
    output_dir: Path,
    *,
    sources: set[str] | None = None,
) -> Path | None:
    """Build/extend ``narrators_canonical.parquet`` directly from narrator bios.

    Args:
        staging_dir: directory holding ``narrators_bio_*.parquet`` files.
        output_dir: directory the canonical Parquet is written to (and read from
            when merging into a pre-existing file). This is the resolve stage's
            ``output_dir`` — the same curated location the graph loader reads
            ``narrators_canonical.parquet`` from (the CLI maps it to
            ``DATA_CURATED_DIR``; see ``load_nodes`` artifact-location contract,
            da#112). Point it at the *curated* dir, never staging.
        sources: if given, only bios whose ``source`` column is in this set are
            promoted (e.g. ``{"itqan"}``).

    Returns the canonical Parquet path, or ``None`` when there are no bios to
    promote (no file written).
    """
    bio_files = sorted(staging_dir.glob("narrators_bio_*.parquet"))
    if not bio_files:
        logger.warning("bio_promote_no_bio_files", staging_dir=str(staging_dir))
        return None

    canonical_path = output_dir / "narrators_canonical.parquet"
    canonical_map = _load_existing_canonical(canonical_path)
    pre_existing = len(canonical_map)

    promoted = 0
    skipped_no_name = 0
    skipped_source = 0

    for bf in bio_files:
        table = pq.read_table(bf)
        for row in table.to_pylist():
            source = safe_str(row.get("source"))
            if sources is not None and source not in sources:
                skipped_source += 1
                continue

            name_ar = safe_str(row.get("name_ar"))
            norm = safe_str(row.get("name_ar_normalized")) or (
                normalize_arabic(name_ar) if name_ar else None
            )
            if not norm:
                skipped_no_name += 1
                continue

            bio_id = safe_str(row.get("bio_id"))
            cid = make_canonical_id(norm)

            if cid not in canonical_map:
                canonical_map[cid] = {
                    "canonical_id": cid,
                    "name_ar": name_ar,
                    "name_en": safe_str(row.get("name_en")),
                    "name_ar_normalized": norm,
                    "aliases": [],
                    "birth_year_ah": row.get("birth_year_ah"),
                    "death_year_ah": row.get("death_year_ah"),
                    "generation": safe_str(row.get("generation")),
                    "gender": safe_str(row.get("gender")),
                    "trustworthiness": safe_str(row.get("trustworthiness")),
                    "source_ids": [bio_id] if bio_id else [],
                    "external_id": safe_str(row.get("external_id")),
                    # mention_count stays 0: a bio is provenance, not a chain hit.
                    "mention_count": 0,
                }
            else:
                rec = canonical_map[cid]
                src_ids = rec.setdefault("source_ids", [])
                if bio_id and bio_id not in src_ids:
                    src_ids.append(bio_id)
                # Back-fill only fields the existing record is missing. Keep the
                # raw Parquet value (already a clean str / int) — do NOT coerce,
                # or an int column like death_year_ah gets stringified.
                for field_name in _BACKFILL_FIELDS:
                    if rec.get(field_name) in (None, "") and row.get(field_name) not in (None, ""):
                        rec[field_name] = row.get(field_name)
            promoted += 1

    table_rows = list(canonical_map.values())
    arrays = {f.name: [r.get(f.name) for r in table_rows] for f in NARRATORS_CANONICAL_SCHEMA}
    out_table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_parquet(out_table, canonical_path, schema=NARRATORS_CANONICAL_SCHEMA)

    logger.info(
        "bio_promote_complete",
        bio_files=len(bio_files),
        promoted=promoted,
        canonical_total=len(canonical_map),
        pre_existing=pre_existing,
        skipped_no_name=skipped_no_name,
        skipped_source=skipped_source,
    )
    return canonical_path
