"""Parse Thaqalayn raw JSON into staging Parquet.

The Thaqalayn REST API v2 was removed circa early 2026 when the site was rebuilt
with Next.js, so the **GitHub repo** (``MohammedArab1/ThaqalaynAPI``) is the sole
acquisition source. That repo ships the dataset SEVERAL times over — a ``V1`` and
a ``V2`` tree, each holding per-book files AND a giant ``allBooks.json`` aggregate
— plus build/index files (``package-lock.json``, ``tsconfig.json``,
``BookNames.json``, ``Ingredients/``). An earlier ``rglob("*.json")`` swept all of
them, producing ~4x-duplicated rows and config files parsed as "collections"
(da#175). We therefore read exactly ONE canonical tree: the per-book numeric files
under ``V2/ThaqalaynData/<n>.json`` (V2 is richer — it carries the
``thaqalaynSanad``/``thaqalaynMatn`` *English* isnad/matn split). :func:`book_json_files`
is the single source of truth for that selection, shared with the acquire adapter.

Each numeric file is a JSON array of hadith records with the real upstream schema::

    id, bookId, book, volume, category, categoryId, chapter, chapterInCategoryId,
    author, translator, englishText, arabicText, frenchText, URL,
    majlisiGrading, behbudiGrading, mohseniGrading, gradingsFull,
    thaqalaynSanad, thaqalaynMatn

Identity (da#175): ``source_id = thaqalayn:<bookId>:<id>`` — ``id`` is the in-book
ordinal and is REQUIRED for collision-safe ids (without it every row in a book
collapsed onto one node). ``collection_name`` is the ``bookId`` slug so the graph
APPEARS_IN loader (which keys collections as ``{corpus}:{collection_name}``) links
each hadith to its volume's Collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from src.parse.base import generate_source_id, safe_int, safe_str, write_parquet
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE_CORPUS = "thaqalayn"
SECT = "shia"

# The one canonical tree we read — see module docstring. Per-book files here are
# named ``<n>.json``; the non-book ``allBooks.json``/``BookNames.json`` siblings
# (and the ``Ingredients/`` subdir) are excluded by requiring a numeric stem.
_DATA_SUBDIR = ("github_clone", "V2", "ThaqalaynData")


def book_json_files(thaq_dir: Path) -> list[Path]:
    """Return the canonical per-book Thaqalayn JSON files under ``thaq_dir``.

    Scoped to ``V2/ThaqalaynData/<numeric>.json`` — the per-book data files —
    excluding the ``allBooks.json``/``BookNames.json`` aggregates, the
    ``Ingredients/`` subdir, and repo build/config files. Shared with
    :mod:`src.acquire.thaqalayn` so acquisition and parsing never drift (da#175).
    """
    data_dir = thaq_dir.joinpath(*_DATA_SUBDIR)
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.glob("*.json") if p.stem.isdigit())


def _primary_grade(rec: dict[str, Any]) -> str | None:
    """Pick a single representative grade from the per-grader fields.

    Thaqalayn carries up to three independent gradings; the staging schema (and
    the Grading node it feeds) holds one ``grade`` per hadith. We prefer
    al-Majlisi's grade (the most-cited classical grading), then Behbudi, then
    Mohseni. ``behdudiGrading`` is accepted as a V1 spelling fallback. Empty
    strings (the upstream "ungraded" sentinel) are treated as absent.
    """
    for key in ("majlisiGrading", "behbudiGrading", "behdudiGrading", "mohseniGrading"):
        val = safe_str(rec.get(key))
        if val:
            return val
    return None


def _hadith_to_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one upstream record to a staging row, or None if unusable."""
    book_id = safe_str(rec.get("bookId"))
    hadith_num = safe_int(rec.get("id"))
    if not book_id or hadith_num is None:
        return None

    source_id = generate_source_id(SOURCE_CORPUS, book_id, hadith_num)

    # Thaqalayn separates isnad from matn only on the ENGLISH translation
    # (``thaqalaynSanad``/``thaqalaynMatn`` are English); the Arabic (``arabicText``)
    # is stored whole, isnad + matn together. So the split feeds the ``*_en`` fields
    # and the Arabic text stays intact in both ``matn_ar`` and ``full_text_ar``
    # rather than mis-routing English prose into an Arabic column (da#175).
    arabic = safe_str(rec.get("arabicText"))
    english = safe_str(rec.get("englishText"))
    sanad_en = safe_str(rec.get("thaqalaynSanad"))
    matn_en = safe_str(rec.get("thaqalaynMatn")) or english

    return {
        "source_id": source_id,
        "source_corpus": SOURCE_CORPUS,
        "collection_name": book_id,
        "book_number": safe_int(rec.get("categoryId")),
        "chapter_number": safe_int(rec.get("chapterInCategoryId")),
        "hadith_number": hadith_num,
        "matn_ar": arabic,
        "matn_en": matn_en,
        "isnad_raw_ar": None,
        "isnad_raw_en": sanad_en,
        "full_text_ar": arabic,
        "full_text_en": english,
        "grade": _primary_grade(rec),
        "chapter_name_ar": None,
        "chapter_name_en": safe_str(rec.get("chapter")),
        "sect": SECT,
    }


def _collection_name_en(rec: dict[str, Any], book_id: str) -> str:
    """Human display name for a book's Collection node.

    Volumes of a multi-volume work share the same ``book`` title, so the volume
    is appended to keep each Collection distinct (and matching the per-volume
    ``bookId``). Falls back to the ``bookId`` slug if no title is present.
    """
    book = safe_str(rec.get("book"))
    if not book:
        return book_id
    volume = safe_int(rec.get("volume"))
    if volume and "volume" not in book.lower():
        return f"{book} (Volume {volume})"
    return book


def run(raw_dir: Path, staging_dir: Path) -> tuple[Path, Path]:
    """Parse Thaqalayn per-book JSONs into hadiths + collections Parquet files."""
    thaq_dir = raw_dir / "thaqalayn"
    json_files = book_json_files(thaq_dir)
    if not json_files:
        msg = f"No Thaqalayn per-book JSON files under {thaq_dir}/{'/'.join(_DATA_SUBDIR)}"
        raise FileNotFoundError(msg)

    logger.info("thaqalayn_parse_start", file_count=len(json_files))

    hadith_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    collections: dict[str, dict[str, Any]] = {}
    total_with_isnad = 0

    for fp in json_files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("thaqalayn_unexpected_shape", file=fp.name, type=type(data).__name__)
            continue

        for rec in data:
            if not isinstance(rec, dict):
                continue
            row = _hadith_to_row(rec)
            if row is None:
                continue
            # (bookId, id) is unique upstream, but guard against any dup so the
            # MERGE-by-id load never silently collapses rows (da#175).
            if row["source_id"] in seen_ids:
                continue
            seen_ids.add(row["source_id"])
            hadith_rows.append(row)
            # Thaqalayn's isnad/matn split is English-only (see _hadith_to_row).
            if row["isnad_raw_en"] is not None:
                total_with_isnad += 1

            book_id = row["collection_name"]
            coll = collections.get(book_id)
            if coll is None:
                collections[book_id] = {
                    "collection_id": generate_source_id(SOURCE_CORPUS, book_id),
                    "name_ar": None,
                    "name_en": _collection_name_en(rec, book_id),
                    "compiler_name": safe_str(rec.get("author")),
                    "compilation_year_ah": None,
                    "sect": SECT,
                    "total_hadiths": 1,
                    "source_corpus": SOURCE_CORPUS,
                }
            else:
                coll["total_hadiths"] += 1

    total_hadiths = len(hadith_rows)
    isnad_rate = (total_with_isnad / total_hadiths * 100) if total_hadiths else 0.0
    logger.info(
        "thaqalayn_isnad_rate",
        total=total_hadiths,
        with_isnad=total_with_isnad,
        rate_pct=round(isnad_rate, 1),
    )

    hadith_table = pa.table(
        {field.name: [r[field.name] for r in hadith_rows] for field in HADITH_SCHEMA},
        schema=HADITH_SCHEMA,
    )
    hadiths_path = write_parquet(
        hadith_table, Path(staging_dir) / "hadiths_thaqalayn.parquet", schema=HADITH_SCHEMA
    )

    coll_rows = list(collections.values())
    coll_table = pa.table(
        {field.name: [r[field.name] for r in coll_rows] for field in COLLECTION_SCHEMA},
        schema=COLLECTION_SCHEMA,
    )
    collections_path = write_parquet(
        coll_table, Path(staging_dir) / "collections_thaqalayn.parquet", schema=COLLECTION_SCHEMA
    )

    logger.info(
        "thaqalayn_parse_complete",
        hadiths=len(hadith_rows),
        collections=len(coll_rows),
    )
    return hadiths_path, collections_path
