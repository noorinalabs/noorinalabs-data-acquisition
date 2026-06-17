"""Parse *Tahdhib al-Ahkam* + *al-Istibsar* (ThaqalaynData clone) into staging Parquet.

Source
------
``narmafraz/ThaqalaynData`` (CC0 1.0) is the original thaqalayn.net data backend.
Unlike ``MohammedArab1/ThaqalaynAPI`` (the ``thaqalayn`` adapter's upstream, a
website scrape that omits these two Books), it carries all four Shia Books with
the real Arabic source text. da#182 sources the two missing Books — Tahdhib
al-Ahkam and al-Istibsar — from here, completing the Four Books.

Markup contract (observed in the clone, verified over all 17,421 verse files)
----------------------------------------------------------------------------
Each Book is a tree of JSON files under ``books/<slug>/`` addressed by a
``part:chapter:hadith`` path. There are three node ``kind``\\ s::

    chapter_list   books/<slug>.json / books/<slug>/<part>.json    (TOC, parts)
    verse_list     books/<slug>/<part>/<chapter>.json              (chapter, refs)
    verse_detail   books/<slug>/<part>/<chapter>/<hadith>.json     (ONE hadith)

We walk by ``kind == "verse_detail"`` (NOT by hard-coded paths — Tahdhib's and
al-Istibsar's nesting differ) and read each hadith from its ``data.verse``:

* ``verse.path``            ``/books/al-istibsar:1:1:1`` -> ``(part, chapter, hadith)``.
* ``verse.narrator_chain``  the leading sanad run (real Arabic), if present.
* ``verse.text``            the rest of the hadith body (real Arabic).
* ``data.chapter_title.ar`` the chapter (bab) heading.

Real Arabic only (no fixture-masking — main#671)
------------------------------------------------
The dataset's non-Arabic translations are **AI-generated** (``verse.ai``,
``model=pipeline_v4``) — they are NOT a human translation of the source, so
da#182 deliberately omits them: the English and grade staging columns stay null.
The genuine Arabic goes to ``matn_ar`` (the field the graph loader surfaces on the
Hadith node — see :func:`_hadith_row`), with the clean ``narrator_chain`` lead-in
also exposed as ``isnad_raw_ar``. The real-upstream fixture test
(``tests/test_parse/test_thaqalayn_data_parser.py``) asserts non-empty Arabic so a
schema drift that silently zeroed the text would fail CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from src.parse.base import generate_source_id, safe_str, write_parquet
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE_CORPUS = "thaqalayn_data"
SECT = "shia"
COMPILER_NAME = "Muhammad ibn al-Hasan al-Tusi"

# The two Books this parser emits. The key is BOTH the collection slug (the
# ``source_id`` / ``collection_id`` segment) AND the hadith row's
# ``collection_name`` — they MUST be the same string so the APPEARS_IN loader's
# ``{corpus}:{collection_name}`` key matches the Collection node's
# ``{corpus}:{slug}`` id (see src/graph/load_edges.py). Slugs are hyphenated and
# never equal the corpus, so an id is never double-prefixed (src/parse/identity.py).
BOOK_META: dict[str, dict[str, str]] = {
    "tahdhib-al-ahkam": {"name_en": "Tahdhib al-Ahkam", "name_ar": "تهذيب الأحكام"},
    "al-istibsar": {"name_en": "al-Istibsar", "name_ar": "الاستبصار"},
}


def _clone_books_root(raw_dir: Path) -> Path:
    """Path to the cloned ``books/`` directory (see src/acquire/thaqalayn_data.py)."""
    return raw_dir / "thaqalayn_data" / "clone" / "books"


def _verse_path_parts(path: str, slug: str) -> tuple[int, int, int] | None:
    """Parse ``/books/<slug>:<part>:<chapter>:<hadith>`` into ``(part, chapter, hadith)``.

    The slug carries no ``:`` so it is one segment; returns ``None`` for any path
    that does not match the expected four-segment grammar.
    """
    prefix = "/books/"
    if not path.startswith(prefix):
        return None
    segments = path[len(prefix) :].split(":")
    if len(segments) != 4 or segments[0] != slug:
        return None
    try:
        return int(segments[1]), int(segments[2]), int(segments[3])
    except ValueError:
        return None


def _chain_arabic(verse: dict[str, Any]) -> str | None:
    """The genuine Arabic sanad lead-in from ``narrator_chain`` (~98% of verses)."""
    chain = verse.get("narrator_chain") or {}
    parts = [
        part["text"].strip()
        for part in chain.get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
    ]
    return " ".join(parts).strip() or None


def _body_arabic(verse: dict[str, Any]) -> str | None:
    """The genuine Arabic body from ``verse.text`` (continued isnad + matn)."""
    body = verse.get("text")
    parts: list[str] = []
    if isinstance(body, list):
        parts = [seg.strip() for seg in body if isinstance(seg, str) and seg.strip()]
    elif isinstance(body, str) and body.strip():
        parts = [body.strip()]
    return " ".join(parts).strip() or None


def _hadith_row(
    slug: str,
    part: int,
    chapter: int,
    hadith: int,
    chapter_name_ar: str | None,
    isnad_ar: str | None,
    body_ar: str | None,
) -> dict[str, Any]:
    """Assemble one HADITH_SCHEMA row from a verse's genuine Arabic (English null).

    The graph loader (:mod:`src.graph.load_nodes` ``_HADITH_MERGE``) surfaces only
    ``matn_ar`` / ``isnad_raw_ar`` (NOT ``full_text_ar``) on the Hadith node, so
    the readable Arabic MUST live in ``matn_ar``. Upstream does not cleanly split
    isnad from matn (``verse.text`` carries continued isnad + matn), so — as in the
    sibling ``thaqalayn`` parser — ``matn_ar`` holds the full hadith Arabic
    (sanad lead-in + body); ``isnad_raw_ar`` additionally surfaces the clean
    ``narrator_chain`` lead-in. ``full_text_ar`` mirrors ``matn_ar`` for Phase-2.
    """
    full_ar = " ".join(p for p in (isnad_ar, body_ar) if p).strip() or None
    return {
        "source_id": generate_source_id(SOURCE_CORPUS, slug, part, chapter, hadith),
        "source_corpus": SOURCE_CORPUS,
        # MUST equal the slug so APPEARS_IN's {corpus}:{collection_name} key
        # matches the Collection node id {corpus}:{slug} (load_edges.py).
        "collection_name": slug,
        "book_number": part,
        "chapter_number": chapter,
        "hadith_number": hadith,
        "matn_ar": full_ar,
        "matn_en": None,
        "isnad_raw_ar": isnad_ar,
        "isnad_raw_en": None,
        "full_text_ar": full_ar,
        "full_text_en": None,
        "grade": None,
        "chapter_name_ar": chapter_name_ar,
        "chapter_name_en": None,
        "sect": SECT,
    }


def _parse_book(book_dir: Path, slug: str) -> list[dict[str, Any]]:
    """Walk one Book's ``verse_detail`` files into hadith rows."""
    rows: list[dict[str, Any]] = []
    for json_path in sorted(book_dir.rglob("*.json")):
        with open(json_path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if not isinstance(doc, dict) or doc.get("kind") != "verse_detail":
            continue
        data = doc.get("data") or {}
        verse = data.get("verse") or {}
        coords = _verse_path_parts(safe_str(verse.get("path")) or "", slug)
        if coords is None:
            logger.warning("thaqalayn_data_unparseable_path", file=json_path.name)
            continue
        part, chapter, hadith = coords
        chapter_title = data.get("chapter_title") or {}
        rows.append(
            _hadith_row(
                slug,
                part,
                chapter,
                hadith,
                safe_str(chapter_title.get("ar")),
                _chain_arabic(verse),
                _body_arabic(verse),
            )
        )
    return rows


def _collection_row(slug: str, total_hadiths: int) -> dict[str, Any]:
    """The single COLLECTION_SCHEMA row for one Book."""
    meta = BOOK_META[slug]
    return {
        "collection_id": generate_source_id(SOURCE_CORPUS, slug),
        "name_ar": meta["name_ar"],
        "name_en": meta["name_en"],
        "compiler_name": COMPILER_NAME,
        "compilation_year_ah": None,
        "sect": SECT,
        "total_hadiths": total_hadiths,
        "source_corpus": SOURCE_CORPUS,
    }


def run(raw_dir: Path, staging_dir: Path) -> tuple[Path, Path]:
    """Parse the ThaqalaynData clone's two Books into hadiths + collections Parquet."""
    books_root = _clone_books_root(raw_dir)
    if not books_root.is_dir():
        msg = f"No ThaqalaynData clone found at {books_root}"
        raise FileNotFoundError(msg)

    logger.info("thaqalayn_data_parse_start", books=list(BOOK_META))

    hadith_rows: list[dict[str, Any]] = []
    coll_rows: list[dict[str, Any]] = []
    for slug in BOOK_META:
        book_dir = books_root / slug
        if not book_dir.is_dir():
            msg = f"Expected Book directory {book_dir} missing in clone"
            raise FileNotFoundError(msg)
        book_rows = _parse_book(book_dir, slug)
        logger.info("thaqalayn_data_parsed_book", book=slug, hadiths=len(book_rows))
        hadith_rows.extend(book_rows)
        coll_rows.append(_collection_row(slug, len(book_rows)))

    if not hadith_rows:
        msg = "ThaqalaynData parse produced 0 hadiths — check the verse_detail contract"
        raise ValueError(msg)

    hadith_table = pa.table(
        {field.name: [r[field.name] for r in hadith_rows] for field in HADITH_SCHEMA},
        schema=HADITH_SCHEMA,
    )
    hadiths_path = write_parquet(
        hadith_table, Path(staging_dir) / "hadiths_thaqalayn_data.parquet", schema=HADITH_SCHEMA
    )

    coll_table = pa.table(
        {field.name: [r[field.name] for r in coll_rows] for field in COLLECTION_SCHEMA},
        schema=COLLECTION_SCHEMA,
    )
    collections_path = write_parquet(
        coll_table,
        Path(staging_dir) / "collections_thaqalayn_data.parquet",
        schema=COLLECTION_SCHEMA,
    )

    logger.info(
        "thaqalayn_data_parse_complete",
        hadiths=len(hadith_rows),
        collections=len(coll_rows),
    )
    return hadiths_path, collections_path
