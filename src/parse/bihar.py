"""Parse Bihar al-Anwar (hubeali.com) raw HTML into staging Parquet.

Source
------
Bihar al-Anwar (بحار الأنوار), Allama Muhammad Baqir al-Majlisi's ~100k-hadith
Shia encyclopedia, is NOT carried by the Thaqalayn data source (the Four Books
+ ~20 secondary texts only). The one openly-readable, per-hadith, bilingual
(Arabic + English) edition is hubeali.com's "Read Online" set, one big WordPress
page per *volume / part* under
``/books-library/bihar-al-anwaar/volume-<V>/part-<P>/...``. ``src/acquire/bihar.py``
caches those pages; this module turns the cached HTML into hadith rows.

Markup contract (observed on the live pages)
--------------------------------------------
The hadith text lives in ``div.entry-content`` as a flat run of ``<p>``
elements that *alternate* an Arabic line and its English translation::

    <p>كتاب العقل و العلم و الجهل</p>          # كتاب  — book heading
    <p>باب 1 فضل العقل و ذم الجهل</p>          # باب N — chapter heading
    <p>الآيات ...</p> <p>The Verses ...</p>     # chapter preamble (Quranic verses)
    <p>1- مع، معاني الأخبار ... الإسناد</p>     # hadith 1: "<N>- <isnad+matn AR>"
    <p>In accordance to ...</p>                 #   its English translation
    <p>From Al-Sadiq ...</p>                     #   (continuation paragraphs)
    <p>2- لي، الأمالي ...</p>                    # hadith 2 ...

So a hadith **starts** at a paragraph matching ``^<N>- `` whose text is Arabic,
and runs until the next such marker or the next ``باب`` heading. ``<N>`` is the
in-chapter ordinal; ``source_id`` is namespaced ``bihar:bihar-al-anwar:<volume>:
<chapter>:<ordinal>`` (the corpus is ``bihar``; the collection slug is
``bihar-al-anwar`` — deliberately NOT ``bihar`` so the id is not mistaken for a
doubled-corpus prefix by :func:`src.parse.identity.is_double_prefixed`).

Like Thaqalayn, hubeali does not separate isnad from matn in a machine field, so
the whole Arabic body goes to ``full_text_ar`` and the English to
``full_text_en`` (``matn_*`` / ``isnad_raw_*`` stay null for Phase-2 extraction).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow as pa
from bs4 import BeautifulSoup

from src.parse.base import generate_source_id, safe_str, write_parquet
from src.parse.schemas import COLLECTION_SCHEMA, HADITH_SCHEMA
from src.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE_CORPUS = "bihar"
# Collection slug is NOT "bihar": a "bihar:bihar:..." source_id would trip
# identity.is_double_prefixed (leading segment == second segment == a known
# corpus) and be collapsed by the loader. "bihar-al-anwar" keeps it distinct.
COLLECTION_SLUG = "bihar-al-anwar"
COLLECTION_NAME = "Bihar al-Anwar"
COLLECTION_NAME_AR = "بحار الأنوار"
COMPILER_NAME = "Muhammad Baqir al-Majlisi"
SECT = "shia"

# Unicode ranges that mark a paragraph as Arabic rather than English.
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
# A hadith opener: "<N>- ..." / "<N> – ..." at the very start of an Arabic line.
_HADITH_START_RE = re.compile(r"^\s*(\d+)\s*[-–—]\s*")
# A chapter heading: Arabic "باب <N> ...".
_BAB_RE = re.compile(r"^\s*باب\s*(\d+)")
# Volume / part numbers parsed from the cached filename, e.g. "volume-1-part-1".
_VOL_PART_RE = re.compile(r"volume-(\d+)-part-(\d+)", re.IGNORECASE)
# RTL/LTR marks that pad the scraped Arabic and would defeat str.startswith.
_BIDI_MARKS = "‎‏‪‫‬​"


def _clean(text: str) -> str:
    """Normalise a scraped paragraph: drop bidi marks, collapse whitespace."""
    for mark in _BIDI_MARKS:
        text = text.replace(mark, "")
    return re.sub(r"\s+", " ", text).strip()


def _is_arabic(text: str) -> bool:
    """True if *text* contains Arabic-script characters."""
    return bool(_ARABIC_RE.search(text))


def _volume_part(html_path: Path) -> tuple[int | None, int | None]:
    """Parse ``(volume, part)`` from a cached filename like ``volume-1-part-1.html``."""
    match = _VOL_PART_RE.search(html_path.stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _paragraphs(html: str) -> list[str]:
    """Cleaned, non-empty ``<p>`` texts from the page's ``div.entry-content``.

    ``<sup>`` honorific glyphs (``asws`` etc.) are dropped before extraction so
    they do not pollute the text; falls back to the whole document if the page
    has no ``entry-content`` wrapper.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="entry-content") or soup
    for sup in container.find_all("sup"):
        sup.decompose()
    out: list[str] = []
    for para in container.find_all("p"):
        text = _clean(para.get_text(" ", strip=True))
        if text:
            out.append(text)
    return out


def _hadith_row(
    number: int,
    chapter_number: int | None,
    volume: int | None,
    chapter_name_ar: str | None,
    arabic_parts: list[str],
    english_parts: list[str],
) -> dict[str, Any]:
    """Assemble one HADITH_SCHEMA row from a hadith's collected paragraphs."""
    # Strip the leading "<N>- " marker off the first Arabic line.
    if arabic_parts:
        arabic_parts = [_HADITH_START_RE.sub("", arabic_parts[0], count=1), *arabic_parts[1:]]
    full_ar = safe_str("\n".join(p for p in arabic_parts if p))
    full_en = safe_str("\n".join(p for p in english_parts if p))
    source_id = generate_source_id(
        SOURCE_CORPUS,
        COLLECTION_SLUG,
        volume or 0,
        chapter_number or 0,
        number,
    )
    return {
        "source_id": source_id,
        "source_corpus": SOURCE_CORPUS,
        "collection_name": COLLECTION_NAME,
        "book_number": volume,
        "chapter_number": chapter_number,
        "hadith_number": number,
        "matn_ar": None,
        "matn_en": None,
        "isnad_raw_ar": None,
        "isnad_raw_en": None,
        "full_text_ar": full_ar,
        "full_text_en": full_en,
        "grade": None,
        "chapter_name_ar": chapter_name_ar,
        "chapter_name_en": None,
        "sect": SECT,
    }


def _parse_page(html: str, volume: int | None) -> list[dict[str, Any]]:
    """Segment one cached volume-part page into hadith rows.

    Walks the alternating Arabic/English paragraph run, tracking the current
    ``باب`` chapter and accumulating each ``<N>- ...`` hadith's Arabic and
    English paragraphs until the next hadith marker or chapter heading.
    """
    rows: list[dict[str, Any]] = []
    chapter_number: int | None = None
    chapter_name_ar: str | None = None
    in_chapter = False  # only collect hadiths once inside a باب (skips the index)

    cur_number: int | None = None
    cur_ar: list[str] = []
    cur_en: list[str] = []

    def flush() -> None:
        nonlocal cur_number, cur_ar, cur_en
        if cur_number is not None and (cur_ar or cur_en):
            rows.append(
                _hadith_row(cur_number, chapter_number, volume, chapter_name_ar, cur_ar, cur_en)
            )
        cur_number, cur_ar, cur_en = None, [], []

    for para in _paragraphs(html):
        bab = _BAB_RE.match(para)
        if bab:
            flush()
            chapter_number = int(bab.group(1))
            chapter_name_ar = para
            in_chapter = True
            continue

        start = _HADITH_START_RE.match(para)
        if start and in_chapter and _is_arabic(para):
            flush()
            cur_number = int(start.group(1))
            cur_ar = [para]
            continue

        if cur_number is None:
            continue  # chapter preamble (verses) or pre-content index — skip
        (cur_ar if _is_arabic(para) else cur_en).append(para)

    flush()
    return rows


def _collection_row(total_hadiths: int) -> dict[str, Any]:
    """The single Bihar al-Anwar COLLECTION_SCHEMA row."""
    return {
        "collection_id": generate_source_id(SOURCE_CORPUS, COLLECTION_SLUG),
        "name_ar": COLLECTION_NAME_AR,
        "name_en": COLLECTION_NAME,
        "compiler_name": COMPILER_NAME,
        "compilation_year_ah": None,
        "sect": SECT,
        "total_hadiths": total_hadiths,
        "source_corpus": SOURCE_CORPUS,
    }


def run(raw_dir: Path, staging_dir: Path) -> tuple[Path, Path]:
    """Parse cached hubeali Bihar HTML into hadiths + collection Parquet."""
    bihar_dir = raw_dir / "bihar"
    html_files = sorted(bihar_dir.glob("*.html"))
    if not html_files:
        msg = f"No Bihar al-Anwar HTML files found under {bihar_dir}"
        raise FileNotFoundError(msg)

    logger.info("bihar_parse_start", file_count=len(html_files))

    hadith_rows: list[dict[str, Any]] = []
    for html_path in html_files:
        volume, _part = _volume_part(html_path)
        page_rows = _parse_page(html_path.read_text(encoding="utf-8"), volume)
        logger.info("bihar_parsed_page", file=html_path.name, hadiths=len(page_rows))
        hadith_rows.extend(page_rows)

    if not hadith_rows:
        msg = "Bihar al-Anwar parse produced 0 hadiths — check the page markup contract"
        raise ValueError(msg)

    hadith_table = pa.table(
        {field.name: [r[field.name] for r in hadith_rows] for field in HADITH_SCHEMA},
        schema=HADITH_SCHEMA,
    )
    hadiths_path = write_parquet(
        hadith_table, Path(staging_dir) / "hadiths_bihar.parquet", schema=HADITH_SCHEMA
    )

    coll_rows = [_collection_row(len(hadith_rows))]
    coll_table = pa.table(
        {field.name: [r[field.name] for r in coll_rows] for field in COLLECTION_SCHEMA},
        schema=COLLECTION_SCHEMA,
    )
    collections_path = write_parquet(
        coll_table, Path(staging_dir) / "collections_bihar.parquet", schema=COLLECTION_SCHEMA
    )

    logger.info("bihar_parse_complete", hadiths=len(hadith_rows), collections=len(coll_rows))
    return hadiths_path, collections_path
