"""Web scraper for sunnah.com hadith collections.

Scrapes collections not available via the Fawaz dataset or when the
Sunnah.com API key is not configured.  Uses BeautifulSoup + httpx with
rate limiting (500 ms+ between requests).  Resumable: tracks per-collection
progress and skips already-scraped pages.

Output directory: ``data/raw/sunnah_scraped/``
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Tag

from src.acquire.base import ensure_dir, select_first, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://sunnah.com"
RATE_LIMIT_SECONDS = 0.5
REQUEST_TIMEOUT = 30.0
USER_AGENT = "isnad-graph/1.0 (hadith-research)"

# CSS selectors with fallbacks, ordered by preference.
# If sunnah.com redesigns, add new selectors at the front of each list.
#
# Hadith number: sunnah.com's current layout no longer exposes a bare
# ``.hadith_num`` span.  Both numbers we care about live in the
# ``table.hadith_reference`` block at the bottom of each container, e.g.
#     Reference         : Riyad as-Salihin 680   (collection-wide)
#     In-book reference : Book 1, Hadith 1        (book + in-book ordinal)
# We want the IN-BOOK ordinal (1) for ``hadith_number`` because the downstream
# APPEARS_IN edge maps it into ``hadith_number_in_book`` (documented as "Hadith
# number within the book" — da#77).  The in-book ordinal is unique within a
# book, so ``source_id = sunnah:<coll>:<book>:<chapter>:<ordinal>`` stays
# collision-free.  If the "In-book reference" row is absent we fall back to the
# collection-wide reference (sticky label / link href) so source_id keying still
# gets a non-null, per-collection-unique value, then to the legacy span.
HADITH_REFERENCE_SELECTORS = ["table.hadith_reference"]
HADITH_NUMBER_STICKY_SELECTORS = [".hadith_reference_sticky"]
HADITH_NUMBER_LINK_SELECTORS = ["table.hadith_reference tr td a[href]"]
HADITH_NUMBER_SELECTORS = [".hadith_reference .hadith_num", ".hadithNar498"]

# In-book reference row: "In-book reference : Book 1, Hadith 1".
_IN_BOOK_REF_RE = re.compile(
    r"In-book reference\s*:?\s*Book\s+(\d+)\s*,\s*Hadith\s+(\d+)", re.IGNORECASE
)
ARABIC_TEXT_SELECTORS = [".arabic_hadith_full", ".text_details .arabic_text_details"]
ENGLISH_TEXT_SELECTORS = [".english_hadith_full", ".text_details .english_hadith_full"]
GRADE_SELECTORS = [".hadith_grade", ".hadith-grade"]
CHAPTER_EN_SELECTORS = [".book_page_english_name", ".englishchapter"]
CHAPTER_AR_SELECTORS = [".book_page_arabic_name", ".arabicchapter"]
HADITH_CONTAINER_SELECTORS = [".actualHadithContainer", ".hadith"]

# Collections not in Fawaz dataset that we need to scrape.
SCRAPE_COLLECTIONS = [
    "musnad-ahmad",
    "sunan-darimi",
    "riyadussalihin",
    "adab",
    "shamail",
    "mishkat",
    "bulugh",
    "hisn",
]


def _check_robots_txt(client: httpx.Client) -> bool:
    """Check robots.txt to ensure scraping is allowed."""
    try:
        rp = RobotFileParser()
        resp = client.get(f"{BASE_URL}/robots.txt")
        rp.parse(resp.text.splitlines())
        allowed: bool = rp.can_fetch(USER_AGENT, f"{BASE_URL}/bukhari")
        if not allowed:
            logger.warning("sunnah_scraper_robots_denied")
        return allowed
    except Exception:  # noqa: BLE001
        # If we cannot fetch robots.txt, proceed cautiously.
        logger.warning("sunnah_scraper_robots_unavailable")
        return True


def _fetch_page(client: httpx.Client, url: str) -> BeautifulSoup | None:
    """Fetch a page and return parsed BeautifulSoup, or None on error."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        logger.warning("sunnah_scraper_http_error", url=url, status=exc.response.status_code)
        return None
    except httpx.HTTPError:
        logger.warning("sunnah_scraper_request_error", url=url)
        return None


def _extract_in_book_ref(row: Tag) -> tuple[int | None, int | None]:
    """Extract ``(book_number, in_book_ordinal)`` from the reference table.

    Reads the "In-book reference : Book N, Hadith M" row of
    ``table.hadith_reference``.  Returns ``(None, None)`` when that row is
    absent (e.g. a collection whose pages omit it).
    """
    ref_tag = select_first(row, HADITH_REFERENCE_SELECTORS)
    if ref_tag is None:
        return None, None
    match = _IN_BOOK_REF_RE.search(ref_tag.get_text(" ", strip=True))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _extract_collection_ref_number(row: Tag) -> int | None:
    """Extract the collection-wide reference number (e.g. 680).

    Fallback for ``hadith_number`` when the in-book ordinal is unavailable: it
    is unique per collection, so source_id keying still gets a non-null value.

    1. ``.hadith_reference_sticky`` label — trailing number ("... 680").
    2. ``table.hadith_reference`` reference link — ``href="/<coll>:680"``.
    3. Legacy ``.hadith_num`` / ``.hadithNar498`` span (pre-redesign markup).
    """
    # 1. Sticky reference label: "<Collection Name> <number>".
    sticky_tag = select_first(row, HADITH_NUMBER_STICKY_SELECTORS)
    if sticky_tag:
        match = re.search(r"(\d+)\s*$", sticky_tag.get_text(strip=True))
        if match:
            return int(match.group(1))

    # 2. Reference link href: "/<collection>:<number>" (or "...#<number>").
    link_tag = select_first(row, HADITH_NUMBER_LINK_SELECTORS)
    if link_tag:
        href = link_tag.get("href", "")
        if isinstance(href, str):
            match = re.search(r"[:#](\d+)\s*$", href)
            if match:
                return int(match.group(1))

    # 3. Legacy bare-number span.
    num_tag = select_first(row, HADITH_NUMBER_SELECTORS)
    if num_tag:
        text = num_tag.get_text(strip=True).replace(":", "").strip()
        try:
            return int(text)
        except ValueError:
            pass

    return None


def _extract_hadith_from_row(row: Tag) -> dict[str, Any] | None:
    """Extract a single hadith record from a hadith container element."""
    # Hadith number: prefer the IN-BOOK ordinal (what hadith_number_in_book
    # promises downstream); fall back to the collection-wide reference so
    # source_id keying still gets a non-null, per-collection-unique value.
    book_number, in_book_ordinal = _extract_in_book_ref(row)
    hadith_number = (
        in_book_ordinal if in_book_ordinal is not None else _extract_collection_ref_number(row)
    )

    # Arabic text
    ar_tag = select_first(row, ARABIC_TEXT_SELECTORS)
    text_ar = ar_tag.get_text(strip=True) if ar_tag else None

    # English text
    en_tag = select_first(row, ENGLISH_TEXT_SELECTORS)
    text_en = en_tag.get_text(strip=True) if en_tag else None

    # Grade
    grade_tag = select_first(row, GRADE_SELECTORS)
    grade = grade_tag.get_text(strip=True) if grade_tag else None

    if not text_ar and not text_en:
        return None

    return {
        "hadith_number": hadith_number,
        # Book number parsed from the in-book reference ("Book N, ..."); the
        # caller falls back to the page URL's book number when this is None.
        "book_number": book_number,
        "text_ar": text_ar,
        "text_en": text_en,
        "grade": grade,
    }


def _scrape_book_page(
    client: httpx.Client,
    collection: str,
    book_number: int,
) -> list[dict[str, Any]]:
    """Scrape all hadiths from a single book page."""
    url = f"{BASE_URL}/{collection}/{book_number}"
    soup = _fetch_page(client, url)
    if soup is None:
        return []

    # Chapter info
    chapter_name_en: str | None = None
    chapter_name_ar: str | None = None
    chapter_tag = select_first(soup, CHAPTER_EN_SELECTORS)
    if chapter_tag:
        chapter_name_en = chapter_tag.get_text(strip=True)
    chapter_ar_tag = select_first(soup, CHAPTER_AR_SELECTORS)
    if chapter_ar_tag:
        chapter_name_ar = chapter_ar_tag.get_text(strip=True)

    # Warn if no selectors matched — may indicate a site redesign
    container_matched = False
    hadiths: list[dict[str, Any]] = []
    rows: list[Tag] = []
    for selector in HADITH_CONTAINER_SELECTORS:
        rows = soup.select(selector)
        if rows:
            container_matched = True
            break

    if not container_matched and soup.get_text(strip=True):
        logger.warning(
            "sunnah_scraper_no_selectors_matched",
            collection=collection,
            book_number=book_number,
            hint="CSS selectors may be outdated — check for site redesign",
        )

    # Track chapter numbers within a book page. Sunnah.com sometimes has
    # multiple chapters per book, marked by chapter header elements.
    current_chapter_number = book_number
    chapter_counter = 0
    for row in rows:
        chapter_heading = row.find_previous_sibling(
            class_=lambda c: c and ("chapter" in c or "achapter" in c)
        )
        if chapter_heading:
            chapter_counter += 1
            current_chapter_number = book_number * 100 + chapter_counter

        record = _extract_hadith_from_row(row)
        if record is not None:
            # Prefer the book number parsed from the in-book reference; fall
            # back to the page URL's book number when the row was absent.
            if record.get("book_number") is None:
                record["book_number"] = book_number
            record["chapter_number"] = current_chapter_number
            record["chapter_name_ar"] = chapter_name_ar
            record["chapter_name_en"] = chapter_name_en
            hadiths.append(record)

    return hadiths


def _get_book_numbers(client: httpx.Client, collection: str) -> list[int]:
    """Get list of book numbers for a collection from its index page."""
    soup = _fetch_page(client, f"{BASE_URL}/{collection}")
    if soup is None:
        return []

    book_numbers: list[int] = []
    # Books are linked as /{collection}/{number}
    links = soup.select("a[href]")
    prefix = f"/{collection}/"
    seen: set[int] = set()
    for link in links:
        href = link.get("href", "")
        if isinstance(href, str) and href.startswith(prefix):
            segment = href[len(prefix) :].rstrip("/")
            if segment.isdigit():
                num = int(segment)
                if num not in seen:
                    seen.add(num)
                    book_numbers.append(num)

    book_numbers.sort()
    return book_numbers


def _scrape_collection(
    client: httpx.Client,
    collection: str,
    dest: Path,
) -> Path | None:
    """Scrape all hadiths for a single collection. Returns output path."""
    output_path = dest / f"{collection}.json"

    # Idempotent: skip if already scraped
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("sunnah_scraper_cached", collection=collection)
        return output_path

    # Progress file for resumability
    progress_path = dest / f".{collection}_progress.json"
    scraped_books: dict[str, list[dict[str, Any]]] = {}
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            scraped_books = json.load(f)

    book_numbers = _get_book_numbers(client, collection)
    if not book_numbers:
        logger.warning("sunnah_scraper_no_books", collection=collection)
        return None

    time.sleep(RATE_LIMIT_SECONDS)

    logger.info(
        "sunnah_scraper_starting",
        collection=collection,
        total_books=len(book_numbers),
        already_scraped=len(scraped_books),
    )

    for book_num in book_numbers:
        book_key = str(book_num)
        if book_key in scraped_books:
            continue

        hadiths = _scrape_book_page(client, collection, book_num)
        scraped_books[book_key] = hadiths

        # Save progress after each book
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(scraped_books, f, ensure_ascii=False)

        time.sleep(RATE_LIMIT_SECONDS)

    # Flatten all hadiths into a single list
    all_hadiths: list[dict[str, Any]] = []
    for hadiths in scraped_books.values():
        all_hadiths.extend(hadiths)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_hadiths, f, ensure_ascii=False, indent=2)

    # Clean up progress file
    if progress_path.exists():
        progress_path.unlink()

    logger.info(
        "sunnah_scraper_complete",
        collection=collection,
        books=len(scraped_books),
        hadiths=len(all_hadiths),
    )
    return output_path


def run(raw_dir: Path) -> Path | None:
    """Scrape hadiths from sunnah.com.

    Returns the output directory on success, or ``None`` if robots.txt
    disallows scraping.
    """
    dest = ensure_dir(raw_dir / "sunnah_scraped")

    client = httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    )

    try:
        if not _check_robots_txt(client):
            return None

        saved_files: list[Path] = []
        total_hadiths = 0

        for collection in SCRAPE_COLLECTIONS:
            output_path = _scrape_collection(client, collection, dest)
            if output_path is not None:
                saved_files.append(output_path)
                with open(output_path, encoding="utf-8") as f:
                    data: list[Any] = json.load(f)
                total_hadiths += len(data)

        if saved_files:
            write_manifest(dest, saved_files)
            emit_raw_new_for_manifest(source="sunnah_scraper", local_dir=dest, files=saved_files)

        logger.info(
            "sunnah_scraper_acquired",
            collections=len(saved_files),
            total_hadiths=total_hadiths,
        )
        return dest
    finally:
        client.close()
