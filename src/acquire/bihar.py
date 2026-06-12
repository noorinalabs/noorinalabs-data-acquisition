"""Acquire Bihar al-Anwar (بحار الأنوار) from hubeali.com.

Bihar al-Anwar — al-Majlisi's ~100k-hadith Shia encyclopedia — is NOT carried by
the Thaqalayn data source (which is the Four Books + ~20 secondary texts only;
verified da#95). hubeali.com's "Read Online" edition is the one openly-readable,
per-hadith, bilingual (Arabic + English) copy. It is published as one large
WordPress page per *volume / part*::

    /books-library/bihar-al-anwaar/volume-<V>/                       (volume index)
    /books-library/bihar-al-anwaar/volume-<V>/part-<P>/
        bihar-al-anwaar-volume-<V>-part-<P>/                         (content page)

This downloader is a *polite, bounded* scraper modelled on
:mod:`src.acquire.sunnah_scraper`: it honours ``robots.txt`` (hubeali only
disallows ``/wp-admin/``), throttles between requests, and caches each content
page's raw HTML to ``data/raw/bihar/volume-<V>-part-<P>.html`` (idempotent — a
cached page is skipped). :mod:`src.parse.bihar` parses those cached pages.

``DEFAULT_VOLUMES`` is deliberately small (volume 1): a PR-time acquisition pulls
a bounded, representative slice rather than the whole ~100k-hadith corpus. Pass
``volumes=range(1, N)`` (or set it in a job) to widen the crawl for a real
data-load run.

Licensing (owner-approved da#95, same posture as Itqan da#92a): hubeali content
carries no machine-readable upstream licence; use is non-profit, the hadith facts
are re-expressed in our own schema, and everything is cleanly removable via the
``source_corpus="bihar"`` provenance tag.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from src.acquire.base import ensure_dir, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://hubeali.com"
BOOK_ROOT = f"{BASE_URL}/books-library/bihar-al-anwaar"
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 60.0
USER_AGENT = "isnad-graph/1.0 (hadith-research)"

# A PR-time crawl pulls a bounded slice; widen for a full data-load run.
DEFAULT_VOLUMES: tuple[int, ...] = (1,)

# The deep "content" link on a volume index page, e.g.
# ".../volume-1/part-1/bihar-al-anwaar-volume-1-part-1/". The shorter
# ".../part-1/" part-index link is ignored — we want the page with the text.
_CONTENT_LINK_RE = re.compile(
    r"/books-library/bihar-al-anwaar/volume-(\d+)/part-(\d+)/"
    r"bihar-al-anwaar-volume-\d+-part-\d+/?",
    re.IGNORECASE,
)


def _check_robots_txt(client: httpx.Client) -> bool:
    """Return True if ``robots.txt`` permits crawling the book pages."""
    try:
        resp = client.get(f"{BASE_URL}/robots.txt")
        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        allowed: bool = parser.can_fetch(USER_AGENT, f"{BOOK_ROOT}/volume-1/")
        if not allowed:
            logger.warning("bihar_robots_denied")
        return allowed
    except Exception:  # noqa: BLE001 — robots unreachable: proceed cautiously, like sunnah_scraper
        logger.warning("bihar_robots_unavailable")
        return True


def _get(client: httpx.Client, url: str) -> str | None:
    """GET *url*, returning the response text or ``None`` on any HTTP error."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        logger.warning("bihar_fetch_error", url=url, error=str(exc))
        return None


def _discover_content_links(client: httpx.Client, volume: int) -> list[tuple[int, str]]:
    """Find ``(part_number, url)`` content links on a volume index page."""
    html = _get(client, f"{BOOK_ROOT}/volume-{volume}/")
    if html is None:
        return []
    seen: dict[int, str] = {}
    for match in _CONTENT_LINK_RE.finditer(html):
        if int(match.group(1)) != volume:
            continue
        part = int(match.group(2))
        url = match.group(0)
        if not url.startswith("http"):
            url = BASE_URL + url
        seen.setdefault(part, url.rstrip("/") + "/")
    return sorted(seen.items())


def run(raw_dir: Path, volumes: Iterable[int] = DEFAULT_VOLUMES) -> Path | None:
    """Scrape bounded Bihar al-Anwar volume/part pages from hubeali.com.

    Caches each content page's raw HTML to ``<raw_dir>/bihar/volume-<V>-part-<P>.html``
    (idempotent). Returns the ``bihar`` raw directory, or ``None`` if
    ``robots.txt`` disallows the crawl or nothing could be fetched.
    """
    dest = ensure_dir(raw_dir / "bihar")
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    )
    try:
        if not _check_robots_txt(client):
            return None

        saved: list[Path] = []
        for volume in volumes:
            for part, url in _discover_content_links(client, volume):
                out_path = dest / f"volume-{volume}-part-{part}.html"
                if out_path.exists() and out_path.stat().st_size > 0:
                    logger.info("bihar_cached", volume=volume, part=part)
                    saved.append(out_path)
                    continue

                time.sleep(RATE_LIMIT_SECONDS)
                html = _get(client, url)
                if html is None:
                    continue
                out_path.write_text(html, encoding="utf-8")
                logger.info("bihar_fetched", volume=volume, part=part, bytes=len(html))
                saved.append(out_path)

        if not saved:
            logger.warning("bihar_no_pages_acquired")
            return None

        write_manifest(dest, saved)
        emit_raw_new_for_manifest(source="bihar", local_dir=dest, files=saved)
        logger.info("bihar_acquired", pages=len(saved))
        return dest
    finally:
        client.close()
