"""Tests for the Bihar al-Anwar (hubeali) acquire scraper.

Two legs, mirroring the project's acquire-test convention:

* **Mocked unit legs** (always run, no network) — patch ``httpx.Client.get`` to
  serve a robots.txt, a real-shaped volume-index page, and a content page, then
  assert robots-gating, content-link discovery, raw caching/idempotency, and the
  manifest write.
* **Live smoke** (skip-guarded behind ``BIHAR_LIVE_TEST=1``) — actually crawls a
  bounded slice of hubeali.com and parses it, proving the live→parse contract
  without pulling the whole ~100k-hadith corpus.  Off by default so CI never
  depends on a third-party site being up.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pyarrow.parquet as pq
import pytest

from src.acquire import bihar
from src.parse.bihar import run as parse_run
from src.parse.schemas import HADITH_SCHEMA

ROBOTS_ALLOW = "User-agent: *\nDisallow: /wp-admin/\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /\n"

# A volume-index page carrying the real deep "content" link shape plus the
# shorter part-index link that must be ignored.
VOLUME_INDEX_HTML = """
<html><body>
  <a href="https://hubeali.com/books-library/bihar-al-anwaar/volume-1/part-1/">Part 1</a>
  <a href="https://hubeali.com/books-library/bihar-al-anwaar/volume-1/part-1/bihar-al-anwaar-volume-1-part-1/">Read</a>
  <a href="https://hubeali.com/books-library/bihar-al-anwaar/volume-2/">Volume 2</a>
</body></html>
"""

CONTENT_HTML = """
<html><body><div class="entry-content">
<p>باب 1 فضل العقل</p>
<p>1- مع، معاني الأخبار الإسناد العربي</p>
<p>In accordance to the chain, English translation.</p>
</div></body></html>
"""


def _response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status, text=text, request=httpx.Request("GET", "https://hubeali.com/x")
    )


def _router(robots: str = ROBOTS_ALLOW):
    """Return a fake ``Client.get`` that routes by URL to canned responses."""

    def _get(self: httpx.Client, url: str, *args: object, **kwargs: object) -> httpx.Response:
        if url.endswith("robots.txt"):
            return _response(robots)
        if url.endswith("/volume-1/"):
            return _response(VOLUME_INDEX_HTML)
        if "bihar-al-anwaar-volume-1-part-1" in url:
            return _response(CONTENT_HTML)
        if url.endswith("/volume-2/"):
            return _response("<html><body>no parts</body></html>")
        return _response("not found", status=404)

    return _get


class TestBiharAcquireMocked:
    def test_robots_denied_returns_none(self, tmp_path: Path) -> None:
        with patch.object(httpx.Client, "get", _router(robots=ROBOTS_DENY)):
            assert bihar.run(tmp_path / "raw", volumes=[1]) is None

    def test_discovers_and_caches_content_page(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        with patch.object(httpx.Client, "get", _router()), patch.object(bihar.time, "sleep"):
            result = bihar.run(raw_dir, volumes=[1])
        assert result == raw_dir / "bihar"
        cached = raw_dir / "bihar" / "volume-1-part-1.html"
        assert cached.exists()
        assert "entry-content" in cached.read_text(encoding="utf-8")
        assert (raw_dir / "bihar" / "manifest.json").exists()

    def test_idempotent_skip_on_second_run(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        get = _router()
        calls: list[str] = []

        def _counting_get(self: httpx.Client, url: str, *a: object, **k: object) -> httpx.Response:
            calls.append(url)
            return get(self, url, *a, **k)

        with patch.object(httpx.Client, "get", _counting_get), patch.object(bihar.time, "sleep"):
            bihar.run(raw_dir, volumes=[1])
            first = len([u for u in calls if "part-1/bihar" in u])
            bihar.run(raw_dir, volumes=[1])
            second = len([u for u in calls if "part-1/bihar" in u])
        # The content page is fetched on the first run, served from cache on the second.
        assert first == 1
        assert second == 1

    def test_no_pages_returns_none(self, tmp_path: Path) -> None:
        # Volume 2 in the fixture has no content links → nothing acquired.
        with patch.object(httpx.Client, "get", _router()), patch.object(bihar.time, "sleep"):
            assert bihar.run(tmp_path / "raw", volumes=[2]) is None


@pytest.mark.skipif(
    os.getenv("BIHAR_LIVE_TEST") != "1",
    reason="live hubeali crawl skip-guarded behind BIHAR_LIVE_TEST=1",
)
class TestBiharAcquireLive:
    def test_live_bounded_acquire_and_parse(self, tmp_path: Path) -> None:
        """Crawl volume 1 from the real site and parse it to conforming Parquet."""
        raw_dir = tmp_path / "raw"
        result = bihar.run(raw_dir, volumes=[1])
        assert result is not None, "live crawl acquired nothing (site down or markup changed)"

        hadiths_path, _ = parse_run(raw_dir, tmp_path / "staging")
        table = pq.read_table(hadiths_path)
        assert table.schema == HADITH_SCHEMA
        assert table.num_rows > 0
        rows = table.to_pylist()
        assert all(r["sect"] == "shia" and r["source_corpus"] == "bihar" for r in rows)
