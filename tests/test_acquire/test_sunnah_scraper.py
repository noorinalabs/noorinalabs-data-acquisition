"""Tests for the sunnah.com web scraper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.acquire.sunnah_scraper import (
    REQUEST_TIMEOUT,
    SCRAPE_COLLECTIONS,
    USER_AGENT,
    _extract_collection_ref_number,
    _extract_hadith_from_row,
    _extract_in_book_ref,
    _get_book_numbers,
    _scrape_book_page,
    _scrape_collection,
    run,
)

# Opt-in live leg: hits the real sunnah.com (keyless). Off by default so CI stays
# offline-deterministic; set RUN_LIVE_SUNNAH_SCRAPE=1 to exercise the real parse
# path end-to-end (the no-fixture-masking guard, main#671).
_RUN_LIVE = os.getenv("RUN_LIVE_SUNNAH_SCRAPE") == "1"
_LIVE_SKIP_REASON = "set RUN_LIVE_SUNNAH_SCRAPE=1 to run the live sunnah.com scrape leg"

FIXTURES_DIR = Path(__file__).parent / "fixtures"

SAMPLE_COLLECTION_HTML = """
<html><body>
<a href="/musnad-ahmad/1">Book 1</a>
<a href="/musnad-ahmad/2">Book 2</a>
<a href="/musnad-ahmad/3">Book 3</a>
<a href="/other-link">Other</a>
</body></html>
"""

SAMPLE_BOOK_HTML = """
<html><body>
<div class="book_page_english_name">The Book of Purification</div>
<div class="book_page_arabic_name">كتاب الطهارة</div>
<div class="actualHadithContainer">
  <div class="hadith_reference"><span class="hadith_num">1</span></div>
  <div class="arabic_hadith_full">نص الحديث بالعربية</div>
  <div class="english_hadith_full">The hadith text in English</div>
  <div class="hadith_grade">Sahih</div>
</div>
<div class="actualHadithContainer">
  <div class="hadith_reference"><span class="hadith_num">2</span></div>
  <div class="arabic_hadith_full">حديث ثاني</div>
  <div class="english_hadith_full">Second hadith</div>
</div>
</body></html>
"""

ROBOTS_TXT = "User-agent: *\nAllow: /\n"


def _mock_response(text: str, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        text=text,
        request=httpx.Request("GET", "https://sunnah.com/test"),
    )


class TestExtractHadithFromRow:
    def test_extracts_full_record(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <div class="actualHadithContainer">
          <div class="hadith_reference"><span class="hadith_num">42</span></div>
          <div class="arabic_hadith_full">عربي</div>
          <div class="english_hadith_full">English text</div>
          <div class="hadith_grade">Hasan</div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        result = _extract_hadith_from_row(row)

        assert result is not None
        assert result["hadith_number"] == 42
        assert result["text_ar"] == "عربي"
        assert result["text_en"] == "English text"
        assert result["grade"] == "Hasan"

    def test_returns_none_when_no_text(self) -> None:
        from bs4 import BeautifulSoup

        html = '<div class="actualHadithContainer"><span>empty</span></div>'
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        result = _extract_hadith_from_row(row)
        assert result is None


class TestGetBookNumbers:
    def test_parses_book_links(self) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_response(SAMPLE_COLLECTION_HTML)

        books = _get_book_numbers(client, "musnad-ahmad")
        assert books == [1, 2, 3]

    def test_returns_empty_on_404(self) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("GET", "https://sunnah.com/test"),
            response=httpx.Response(404),
        )

        books = _get_book_numbers(client, "nonexistent")
        assert books == []

    def test_real_index_markup_includes_introduction(self) -> None:
        """Real-upstream guard (da#177): the saved sunnah.com riyadussalihin
        index lists a NAMED ``introduction`` book ("The Book of Miscellany")
        alongside /1../19. The old digits-only enumerator dropped it, silently
        truncating the collection by its first 679 hadiths."""
        index_html = (FIXTURES_DIR / "riyadussalihin_index_sample.html").read_text(encoding="utf-8")
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_response(index_html)

        books = _get_book_numbers(client, "riyadussalihin")

        # Named books first, then numbered books ascending.
        assert books == ["introduction", *range(1, 20)]
        # Non-book links (the /riyadussalihin home, /contactus) are excluded.
        assert "" not in books

    def test_folds_anchor_variant_and_skips_subpaths(self) -> None:
        """A named book can appear with an #anchor (sunnah.com hisn:
        ``introduction#C0.00``); fold it onto the bare segment and dedupe.
        Multi-component hrefs (.../x/y) are not book pages and are skipped."""
        html = """
        <html><body>
          <a href="/hisn/introduction">Introduction</a>
          <a href="/hisn/introduction#C0.00">Introduction (anchor)</a>
          <a href="/hisn/1">Book 1</a>
          <a href="/hisn/1/5">A hadith permalink</a>
          <a href="/hisn/">collection home</a>
        </body></html>
        """
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_response(html)

        books = _get_book_numbers(client, "hisn")
        assert books == ["introduction", 1]


class TestScrapeBookPage:
    def test_extracts_hadiths(self) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_response(SAMPLE_BOOK_HTML)

        hadiths = _scrape_book_page(client, "musnad-ahmad", 1)
        assert len(hadiths) == 2
        assert hadiths[0]["hadith_number"] == 1
        assert hadiths[0]["text_ar"] == "نص الحديث بالعربية"
        assert hadiths[0]["book_number"] == 1
        assert hadiths[0]["chapter_name_en"] == "The Book of Purification"
        assert hadiths[0]["chapter_name_ar"] == "كتاب الطهارة"

    def test_named_introduction_segment(self) -> None:
        """A named book segment ("introduction") builds the right URL and keys to
        book 0; its hadith_number falls back to the collection-wide reference
        because the in-book row reads "Introduction, Hadith N" (not "Book N")."""
        intro_html = """
        <html><body>
        <div class="actualHadithContainer">
          <div class="hadith_reference_sticky">Riyad as-Salihin 1</div>
          <div class="arabic_hadith_full">إنما الأعمال بالنيات</div>
          <div class="english_hadith_full">Actions are judged by intentions</div>
          <table class="hadith_reference">
            <tr><td>In-book reference</td><td>: Introduction, Hadith 1</td></tr>
          </table>
        </div>
        </body></html>
        """
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_response(intro_html)

        hadiths = _scrape_book_page(client, "riyadussalihin", "introduction")

        client.get.assert_called_once_with("https://sunnah.com/riyadussalihin/introduction")
        assert len(hadiths) == 1
        # "Introduction, Hadith 1" doesn't match the "Book N" in-book pattern, so
        # hadith_number falls back to the collection-wide ref (1), and book_number
        # falls back to the named-segment key (0).
        assert hadiths[0]["hadith_number"] == 1
        assert hadiths[0]["book_number"] == 0


class TestRun:
    @patch("src.acquire.sunnah_scraper.SCRAPE_COLLECTIONS", ["test-collection"])
    @patch("src.acquire.sunnah_scraper.RATE_LIMIT_SECONDS", 0)
    def test_scrapes_and_saves_json(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"

        with patch("src.acquire.sunnah_scraper.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value = client

            # robots.txt
            client.get.side_effect = [
                _mock_response(ROBOTS_TXT),  # robots.txt
                _mock_response(SAMPLE_COLLECTION_HTML.replace("musnad-ahmad", "test-collection")),
                _mock_response(SAMPLE_BOOK_HTML),  # book 1
                _mock_response(SAMPLE_BOOK_HTML),  # book 2
                _mock_response(SAMPLE_BOOK_HTML),  # book 3
            ]

            result = run(raw_dir)

        assert result is not None
        assert (raw_dir / "sunnah_scraped" / "test-collection.json").exists()
        with open(raw_dir / "sunnah_scraped" / "test-collection.json") as f:
            data = json.load(f)
        assert len(data) == 6  # 2 hadiths per book * 3 books

    @patch("src.acquire.sunnah_scraper.SCRAPE_COLLECTIONS", ["test-collection"])
    @patch("src.acquire.sunnah_scraper.RATE_LIMIT_SECONDS", 0)
    def test_idempotent_skips_existing(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        dest = raw_dir / "sunnah_scraped"
        dest.mkdir(parents=True)
        (dest / "test-collection.json").write_text('[{"hadith_number": 1}]')

        with patch("src.acquire.sunnah_scraper.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value = client
            client.get.return_value = _mock_response(ROBOTS_TXT)

            result = run(raw_dir)

        assert result is not None

    @patch("src.acquire.sunnah_scraper.RATE_LIMIT_SECONDS", 0)
    def test_robots_denied(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"

        with patch("src.acquire.sunnah_scraper.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value = client
            client.get.return_value = _mock_response("User-agent: *\nDisallow: /\n")

            result = run(raw_dir)

        assert result is None

    def test_target_collections_defined(self) -> None:
        assert len(SCRAPE_COLLECTIONS) == 8
        assert "musnad-ahmad" in SCRAPE_COLLECTIONS
        assert "riyadussalihin" in SCRAPE_COLLECTIONS


# Container markup matching sunnah.com's CURRENT layout (no bare `.hadith_num`
# span). The reference table carries BOTH numbers: the collection-wide ref in
# the first row / sticky label, and the in-book ordinal in the "In-book
# reference" row. da#72 extracts the IN-BOOK ordinal into `hadith_number` so the
# downstream APPEARS_IN `hadith_number_in_book` edge prop is semantically honest
# (da#77); the collection-wide ref is a non-null fallback for source_id keying.
CURRENT_MARKUP_HADITH_HTML = """
<div class="actualHadithContainer">
  <div class="hadith_reference_sticky">Riyad as-Salihin 680</div>
  <div class="english_hadith_full">English text</div>
  <div class="arabic_hadith_full arabic">عربي</div>
  <div class="bottomItems">
    <table class="hadith_reference">
      <tr><td><b>Reference</b></td>
        <td>: <a href="/riyadussalihin:680">Riyad as-Salihin 680</a></td></tr>
      <tr><td>In-book reference</td><td>: Book 1, Hadith 5</td></tr>
    </table>
  </div>
</div>
"""


class TestExtractInBookRef:
    """Regression coverage for da#72/da#77 — in-book ordinal on current markup."""

    def test_extracts_book_and_in_book_ordinal(self) -> None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(CURRENT_MARKUP_HADITH_HTML, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        # "In-book reference : Book 1, Hadith 5" -> (book=1, ordinal=5).
        assert _extract_in_book_ref(row) == (1, 5)

    def test_returns_none_pair_when_in_book_row_absent(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <div class="actualHadithContainer">
          <div class="english_hadith_full">English text</div>
          <table class="hadith_reference"><tr>
            <td><b>Reference</b></td>
            <td>: <a href="/riyadussalihin:681">Riyad as-Salihin 681</a></td>
          </tr></table>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        assert _extract_in_book_ref(row) == (None, None)


class TestExtractCollectionRefNumber:
    """The collection-wide ref is the source_id-keying fallback when no ordinal."""

    def test_extracts_from_sticky_reference_label(self) -> None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(CURRENT_MARKUP_HADITH_HTML, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        assert _extract_collection_ref_number(row) == 680

    def test_extracts_from_reference_link_href(self) -> None:
        from bs4 import BeautifulSoup

        # No sticky label — must fall back to the reference-link href.
        html = """
        <div class="actualHadithContainer">
          <div class="english_hadith_full">English text</div>
          <table class="hadith_reference"><tr>
            <td><b>Reference</b></td>
            <td>: <a href="/riyadussalihin:681">Riyad as-Salihin 681</a></td>
          </tr></table>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        assert _extract_collection_ref_number(row) == 681

    def test_legacy_span_still_supported(self) -> None:
        from bs4 import BeautifulSoup

        # Pre-redesign markup must keep working for collections still on it.
        html = """
        <div class="actualHadithContainer">
          <div class="hadith_reference"><span class="hadith_num">42</span></div>
          <div class="english_hadith_full">English text</div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        assert _extract_collection_ref_number(row) == 42


class TestExtractHadithFromRowCurrentMarkup:
    """`hadith_number` carries the in-book ordinal; book parsed from the ref."""

    def test_record_uses_in_book_ordinal(self) -> None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(CURRENT_MARKUP_HADITH_HTML, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        record = _extract_hadith_from_row(row)
        assert record is not None
        # The pre-fix bug: hadith_number was None for every row on the live site.
        # Now it is the IN-BOOK ordinal (5), not the collection-wide ref (680).
        assert record["hadith_number"] == 5
        assert record["book_number"] == 1

    def test_falls_back_to_collection_ref_when_no_in_book_row(self) -> None:
        from bs4 import BeautifulSoup

        # No "In-book reference" row: hadith_number must still be non-null,
        # using the collection-wide ref so source_id keying never collapses.
        html = """
        <div class="actualHadithContainer">
          <div class="hadith_reference_sticky">Riyad as-Salihin 681</div>
          <div class="english_hadith_full">English text</div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        row = soup.select_one(".actualHadithContainer")
        assert row is not None
        record = _extract_hadith_from_row(row)
        assert record is not None
        assert record["hadith_number"] == 681
        assert record["book_number"] is None


class TestRiyadussalihinFixture:
    """End-to-end selector guard over a saved current-markup fixture page."""

    def test_in_book_ordinals_parsed_and_source_ids_unique(self) -> None:
        from bs4 import BeautifulSoup

        from src.parse.base import generate_source_id

        html = (FIXTURES_DIR / "riyadussalihin_book1_sample.html").read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select(".actualHadithContainer")
        assert rows, "fixture must contain hadith containers"

        records = [rec for r in rows if (rec := _extract_hadith_from_row(r)) is not None]
        assert records, "fixture must yield parseable hadith records"

        # Acceptance: a non-null number for every sampled hadith ...
        assert all(r["hadith_number"] is not None for r in records)
        # ... and it is the in-book ordinal (1, 2, 3, ...), not the
        # collection-wide ref (680, 681, ...).
        assert [r["hadith_number"] for r in records] == list(range(1, len(records) + 1))

        # Acceptance: source_id uniqueness holds. The in-book ordinal is unique
        # within a book, so book+chapter+ordinal keeps every record distinct and
        # dedup never merges them — even with chapter held constant here.
        source_ids = {
            generate_source_id(
                "sunnah",
                "riyadussalihin",
                r["book_number"] or 0,
                1,
                r["hadith_number"] or 0,
            )
            for r in records
        }
        assert len(source_ids) == len(records)


@pytest.mark.skipif(not _RUN_LIVE, reason=_LIVE_SKIP_REASON)
class TestRiyadussalihinLiveUpstream:
    """Live, opt-in scrape of the REAL sunnah.com riyadussalihin (no key needed).

    This is the no-fixture-masking guard (main#671): a saved fixture proves the
    parser handles known markup, but only a live fetch proves the *enumerator*
    still discovers every book — the exact axis where the introduction book was
    being dropped. Gated on RUN_LIVE_SUNNAH_SCRAPE=1 so CI stays offline.
    """

    def _client(self) -> httpx.Client:
        return httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True
        )

    def test_introduction_is_enumerated(self) -> None:
        with self._client() as client:
            books = _get_book_numbers(client, "riyadussalihin")
        assert "introduction" in books, "the introduction book must be discovered"
        assert sum(1 for b in books if isinstance(b, int)) >= 19

    def test_full_collection_reaches_expected_total(self, tmp_path: Path) -> None:
        with self._client() as client:
            out = _scrape_collection(client, "riyadussalihin", tmp_path)
        assert out is not None
        hadiths = json.loads(out.read_text(encoding="utf-8"))
        # Riyad as-Salihin is 1,896 hadiths (679 in the introduction + 1,217 in
        # books 1-19). Allow a small tolerance for upstream markup churn.
        assert len(hadiths) >= 1890, f"expected ~1896, got {len(hadiths)}"
        assert all(h.get("text_ar") and h.get("text_en") for h in hadiths)
        assert all(h.get("hadith_number") is not None for h in hadiths)
