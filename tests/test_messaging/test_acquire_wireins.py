"""Wire-in tests: every acquire connector calls ``emit_raw_new_for_manifest``.

These tests stub out network and git operations so each connector's
``run()`` reaches the emit call on the happy path. The assertion is
narrow: the emit helper was called with the expected ``source=`` value.
The per-file payload correctness is covered by
``test_kafka_producer.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _seed_csvs(dest: Path, names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dest / n).write_text("a,b\n1,2\n", encoding="utf-8")


def _seed_jsons(dest: Path, names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dest / n).write_text("{}", encoding="utf-8")


class TestLkCorpusWireIn:
    def test_run_emits_with_source_lk_corpus(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        csvs = [
            "bukhari.csv",
            "muslim.csv",
            "abudawud.csv",
            "tirmidhi.csv",
            "nasai.csv",
            "ibnmajah.csv",
        ]

        def fake_clone_repo(url: str, dest: Path, **kw: Any) -> Path:
            _seed_csvs(dest, csvs)
            return dest

        with (
            patch("src.acquire.lk_corpus.clone_repo", side_effect=fake_clone_repo),
            patch("src.acquire.lk_corpus.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import lk_corpus

            lk_corpus.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "lk_corpus"
        assert len(emit.call_args.kwargs["files"]) >= 6


class TestOpenHadithWireIn:
    def test_run_emits_with_source_open_hadith(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        names = [f"book{i}.csv" for i in range(9)]

        def fake_clone_repo(url: str, dest: Path, **kw: Any) -> Path:
            _seed_csvs(dest, names)
            return dest

        with (
            patch("src.acquire.open_hadith.clone_repo", side_effect=fake_clone_repo),
            patch("src.acquire.open_hadith.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import open_hadith

            open_hadith.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "open_hadith"


class TestMuhaddithatWireIn:
    def test_run_emits_with_source_muhaddithat(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"

        def fake_clone_repo(url: str, dest: Path, **kw: Any) -> Path:
            _seed_csvs(dest, ["hadiths.csv", "narrators.csv"])
            return dest

        with (
            patch("src.acquire.muhaddithat.clone_repo", side_effect=fake_clone_repo),
            patch("src.acquire.muhaddithat.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import muhaddithat

            muhaddithat.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "muhaddithat"


class TestThaqalaynWireIn:
    def test_run_emits_with_source_thaqalayn_on_skip_path(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        dest = raw / "thaqalayn"
        # Pre-seed enough book_*.json files to hit the idempotent skip branch.
        _seed_jsons(dest, [f"book_{i:03d}.json" for i in range(16)])

        with patch("src.acquire.thaqalayn.emit_raw_new_for_manifest") as emit:
            from src.acquire import thaqalayn

            thaqalayn.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "thaqalayn"

    def test_run_emits_with_source_thaqalayn_on_fresh_clone(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"

        def fake_download(dest: Path) -> list[Path]:
            clone_dest = dest / "github_clone"
            _seed_jsons(clone_dest, [f"book_{i:03d}.json" for i in range(16)])
            return list(clone_dest.glob("*.json"))

        with (
            patch(
                "src.acquire.thaqalayn._download_via_github",
                side_effect=fake_download,
            ),
            patch("src.acquire.thaqalayn.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import thaqalayn

            thaqalayn.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "thaqalayn"


class TestSunnahApiWireIn:
    def test_run_emits_with_source_sunnah_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUNNAH_API_KEY", "test-key")
        # Clear the settings cache so the env var is picked up.
        from src.config import get_settings

        get_settings.cache_clear()

        raw = tmp_path / "raw"

        def fake_fetch_json_paginated(url: str, **kw: Any) -> list[dict[str, Any]]:
            if "/collections" in url and "/hadiths" not in url:
                return [{"name": "bukhari"}, {"name": "muslim"}]
            return [{"hadith": "x"}]

        with (
            patch(
                "src.acquire.sunnah_api.fetch_json_paginated",
                side_effect=fake_fetch_json_paginated,
            ),
            patch("src.acquire.sunnah_api.time.sleep"),
            patch("src.acquire.sunnah_api.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import sunnah_api

            sunnah_api.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "sunnah_api"


class TestFawazWireIn:
    def test_run_emits_with_source_fawaz_on_skip_path(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        dest = raw / "fawaz"
        _seed_jsons(dest, [f"eng-{i}.json" for i in range(10)])
        _seed_jsons(dest, [f"ara-{i}.json" for i in range(10)])

        with patch("src.acquire.fawaz.emit_raw_new_for_manifest") as emit:
            from src.acquire import fawaz

            fawaz.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "fawaz"


class TestSanadsetWireIn:
    def test_run_emits_with_source_sanadset(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        dest = raw / "sanadset"
        _seed_csvs(dest, ["sanadset.csv", "books.csv"])
        # Write ~MIN_EXPECTED_ROWS worth of rows so the warning path is stable.
        (dest / "sanadset.csv").write_text("h,b\n" + "\n".join(["x,y"] * 10), encoding="utf-8")

        with patch("src.acquire.sanadset.emit_raw_new_for_manifest") as emit:
            from src.acquire import sanadset

            sanadset.download_sanadset(dest=dest)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "sanadset"


class TestSunnahScraperWireIn:
    def test_run_emits_with_source_sunnah_scraper(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"

        def fake_scrape(client: Any, collection: str, dest_: Path) -> Path | None:
            dest_.mkdir(parents=True, exist_ok=True)
            path = dest_ / f"{collection}.json"
            path.write_text("[]", encoding="utf-8")
            return path

        with (
            patch("src.acquire.sunnah_scraper._check_robots_txt", return_value=True),
            patch(
                "src.acquire.sunnah_scraper._scrape_collection",
                side_effect=fake_scrape,
            ),
            patch("src.acquire.sunnah_scraper.emit_raw_new_for_manifest") as emit,
        ):
            from src.acquire import sunnah_scraper

            sunnah_scraper.run(raw)

        emit.assert_called_once()
        assert emit.call_args.kwargs["source"] == "sunnah_scraper"


class TestFailureInjection:
    """Emit failure must not fail acquire — documented invariant for all
    connectors. We exercise the invariant on one representative connector
    (lk_corpus) since the emit helper is shared code."""

    def test_emit_failure_does_not_fail_lk_corpus_run(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        csvs = [f"book{i}.csv" for i in range(6)]

        def fake_clone_repo(url: str, dest: Path, **kw: Any) -> Path:
            _seed_csvs(dest, csvs)
            return dest

        # If the emit helper raised, acquire would bubble it. But the helper
        # is contractually no-raise — the sentinel here is that a mock that
        # returns 0 (total failure) still lets run() return normally.
        with (
            patch("src.acquire.lk_corpus.clone_repo", side_effect=fake_clone_repo),
            patch(
                "src.acquire.lk_corpus.emit_raw_new_for_manifest",
                return_value=0,
            ) as emit,
        ):
            from src.acquire import lk_corpus

            result = lk_corpus.run(raw)

        assert result == raw / "lk"
        emit.assert_called_once()

    def test_producer_exception_is_swallowed_by_emit_raw_new(self) -> None:
        """Direct contract check: even if the injected producer explodes on
        send, emit_raw_new returns False without raising."""
        from src.messaging.kafka_producer import emit_raw_new

        broken = MagicMock()
        broken.send.side_effect = RuntimeError("broker down")
        ok = emit_raw_new(
            source="lk_corpus",
            b2_key="raw/lk_corpus/2026-04-21/a.csv",
            content_type="text/csv",
            size_bytes=1,
            checksum_sha256="x",
            producer=broken,
        )
        assert ok is False
