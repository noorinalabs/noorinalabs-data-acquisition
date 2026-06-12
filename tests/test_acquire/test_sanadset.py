"""Tests for the Sanadset acquire-side downloader (da#89).

Closes the acquire-side coverage gap: before this, only ``base`` and
``sunnah_scraper`` had acquire-layer tests. Network access (Mendeley, Kaggle) is
fully mocked — these assert the downloader's *orchestration* logic: idempotent
skip-when-present, the Mendeley-primary path, Kaggle credential gating, row-count
validation, and manifest/messaging emission.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.acquire.sanadset import (
    _count_csv_rows,
    _kaggle_credentials_available,
    download_sanadset,
)


def _fake_download(url: str, path: Path, **_kwargs: object) -> None:
    """Stand-in for ``download_file`` — materializes a tiny CSV at *path*."""
    path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")


def _settings(*, kaggle_username: str | None = None, kaggle_key: str | None = None) -> MagicMock:
    s = MagicMock()
    s.kaggle_username = kaggle_username
    s.kaggle_key = kaggle_key
    return s


class TestCountCsvRows:
    def test_counts_data_rows_excluding_headers(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text("h1,h2\n1,2\n3,4\n", encoding="utf-8")
        (tmp_path / "b.csv").write_text("h\nx\n", encoding="utf-8")
        # (2 data rows) + (1 data row) = 3
        assert _count_csv_rows(tmp_path) == 3


class TestKaggleCredentials:
    def test_true_from_settings(self) -> None:
        with patch(
            "src.acquire.sanadset.get_settings",
            return_value=_settings(kaggle_username="u", kaggle_key="k"),
        ):
            assert _kaggle_credentials_available() is True

    def test_false_when_absent(self, tmp_path: Path) -> None:
        with (
            patch("src.acquire.sanadset.get_settings", return_value=_settings()),
            patch("src.acquire.sanadset.Path.home", return_value=tmp_path),
        ):
            assert _kaggle_credentials_available() is False

    def test_true_from_kaggle_json(self, tmp_path: Path) -> None:
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text(
            json.dumps({"username": "u", "key": "k"}), encoding="utf-8"
        )
        with (
            patch("src.acquire.sanadset.get_settings", return_value=_settings()),
            patch("src.acquire.sanadset.Path.home", return_value=tmp_path),
        ):
            assert _kaggle_credentials_available() is True


class TestDownloadSanadset:
    def test_downloads_mendeley_when_empty(self, tmp_path: Path) -> None:
        dest = tmp_path / "sanadset"
        with (
            patch("src.acquire.sanadset.get_settings", return_value=_settings()),
            patch("src.acquire.sanadset.Path.home", return_value=tmp_path / "nohome"),
            patch("src.acquire.sanadset.download_file", side_effect=_fake_download) as dl,
            patch("src.acquire.sanadset.write_manifest") as manifest,
            patch("src.acquire.sanadset.emit_raw_new_for_manifest") as emit,
        ):
            out = download_sanadset(dest=dest)

        assert out == dest
        # Both Mendeley files (sanadset.csv, books.csv) were fetched.
        assert dl.call_count == 2
        manifest.assert_called_once()
        emit.assert_called_once()

    def test_skips_mendeley_when_csv_present(self, tmp_path: Path) -> None:
        dest = tmp_path / "sanadset"
        dest.mkdir()
        (dest / "sanadset.csv").write_text("h\n1\n", encoding="utf-8")
        with (
            patch("src.acquire.sanadset.get_settings", return_value=_settings()),
            patch("src.acquire.sanadset.Path.home", return_value=tmp_path / "nohome"),
            patch("src.acquire.sanadset.download_file", side_effect=_fake_download) as dl,
            patch("src.acquire.sanadset.write_manifest"),
            patch("src.acquire.sanadset.emit_raw_new_for_manifest"),
        ):
            download_sanadset(dest=dest)

        # Idempotent: existing CSVs short-circuit the Mendeley download.
        dl.assert_not_called()

    def test_narrators_skipped_without_credentials(self, tmp_path: Path) -> None:
        dest = tmp_path / "sanadset"
        with (
            patch("src.acquire.sanadset.get_settings", return_value=_settings()),
            patch("src.acquire.sanadset.Path.home", return_value=tmp_path / "nohome"),
            patch("src.acquire.sanadset.download_file", side_effect=_fake_download),
            patch("src.acquire.sanadset.write_manifest"),
            patch("src.acquire.sanadset.emit_raw_new_for_manifest"),
            patch("src.acquire.sanadset._run_kaggle_download") as kaggle,
        ):
            download_sanadset(dest=dest)

        # No Kaggle creds → narrator-bio download is never attempted.
        kaggle.assert_not_called()

    def test_narrators_downloaded_with_credentials(self, tmp_path: Path) -> None:
        dest = tmp_path / "sanadset"
        with (
            patch(
                "src.acquire.sanadset.get_settings",
                return_value=_settings(kaggle_username="u", kaggle_key="k"),
            ),
            patch("src.acquire.sanadset.download_file", side_effect=_fake_download),
            patch("src.acquire.sanadset.write_manifest"),
            patch("src.acquire.sanadset.emit_raw_new_for_manifest"),
            patch("src.acquire.sanadset._run_kaggle_download") as kaggle,
        ):
            download_sanadset(dest=dest)

        # Credentials present → narrator-bio dataset is fetched into narrators/.
        kaggle.assert_called_once()
