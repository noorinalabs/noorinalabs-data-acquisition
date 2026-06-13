"""Tests for the MIS acquire downloader (da#97) — no network.

The Mendeley files API and the file downloads are stubbed; the test asserts the
downloader enumerates the file set, fetches **only** the spreadsheet workbooks
(skipping non-.xlsx artefacts), and fails fast when no workbook is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.acquire import mis

_FILES_RESPONSE = [
    {
        "filename": "Hadith_SahihMuslim_CoreInfo.xlsx",
        "content_details": {
            "download_url": "https://data.mendeley.com/f/core",
            "sha256_hash": None,
        },
    },
    {
        "filename": "Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.xlsx",
        "content_details": {
            "download_url": "https://data.mendeley.com/f/detail",
            "sha256_hash": None,
        },
    },
    {
        "filename": "README.pdf",
        "content_details": {"download_url": "https://data.mendeley.com/f/readme"},
    },
]


@pytest.fixture
def fake_mendeley(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the files API + downloads; record requested download URLs."""
    requested: list[str] = []

    def _fetch_json(url: str, **_kw: object) -> object:
        assert "gzprcr93zn" in url
        return _FILES_RESPONSE

    def _download(url: str, dest: Path, **_kw: object) -> Path:
        requested.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PK\x03\x04 fake-xlsx")
        return dest

    monkeypatch.setattr(mis, "fetch_json", _fetch_json)
    monkeypatch.setattr(mis, "download_file", _download)
    monkeypatch.setattr(mis, "emit_raw_new_for_manifest", lambda **_kw: None)
    return requested


def test_downloads_only_xlsx_workbooks(fake_mendeley: list[str], tmp_path: Path) -> None:
    dest = mis.run(tmp_path / "raw")

    assert dest == tmp_path / "raw" / "mis"
    # Both workbooks fetched; the PDF is NOT.
    assert any(u.endswith("/core") for u in fake_mendeley)
    assert any(u.endswith("/detail") for u in fake_mendeley)
    assert not any(u.endswith("/readme") for u in fake_mendeley)

    assert (dest / "Hadith_SahihMuslim_CoreInfo.xlsx").exists()
    assert (dest / "Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.xlsx").exists()
    # write_manifest output present.
    assert (dest / "manifest.json").exists()


def test_raises_when_no_workbook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mis, "fetch_json", lambda *_a, **_kw: [{"filename": "README.pdf"}])
    monkeypatch.setattr(mis, "emit_raw_new_for_manifest", lambda **_kw: None)

    with pytest.raises(AssertionError, match="no .xlsx"):
        mis.run(tmp_path / "raw")
